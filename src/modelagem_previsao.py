"""Previsão multivariada, validação walk-forward (horizonte 1-3 meses) e avaliação."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
FIGURE_DIR = ROOT / "figuras"
TARGET = "variacao_ipca_mensal_pct"
EXOG_FEATURES = [
    "selic_efetiva_aa",
    "cambio_usd_brl",
    "ic_br",
    "ibc_br_sa",
    "desocupacao",
]
TARGET_EXOG_LAGS = [
    "selic_d1_lag3",
    "ic_d1_lag11",
    "ibc_d1_lag4",
    "desoc_d1_lag4",
]
ANCHOR_FOCUS_WEIGHT = 0.75
ANCHOR_SEASONAL_WEIGHT = 0.25
HORIZONS = [1, 2, 3]
TEST_MONTHS = 12
LSTM_LOOKBACK = 12


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.output(sequence[:, -1, :]).squeeze(-1)


def _load_base() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "painel_ipca_macro.csv", parse_dates=["data"])
    macro = pd.read_csv(DATA_DIR / "macro_mensal.csv", parse_dates=["data"])
    series = (
        panel.groupby("data", as_index=False)[TARGET]
        .median()
        .rename(columns={TARGET: "ipca_mensal_median"})
    )
    merged = series.merge(macro, on="data", how="inner")
    sentiment_path = DATA_DIR / "sentimento_mensal.csv"
    if sentiment_path.exists():
        sentiment = pd.read_csv(
            sentiment_path, parse_dates=["data"], encoding="utf-8-sig"
        )
        if "escore_llm" in sentiment and sentiment["escore_llm"].notna().any():
            sentiment = sentiment.sort_values("data")
            merged = merged.merge(
                sentiment[["data", "escore_llm", "n_documentos_llm"]],
                on="data",
                how="left",
                validate="one_to_one",
            )
            merged["sentimento_llm_lag1"] = (
                merged["escore_llm"].shift(1).ffill()
            )
            merged["n_documentos_llm_lag1"] = (
                merged["n_documentos_llm"].shift(1).fillna(0)
            )
    return merged.sort_values("data").reset_index(drop=True)


def _load_focus_benchmark() -> pd.DataFrame:
    """Carrega mediana do Focus por mês de coleta (origem) e mês de referência."""
    focus = pd.read_csv(
        DATA_DIR / "focus_expectativas.csv",
        parse_dates=["data_coleta", "mes_referencia"],
        encoding="utf-8-sig",
    )
    focus = focus.loc[focus["base_calculo"] == 0].copy()
    focus["origem_mes"] = focus["data_coleta"].dt.to_period("M").dt.to_timestamp()
    focus = (
        focus.sort_values("data_coleta")
        .drop_duplicates(["origem_mes", "mes_referencia"], keep="last")
        .reset_index(drop=True)
    )
    return focus[["origem_mes", "mes_referencia", "mediana_pct"]]


def _focus_lookup(focus: pd.DataFrame) -> dict[tuple[pd.Timestamp, pd.Timestamp], float]:
    """Indexa mediana do Focus por (mês de origem, mês de referência)."""
    return focus.set_index(["origem_mes", "mes_referencia"])["mediana_pct"].to_dict()


def _attach_historical_focus_h1(
    df: pd.DataFrame,
    focus_lookup: dict[tuple[pd.Timestamp, pd.Timestamp], float],
) -> pd.Series:
    """Expectativa Focus coletada em tau - 1 mês para o mês tau (sem vazamento)."""
    values = [
        focus_lookup.get((tau - pd.DateOffset(months=1), tau), np.nan)
        for tau in df["data"]
    ]
    return pd.Series(values, index=df.index, dtype=float)


def _compute_anchor(
    focus_val: pd.Series | np.ndarray | float,
    seasonal_val: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """Combina expectativa Focus (75%) e âncora sazonal y_{t-12} (25%)."""
    return ANCHOR_FOCUS_WEIGHT * focus_val + ANCHOR_SEASONAL_WEIGHT * seasonal_val


def _build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["ipca_mensal_median"] = working["ipca_mensal_median"].astype(float)
    # Propaga somente exógenas passadas (sem olhar o futuro) para não descartar 2026-09 por ibc_br_sa nulo
    for feature in EXOG_FEATURES:
        working[feature] = working[feature].ffill()
    for lag in [1, 3, 6, 12]:
        working[f"ipca_lag_{lag}"] = working["ipca_mensal_median"].shift(lag)
        for feature in EXOG_FEATURES:
            working[f"{feature}_lag_{lag}"] = working[feature].shift(lag)
    working["ipca_mm_3m"] = working["ipca_mensal_median"].rolling(3).mean()
    working["ipca_mm_6m"] = working["ipca_mensal_median"].rolling(6).mean()
    working["ipca_std_3m"] = working["ipca_mensal_median"].rolling(3).std()
    for feature in EXOG_FEATURES:
        working[f"{feature}_diff1"] = working[feature].diff(1)
        working[f"{feature}_mm_3m"] = working[feature].rolling(3).mean()
    # Defasagens >= 3 alinhadas ao mês-alvo guiadas pela Etapa 2 (conclusao_selecao_variaveis.csv)
    working["y_lag12"] = working["ipca_mensal_median"].shift(12)
    working["selic_d1_lag3"] = working["selic_efetiva_aa_diff1"].shift(3)
    working["cambio_d1_lag3"] = working["cambio_usd_brl_diff1"].shift(3)
    working["ic_d1_lag11"] = working["ic_br_diff1"].shift(11)
    working["ibc_d1_lag4"] = working["ibc_br_sa_diff1"].shift(4)
    working["desoc_d1_lag4"] = working["desocupacao_diff1"].shift(4)
    return working


def _feature_columns_at_origin(df: pd.DataFrame) -> list[str]:
    """Colunas conhecidas no momento da origem t para prever t+h."""
    cols = [
        "ipca_mensal_median",
        "ipca_mm_3m",
        "ipca_mm_6m",
        "ipca_std_3m",
        *EXOG_FEATURES,
        *[f"{f}_diff1" for f in EXOG_FEATURES],
        *[f"{f}_mm_3m" for f in EXOG_FEATURES],
    ]
    for lag in [1, 3, 6, 12]:
        cols.append(f"ipca_lag_{lag}")
        for feature in EXOG_FEATURES:
            cols.append(f"{feature}_lag_{lag}")
    return [c for c in cols if c in df.columns]


def _as_float(series: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(series, dtype=float)


def _metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, naive_scale: float | None = None) -> dict[str, float]:
    actual_arr = _as_float(actual)
    pred_arr = _as_float(predicted)
    mask = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    actual_arr = actual_arr[mask]
    pred_arr = pred_arr[mask]
    if len(actual_arr) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape_pct": np.nan, "smape_pct": np.nan, "mase": np.nan}
    error = actual_arr - pred_arr
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom_mape = np.clip(np.abs(actual_arr), 1e-6, None)
    mape = float(np.mean(np.abs(error) / denom_mape) * 100.0)
    denom_smape = np.clip(np.abs(actual_arr) + np.abs(pred_arr), 1e-6, None)
    smape = float(np.mean(2.0 * np.abs(error) / denom_smape) * 100.0)
    if naive_scale is None or not np.isfinite(naive_scale) or naive_scale <= 0:
        naive_denom = np.mean(np.abs(np.diff(actual_arr))) if len(actual_arr) > 1 else np.nan
    else:
        naive_denom = naive_scale
    mase = float(mae / naive_denom) if np.isfinite(naive_denom) and naive_denom > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "smape_pct": smape,
        "mase": mase,
    }


def _diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Teste de Diebold-Mariano bilateral com variância HAC (Bartlett) para horizonte h."""
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)
    if n < 5:
        return float("nan"), float("nan")
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = float(np.mean(d))
    d_centered = d - d_mean
    gamma0 = float(np.mean(d_centered ** 2))
    var = gamma0
    for k in range(1, min(h, n)):
        cov_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
        weight = 1.0 - k / float(h)
        var += 2.0 * weight * cov_k
    var = max(var, 1e-12)
    dm_stat = d_mean / np.sqrt(var / n)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1)))
    return float(dm_stat), p_val


def _fit_lstm_and_predict(
    df: pd.DataFrame,
    split_index: int,
    focus: pd.DataFrame | None = None,
) -> dict[tuple[int, int], float]:
    """Treina LSTM regularizada sobre resíduo da âncora até split_index e prevê h=1,2,3."""
    focus_df = focus if focus is not None else _load_focus_benchmark()
    focus_lookup = _focus_lookup(focus_df)

    work = df.copy()
    work["y_prev"] = work["ipca_mensal_median"].shift(1)
    work["focus_tau"] = _attach_historical_focus_h1(work, focus_lookup)
    work["anchor"] = _compute_anchor(work["focus_tau"], work["y_lag12"])
    work["resid"] = work["ipca_mensal_median"] - work["anchor"]

    seq_cols = ["y_prev", "focus_tau", "y_lag12", *TARGET_EXOG_LAGS]
    clean = work[["data", "ipca_mensal_median", "anchor", "resid", *seq_cols]].dropna().reset_index(drop=True)
    split_date = df.iloc[split_index]["data"]
    n_train = int((clean["data"] < split_date).sum())
    if n_train <= LSTM_LOOKBACK:
        return {}

    train_x_raw = clean.iloc[:n_train][seq_cols].to_numpy(dtype=np.float32)
    input_mean = train_x_raw.mean(axis=0)
    input_scale = train_x_raw.std(axis=0)
    input_scale[input_scale == 0] = 1.0

    resid_tr = clean.iloc[:n_train]["resid"].to_numpy(dtype=np.float32)
    r_mean = float(resid_tr.mean())
    r_std = float(resid_tr.std())
    if r_std == 0.0:
        r_std = 1.0

    all_x_scaled = (clean[seq_cols].to_numpy(dtype=np.float32) - input_mean) / input_scale
    sequences = np.stack(
        [all_x_scaled[i - LSTM_LOOKBACK + 1 : i + 1] for i in range(LSTM_LOOKBACK - 1, n_train)]
    )
    labels = (resid_tr[LSTM_LOOKBACK - 1 : n_train] - r_mean) / r_std

    torch.manual_seed(42)
    model = _LSTMRegressor(input_size=len(seq_cols), hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.25)
    loss_fn = nn.MSELoss()
    train_x = torch.as_tensor(sequences, dtype=torch.float32)
    train_y = torch.as_tensor(labels, dtype=torch.float32)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    date_to_cidx = {d: idx for idx, d in enumerate(clean["data"])}
    preds: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for origin_idx in range(split_index - 1, len(df) - 1):
            origin_date = df.iloc[origin_idx]["data"]
            c_idx = date_to_cidx.get(origin_date)
            if c_idx is None or c_idx < LSTM_LOOKBACK - 1:
                continue
            window = all_x_scaled[c_idx - LSTM_LOOKBACK + 1 : c_idx + 1].copy()
            prev_y = float(df.iloc[origin_idx]["ipca_mensal_median"])
            for h in HORIZONS:
                t_idx = origin_idx + h
                if t_idx >= len(df):
                    continue
                t_date = df.iloc[t_idx]["data"]
                focus_p = focus_lookup.get((origin_date, t_date), np.nan)
                y_l12 = float(df.iloc[t_idx]["y_lag12"])
                next_raw = np.array(
                    [
                        prev_y,
                        float(focus_p),
                        y_l12,
                        *[float(df.iloc[t_idx][col]) for col in TARGET_EXOG_LAGS],
                    ],
                    dtype=np.float32,
                )
                if not np.all(np.isfinite(next_raw)):
                    continue
                next_scaled = (next_raw - input_mean) / input_scale
                window = np.vstack([window[1:], next_scaled])
                seq = torch.as_tensor(window[None, :, :], dtype=torch.float32)
                pred_resid = float(model(seq).item()) * r_std + r_mean
                pred_val = float(_compute_anchor(float(focus_p), y_l12) + pred_resid)
                preds[(origin_idx, h)] = pred_val
                prev_y = pred_val
    return preds


def _walk_forward_monthly(df: pd.DataFrame, focus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa validação walk-forward (origem móvel) para horizontes h=1,2,3 meses."""
    split_index = len(df) - TEST_MONTHS
    focus_lookup = _focus_lookup(focus)
    df_work = df.copy()
    df_work["focus_h1_self"] = _attach_historical_focus_h1(df_work, focus_lookup)

    # Escala do MASE baseada no naïve sazonal in-sample do treino
    train_y = df_work.iloc[:split_index]["ipca_mensal_median"].dropna()
    naive_scale = float(np.mean(np.abs(train_y.diff(12).dropna())))

    lstm_preds = _fit_lstm_and_predict(df_work, split_index, focus=focus)

    exog_tau_cols = ["focus_tau", "y_lag12", "perfil_mes", *TARGET_EXOG_LAGS]
    var_endog_cols = ["ipca_mensal_median", *[f"{f}_diff1" for f in EXOG_FEATURES]]
    var_exog_cols = ["focus_tau", "y_lag12", "perfil_mes"]
    direct_cols = ["focus_tau", "y_lag12", "perfil_mes", "ipca_mensal_median", "ipca_mm_3m", *TARGET_EXOG_LAGS]

    records: list[dict[str, object]] = []
    # Origens móveis: de (split_index - 1) até (len(df_work) - 2)
    for origin_idx in range(split_index - 1, len(df_work) - 1):
        origin_date = df_work.iloc[origin_idx]["data"]
        train_slice = df_work.iloc[: origin_idx + 1].copy()
        month_medians = (
            train_slice.groupby(train_slice["data"].dt.month)["ipca_mensal_median"]
            .median()
            .to_dict()
        )
        global_median = float(train_slice["ipca_mensal_median"].median())
        train_slice["focus_tau"] = train_slice["focus_h1_self"]
        train_slice["perfil_mes"] = (
            train_slice["data"].dt.month.map(month_medians).fillna(global_median)
        )

        fut_rows: list[tuple[int, int, pd.Timestamp, pd.Series]] = []
        for h in HORIZONS:
            t_idx = origin_idx + h
            if t_idx < len(df_work):
                t_date = df_work.iloc[t_idx]["data"]
                row_tau = df_work.iloc[t_idx].copy()
                row_tau["focus_tau"] = focus_lookup.get((origin_date, t_date), np.nan)
                row_tau["perfil_mes"] = float(month_medians.get(t_date.month, global_median))
                fut_rows.append((h, t_idx, t_date, row_tau))

        # 1) SARIMAX na origem com exógenas alinhadas ao mês-alvo (observadas até origin_idx)
        sarimax_preds: dict[int, float] = {}
        s_frame = train_slice[["ipca_mensal_median", *exog_tau_cols]].dropna()
        if len(s_frame) >= 36 and fut_rows:
            try:
                fut_exog = pd.DataFrame([r[3][exog_tau_cols] for r in fut_rows]).astype(float)
                if not fut_exog.isna().any(axis=None):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        s_model = SARIMAX(
                            s_frame["ipca_mensal_median"].astype(float),
                            exog=s_frame[exog_tau_cols].astype(float),
                            order=(1, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    s_fc = s_model.forecast(steps=len(fut_rows), exog=fut_exog)
                    for idx_h, (h, _, _, _) in enumerate(fut_rows):
                        sarimax_preds[h] = float(s_fc.iloc[idx_h])
            except Exception:
                pass

        # 2) VAR na origem (primeiras diferenças das macroeconômicas + exógenas sazonais/Focus)
        var_preds: dict[int, float] = {}
        var_frame = train_slice[[*var_endog_cols, *var_exog_cols]].dropna()
        if len(var_frame) >= 36 and fut_rows:
            try:
                fut_vexog = pd.DataFrame([r[3][var_exog_cols] for r in fut_rows]).to_numpy(dtype=float)
                if np.all(np.isfinite(fut_vexog)):
                    v_fit = VAR(
                        endog=var_frame[var_endog_cols].astype(float),
                        exog=var_frame[var_exog_cols].astype(float),
                    ).fit(1)
                    v_fc = v_fit.forecast(
                        y=var_frame[var_endog_cols].tail(v_fit.k_ar).to_numpy(dtype=float),
                        steps=len(fut_rows),
                        exog_future=fut_vexog,
                    )
                    for idx_h, (h, _, _, _) in enumerate(fut_rows):
                        var_preds[h] = float(v_fc[idx_h, 0])
            except Exception:
                pass

        # 3) Modelos diretos por horizonte h (RidgeCV e Random Forest residual sobre âncora)
        for h, target_idx, target_date, row_tau in fut_rows:
            actual_val = float(df_work.iloc[target_idx]["ipca_mensal_median"])

            naive_idx = target_idx - 12
            naive_pred = float(df_work.iloc[naive_idx]["ipca_mensal_median"]) if naive_idx >= 0 else np.nan
            focus_pred = float(focus_lookup.get((origin_date, target_date), np.nan))

            # Treino direto apenas com origens i <= origin_idx - h (i + h <= origin_idx, zero vazamento)
            valid_end = origin_idx - h + 1
            ridge_pred = np.nan
            rf_pred = np.nan
            if valid_end >= 36:
                tr_h = df_work.iloc[:valid_end].copy()
                tr_h["target_h"] = df_work["ipca_mensal_median"].shift(-h).iloc[:valid_end]
                tr_h["focus_tau"] = [
                    focus_lookup.get((df_work.loc[i, "data"], df_work.loc[i + h, "data"]), np.nan)
                    for i in range(valid_end)
                ]
                for col in ["y_lag12", *TARGET_EXOG_LAGS]:
                    tr_h[col] = df_work[col].shift(-h).iloc[:valid_end]
                tr_h["perfil_mes"] = (
                    df_work["data"].shift(-h).iloc[:valid_end].dt.month.map(month_medians).fillna(global_median)
                )
                tr_h["anchor"] = _compute_anchor(tr_h["focus_tau"], tr_h["y_lag12"])
                tr_h["resid_target"] = tr_h["target_h"] - tr_h["anchor"]
                direct_train = tr_h[[*direct_cols, "target_h", "resid_target", "anchor"]].dropna()

                x_te_df = pd.DataFrame(
                    [
                        {
                            "focus_tau": focus_pred,
                            "y_lag12": float(row_tau["y_lag12"]),
                            "perfil_mes": float(row_tau["perfil_mes"]),
                            "ipca_mensal_median": float(df_work.iloc[origin_idx]["ipca_mensal_median"]),
                            "ipca_mm_3m": float(df_work.iloc[origin_idx]["ipca_mm_3m"]),
                            **{col: float(row_tau[col]) for col in TARGET_EXOG_LAGS},
                        }
                    ]
                )[direct_cols]
                anchor_te = float(_compute_anchor(focus_pred, float(row_tau["y_lag12"])))

                if (
                    len(direct_train) >= 36
                    and not x_te_df.isna().any(axis=None)
                    and np.isfinite(anchor_te)
                ):
                    x_tr = direct_train[direct_cols].to_numpy(dtype=float)
                    y_tr = direct_train["target_h"].to_numpy(dtype=float)
                    y_resid_tr = direct_train["resid_target"].to_numpy(dtype=float)
                    x_te = x_te_df.to_numpy(dtype=float)

                    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 3, 25)))
                    ridge.fit(x_tr, y_tr)
                    ridge_pred = float(ridge.predict(x_te)[0])

                    rf = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=3,
                        min_samples_leaf=8,
                        random_state=42,
                        n_jobs=-1,
                    )
                    rf.fit(x_tr, y_resid_tr)
                    rf_pred = float(anchor_te + rf.predict(x_te)[0])

            model_outputs = {
                "baseline_naive_sazonal": naive_pred,
                "focus_mediana": focus_pred,
                "regressao_ridge": ridge_pred,
                "sarimax_exogeno": sarimax_preds.get(h, np.nan),
                "var_multivariado": var_preds.get(h, np.nan),
                "random_forest": rf_pred,
                "lstm_multivariado": lstm_preds.get((origin_idx, h), np.nan),
            }
            for model_name, pred_val in model_outputs.items():
                records.append(
                    {
                        "modelo": model_name,
                        "data_origem": origin_date,
                        "data_alvo": target_date,
                        "horizonte_meses": h,
                        "observado": actual_val,
                        "previsto": pred_val,
                        "erro": actual_val - pred_val if np.isfinite(pred_val) else np.nan,
                    }
                )

    wf_df = pd.DataFrame(records).sort_values(["modelo", "horizonte_meses", "data_alvo"]).reset_index(drop=True)

    # Calcula intervalos de previsão empíricos (80% e 95%) por modelo e horizonte via resíduos passados
    wf_df["ic80_inf"] = np.nan
    wf_df["ic80_sup"] = np.nan
    wf_df["ic95_inf"] = np.nan
    wf_df["ic95_sup"] = np.nan
    for (model_name, h), grp in wf_df.groupby(["modelo", "horizonte_meses"]):
        idxs = grp.index.tolist()
        abs_errors: list[float] = []
        for pos, idx in enumerate(idxs):
            if pos >= 3:
                q80 = float(np.quantile(abs_errors, 0.80))
                q95 = float(np.quantile(abs_errors, 0.95))
            else:
                q80 = 1.2816 * naive_scale * np.sqrt(h)
                q95 = 1.9600 * naive_scale * np.sqrt(h)
            pred_v = wf_df.loc[idx, "previsto"]
            if np.isfinite(pred_v):
                wf_df.loc[idx, "ic80_inf"] = pred_v - q80
                wf_df.loc[idx, "ic80_sup"] = pred_v + q80
                wf_df.loc[idx, "ic95_inf"] = pred_v - q95
                wf_df.loc[idx, "ic95_sup"] = pred_v + q95
            err_v = wf_df.loc[idx, "erro"]
            if np.isfinite(err_v):
                abs_errors.append(abs(float(err_v)))

    # Métricas consolidadas e por horizonte
    eval_rows = []
    for model_name, grp in wf_df.groupby("modelo", sort=False):
        m = _metrics(grp["observado"], grp["previsto"], naive_scale=naive_scale)
        eval_rows.append(
            {
                "modelo": model_name,
                "protocolo": "walk-forward origem móvel (h=1,2,3)",
                "n_previsoes": int(grp["previsto"].notna().sum()),
                "periodo_teste_inicio": grp["data_alvo"].min().date(),
                "periodo_teste_fim": grp["data_alvo"].max().date(),
                **m,
            }
        )
    eval_all = pd.DataFrame(eval_rows).sort_values("rmse").reset_index(drop=True)

    eval_h_rows = []
    for (model_name, h), grp in wf_df.groupby(["modelo", "horizonte_meses"], sort=False):
        m = _metrics(grp["observado"], grp["previsto"], naive_scale=naive_scale)
        eval_h_rows.append(
            {
                "modelo": model_name,
                "horizonte_meses": h,
                "n_previsoes": int(grp["previsto"].notna().sum()),
                **m,
            }
        )
    eval_by_h = pd.DataFrame(eval_h_rows).sort_values(["horizonte_meses", "rmse"]).reset_index(drop=True)

    # Diebold-Mariano: compara os 2 melhores modelos próprios e o melhor contra Focus
    own_models = [m for m in eval_all["modelo"] if m not in {"focus_mediana", "baseline_naive_sazonal"}]
    dm_rows = []
    pairs: list[tuple[str, str]] = []
    if len(own_models) >= 2:
        pairs.append((own_models[0], own_models[1]))
    if len(own_models) >= 1:
        pairs.append((own_models[0], "focus_mediana"))
        pairs.append((own_models[0], "baseline_naive_sazonal"))

    for m1, m2 in pairs:
        for h in HORIZONS:
            g1 = wf_df.loc[(wf_df["modelo"] == m1) & (wf_df["horizonte_meses"] == h)].set_index("data_alvo")["erro"]
            g2 = wf_df.loc[(wf_df["modelo"] == m2) & (wf_df["horizonte_meses"] == h)].set_index("data_alvo")["erro"]
            aligned = pd.concat([g1, g2], axis=1, keys=["e1", "e2"]).dropna()
            dm_sq, p_sq = _diebold_mariano(aligned["e1"].to_numpy(), aligned["e2"].to_numpy(), h=h, power=2)
            dm_abs, p_abs = _diebold_mariano(aligned["e1"].to_numpy(), aligned["e2"].to_numpy(), h=h, power=1)
            dm_rows.append(
                {
                    "modelo_1": m1,
                    "modelo_2": m2,
                    "horizonte_meses": h,
                    "n_pares": len(aligned),
                    "dm_stat_mse": round(dm_sq, 4),
                    "p_valor_mse": round(p_sq, 4),
                    "dm_stat_mae": round(dm_abs, 4),
                    "p_valor_mae": round(p_abs, 4),
                }
            )
    dm_df = pd.DataFrame(dm_rows)
    return wf_df, eval_all, eval_by_h, dm_df


def _panel_forecast_evaluation(focus: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia previsão walk-forward (h=1,2,3) no painel completo (área × grupo)."""
    panel = pd.read_csv(DATA_DIR / "painel_ipca_macro.csv", parse_dates=["data"])
    panel = panel.sort_values(["area_id", "grupo_id", "data"]).reset_index(drop=True)
    focus_df = focus if focus is not None else _load_focus_benchmark()
    focus_lookup = _focus_lookup(focus_df)

    for lag in [1, 3, 6, 12]:
        panel[f"ipca_lag_{lag}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(lag)
    panel["ipca_mm_3m"] = (
        panel.groupby(["area_id", "grupo_id"])[TARGET].transform(lambda s: s.rolling(3).mean())
    )
    for feat in EXOG_FEATURES:
        panel[f"{feat}_diff1"] = (
            panel.groupby(["area_id", "grupo_id"])[feat].transform(lambda s: s.ffill().diff(1))
        )

    unique_dates = sorted(panel["data"].unique())
    split_date = unique_dates[-TEST_MONTHS]
    test_origins = unique_dates[-TEST_MONTHS - 1 : -1]

    # Perfil sazonal local calculado exclusivamente no histórico anterior a split_date (zero vazamento)
    train_hist = panel.loc[panel["data"] < split_date].copy()
    train_hist["mes"] = train_hist["data"].dt.month
    local_saz = train_hist.groupby(["area_id", "grupo_id", "mes"])[TARGET].median().to_dict()
    grupo_saz = train_hist.groupby(["grupo_id", "mes"])[TARGET].median().to_dict()
    global_saz = float(train_hist[TARGET].median())

    num_base_features = [
        TARGET,
        "ipca_lag_1",
        "ipca_lag_3",
        "ipca_lag_6",
        "ipca_lag_12",
        "ipca_mm_3m",
        *[f"{f}_diff1" for f in EXOG_FEATURES],
    ]
    cat_features = ["area_id", "grupo_id"]

    rows: list[dict[str, object]] = []
    for h in HORIZONS:
        panel[f"target_h{h}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(-h)
        panel[f"target_date_h{h}"] = panel.groupby(["area_id", "grupo_id"])["data"].shift(-h)
        panel[f"naive_h{h}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(12 - h)

        t_months = panel[f"target_date_h{h}"].dt.month
        panel[f"local_saz_h{h}"] = [
            local_saz.get((a, g, m), grupo_saz.get((g, m), global_saz))
            for a, g, m in zip(panel["area_id"], panel["grupo_id"], t_months)
        ]
        panel[f"focus_h{h}"] = [
            focus_lookup.get((d, td), np.nan)
            for d, td in zip(panel["data"], panel[f"target_date_h{h}"])
        ]

        h_num_features = [*num_base_features, f"naive_h{h}", f"local_saz_h{h}", f"focus_h{h}"]
        panel_features = [*cat_features, *h_num_features]

        # Garante que tanto a origem quanto a data-alvo de treino sejam estritamente anteriores a split_date
        train_df = panel.loc[
            (panel["data"] < split_date) & (panel[f"target_date_h{h}"] < split_date),
            [*panel_features, f"target_h{h}"],
        ].dropna()
        x_tr = train_df[panel_features]
        y_tr = train_df[f"target_h{h}"].to_numpy(dtype=float)

        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
                ("num", StandardScaler(), h_num_features),
            ]
        )
        ridge = make_pipeline(preprocessor, RidgeCV(alphas=np.logspace(0, 3, 15)))
        ridge.fit(x_tr, y_tr)

        rf = make_pipeline(
            preprocessor,
            RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1,
            ),
        )
        rf.fit(x_tr, y_tr)

        eval_slice = panel.loc[
            panel["data"].isin(test_origins) & panel[f"target_h{h}"].notna()
        ].copy()
        valid_mask = eval_slice[panel_features].notna().all(axis=1)
        eval_valid = eval_slice.loc[valid_mask].copy()
        x_te = eval_valid[panel_features]
        eval_valid["pred_ridge"] = ridge.predict(x_te)
        eval_valid["pred_rf"] = rf.predict(x_te)

        for _, r in eval_valid.iterrows():
            for model_name, col_pred in [
                ("baseline_naive_sazonal_painel", f"naive_h{h}"),
                ("ridge_painel", "pred_ridge"),
                ("random_forest_painel", "pred_rf"),
            ]:
                rows.append(
                    {
                        "modelo": model_name,
                        "area_id": r["area_id"],
                        "area": r["area"],
                        "grupo_id": r["grupo_id"],
                        "grupo_despesa": r["grupo_despesa"],
                        "data_origem": r["data"],
                        "data_alvo": r[f"target_date_h{h}"],
                        "horizonte_meses": h,
                        "observado": r[f"target_h{h}"],
                        "previsto": r[col_pred],
                    }
                )

    pred_panel = pd.DataFrame(rows)
    eval_rows = []
    for (model_name, h), grp in pred_panel.groupby(["modelo", "horizonte_meses"]):
        m = _metrics(grp["observado"], grp["previsto"])
        eval_rows.append(
            {
                "modelo": model_name,
                "horizonte_meses": h,
                "n_previsoes": len(grp),
                "n_areas": grp["area_id"].nunique(),
                "n_grupos": grp["grupo_id"].nunique(),
                **m,
            }
        )
    eval_panel = pd.DataFrame(eval_rows).sort_values(["horizonte_meses", "rmse"]).reset_index(drop=True)
    return pred_panel, eval_panel


def _plot_forecasts_with_intervals(wf_df: pd.DataFrame, eval_all: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    own_best = [
        m for m in eval_all["modelo"] if m not in {"focus_mediana", "baseline_naive_sazonal"}
    ][:2]
    h1 = wf_df.loc[wf_df["horizonte_meses"] == 1].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    obs = h1.drop_duplicates("data_alvo").sort_values("data_alvo")
    ax.plot(obs["data_alvo"], obs["observado"], color="black", linewidth=2, marker="o", label="Observado (IPCA mediana)")

    focus_h1 = h1.loc[h1["modelo"] == "focus_mediana"].sort_values("data_alvo")
    if not focus_h1.empty:
        ax.plot(focus_h1["data_alvo"], focus_h1["previsto"], linestyle="--", color="tab:gray", label="Focus (mediana)")

    colors = ["tab:blue", "tab:orange"]
    for idx, model_name in enumerate(own_best):
        sub = h1.loc[h1["modelo"] == model_name].sort_values("data_alvo")
        c = colors[idx % len(colors)]
        ax.plot(sub["data_alvo"], sub["previsto"], color=c, marker="s", markersize=4, label=model_name)
        if idx == 0:
            ax.fill_between(
                sub["data_alvo"],
                sub["ic95_inf"],
                sub["ic95_sup"],
                color=c,
                alpha=0.15,
                label=f"IC 95% ({model_name})",
            )
            ax.fill_between(
                sub["data_alvo"],
                sub["ic80_inf"],
                sub["ic80_sup"],
                color=c,
                alpha=0.25,
                label=f"IC 80% ({model_name})",
            )

    ax.set_title("Previsão Walk-Forward (h = 1 mês) com Intervalos de Previsão")
    ax.set_ylabel("Variação mensal do IPCA (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "previsoes_intervalos_modelos.png", dpi=150)
    plt.close(fig)


def _sarimax_walk_forward(
    df: pd.DataFrame, split_index: int, features: list[str]
) -> pd.Series:
    """Issue a one-month forecast at each origin, using only then-known history."""
    forecast_values: list[float] = []
    for current_index in range(split_index, len(df)):
        training = df.iloc[:current_index][
            ["ipca_mensal_median", *features]
        ].dropna()
        current = df.iloc[[current_index]][features]
        if len(training) < 36 or current.isna().any(axis=None):
            forecast_values.append(float("nan"))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                training["ipca_mensal_median"].astype(float),
                exog=training[features].astype(float),
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=1, exog=current.astype(float))
        forecast_values.append(float(forecast.iloc[0]))
    return pd.Series(forecast_values, index=df.index[split_index:], dtype=float)


def _evaluate_sentiment_effect(df: pd.DataFrame) -> pd.DataFrame | None:
    sentiment_features = [
        "sentimento_llm_lag1",
        "n_documentos_llm_lag1",
    ]
    if any(feature not in df for feature in sentiment_features):
        return None

    split_index = len(df) - TEST_MONTHS
    regular_features = [
        column
        for column in df.columns
        if column.startswith("ipca_lag_")
        or any(column.startswith(f"{feature}_lag_") for feature in EXOG_FEATURES)
    ]
    complete_training = df.iloc[:split_index][
        ["ipca_mensal_median", *regular_features, *sentiment_features]
    ].dropna()
    if len(complete_training) < 36:
        return None
    predictions = {
        "sarimax_sem_sentimento": _sarimax_walk_forward(
            df, split_index, regular_features
        ),
        "sarimax_com_sentimento": _sarimax_walk_forward(
            df, split_index, [*regular_features, *sentiment_features]
        ),
    }
    forecast_frame = pd.DataFrame(
        {
            "observado": df.iloc[split_index:]["ipca_mensal_median"].to_numpy(
                dtype=float
            ),
            **{
                model_name: forecast.to_numpy(dtype=float)
                for model_name, forecast in predictions.items()
            },
        }
    ).dropna()
    if forecast_frame.empty:
        return None
    rows = []
    for model_name in predictions:
        rows.append(
            {
                "modelo": model_name,
                "protocolo": "walk-forward, horizonte de 1 mês",
                "n_previsoes": len(forecast_frame),
                "periodo_teste_inicio": df.iloc[
                    split_index + forecast_frame.index.min()
                ]["data"].date(),
                "periodo_teste_fim": df.iloc[
                    split_index + forecast_frame.index.max()
                ]["data"].date(),
                **_metrics(forecast_frame["observado"], forecast_frame[model_name]),
            }
        )
    return pd.DataFrame(rows)


def gerar_modelos() -> pd.DataFrame:
    np.random.seed(42)
    torch.manual_seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _build_lagged_features(_load_base())
    if len(df) < 48:
        raise ValueError("A série deve ter pelo menos 48 meses para validação walk-forward.")
    focus = _load_focus_benchmark()

    wf_df, eval_all, eval_by_h, dm_df = _walk_forward_monthly(df, focus)
    pred_panel, eval_panel = _panel_forecast_evaluation(focus=focus)

    wf_df.to_csv(DATA_DIR / "previsoes_walk_forward.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(DATA_DIR / "avaliacao_modelos.csv", index=False, encoding="utf-8-sig")
    eval_by_h.to_csv(DATA_DIR / "avaliacao_modelos_por_horizonte.csv", index=False, encoding="utf-8-sig")
    dm_df.to_csv(DATA_DIR / "teste_diebold_mariano.csv", index=False, encoding="utf-8-sig")
    pred_panel.to_csv(DATA_DIR / "previsoes_painel_area_grupo.csv", index=False, encoding="utf-8-sig")
    eval_panel.to_csv(DATA_DIR / "avaliacao_painel_area_grupo.csv", index=False, encoding="utf-8-sig")
    _plot_forecasts_with_intervals(wf_df, eval_all)

    sentiment_comparison = _evaluate_sentiment_effect(df)
    sentiment_output = DATA_DIR / "impacto_sentimento_sarimax.csv"
    if sentiment_comparison is not None:
        sentiment_comparison.to_csv(sentiment_output, index=False, encoding="utf-8-sig")
        print("\nComparação SARIMAX com e sem sentimento:")
        print(sentiment_comparison.to_string(index=False))
    else:
        print(
            "\nComparação com sentimento não executada: faltam escores LLM "
            "com cobertura suficiente no treino e no teste."
        )

    print("\nAvaliação consolidada (walk-forward h=1,2,3 meses):")
    print(eval_all.to_string(index=False))
    print("\nTeste de Diebold-Mariano:")
    print(dm_df.to_string(index=False))
    print("\nAvaliação no painel (16 áreas × 9 grupos):")
    print(eval_panel.to_string(index=False))
    return eval_all


if __name__ == "__main__":
    gerar_modelos()
