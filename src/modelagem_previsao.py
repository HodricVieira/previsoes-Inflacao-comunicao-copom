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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


def _build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["ipca_mensal_median"] = working["ipca_mensal_median"].astype(float)
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
) -> dict[tuple[int, int], float]:
    """Treina LSTM na base de treino até split_index e prevê h=1,2,3 em walk-forward."""
    feature_columns = ["ipca_mensal_median", *EXOG_FEATURES]
    clean = df[["data", *feature_columns]].dropna().reset_index(drop=True)
    split_date = df.iloc[split_index]["data"]
    train_mask = clean["data"] < split_date
    train_values = clean.loc[train_mask, feature_columns].to_numpy(dtype=np.float32)
    all_values = clean[feature_columns].to_numpy(dtype=np.float32)
    n_train = len(train_values)

    input_mean = train_values.mean(axis=0)
    input_scale = train_values.std(axis=0)
    input_scale[input_scale == 0] = 1.0
    target_mean = float(input_mean[0])
    target_scale = float(input_scale[0])
    scaled = (all_values - input_mean) / input_scale

    sequences = np.stack(
        [scaled[i - LSTM_LOOKBACK : i] for i in range(LSTM_LOOKBACK, n_train)]
    )
    labels = scaled[LSTM_LOOKBACK:n_train, 0]

    torch.manual_seed(42)
    model = _LSTMRegressor(input_size=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    train_x = torch.as_tensor(sequences, dtype=torch.float32)
    train_y = torch.as_tensor(labels, dtype=torch.float32)
    model.train()
    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    date_to_orig_idx = {d: idx for idx, d in enumerate(df["data"])}
    preds: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for clean_origin_idx in range(n_train - 1, len(clean) - 1):
            origin_date = clean.iloc[clean_origin_idx]["data"]
            orig_idx = date_to_orig_idx[origin_date]
            window = scaled[clean_origin_idx - LSTM_LOOKBACK + 1 : clean_origin_idx + 1].copy()
            if len(window) < LSTM_LOOKBACK:
                continue
            for h in HORIZONS:
                if orig_idx + h >= len(df):
                    continue
                seq = torch.as_tensor(window[None, :, :], dtype=torch.float32)
                pred_scaled = float(model(seq).item())
                preds[(orig_idx, h)] = pred_scaled * target_scale + target_mean
                # Atualiza janela recursivamente mantendo exógenas no último valor conhecido
                next_row = window[-1].copy()
                next_row[0] = pred_scaled
                window = np.vstack([window[1:], next_row])
    return preds


def _walk_forward_monthly(df: pd.DataFrame, focus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa validação walk-forward (origem móvel) para horizontes h=1,2,3 meses."""
    split_index = len(df) - TEST_MONTHS
    feature_cols = _feature_columns_at_origin(df)
    focus_lookup = focus.set_index(["origem_mes", "mes_referencia"])["mediana_pct"].to_dict()

    # Escala do MASE baseada no naïve sazonal in-sample do treino
    train_y = df.iloc[:split_index]["ipca_mensal_median"].dropna()
    naive_scale = float(np.mean(np.abs(train_y.diff(12).dropna())))

    lstm_preds = _fit_lstm_and_predict(df, split_index)

    records: list[dict[str, object]] = []
    # Origens móveis: de (split_index - 1) até (len(df) - 2)
    for origin_idx in range(split_index - 1, len(df) - 1):
        origin_date = df.iloc[origin_idx]["data"]
        train_slice = df.iloc[: origin_idx + 1].copy()

        # 1) SARIMAX na origem (com exógenas defasadas já disponíveis em origin_idx)
        sarimax_preds: dict[int, float] = {}
        sarimax_exog_cols = [f"{f}_lag_1" for f in EXOG_FEATURES]
        s_frame = train_slice[["ipca_mensal_median", *sarimax_exog_cols]].dropna()
        if len(s_frame) >= 36:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s_model = SARIMAX(
                        s_frame["ipca_mensal_median"].astype(float),
                        exog=s_frame[sarimax_exog_cols].astype(float),
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 0, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                # Para h=1..3, usa exógenas conhecidas na origem (últimas observações disponíveis)
                last_known_exog = np.repeat(
                    df.iloc[[origin_idx]][EXOG_FEATURES].to_numpy(dtype=float),
                    repeats=max(HORIZONS),
                    axis=0,
                )
                s_fc = s_model.forecast(steps=max(HORIZONS), exog=last_known_exog)
                for h in HORIZONS:
                    sarimax_preds[h] = float(s_fc.iloc[h - 1])
            except Exception:
                pass

        # 2) VAR na origem (com primeiras diferenças das macroeconômicas para estacionariedade)
        var_preds: dict[int, float] = {}
        var_cols = ["ipca_mensal_median", *[f"{f}_diff1" for f in EXOG_FEATURES]]
        var_frame = train_slice[var_cols].dropna()
        if len(var_frame) >= 36:
            try:
                v_fit = VAR(var_frame).fit(4)
                v_fc = v_fit.forecast(
                    y=var_frame.tail(v_fit.k_ar).to_numpy(dtype=float),
                    steps=max(HORIZONS),
                )
                for h in HORIZONS:
                    var_preds[h] = float(v_fc[h - 1, 0])
            except Exception:
                pass

        # 3) Modelos diretos por horizonte h (Ridge e Random Forest)
        for h in HORIZONS:
            target_idx = origin_idx + h
            if target_idx >= len(df):
                continue
            target_date = df.iloc[target_idx]["data"]
            actual_val = float(df.iloc[target_idx]["ipca_mensal_median"])

            # Baseline naïve sazonal: t + h - 12
            naive_idx = target_idx - 12
            naive_pred = float(df.iloc[naive_idx]["ipca_mensal_median"]) if naive_idx >= 0 else np.nan
            focus_pred = float(focus_lookup.get((origin_date, target_date), np.nan))

            # Monta base direta y_{t+h} ~ X_t usando apenas dados <= origin_idx
            direct_df = train_slice[feature_cols].copy()
            direct_df["target_h"] = train_slice["ipca_mensal_median"].shift(-h)
            direct_train = direct_df.dropna()
            x_origin = df.iloc[[origin_idx]][feature_cols]

            ridge_pred = np.nan
            rf_pred = np.nan
            if len(direct_train) >= 36 and not x_origin.isna().any(axis=None):
                x_tr = direct_train[feature_cols].to_numpy(dtype=float)
                y_tr = direct_train["target_h"].to_numpy(dtype=float)
                x_te = x_origin.to_numpy(dtype=float)

                ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                ridge.fit(x_tr, y_tr)
                ridge_pred = float(ridge.predict(x_te)[0])

                rf = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(x_tr, y_tr)
                rf_pred = float(rf.predict(x_te)[0])

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


def _panel_forecast_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia previsão walk-forward (h=1,2,3) no painel completo (área × grupo)."""
    panel = pd.read_csv(DATA_DIR / "painel_ipca_macro.csv", parse_dates=["data"])
    panel = panel.sort_values(["area_id", "grupo_id", "data"]).reset_index(drop=True)
    panel["area_cat"] = panel["area_id"].astype("category").cat.codes
    panel["grupo_cat"] = panel["grupo_id"].astype("category").cat.codes

    for lag in [1, 3, 6, 12]:
        panel[f"ipca_lag_{lag}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(lag)
    panel["ipca_mm_3m"] = (
        panel.groupby(["area_id", "grupo_id"])[TARGET].transform(lambda s: s.rolling(3).mean())
    )
    for feat in EXOG_FEATURES:
        panel[f"{feat}_diff1"] = panel.groupby(["area_id", "grupo_id"])[feat].diff(1)

    panel["naive_sazonal"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(12)
    panel_features = [
        "area_cat",
        "grupo_cat",
        TARGET,
        "ipca_lag_1",
        "ipca_lag_3",
        "ipca_lag_6",
        "ipca_lag_12",
        "ipca_mm_3m",
        *EXOG_FEATURES,
        *[f"{f}_diff1" for f in EXOG_FEATURES],
    ]

    unique_dates = sorted(panel["data"].unique())
    split_date = unique_dates[-TEST_MONTHS]
    test_origins = unique_dates[-TEST_MONTHS - 1 : -1]

    rows: list[dict[str, object]] = []
    for h in HORIZONS:
        panel[f"target_h{h}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(-h)
        panel[f"target_date_h{h}"] = panel.groupby(["area_id", "grupo_id"])["data"].shift(-h)
        panel[f"naive_h{h}"] = panel.groupby(["area_id", "grupo_id"])[TARGET].shift(12 - h)

        # Treina modelo global de painel uma vez antes do bloco de teste (sem vazamento) e avalia nas origens
        train_df = panel.loc[panel["data"] < split_date, [*panel_features, f"target_h{h}"]].dropna()
        x_tr = train_df[panel_features].to_numpy(dtype=float)
        y_tr = train_df[f"target_h{h}"].to_numpy(dtype=float)

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        ridge.fit(x_tr, y_tr)

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(x_tr, y_tr)

        eval_slice = panel.loc[
            panel["data"].isin(test_origins) & panel[f"target_h{h}"].notna()
        ].copy()
        valid_mask = eval_slice[panel_features].notna().all(axis=1)
        eval_valid = eval_slice.loc[valid_mask].copy()
        x_te = eval_valid[panel_features].to_numpy(dtype=float)
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _build_lagged_features(_load_base())
    if len(df) < 48:
        raise ValueError("A série deve ter pelo menos 48 meses para validação walk-forward.")
    focus = _load_focus_benchmark()

    wf_df, eval_all, eval_by_h, dm_df = _walk_forward_monthly(df, focus)
    pred_panel, eval_panel = _panel_forecast_evaluation()

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
