"""Previsão multivariada e validação temporal do IPCA mensal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
TARGET = "variacao_ipca_mensal_pct"
EXOG_FEATURES = [
    "selic_efetiva_aa",
    "cambio_usd_brl",
    "ic_br",
    "ibc_br_sa",
    "desocupacao",
]
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


def _build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["ipca_mensal_median"] = working["ipca_mensal_median"].astype(float)
    for lag in [1, 3, 6, 12]:
        working[f"ipca_lag_{lag}"] = working["ipca_mensal_median"].shift(lag)
        for feature in EXOG_FEATURES:
            working[f"{feature}_lag_{lag}"] = working[feature].shift(lag)
    return working


def _as_float(series: pd.Series) -> np.ndarray:
    return np.asarray(series.to_numpy(dtype=float), dtype=float)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_arr = _as_float(actual)
    pred_arr = _as_float(predicted)
    error = actual_arr - pred_arr
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = np.clip(np.abs(actual_arr), 1e-6, None)
    mape = float(np.mean(np.abs(error) / denominator) * 100.0)
    naive_denom = np.mean(np.abs(np.diff(actual_arr, prepend=actual_arr[0])))
    mase = float(np.mean(np.abs(error)) / naive_denom) if naive_denom > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "mase": mase,
    }


def _baseline_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    observed = train["ipca_mensal_median"].copy()
    predicted = pd.Series(np.nan, index=test.index, dtype=float)
    if len(observed) >= 12:
        predicted[:] = observed.iloc[-12]
    else:
        predicted[:] = observed.iloc[-1]
    return predicted


def _ols_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    features = [
        col
        for col in train.columns
        if col.startswith("ipca_lag_")
        or any(col.startswith(f"{feature}_lag_") for feature in EXOG_FEATURES)
    ]
    model_frame = train[["ipca_mensal_median", *features]].dropna().copy()
    if model_frame.empty:
        return pd.Series(np.nan, index=test.index)
    X_train = model_frame[features].astype(float)
    y_train = model_frame["ipca_mensal_median"].astype(float)
    model = OLS(y_train, X_train).fit()
    X_test = test[features].astype(float).copy()
    X_test = X_test.bfill().ffill()
    predictions = model.predict(X_test)
    return pd.Series(predictions.to_numpy(dtype=float), index=test.index)


def _sarimax_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    features = [
        col
        for col in train.columns
        if col.startswith("ipca_lag_")
        or any(col.startswith(f"{feature}_lag_") for feature in EXOG_FEATURES)
    ]
    model_frame = train[["ipca_mensal_median", *features]].dropna().copy()
    if model_frame.empty:
        return pd.Series(np.nan, index=test.index)
    y_train = model_frame["ipca_mensal_median"].astype(float)
    x_train = model_frame[features].astype(float)
    x_test = test[features].astype(float).copy()
    x_test = x_test.bfill().ffill()
    model = SARIMAX(
        y_train,
        exog=x_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test), exog=x_test)
    return pd.Series(forecast.to_numpy(dtype=float), index=test.index)


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

    split_index = len(df) - 12
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


def _var_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    target = "ipca_mensal_median"
    selected = [target, *[f"{feature}_lag_1" for feature in EXOG_FEATURES]]
    var_df = train[selected].dropna()
    if len(var_df) < 12:
        return pd.Series(np.nan, index=test.index)
    model = VAR(var_df)
    fitted = model.fit(6)
    actual_test = test[target].reset_index(drop=True)
    forecast = fitted.forecast(y=var_df.tail(fitted.k_ar).to_numpy(dtype=float), steps=len(actual_test))
    return pd.Series(forecast[:, 0], index=test.index)


def _lstm_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Train on 12-month sequences and issue one-step-ahead walk-forward forecasts."""
    feature_columns = ["ipca_mensal_median", *EXOG_FEATURES]
    train_end = train["data"].iloc[-1]
    history = pd.concat([train, test], ignore_index=True).sort_values("data")
    complete_training = history.loc[
        (history["data"] <= train_end)
        & history[feature_columns].notna().all(axis=1),
        "data",
    ]
    if complete_training.empty:
        raise ValueError("LSTM não encontrou observações completas para treino.")
    history = history.loc[history["data"] >= complete_training.min()].reset_index(
        drop=True
    )
    train_mask = history["data"] <= train_end
    train_values = history.loc[train_mask, feature_columns].to_numpy(dtype=np.float32)
    all_values = history[feature_columns].to_numpy(dtype=np.float32)
    n_train = len(train_values)

    if not np.isfinite(train_values).all():
        raise ValueError("LSTM encontrou lacunas internas nas variáveis de treino.")
    if n_train <= LSTM_LOOKBACK:
        raise ValueError(
            f"LSTM requer mais de {LSTM_LOOKBACK} meses completos no treino."
        )
    if len(history) - n_train != len(test):
        raise ValueError("LSTM encontrou meses de teste sem variáveis completas.")

    input_mean = train_values.mean(axis=0)
    input_scale = train_values.std(axis=0)
    input_scale[input_scale == 0] = 1.0
    target_mean = input_mean[0]
    target_scale = input_scale[0]
    scaled_values = (all_values - input_mean) / input_scale

    sequences = np.stack(
        [
            scaled_values[index - LSTM_LOOKBACK : index]
            for index in range(LSTM_LOOKBACK, n_train)
        ]
    )
    if not np.isfinite(sequences).all():
        raise ValueError("LSTM encontrou valores ausentes nas janelas de treino.")
    labels = scaled_values[LSTM_LOOKBACK:n_train, 0]
    train_x = torch.as_tensor(sequences, dtype=torch.float32)
    train_y = torch.as_tensor(labels, dtype=torch.float32)

    torch.manual_seed(42)
    model = _LSTMRegressor(input_size=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        predictions = model(train_x)
        loss = loss_function(predictions, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    forecast_values = []
    with torch.no_grad():
        for index in range(n_train, len(all_values)):
            input_window = scaled_values[index - LSTM_LOOKBACK : index]
            if not np.isfinite(input_window).all():
                raise ValueError(
                    "LSTM encontrou valores ausentes na janela de previsão."
                )
            sequence = torch.as_tensor(
                input_window[None, :, :],
                dtype=torch.float32,
            )
            forecast = model(sequence).item()
            forecast_values.append(forecast * target_scale + target_mean)

    return pd.Series(forecast_values, index=test.index, dtype=float)


def _evaluate_models() -> pd.DataFrame:
    df = _build_lagged_features(_load_base())
    if len(df) < 36:
        raise ValueError("A série deve ter pelo menos 36 meses para validação temporal.")

    split_index = len(df) - 12
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    actual = test["ipca_mensal_median"].reset_index(drop=True)

    models = {
        "baseline_naive_sazonal": _baseline_forecast(train, test),
        "regressao_lag_ols": _ols_forecast(train, test),
        "sarimax_exogeno": _sarimax_forecast(train, test),
        "var_multivariado": _var_forecast(train, test),
        "lstm_multivariado": _lstm_forecast(train, test),
    }

    rows = []
    for name, prediction in models.items():
        prediction = pd.Series(np.asarray(prediction, dtype=float), index=actual.index)
        metrics = _metrics(actual, prediction)
        rows.append(
            {
                "modelo": name,
                "periodo_treino_inicio": train["data"].iloc[0].date(),
                "periodo_treino_fim": train["data"].iloc[-1].date(),
                "periodo_teste_inicio": test["data"].iloc[0].date(),
                "periodo_teste_fim": test["data"].iloc[-1].date(),
                "horizonte_meses": len(actual),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def gerar_modelos() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _evaluate_models()
    metrics.to_csv(DATA_DIR / "avaliacao_modelos.csv", index=False, encoding="utf-8-sig")
    sentiment_comparison = _evaluate_sentiment_effect(
        _build_lagged_features(_load_base())
    )
    sentiment_output = DATA_DIR / "impacto_sentimento_sarimax.csv"
    if sentiment_comparison is not None:
        sentiment_comparison.to_csv(
            sentiment_output,
            index=False,
            encoding="utf-8-sig",
        )
        print("\nComparação SARIMAX com e sem sentimento:")
        print(sentiment_comparison.to_string(index=False))
    else:
        print(
            "\nComparação com sentimento não executada: faltam escores LLM "
            "com cobertura suficiente no treino e no teste."
        )
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    gerar_modelos()
