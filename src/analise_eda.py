"""Gera estatísticas e diagnósticos exploratórios para o painel mensal do IPCA."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
FIGURE_DIR = ROOT / "figuras"
TARGET = "variacao_ipca_mensal_pct"
KEYS = ["area_id", "area", "nivel_geografico", "grupo_id", "grupo_despesa"]


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ipca = pd.read_csv(
        DATA_DIR / "ipca_painel.csv",
        encoding="utf-8-sig",
        parse_dates=["data"],
    )
    macro = pd.read_csv(
        DATA_DIR / "macro_mensal.csv",
        encoding="utf-8-sig",
        parse_dates=["data"],
    )
    if ipca.empty or macro.empty:
        raise ValueError("A base de IPCA ou macroeconômica está vazia.")
    if ipca.duplicated(["data", "area_id", "grupo_id"]).any():
        raise ValueError("A base de IPCA contém chaves duplicadas.")
    return ipca, macro


def _series_groups(ipca: pd.DataFrame):
    yield from ipca.sort_values("data").groupby(KEYS, sort=True, dropna=False)


def _observed_segment(group: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    observed = group.loc[group[TARGET].notna()].sort_values("data")
    if observed.empty:
        return observed, False
    expected_dates = pd.date_range(
        observed["data"].iloc[0], observed["data"].iloc[-1], freq="MS"
    )
    contiguous = pd.DatetimeIndex(observed["data"]).equals(expected_dates)
    return observed, contiguous


def _descriptive_statistics(ipca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in _series_groups(ipca):
        values = group[TARGET]
        observed = values.dropna()
        rows.append(
            dict(
                zip(KEYS, key),
                n_periodos=len(group),
                n_observados=observed.size,
                n_ausentes=values.isna().sum(),
                media=observed.mean(),
                desvio_padrao=observed.std(),
                mediana=observed.median(),
                minimo=observed.min(),
                maximo=observed.max(),
            )
        )
    return pd.DataFrame(rows)


def _missing_profile(ipca: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in _series_groups(ipca):
        rows.append(
            dict(
                zip(KEYS, key),
                fonte="IBGE/SIDRA",
                n_periodos=len(group),
                n_ausentes=group[TARGET].isna().sum(),
            )
        )
    for variable in macro.columns:
        if variable == "data":
            continue
        rows.append(
            {
                "area_id": "",
                "area": "Brasil",
                "nivel_geografico": "nacional",
                "grupo_id": "",
                "grupo_despesa": variable,
                "fonte": "BCB/SGS",
                "n_periodos": len(macro),
                "n_ausentes": macro[variable].isna().sum(),
            }
        )
    return pd.DataFrame(rows)


def _iqr_outliers(ipca: pd.DataFrame) -> pd.DataFrame:
    outliers = []
    for key, group in _series_groups(ipca):
        values = group[TARGET].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = group.loc[
            group[TARGET].notna()
            & ((group[TARGET] < lower) | (group[TARGET] > upper))
        ]
        for row in flagged.itertuples(index=False):
            outliers.append(
                {
                    **dict(zip(KEYS, key)),
                    "data": row.data,
                    "valor_pct": getattr(row, TARGET),
                    "limite_inferior_iqr": lower,
                    "limite_superior_iqr": upper,
                }
            )
    return pd.DataFrame(
        outliers,
        columns=KEYS
        + [
            "data",
            "valor_pct",
            "limite_inferior_iqr",
            "limite_superior_iqr",
        ],
    )


def _stationarity_and_correlation(ipca: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests: list[dict] = []
    correlations: list[dict] = []
    for key, group in _series_groups(ipca):
        segment, contiguous = _observed_segment(group)
        values = segment[TARGET]
        test_row = dict(zip(KEYS, key))
        correlation_row = dict(zip(KEYS, key))
        if not contiguous:
            test_row.update(
                status="não executado: lacuna interna ou série sem dados; sem imputação"
            )
            correlation_row.update(
                status="não executado: lacuna interna ou série sem dados; sem imputação"
            )
            tests.append(test_row)
            correlations.append(correlation_row)
            continue

        if len(values) < 24 or values.nunique() < 2:
            test_row.update(status="não executado: série curta ou constante")
            correlation_row.update(status="não executado: série curta ou constante")
            tests.append(test_row)
            correlations.append(correlation_row)
            continue

        test_row.update(
            periodo_inicio=segment["data"].iloc[0],
            periodo_fim=segment["data"].iloc[-1],
            n_observacoes=len(segment),
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="adfuller currently returns a plain tuple",
                    category=FutureWarning,
                )
                adf = adfuller(values.to_numpy(dtype=float), autolag="AIC")
            test_row.update(
                status="ok",
                adf_estatistica=adf[0],
                adf_pvalor=adf[1],
                adf_lags=adf[2],
                adf_nobs=adf[3],
                adf_conclusao_5pct=(
                    "estacionária (rejeita raiz unitária)"
                    if adf[1] < 0.05
                    else "não estacionária (não rejeita raiz unitária)"
                ),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                kpss_result = kpss(
                    values.to_numpy(dtype=float), regression="c", nlags="auto"
                )
            test_row.update(
                kpss_estatistica=kpss_result[0],
                kpss_pvalor=kpss_result[1],
                kpss_lags=kpss_result[2],
                kpss_conclusao_5pct=(
                    "não estacionária (rejeita estacionariedade)"
                    if kpss_result[1] < 0.05
                    else "estacionária (não rejeita estacionariedade)"
                ),
                kpss_aviso="; ".join(str(item.message) for item in caught),
            )
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
            test_row.update(status=f"falha no teste: {type(exc).__name__}: {exc}")
        tests.append(test_row)

        max_lags = min(24, len(values) // 2 - 1)
        try:
            acf_values = acf(values.to_numpy(dtype=float), nlags=max_lags, fft=True)
            pacf_values = pacf(
                values.to_numpy(dtype=float), nlags=max_lags, method="ywm"
            )
            for lag in range(1, max_lags + 1):
                correlations.append(
                    {
                        **dict(zip(KEYS, key)),
                        "status": "ok",
                        "defasagem_meses": lag,
                        "acf": acf_values[lag],
                        "pacf": pacf_values[lag],
                    }
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            correlation_row.update(
                status=f"falha no cálculo: {type(exc).__name__}: {exc}"
            )
            correlations.append(correlation_row)

    return pd.DataFrame(tests), pd.DataFrame(correlations)


def _stl_decomposition(ipca: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_frames = []
    summaries = []
    for key, group in _series_groups(ipca):
        segment, contiguous = _observed_segment(group)
        values = segment[TARGET]
        row = dict(zip(KEYS, key))
        if not contiguous:
            row.update(
                status="não executado: lacuna interna ou série sem dados; sem imputação"
            )
            summaries.append(row)
            continue
        if len(values) < 24:
            row.update(status="não executado: menos de 24 observações")
            summaries.append(row)
            continue
        row.update(
            periodo_inicio=segment["data"].iloc[0],
            periodo_fim=segment["data"].iloc[-1],
            n_observacoes=len(segment),
        )
        result = STL(values.to_numpy(dtype=float), period=12, robust=True).fit()
        row.update(
            status="ok",
            desvio_padrao_tendencia=np.std(result.trend, ddof=1),
            desvio_padrao_sazonalidade=np.std(result.seasonal, ddof=1),
            desvio_padrao_residuo=np.std(result.resid, ddof=1),
        )
        summaries.append(row)
        component_frames.append(
            pd.DataFrame(
                {
                    "data": segment["data"].to_numpy(),
                    **dict(zip(KEYS, key)),
                    "observado": values.to_numpy(),
                    "tendencia": result.trend,
                    "sazonalidade": result.seasonal,
                    "residuo": result.resid,
                }
            )
        )
    components = (
        pd.concat(component_frames, ignore_index=True)
        if component_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(summaries), components


def _plot_series(ipca: pd.DataFrame) -> None:
    groups = (
        ipca[["grupo_id", "grupo_despesa"]]
        .drop_duplicates()
        .sort_values("grupo_id")
    )
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    for ax, (_, group_info) in zip(axes.flat, groups.iterrows()):
        group_data = ipca.loc[ipca["grupo_id"] == group_info["grupo_id"]]
        for _, area_data in group_data.groupby("area_id"):
            ax.plot(
                area_data["data"],
                area_data[TARGET],
                color="#7f8c8d",
                alpha=0.28,
                linewidth=0.7,
            )
        median = group_data.groupby("data")[TARGET].median()
        ax.plot(median.index, median.values, color="#1f4e79", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(group_info["grupo_despesa"])
        ax.set_ylabel("% mensal")
        ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Variação mensal do IPCA por grupo — linhas finas: áreas; linha azul: mediana",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ipca_por_grupo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_macro(macro: pd.DataFrame) -> None:
    variables = [column for column in macro.columns if column != "data"]
    fig, axes = plt.subplots(
        len(variables), 1, figsize=(13, 2.5 * len(variables)), sharex=True
    )
    for ax, variable in zip(np.atleast_1d(axes), variables):
        ax.plot(macro["data"], macro[variable], color="#1f4e79", linewidth=1.1)
        ax.set_title(variable)
        ax.grid(alpha=0.2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Variáveis macroeconômicas mensais — Brasil", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "variaveis_macro.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_stl_and_acf(
    ipca: pd.DataFrame, components: pd.DataFrame
) -> tuple[Path | None, Path | None]:
    if components.empty:
        return None, None
    chosen = (
        components.sort_values(["area_id", "grupo_id"])
        .iloc[0][KEYS]
        .to_dict()
    )
    selected = components
    for key in KEYS:
        selected = selected.loc[selected[key] == chosen[key]]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(selected["data"], selected["observado"], color="#333333")
    axes[0].set_title(
        f"STL: {chosen['area']} — {chosen['grupo_despesa']} "
        "(sazonalidade mensal, período=12)"
    )
    axes[0].set_ylabel("Observado")
    for ax, column, label in zip(
        axes[1:], ["tendencia", "sazonalidade", "residuo"], ["Tendência", "Sazonalidade", "Resíduo"]
    ):
        ax.plot(selected["data"], selected[column], color="#1f4e79")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    stl_path = FIGURE_DIR / "decomposicao_stl_exemplo.png"
    fig.savefig(stl_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    values = selected.sort_values("data")["observado"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(values, lags=24, zero=False, ax=axes[0])
    plot_pacf(values, lags=24, zero=False, method="ywm", ax=axes[1])
    axes[0].set_title(f"ACF — {chosen['area']}, {chosen['grupo_despesa']}")
    axes[1].set_title("PACF")
    fig.tight_layout()
    acf_path = FIGURE_DIR / "acf_pacf_exemplo.png"
    fig.savefig(acf_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stl_path, acf_path


def gerar_eda() -> dict[str, pd.DataFrame]:
    """Calcula diagnósticos, grava tabelas e salva gráficos da análise."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ipca, macro = _read_inputs()

    descriptive = _descriptive_statistics(ipca)
    missing = _missing_profile(ipca, macro)
    outliers = _iqr_outliers(ipca)
    stationarity, correlations = _stationarity_and_correlation(ipca)
    stl_summary, stl_components = _stl_decomposition(ipca)
    panel = ipca.merge(macro, on="data", how="left", validate="many_to_one")

    outputs = {
        "resumo_descritivo_unidade_grupo": descriptive,
        "perfil_ausentes": missing,
        "outliers_iqr": outliers,
        "testes_estacionariedade": stationarity,
        "acf_pacf": correlations,
        "resumo_decomposicao_stl": stl_summary,
        "componentes_stl": stl_components,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            DATA_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    panel.to_csv(
        DATA_DIR / "painel_ipca_macro.csv", index=False, encoding="utf-8-sig"
    )
    _plot_series(ipca)
    _plot_macro(macro)
    _plot_stl_and_acf(ipca, stl_components)

    print(
        f"EDA concluída: {len(descriptive)} séries (área × grupo), "
        f"{len(outliers)} observações sinalizadas pela regra IQR; "
        f"{stationarity['status'].eq('ok').sum()} séries testadas; "
        f"{stl_summary['status'].eq('ok').sum()} decomposições STL."
    )
    print(
        "Nenhum valor ausente foi imputado e nenhum outlier foi removido; "
        "os sinalizados são mantidos para inspeção contextual."
    )
    return outputs


if __name__ == "__main__":
    gerar_eda()
