"""Calcula correlações, defasagens e seleção de variáveis exógenas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
TARGET = "variacao_ipca_mensal_pct"
FEATURES = [
    "selic_efetiva_aa",
    "cambio_usd_brl",
    "ic_br",
    "ibc_br_sa",
    "desocupacao",
    "ipca_brasil",
]


def _load_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "painel_ipca_macro.csv", parse_dates=["data"])
    panel = panel.sort_values(["area_id", "grupo_id", "data"]).reset_index(drop=True)
    return panel


def _series_lagged_correlations(panel: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (area_id, area, nivel, grupo_id, grupo), group in panel.groupby(
        ["area_id", "area", "nivel_geografico", "grupo_id", "grupo_despesa"],
        sort=True,
        dropna=False,
    ):
        grouped = group.sort_values("data").copy()
        for feature in FEATURES:
            for lag in range(0, max_lag + 1):
                series = grouped[feature].shift(lag)
                target = grouped[TARGET]
                aligned = pd.DataFrame({"feature": series, "target": target}).dropna()
                if aligned.empty or aligned["feature"].nunique() < 2:
                    continue
                pearson = aligned["target"].corr(aligned["feature"], method="pearson")
                spearman = aligned["target"].corr(aligned["feature"], method="spearman")
                rows.append(
                    {
                        "area_id": area_id,
                        "area": area,
                        "nivel_geografico": nivel,
                        "grupo_id": grupo_id,
                        "grupo_despesa": grupo,
                        "variavel_exogena": feature,
                        "defasagem_meses": lag,
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_pearson": abs(float(pearson)) if pd.notna(pearson) else np.nan,
                        "abs_spearman": abs(float(spearman)) if pd.notna(spearman) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _summary_for_selection(correlations: pd.DataFrame) -> pd.DataFrame:
    summary = (
        correlations.groupby(["variavel_exogena", "defasagem_meses"], as_index=False)
        .agg(
            media_abs_pearson=("abs_pearson", "mean"),
            media_abs_spearman=("abs_spearman", "mean"),
            mediana_abs_pearson=("abs_pearson", "median"),
            mediana_abs_spearman=("abs_spearman", "median"),
            n_series=("area_id", "size"),
        )
    )
    summary["evidencia_total"] = (
        summary["mediana_abs_pearson"] + summary["mediana_abs_spearman"]
    ) / 2.0
    return summary.sort_values(["variavel_exogena", "evidencia_total"], ascending=[True, False])


def _vif_for_selected(correlations: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    ranked = (
        correlations.groupby("variavel_exogena", as_index=False)
        .agg(media_abs_pearson=("abs_pearson", "mean"), media_abs_spearman=("abs_spearman", "mean"))
        .assign(score=lambda df: (df["media_abs_pearson"] + df["media_abs_spearman"]) / 2.0)
        .sort_values("score", ascending=False)
    )
    selected = ranked["variavel_exogena"].head(top_n).tolist()
    if not selected:
        return pd.DataFrame(columns=["variavel_exogena", "vif"])
    panel = _load_panel().sort_values("data")
    subset = panel[[TARGET, *selected]].dropna().copy()
    if subset.empty:
        return pd.DataFrame(columns=["variavel_exogena", "vif"])
    x = subset[selected].to_numpy(dtype=float)
    vif_rows = []
    for idx in range(x.shape[1]):
        vif_rows.append({
            "variavel_exogena": selected[idx],
            "vif": float(variance_inflation_factor(x, idx)),
        })
    return pd.DataFrame(vif_rows)


def gerar_correlacoes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _load_panel()
    correlations = _series_lagged_correlations(panel)
    selection = _summary_for_selection(correlations)
    vif = _vif_for_selected(correlations)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(DATA_DIR / "correlacoes_defasadas.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(DATA_DIR / "selecao_variaveis_correlacao.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(DATA_DIR / "vif_variaveis.csv", index=False, encoding="utf-8-sig")

    print(
        f"Correlação concluída: {len(correlations):,} combinações (área × grupo × variável × lag), "
        f"{len(selection):,} opções de seleção e {len(vif):,} VIFs calculados."
    )
    return correlations, selection, vif


if __name__ == "__main__":
    gerar_correlacoes()
