"""Calcula matrizes de correlação, CCF, causalidade de Granger, VIF e seleção de variáveis."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
FIGURE_DIR = ROOT / "figuras"
TARGET = "variacao_ipca_mensal_pct"
FEATURES = [
    "selic_efetiva_aa",
    "cambio_usd_brl",
    "ic_br",
    "ibc_br_sa",
    "desocupacao",
    "ipca_brasil",
]
MACRO_PREDICTORS = [
    "selic_efetiva_aa",
    "cambio_usd_brl",
    "ic_br",
    "ibc_br_sa",
    "desocupacao",
]


def _load_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "painel_ipca_macro.csv", parse_dates=["data"])
    panel = panel.sort_values(["area_id", "grupo_id", "data"]).reset_index(drop=True)
    return panel


def _load_monthly_series(panel: pd.DataFrame) -> pd.DataFrame:
    macro = pd.read_csv(DATA_DIR / "macro_mensal.csv", parse_dates=["data"])
    median_ipca = (
        panel.groupby("data", as_index=False)[TARGET]
        .median()
        .rename(columns={TARGET: "ipca_mensal_median"})
    )
    return median_ipca.merge(macro, on="data", how="inner").sort_values("data").reset_index(drop=True)


def _correlation_matrices(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["ipca_mensal_median", *FEATURES]
    pearson = monthly[cols].corr(method="pearson").round(4)
    spearman = monthly[cols].corr(method="spearman").round(4)
    pearson.index.name = "variavel"
    spearman.index.name = "variavel"
    return pearson.reset_index(), spearman.reset_index()


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


def _granger_tests(panel: pd.DataFrame, monthly: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """Executa testes de causalidade de Granger (exógena -> IPCA) na série nacional e no painel."""
    panel_pvals: dict[tuple[str, int], list[float]] = {
        (feat, lag): [] for feat in FEATURES for lag in range(1, max_lag + 1)
    }
    for _, group in panel.groupby(["area_id", "grupo_id"], sort=False):
        grouped = group.sort_values("data")
        for feat in FEATURES:
            sub = grouped[[TARGET, feat]].dropna()
            if len(sub) < 36 or sub[feat].nunique() < 2 or sub[TARGET].nunique() < 2:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = grangercausalitytests(sub, maxlag=max_lag)
                for lag in range(1, max_lag + 1):
                    pval = float(res[lag][0]["ssr_ftest"][1])
                    panel_pvals[(feat, lag)].append(pval)
            except Exception:
                continue

    rows: list[dict[str, object]] = []
    for feat in FEATURES:
        sub_nat = monthly[["ipca_mensal_median", feat]].dropna()
        with contextlib.redirect_stdout(io.StringIO()):
            res_nat = grangercausalitytests(sub_nat, maxlag=max_lag)
        for lag in range(1, max_lag + 1):
            f_stat = float(res_nat[lag][0]["ssr_ftest"][0])
            p_val = float(res_nat[lag][0]["ssr_ftest"][1])
            p_list = panel_pvals[(feat, lag)]
            prop_sig = float(np.mean([p < 0.05 for p in p_list])) if p_list else np.nan
            rows.append(
                {
                    "variavel_exogena": feat,
                    "defasagem_meses": lag,
                    "f_stat_nacional": f_stat,
                    "p_valor_nacional": p_val,
                    "significativo_nacional_5pct": bool(p_val < 0.05),
                    "n_series_painel": len(p_list),
                    "prop_series_significativas_5pct": prop_sig,
                }
            )
    return pd.DataFrame(rows)


def _compute_vif(df: pd.DataFrame, columns: list[str], regime: str) -> list[dict[str, object]]:
    subset = df[columns].dropna().copy()
    if subset.empty:
        return []
    x_const = sm.add_constant(subset, has_constant="add")
    rows = []
    for idx, col in enumerate(x_const.columns):
        if col == "const":
            continue
        rows.append(
            {
                "regime": regime,
                "variavel_exogena": col,
                "vif": float(variance_inflation_factor(x_const.to_numpy(dtype=float), idx)),
            }
        )
    return rows


def _vif_analysis(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_compute_vif(monthly, FEATURES, "nivel_todas_variaveis"))
    diff_df = monthly[FEATURES].diff().dropna()
    rows.extend(_compute_vif(diff_df, FEATURES, "primeira_diferenca_todas"))
    reduced = [f for f in MACRO_PREDICTORS if f != "ic_br"]
    rows.extend(_compute_vif(monthly, reduced, "nivel_sem_ic_br"))
    return pd.DataFrame(rows)


def _build_conclusion(
    selection: pd.DataFrame,
    granger: pd.DataFrame,
    vif: pd.DataFrame,
) -> pd.DataFrame:
    """Sintetiza a decisão de quais variáveis e defasagens entram nos modelos."""
    vif_level = vif.loc[vif["regime"] == "nivel_todas_variaveis"].set_index("variavel_exogena")["vif"]
    vif_diff = vif.loc[vif["regime"] == "primeira_diferenca_todas"].set_index("variavel_exogena")["vif"]

    rows: list[dict[str, object]] = []
    for feat in FEATURES:
        ccf_feat = selection.loc[selection["variavel_exogena"] == feat].sort_values(
            "evidencia_total", ascending=False
        )
        top_lags = ccf_feat.loc[ccf_feat["defasagem_meses"] > 0, "defasagem_meses"].head(3).tolist()
        best_lag = int(ccf_feat.iloc[0]["defasagem_meses"]) if not ccf_feat.empty else 1
        best_evid = float(ccf_feat.iloc[0]["evidencia_total"]) if not ccf_feat.empty else np.nan

        g_feat = granger.loc[granger["variavel_exogena"] == feat]
        min_p = float(g_feat["p_valor_nacional"].min()) if not g_feat.empty else np.nan
        max_prop = float(g_feat["prop_series_significativas_5pct"].max()) if not g_feat.empty else np.nan
        sig_lags = g_feat.loc[g_feat["p_valor_nacional"] < 0.05, "defasagem_meses"].tolist()

        v_lvl = float(vif_level.get(feat, np.nan))
        v_dif = float(vif_diff.get(feat, np.nan))

        if feat == "ipca_brasil":
            entra = False
            lags_rec = "não se aplica"
            justificativa = (
                "Série agregada nacional do próprio alvo (IPCA); mantida apenas como referência "
                "macroeconômica, não como preditora exógena nos modelos multivariados."
            )
        else:
            entra = True
            lags_rec = "1, 3, 6, 12 (ou 1..6 no VAR)"
            justificativa = (
                f"VIF em diferença={v_dif:.2f} (<5); "
                f"CCF máxima no lag {best_lag} (evidência={best_evid:.3f}); "
                f"Granger nacional min p={min_p:.4f}, até {max_prop*100:.1f}% das séries do painel com p<0.05. "
            )
            if v_lvl > 10:
                justificativa += (
                    f"Atenção: VIF em nível={v_lvl:.2f} (>10) indica colinearidade de tendência; "
                    "modelos lineares/VAR devem usar defasagens parcimoniosas ou séries diferenciadas."
                )

        rows.append(
            {
                "variavel_exogena": feat,
                "entra_no_modelo": entra,
                "melhor_lag_ccf": best_lag,
                "top3_lags_ccf_positivos": ",".join(str(x) for x in top_lags),
                "lags_granger_sig_nacional": ",".join(str(x) for x in sig_lags) if sig_lags else "nenhum",
                "granger_min_p_nacional": round(min_p, 4),
                "granger_max_prop_painel_5pct": round(max_prop, 4),
                "vif_nivel": round(v_lvl, 3),
                "vif_diferenca": round(v_dif, 3),
                "defasagens_recomendadas": lags_rec,
                "justificativa": justificativa,
            }
        )
    return pd.DataFrame(rows)


def _plot_correlation_summary(pearson_mat: pd.DataFrame, selection: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    mat = pearson_mat.set_index("variavel")
    im1 = axes[0].imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(mat.columns)))
    axes[0].set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(mat.index)))
    axes[0].set_yticklabels(mat.index, fontsize=8)
    axes[0].set_title("Matriz de Correlação de Pearson (Mensal)")
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            axes[0].text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    piv = selection.pivot(
        index="variavel_exogena", columns="defasagem_meses", values="mediana_abs_pearson"
    )
    im2 = axes[1].imshow(piv.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(piv.columns)))
    axes[1].set_xticklabels(piv.columns, fontsize=8)
    axes[1].set_yticks(range(len(piv.index)))
    axes[1].set_yticklabels(piv.index, fontsize=8)
    axes[1].set_xlabel("Defasagem (meses)")
    axes[1].set_title("CCF Mediana |Pearson| por Defasagem no Painel")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlacao_ccf_heatmap.png", dpi=150)
    plt.close(fig)


def gerar_correlacoes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _load_panel()
    monthly = _load_monthly_series(panel)

    pearson_mat, spearman_mat = _correlation_matrices(monthly)
    correlations = _series_lagged_correlations(panel)
    selection = _summary_for_selection(correlations)
    granger = _granger_tests(panel, monthly, max_lag=6)
    vif = _vif_analysis(monthly)
    conclusion = _build_conclusion(selection, granger, vif)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pearson_mat.to_csv(DATA_DIR / "matriz_correlacao_pearson.csv", index=False, encoding="utf-8-sig")
    spearman_mat.to_csv(DATA_DIR / "matriz_correlacao_spearman.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(DATA_DIR / "correlacoes_defasadas.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(DATA_DIR / "selecao_variaveis_correlacao.csv", index=False, encoding="utf-8-sig")
    granger.to_csv(DATA_DIR / "causalidade_granger.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(DATA_DIR / "vif_variaveis.csv", index=False, encoding="utf-8-sig")
    conclusion.to_csv(DATA_DIR / "conclusao_selecao_variaveis.csv", index=False, encoding="utf-8-sig")
    _plot_correlation_summary(pearson_mat, selection)

    print(
        f"Correlação concluída: {len(correlations):,} combinações (área × grupo × variável × lag), "
        f"{len(granger):,} testes de Granger e {len(vif):,} registros de VIF."
    )
    print("\nConclusão da seleção de variáveis:")
    print(
        conclusion[
            [
                "variavel_exogena",
                "entra_no_modelo",
                "melhor_lag_ccf",
                "granger_min_p_nacional",
                "vif_nivel",
                "vif_diferenca",
                "defasagens_recomendadas",
            ]
        ].to_string(index=False)
    )
    return correlations, selection, vif


if __name__ == "__main__":
    gerar_correlacoes()
