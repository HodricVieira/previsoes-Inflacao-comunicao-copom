from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"


class TestModelagemPrevisao(unittest.TestCase):
    def test_todos_modelos_proprios_superam_baseline_nacional(self) -> None:
        eval_all = pd.read_csv(DATA_DIR / "avaliacao_modelos.csv")
        by_model = eval_all.set_index("modelo")
        naive = by_model.loc["baseline_naive_sazonal"]

        own_models = [
            "var_multivariado",
            "sarimax_exogeno",
            "regressao_ridge",
            "random_forest",
            "lstm_multivariado",
        ]
        for model_name in own_models:
            row = by_model.loc[model_name]
            self.assertEqual(int(row["n_previsoes"]), 33, f"{model_name} deve ter 33 previsões")
            self.assertLess(row["rmse"], naive["rmse"], f"{model_name} RMSE ({row['rmse']}) >= naive ({naive['rmse']})")
            self.assertLess(row["mae"], naive["mae"], f"{model_name} MAE ({row['mae']}) >= naive ({naive['mae']})")
            self.assertLess(row["mase"], naive["mase"], f"{model_name} MASE ({row['mase']}) >= naive ({naive['mase']})")

    def test_modelos_painel_superam_baseline_em_todos_horizontes(self) -> None:
        eval_panel = pd.read_csv(DATA_DIR / "avaliacao_painel_area_grupo.csv")
        for h in [1, 2, 3]:
            sub = eval_panel.loc[eval_panel["horizonte_meses"] == h].set_index("modelo")
            naive = sub.loc["baseline_naive_sazonal_painel"]
            for model_name in ["ridge_painel", "random_forest_painel"]:
                row = sub.loc[model_name]
                self.assertLess(row["rmse"], 0.80)
                self.assertLess(row["rmse"], naive["rmse"])
                self.assertLess(row["mae"], 0.55)
                self.assertLess(row["mae"], naive["mae"])
                self.assertLess(row["mase"], naive["mase"])

    def test_validacao_sentimento_200_rotulos_manuais_e_impacto_sarimax(self) -> None:
        valid_path = DATA_DIR / "validacao_sentimento.csv"
        self.assertTrue(valid_path.exists(), "validacao_sentimento.csv deve existir")
        valid_df = pd.read_csv(valid_path).set_index("comparacao")

        self.assertEqual(int(valid_df.loc["manual_vs_llm", "n_textos"]), 200)
        self.assertEqual(int(valid_df.loc["manual_vs_lexico_referencia", "n_textos"]), 200)
        kappa_llm = float(valid_df.loc["manual_vs_llm", "kappa_cohen"])
        kappa_lex = float(valid_df.loc["manual_vs_lexico_referencia", "kappa_cohen"])
        self.assertGreater(kappa_llm, 0.70)
        self.assertGreater(kappa_lex, 0.25)
        self.assertGreater(kappa_llm, kappa_lex)

        impact_df = pd.read_csv(DATA_DIR / "impacto_sentimento_sarimax.csv").set_index("modelo")
        for model_name in ["sarimax_sem_sentimento", "sarimax_com_sentimento"]:
            self.assertEqual(int(impact_df.loc[model_name, "n_previsoes"]), 12)
            self.assertLess(float(impact_df.loc[model_name, "rmse"]), 0.16)
            self.assertLess(float(impact_df.loc[model_name, "mae"]), 0.13)


if __name__ == "__main__":
    unittest.main()
