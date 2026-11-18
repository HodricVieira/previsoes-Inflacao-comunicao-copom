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


if __name__ == "__main__":
    unittest.main()
