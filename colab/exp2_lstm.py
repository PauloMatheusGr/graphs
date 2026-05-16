"""exp2: LSTM em sequências (3 passos × 20 ROIs) com absolutos + taxa desde baseline.

Espelho exp2_xgboost.py: baseline_rate, Optuna + nested CV, SHAP, mesmas saídas em
colab/exp2/{balanced|unbalanced}/lstm/. Alternar DOWNSAMPLE_GROUP_SEX para as duas estratégias.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LSTM_DEVICE", "gpu")
os.environ.setdefault("LSTM_GPU_INDEX", "0")

from exp_lstm_common import LstmExperimentConfig, run_lstm_experiment

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"
EXP2_PATH = ROOT / "exp2.md"
PAIR_ORDER = ["1", "2", "3"]
DT_EPSILON = 0.5
DOWNSAMPLE_GROUP_SEX = False


def main() -> None:
    run_lstm_experiment(
        LstmExperimentConfig(
            exp_name="exp2",
            csv_path=CSV_PATH,
            exp_md_path=EXP2_PATH,
            pair_order=PAIR_ORDER,
            temporal_mode="baseline_rate",
            dt_epsilon=DT_EPSILON,
            downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
            title_prefix="Exp2 LSTM",
        )
    )


if __name__ == "__main__":
    main()
