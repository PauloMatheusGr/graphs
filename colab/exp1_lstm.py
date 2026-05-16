"""exp1: LSTM em sequências (3 passos × 20 ROIs) com deltas taxados (espelho exp1_xgboost.py).

Entrada: tensor (n, 60, p) → (n, 3, 20·p); Optuna + nested CV; SHAP Kernel agregado por ROI/atributo.
Downsample opcional no treino externo (GROUP×SEX). Correr com DOWNSAMPLE_GROUP_SEX True/False
para balanced vs unbalanced (pastas colab/exp1/balanced|unbalanced/lstm/).
"""

from __future__ import annotations

import os
from pathlib import Path

# Antes de importar exp_lstm_common (TensorFlow lê na carga do módulo).
# gpu + use_cudnn=False evita CudnnRNNV3; LSTM_GPU_INDEX escolhe a GPU (0 ou 1).
os.environ.setdefault("LSTM_DEVICE", "gpu")
os.environ.setdefault("LSTM_GPU_INDEX", "0")

from exp_lstm_common import LstmExperimentConfig, run_lstm_experiment

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv"
EXP1_PATH = ROOT / "exp1.md"
PAIR_ORDER = ["12", "13", "23"]
DT_EPSILON = 0.5
DOWNSAMPLE_GROUP_SEX = False


def main() -> None:
    run_lstm_experiment(
        LstmExperimentConfig(
            exp_name="exp1",
            csv_path=CSV_PATH,
            exp_md_path=EXP1_PATH,
            pair_order=PAIR_ORDER,
            temporal_mode="delta_rate",
            dt_epsilon=DT_EPSILON,
            downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
            title_prefix="Exp1 LSTM",
        )
    )


if __name__ == "__main__":
    main()
