cd /mnt/study-data/pgirardi/graphs

# 1) Baseline
.venv/bin/python colab/run_exp2_all.py

# (opcional) smoke checkpoint XGB balanced
.venv/bin/python colab/verify_xgb_checkpoint.py

# 2) PDFs + demografia nos 8 runs (inclui analyze_oof_demographics)
.venv/bin/python colab/postprocess_exp2_runs.py

# 3) Ablação
.venv/bin/python colab/run_roi_ablation_exp2.py

