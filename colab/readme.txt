cd /mnt/study-data/pgirardi/graphs

# 1) Baseline
.venv/bin/python colab/run_exp2_all.py

# (opcional) smoke checkpoint XGB fold 0 vs OOF
.venv/bin/python colab/verify_xgb_checkpoint.py

# 2) PDFs + demografia nos 4 runs balanced
.venv/bin/python colab/postprocess_exp2_runs.py

# 3) Ablação LOO — 5 ROIs (XGB; SVM/LSTM ver abaixo)
ABLATION_ROIS=inf_lateral_ventricle,hippocampus,amygdala,accumbens_area,insula \
  .venv/bin/python colab/run_roi_ablation_exp2.py

# 3b) Ablação SVM (mesmas ROIs; reutiliza C do baseline)
ABLATION_MODEL=svm \
ABLATION_ROIS=inf_lateral_ventricle,hippocampus,amygdala,accumbens_area,insula \
  .venv/bin/python colab/run_roi_ablation_exp2.py

# 3c) Ablação LSTM — requer fold_best_params ou checkpoints no baseline
#     Se o treino LSTM antigo não tem checkpoints:
#       RUN_DIR=colab/exp2/balanced/lstm .venv/bin/python colab/export_fold_best_params.py
#     (falha sem checkpoints; nesse caso re-treine lstm ou use ABLATION_FORCE_OPTUNA=1)
ABLATION_MODEL=lstm \
ABLATION_ROIS=inf_lateral_ventricle,hippocampus,amygdala,accumbens_area,insula \
  .venv/bin/python colab/run_roi_ablation_exp2.py

# 3d) XGB + SVM numa só invocação
ABLATION_MODELS=xgboost,svm \
ABLATION_ROIS=inf_lateral_ventricle,hippocampus,amygdala,accumbens_area,insula \
  .venv/bin/python colab/run_roi_ablation_exp2.py

# 4) Re-postprocess ablação (se ABLATION_SKIP_POSTPROCESS=1 no passo 3)
.venv/bin/python colab/postprocess_exp2_ablation.py
