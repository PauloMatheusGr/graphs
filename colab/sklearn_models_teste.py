# python colab/sklearn_models_teste.py \
#   --csv "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv" \
#   --n-splits 10 \
#   --inner-fold 5 \
#   --top-k 3 \
#   --seed 123 \
#   --feature-selection two_stage \
#   --fs-k-pre 200 \
#   --fs-k-final 30 \
#   --balance downsample \
#   --remove-constant \
#   --corr-threshold 0.95 \
#   --shap \
#   --shap-samples 200 \
#   --shap-background 150 \
#   --outdir "/mnt/study-data/pgirardi/graphs/colab/outputs"

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def filter_by_roi_label(
    df: pd.DataFrame,
    *,
    rois: list[str] | None = None,
    labels: list[int] | None = None,
    roi_label: list[str] | None = None,
) -> pd.DataFrame:
    if ("roi" not in df.columns) or ("label" not in df.columns):
        raise ValueError("CSV precisa ter colunas 'roi' e 'label' para filtrar ROIs.")

    rois = rois or []
    labels = labels or []
    roi_label = roi_label or []

    out = df.copy()
    out["roi"] = out["roi"].astype(str).str.strip()
    out["_label_int"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out[out["_label_int"].notna()].copy()
    out["_label_int"] = out["_label_int"].astype(int)

    if roi_label:
        pairs: list[tuple[str, int]] = []
        for item in roi_label:
            if ":" not in item:
                raise ValueError(f"Use o formato roi:label em --roi-label. Recebi: {item}")
            r, lab = item.split(":", 1)
            pairs.append((r.strip(), int(lab.strip())))

        mask = False
        for r, lab in pairs:
            mask = mask | ((out["roi"] == r) & (out["_label_int"] == lab))
        out = out[mask].copy()
        return out.drop(columns=["_label_int"])

    mask = True
    if rois:
        mask = mask & out["roi"].isin([r.strip() for r in rois])
    if labels:
        mask = mask & out["_label_int"].isin([int(x) for x in labels])

    out = out[mask].copy()
    return out.drop(columns=["_label_int"])


def _pick_positive_class(classes: list[str]) -> str:
    if "pMCI" in classes:
        return "pMCI"
    return sorted(classes)[-1]


def _downsample_patients_by_group_sex(
    df_train: pd.DataFrame,
    *,
    id_col: str = "ID_PT",
    group_col: str = "GROUP",
    sex_col: str = "SEX",
    seed: int = 123,
) -> pd.DataFrame:
    """
    Downsample por paciente: garante o mesmo número de pacientes em cada estrato GROUP+SEX.
    Mantém TODAS as linhas dos pacientes selecionados.
    """
    if group_col not in df_train.columns or sex_col not in df_train.columns or id_col not in df_train.columns:
        raise ValueError("Downsample requer colunas ID_PT, GROUP e SEX.")

    td = df_train.copy()
    td["_strat"] = td[group_col].astype(str) + "_" + td[sex_col].astype(str)
    pt_strat = td.groupby(id_col, sort=False)["_strat"].first()

    rng = np.random.RandomState(int(seed))
    pts_by_strat: dict[str, list[str]] = {}
    for pt, st in pt_strat.items():
        pts_by_strat.setdefault(str(st), []).append(str(pt))

    min_n = min(len(v) for v in pts_by_strat.values())
    selected_pts: set[str] = set()
    for st, pts in pts_by_strat.items():
        pts = list(pts)
        rng.shuffle(pts)
        selected_pts.update(pts[:min_n])

    return td[td[id_col].astype(str).isin(selected_pts)].drop(columns=["_strat"])


def _infer_feature_columns(df: pd.DataFrame, *, ignore: set[str]) -> tuple[list[str], list[str]]:
    """
    Decide quais colunas entram no modelo:
    - num_cols: numéricas
    - cat_cols: categóricas (object/category/bool)
    """
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        elif pd.api.types.is_bool_dtype(df[c]):
            cat_cols.append(c)
        elif pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype):
            cat_cols.append(c)
    return num_cols, cat_cols


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: Callable[[], BaseEstimator]
    supports_proba: bool


def _model_specs(seed: int) -> list[ModelSpec]:
    # Conjunto semelhante ao leaderboard do PyCaret que você mostrou.
    return [
        ModelSpec(
            "Logistic Regression",
            lambda: LogisticRegression(
                max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed
            ),
            True,
        ),
        ModelSpec(
            "Extra Trees Classifier",
            lambda: ExtraTreesClassifier(
                n_estimators=400,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            ),
            True,
        ),
        ModelSpec("Naive Bayes", lambda: GaussianNB(), True),
        # QDA pode falhar com covariância singular; reg_param ajuda a estabilizar
        ModelSpec(
            "Quadratic Discriminant Analysis",
            lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
            True,
        ),
        ModelSpec("K Neighbors Classifier", lambda: KNeighborsClassifier(n_neighbors=11), True),
        ModelSpec("Ridge Classifier", lambda: RidgeClassifier(class_weight="balanced", random_state=seed), False),
        ModelSpec("Linear Discriminant Analysis", lambda: LinearDiscriminantAnalysis(), True),
        ModelSpec(
            "Random Forest Classifier",
            lambda: RandomForestClassifier(
                n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1
            ),
            True,
        ),
        ModelSpec("Ada Boost Classifier", lambda: AdaBoostClassifier(random_state=seed), True),
        ModelSpec("Decision Tree Classifier", lambda: GradientBoostingClassifier(random_state=seed), True),
        ModelSpec(
            "Gradient Boosting Classifier",
            lambda: GradientBoostingClassifier(random_state=seed),
            True,
        ),
        ModelSpec("SVM - Linear Kernel", lambda: LinearSVC(random_state=seed, class_weight="balanced"), False),
        ModelSpec("Dummy Classifier", lambda: DummyClassifier(strategy="most_frequent", random_state=seed), True),
    ]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # roc_auc_score falha se tiver 1 classe só no y_true
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _predict_scores(model: BaseEstimator, X: Any, *, supports_proba: bool) -> np.ndarray:
    """
    Retorna score contínuo para AUC:
    - predict_proba[:,1] quando disponível
    - decision_function quando disponível
    - fallback: predict
    """
    if supports_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)


def _run_shap_for_model(
    fitted_model: BaseEstimator,
    X_train: Any,
    X_test: Any,
    feature_names: list[str],
    *,
    out_dir: str,
    model_name: str,
    samples: int,
    background: int,
) -> None:
    try:
        import shap  # type: ignore
    except Exception as e:
        raise SystemExit(
            "SHAP não está instalado neste ambiente. Instale com: pip install shap\n"
            f"Erro original: {type(e).__name__}: {e}"
        )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib é necessário para salvar plots do SHAP. Instale com: pip install matplotlib\n"
            f"Erro original: {type(e).__name__}: {e}"
        )

    _ensure_dir(out_dir)
    n_bg = max(1, min(int(background), X_train.shape[0]))
    n_samp = max(1, min(int(samples), X_test.shape[0]))
    bg = X_train[:n_bg]
    X_eval = X_test[:n_samp]

    def predict_proba_pos(x_2d: np.ndarray) -> np.ndarray:
        if hasattr(fitted_model, "predict_proba"):
            return np.asarray(fitted_model.predict_proba(x_2d))[:, 1]
        if hasattr(fitted_model, "decision_function"):
            scores = np.asarray(fitted_model.decision_function(x_2d)).reshape(-1)
            return 1.0 / (1.0 + np.exp(-scores))
        return np.asarray(fitted_model.predict(x_2d)).reshape(-1)

    explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg))
    shap_values = explainer(X_eval)
    shap_values = shap.Explanation(
        values=np.asarray(shap_values.values),
        base_values=shap_values.base_values,
        data=X_eval,
        feature_names=feature_names,
    )

    sv = np.asarray(shap_values.values)
    abs_sv = np.abs(sv)
    feat_rank = (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": abs_sv.mean(axis=0).tolist()}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    csv_path = os.path.join(out_dir, f"shap_feature_importance_{safe}.csv")
    feat_rank.to_csv(csv_path, index=False)

    try:
        shap.plots.bar(shap_values, max_display=30, show=False)
        plt.tight_layout()
        p = os.path.join(out_dir, f"shap_bar_{safe}.png")
        plt.savefig(p, dpi=200)
        plt.close()
    except Exception:
        pass

    try:
        shap.plots.beeswarm(shap_values, max_display=30, show=False)
        plt.tight_layout()
        p = os.path.join(out_dir, f"shap_beeswarm_{safe}.png")
        plt.savefig(p, dpi=200)
        plt.close()
    except Exception:
        pass


def _to_dense(a: Any) -> np.ndarray:
    """Converte matriz densa/sparse para np.ndarray (float32)."""
    if hasattr(a, "toarray"):
        return np.asarray(a.toarray(), dtype=np.float32)
    return np.asarray(a, dtype=np.float32)


def _filter_constant_features(
    X_train: Any, X_test: Any, feat_names: list[str]
) -> tuple[Any, Any, list[str]]:
    """
    Remove features constantes usando APENAS o treino (sem vazamento).
    Funciona com matriz densa ou esparsa.
    """
    vt = VarianceThreshold(threshold=0.0)
    X_train2 = vt.fit_transform(X_train)
    X_test2 = vt.transform(X_test)
    mask = vt.get_support()
    names2 = [n for n, keep in zip(feat_names, mask) if bool(keep)]
    return X_train2, X_test2, names2


def _filter_correlated_features(
    X_train: Any,
    X_test: Any,
    feat_names: list[str],
    *,
    threshold: float,
) -> tuple[Any, Any, list[str]]:
    """
    Remove features altamente correlacionadas (|corr| >= threshold) usando APENAS o treino.
    Estratégia greedy: percorre features na ordem atual e remove as que forem muito correlacionadas
    com alguma feature já mantida.

    Observação: aplicamos em versão densa (após preprocessing). Para número de features na casa
    de centenas, é OK.
    """
    thr = float(threshold)
    if thr <= 0.0 or thr >= 1.0:
        return X_train, X_test, feat_names

    Xd = _to_dense(X_train)
    # corrcoef requer variância > 0; assumimos que constantes já foram removidas
    corr = np.corrcoef(Xd, rowvar=False)
    n = corr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # remove j>i se corr alto com i
        for j in range(i + 1, n):
            if keep[j] and np.isfinite(corr[i, j]) and abs(float(corr[i, j])) >= thr:
                keep[j] = False

    idx = np.flatnonzero(keep)
    names2 = [feat_names[i] for i in idx.tolist()]
    X_train2 = _to_dense(X_train)[:, idx]
    X_test2 = _to_dense(X_test)[:, idx]
    return X_train2, X_test2, names2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline scikit-learn equivalente ao models_teste.py (PyCaret), com "
            "nested CV por paciente (ID_PT), seleção de atributos, balanceamento e SHAP."
        )
    )
    parser.add_argument(
        "--csv",
        default="/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv",
        help="Caminho para o CSV de features.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed.")
    parser.add_argument("--n-splits", type=int, default=10, help="Folds externos (StratifiedGroupKFold).")
    parser.add_argument("--inner-fold", type=int, default=5, help="Folds internos (ranking de modelos).")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Quantos modelos selecionar por fold (ranking interno) para avaliar no teste externo.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Lista de modelos (por nome) separados por vírgula. Vazio = todos defaults.",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "kbest", "sfs", "two_stage"],
        default="none",
        help="Seleção de atributos (aplicada por fold externo).",
    )
    parser.add_argument(
        "--remove-constant",
        action="store_true",
        help="Remove features constantes (VarianceThreshold) usando somente o treino do fold.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.0,
        help=(
            "Se > 0, remove features altamente correlacionadas (|corr| >= threshold) "
            "usando somente o treino do fold. Ex.: 0.95."
        ),
    )
    parser.add_argument("--fs-k", type=int, default=30, help="k para kbest/sfs (final).")
    parser.add_argument("--fs-k-pre", type=int, default=200, help="k do estágio 1 do two_stage (kbest).")
    parser.add_argument("--fs-k-final", type=int, default=30, help="k do estágio 2 do two_stage (sfs).")
    parser.add_argument(
        "--sfs-direction",
        choices=["forward", "backward"],
        default="forward",
        help="Direção do SequentialFeatureSelector.",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "downsample"],
        default="none",
        help="Balanceamento no treino do fold externo (por paciente, estratos GROUP+SEX).",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="Lista de ROIs (coluna roi), separadas por vírgula.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Lista de labels (coluna label), separados por vírgula.",
    )
    parser.add_argument(
        "--roi-label",
        type=str,
        default="",
        help="Lista de pares roi:label.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Calcula SHAP para os modelos selecionados (apenas no fold 1, por padrão).",
    )
    parser.add_argument("--shap-samples", type=int, default=200, help="Amostras no SHAP.")
    parser.add_argument("--shap-background", type=int, default=150, help="Background no SHAP.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(os.path.join(os.path.dirname(__file__), "outputs")),
        help="Diretório base de saída.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rois = _parse_csv_list(args.roi)
    labels = [int(x) for x in _parse_csv_list(args.label)]
    roi_label = _parse_csv_list(args.roi_label)
    if rois or labels or roi_label:
        before = df.shape
        df = filter_by_roi_label(df, rois=rois, labels=labels, roi_label=roi_label)
        print(f"[ROI] filtro aplicado: {before} -> {df.shape}")

    required = {"ID_PT", "GROUP", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    # Stratificação multi-nível: GROUP+SEX (como no PyCaret)
    df = df.copy()
    df["strat_col"] = df["GROUP"].astype(str) + "_" + df["SEX"].astype(str)

    classes = sorted(df["GROUP"].astype(str).unique().tolist())
    if len(classes) != 2:
        raise SystemExit(f"Esperado GROUP binário, encontrei {len(classes)} classes: {classes}")
    pos_name = _pick_positive_class(classes)
    y = (df["GROUP"].astype(str) == pos_name).astype(int).to_numpy()
    y_strat = df["strat_col"].astype(str).to_numpy()
    groups = df["ID_PT"].astype(str).to_numpy()

    ignore_cols = {"GROUP", "strat_col", "ID_PT"}
    num_cols, cat_cols = _infer_feature_columns(df, ignore=ignore_cols)
    if not num_cols and not cat_cols:
        raise SystemExit("Não encontrei colunas de features (numéricas ou categóricas).")

    preprocess = _build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

    # Seleciona subconjunto de modelos
    specs = _model_specs(seed=int(args.seed))
    if args.models.strip():
        wanted = {s.strip().lower() for s in args.models.split(",") if s.strip()}
        specs = [sp for sp in specs if sp.name.lower() in wanted]
        if not specs:
            raise SystemExit(f"--models não bateu com nenhum modelo conhecido: {sorted(wanted)}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.outdir, f"sklearn_models_teste_{run_id}")
    _ensure_dir(out_dir)

    outer = StratifiedGroupKFold(
        n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed)
    )
    X_dummy = np.zeros((len(df), 1), dtype=np.int8)

    fold_rows: list[dict[str, Any]] = []
    inner_leaderboards: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X_dummy, y_strat, groups), start=1):
        print(f"\n=== Fold {fold_idx}/{int(args.n_splits)} ===")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)

        if args.balance == "downsample":
            before = train_df.shape
            train_df = _downsample_patients_by_group_sex(train_df, seed=int(args.seed) + fold_idx)
            print(f"[BALANCE] train {before} -> {train_df.shape}")

        # Define X/y por fold
        y_train = (train_df["GROUP"].astype(str) == pos_name).astype(int).to_numpy()
        y_test = (test_df["GROUP"].astype(str) == pos_name).astype(int).to_numpy()
        g_train = train_df["ID_PT"].astype(str).to_numpy()
        strat_train = (train_df["GROUP"].astype(str) + "_" + train_df["SEX"].astype(str)).to_numpy()

        # Fit/transform por fold externo (sem vazamento)
        X_train_all = preprocess.fit_transform(train_df)
        X_test_all = preprocess.transform(test_df)

        # Feature names pós-transform (para SHAP/relatórios)
        try:
            feat_names = list(preprocess.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(X_train_all.shape[1])]

        # Remove constantes / correlacionadas (fit no treino)
        if args.remove_constant:
            X_train_all, X_test_all, feat_names = _filter_constant_features(
                X_train_all, X_test_all, feat_names
            )
        if float(args.corr_threshold) > 0.0:
            X_train_all, X_test_all, feat_names = _filter_correlated_features(
                X_train_all, X_test_all, feat_names, threshold=float(args.corr_threshold)
            )

        # Seleção de atributos por fold externo
        selected_idx = np.arange(X_train_all.shape[1], dtype=np.int64)
        selected_feat_names = feat_names

        if args.feature_selection != "none":
            if args.feature_selection == "kbest":
                k = max(1, min(int(args.fs_k), X_train_all.shape[1]))
                sel = SelectKBest(score_func=f_classif, k=k)
                sel.fit(X_train_all, y_train)
                mask = sel.get_support()
                selected_idx = np.flatnonzero(mask)
            elif args.feature_selection == "two_stage":
                k_pre = max(1, min(int(args.fs_k_pre), X_train_all.shape[1]))
                k_final = max(1, min(int(args.fs_k_final), k_pre))
                sel1 = SelectKBest(score_func=f_classif, k=k_pre)
                sel1.fit(X_train_all, y_train)
                pool_idx = np.flatnonzero(sel1.get_support())
                # SFS no pool (usa LR como estimador base)
                base = LogisticRegression(
                    max_iter=2000, solver="liblinear", class_weight="balanced", random_state=int(args.seed)
                )
                inner_cv = StratifiedGroupKFold(
                    n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
                )
                inner_splits = list(inner_cv.split(X_train_all[:, pool_idx], strat_train, g_train))
                sfs = SequentialFeatureSelector(
                    estimator=base,
                    n_features_to_select=k_final,
                    direction=str(args.sfs_direction),
                    scoring="accuracy",
                    cv=inner_splits,
                    n_jobs=1,
                )
                sfs.fit(X_train_all[:, pool_idx], y_train)
                final_mask = sfs.get_support()
                selected_idx = pool_idx[np.flatnonzero(final_mask)]
            elif args.feature_selection == "sfs":
                k = max(1, min(int(args.fs_k), X_train_all.shape[1]))
                base = LogisticRegression(
                    max_iter=2000, solver="liblinear", class_weight="balanced", random_state=int(args.seed)
                )
                inner_cv = StratifiedGroupKFold(
                    n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
                )
                inner_splits = list(inner_cv.split(X_train_all, strat_train, g_train))
                sfs = SequentialFeatureSelector(
                    estimator=base,
                    n_features_to_select=k,
                    direction=str(args.sfs_direction),
                    scoring="accuracy",
                    cv=inner_splits,
                    n_jobs=1,
                )
                sfs.fit(X_train_all, y_train)
                selected_idx = np.flatnonzero(sfs.get_support())

            selected_feat_names = [feat_names[i] for i in selected_idx.tolist()]

        # Ranking interno de modelos (CV interno por grupos) no treino do fold externo
        inner_cv = StratifiedGroupKFold(
            n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
        )
        inner_splits = list(inner_cv.split(np.zeros((len(train_df), 1), dtype=np.int8), strat_train, g_train))

        inner_rows = []
        for sp in specs:
            scores = []
            failures = 0
            for it, iv in inner_splits:
                est = sp.estimator_factory()
                try:
                    est.fit(X_train_all[it][:, selected_idx], y_train[it])
                    pred = est.predict(X_train_all[iv][:, selected_idx])
                    scores.append(accuracy_score(y_train[iv], pred))
                except Exception:
                    failures += 1
                    continue
            if not scores:
                acc_mean = float("nan")
                acc_std = float("nan")
            else:
                acc_mean = float(np.mean(scores))
                acc_std = float(np.std(scores))
            inner_rows.append(
                {
                    "fold": fold_idx,
                    "Model": sp.name,
                    "Accuracy_inner_mean": acc_mean,
                    "Accuracy_inner_std": acc_std,
                    "inner_fit_failures": int(failures),
                }
            )

        inner_lb = pd.DataFrame(inner_rows).sort_values("Accuracy_inner_mean", ascending=False).reset_index(drop=True)
        inner_leaderboards.append(inner_lb)
        inner_lb.to_csv(os.path.join(out_dir, f"fold_{fold_idx:02d}_leaderboard_inner.csv"), index=False)

        # Seleciona top-k e avalia no teste externo
        top_k = int(args.top_k)
        top_k = max(1, min(top_k, len(inner_lb)))
        # Remove modelos que falharam em todos os splits internos (NaN)
        inner_lb_ok = inner_lb[inner_lb["Accuracy_inner_mean"].notna()].copy()
        if inner_lb_ok.empty:
            print("[WARN] todos os modelos falharam no CV interno deste fold; pulando fold.")
            continue
        top_k = max(1, min(top_k, len(inner_lb_ok)))
        chosen = inner_lb_ok.head(top_k)["Model"].tolist()

        for sp in specs:
            if sp.name not in chosen:
                continue
            model = sp.estimator_factory()
            try:
                model.fit(X_train_all[:, selected_idx], y_train)
                y_pred = model.predict(X_test_all[:, selected_idx])

                y_score = _predict_scores(
                    model, X_test_all[:, selected_idx], supports_proba=sp.supports_proba
                )
                auc = _safe_auc(y_test, y_score)
            except Exception as e:
                print(f"[WARN] modelo falhou no fold externo: {sp.name}: {type(e).__name__}: {e}")
                continue

            row = {
                "fold": fold_idx,
                "Model": sp.name,
                "n_train_rows": int(len(train_df)),
                "n_train_ID_PT": int(train_df["ID_PT"].astype(str).nunique()),
                "n_test_rows": int(len(test_df)),
                "n_test_ID_PT": int(test_df["ID_PT"].astype(str).nunique()),
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "BalancedAccuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "F1_pos": float(f1_score(y_test, y_pred, pos_label=1)),
                "AUC_pos": float(auc),
                "feature_selection": str(args.feature_selection),
                "n_features_total": int(X_train_all.shape[1]),
                "n_features_selected": int(len(selected_idx)),
                "selected_features_json": json.dumps(selected_feat_names, ensure_ascii=False),
                "balance": str(args.balance),
            }
            fold_rows.append(row)
            print(
                f"[fold {fold_idx}] {sp.name} | acc={row['Accuracy']:.4f} "
                f"bacc={row['BalancedAccuracy']:.4f} f1={row['F1_pos']:.4f} auc={row['AUC_pos']:.4f}"
            )

            if args.shap and fold_idx == 1:
                shap_dir = os.path.join(out_dir, "shap")
                _run_shap_for_model(
                    fitted_model=model,
                    X_train=X_train_all[:, selected_idx],
                    X_test=X_test_all[:, selected_idx],
                    feature_names=selected_feat_names,
                    out_dir=shap_dir,
                    model_name=sp.name,
                    samples=int(args.shap_samples),
                    background=int(args.shap_background),
                )

    results = pd.DataFrame(fold_rows)
    results.to_csv(os.path.join(out_dir, "results_by_fold_model.csv"), index=False)

    if not results.empty:
        agg = (
            results.groupby("Model")[["Accuracy", "AUC_pos", "BalancedAccuracy", "F1_pos"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        # achata colunas
        agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
        agg = agg.sort_values("Accuracy_mean", ascending=False).reset_index(drop=True)
        agg.to_csv(os.path.join(out_dir, "leaderboard_cv_agg_by_model_ranked_by_accuracy.csv"), index=False)

    if inner_leaderboards:
        pd.concat(inner_leaderboards, ignore_index=True).to_csv(
            os.path.join(out_dir, "leaderboard_inner_all_folds.csv"), index=False
        )

    print(f"\n[DONE] outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

