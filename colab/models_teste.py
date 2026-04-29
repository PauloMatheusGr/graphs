import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


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
    """
    Filtra linhas pelo(s) ROI(s) e/ou label(s) no CSV (colunas 'roi' e 'label').
    """
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
                raise ValueError(
                    f"Use o formato roi:label em --roi-label. Recebi: {item}"
                )
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


def _run_shap_baseline(
    df: pd.DataFrame,
    *,
    out_dir: str,
    seed: int,
    samples: int,
    background: int,
) -> None:
    """
    SHAP simples (baseline) para tabular: LogisticRegression em features numéricas.
    Gera:
      - shap_feature_importance_models_teste.csv
      - shap_roi_importance_models_teste.csv (se houver roi/label)

    Útil quando o pipeline do PyCaret dificulta SHAP direto no modelo final.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        raise SystemExit(
            "SHAP não está instalado. Instale com: pip install shap\n"
            f"Erro original: {type(e).__name__}: {e}"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if "GROUP" not in df.columns:
        raise SystemExit("CSV sem coluna GROUP (target).")

    out_path = out_dir
    _ensure_dir(out_path)

    base_ignore = {"GROUP", "ID_PT", "strat_col"}
    # Mantém somente numéricas (não inclui roi/label/side/pair etc)
    numeric_cols = _get_numeric_feature_cols(df, ignore=base_ignore)
    if not numeric_cols:
        raise SystemExit("Nenhuma coluna numérica encontrada para SHAP baseline.")

    X = df[numeric_cols].to_numpy(dtype=float)
    y = df["GROUP"].astype(str).to_numpy()
    # Binário: pMCI como positivo se existir
    classes = sorted(np.unique(y).tolist())
    pos = _pick_positive_class(classes)
    y_bin = (y == pos).astype(int)

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X,
        y_bin,
        np.arange(len(df), dtype=np.int64),
        test_size=0.2,
        random_state=seed,
        stratify=y_bin if len(np.unique(y_bin)) == 2 else None,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed),
    )
    clf.fit(X_tr, y_tr)

    n_bg = max(1, min(int(background), X_tr.shape[0]))
    n_samp = max(1, min(int(samples), X_te.shape[0]))
    bg = X_tr[:n_bg]
    X_eval = X_te[:n_samp]
    idx_eval = idx_te[:n_samp]

    def predict_proba_pos(x_2d: np.ndarray) -> np.ndarray:
        return clf.predict_proba(x_2d)[:, 1]

    explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg))
    shap_values = explainer(X_eval)
    sv = np.asarray(shap_values.values)
    abs_sv = np.abs(sv)

    feat_rank = (
        pd.DataFrame({"feature": numeric_cols, "mean_abs_shap": abs_sv.mean(axis=0).tolist()})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    feat_csv = os.path.join(out_path, "shap_feature_importance_models_teste.csv")
    feat_rank.to_csv(feat_csv, index=False)
    print(f"[SHAP] feature importance -> {feat_csv}")

    if ("roi" in df.columns) and ("label" in df.columns):
        meta = df.iloc[idx_eval][["roi", "label"]].copy()
        meta["sample_mean_abs_shap"] = abs_sv.mean(axis=1)
        roi_rank = (
            meta.groupby(["roi", "label"], dropna=False)["sample_mean_abs_shap"]
            .mean()
            .reset_index()
            .sort_values("sample_mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        roi_csv = os.path.join(out_path, "shap_roi_importance_models_teste.csv")
        roi_rank.to_csv(roi_csv, index=False)
        print(f"[SHAP] ROI/label importance -> {roi_csv}")


def _run_shap_pycaret_models(
    exp,
    models: list,
    *,
    out_dir: str,
    fold_idx: int,
    seed: int,
    samples: int,
    background: int,
) -> None:
    """
    Calcula SHAP para modelos retornados pelo PyCaret (após setup/compare_models),
    usando as matrizes pré-processadas (X_train / X_test) do experimento.

    Saída por modelo:
      - shap_feature_importance_pycaret_foldXX_<model>.csv
      - shap_bar_pycaret_foldXX_<model>.png
      - shap_beeswarm_pycaret_foldXX_<model>.png
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        raise SystemExit(
            "SHAP não está instalado. Instale com: pip install shap\n"
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

    # Features já pré-processadas pelo PyCaret (após encoding/imputação/normalize)
    X_train = exp.get_config("X_train")
    X_test = exp.get_config("X_test")

    # Pode ser numpy ou DataFrame, mas geralmente é DataFrame
    if hasattr(X_train, "to_numpy"):
        feat_names = list(X_train.columns)
        X_tr = X_train.to_numpy()
        X_te = X_test.to_numpy()
    else:
        X_tr = np.asarray(X_train)
        X_te = np.asarray(X_test)
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]

    n_bg = max(1, min(int(background), X_tr.shape[0]))
    n_samp = max(1, min(int(samples), X_te.shape[0]))
    bg = X_tr[:n_bg]
    X_eval = X_te[:n_samp]

    for m in models:
        # Nome estável para arquivo
        try:
            mname = type(m).__name__
        except Exception:
            mname = "model"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in mname)

        # Wrapper: algumas classes não têm predict_proba.
        def predict_proba_pos(x_2d: np.ndarray) -> np.ndarray:
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(x_2d)
                # binário -> coluna 1
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return np.asarray(proba).reshape(-1)
            # fallback: decision_function ou predict
            if hasattr(m, "decision_function"):
                scores = m.decision_function(x_2d)
                scores = np.asarray(scores).reshape(-1)
                # converte score em pseudo-probabilidade (sigmoid)
                return 1.0 / (1.0 + np.exp(-scores))
            return np.asarray(m.predict(x_2d)).reshape(-1)

        explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg))
        shap_values = explainer(X_eval)

        sv = np.asarray(shap_values.values)
        abs_sv = np.abs(sv)
        feat_rank = (
            pd.DataFrame(
                {"feature": feat_names, "mean_abs_shap": abs_sv.mean(axis=0).tolist()}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        csv_path = os.path.join(
            out_dir, f"shap_feature_importance_pycaret_fold{fold_idx:02d}_{safe}.csv"
        )
        feat_rank.to_csv(csv_path, index=False)
        print(f"[SHAP][PyCaret] feature importance -> {csv_path}")

        # plots
        try:
            shap.plots.bar(shap_values, max_display=30, show=False)
            plt.tight_layout()
            p = os.path.join(out_dir, f"shap_bar_pycaret_fold{fold_idx:02d}_{safe}.png")
            plt.savefig(p, dpi=200)
            plt.close()
            print(f"[SHAP][PyCaret] bar plot -> {p}")
        except Exception as e:
            print(f"[SHAP][PyCaret][WARN] falha bar plot ({safe}): {e}")

        try:
            shap.plots.beeswarm(shap_values, max_display=30, show=False)
            plt.tight_layout()
            p = os.path.join(
                out_dir, f"shap_beeswarm_pycaret_fold{fold_idx:02d}_{safe}.png"
            )
            plt.savefig(p, dpi=200)
            plt.close()
            print(f"[SHAP][PyCaret] beeswarm plot -> {p}")
        except Exception as e:
            print(f"[SHAP][PyCaret][WARN] falha beeswarm ({safe}): {e}")


def _pick_positive_class(classes: list[str]) -> str:
    # Prefer the domain convention if present; otherwise fall back to the 2nd
    # class in sorted order (works for binary classification).
    if "pMCI" in classes:
        return "pMCI"
    return sorted(classes)[-1]


def _get_numeric_feature_cols(df: pd.DataFrame, *, ignore: set[str]) -> list[str]:
    numeric_cols: list[str] = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            # bools are fine but usually encode categories; keep them numeric anyway
            numeric_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def _select_kbest_features(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    numeric_cols: list[str],
    k: int,
) -> list[str]:
    if not numeric_cols:
        return []
    k = max(1, min(int(k), len(numeric_cols)))
    X = train_df[numeric_cols].to_numpy()
    y = train_df[target_col].astype(str).to_numpy()
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    return [c for c, keep in zip(numeric_cols, mask) if bool(keep)]


def _select_sfs_features(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    groups_col: str,
    numeric_cols: list[str],
    k: int,
    inner_splits: int,
    seed: int,
    direction: str,
) -> list[str]:
    if not numeric_cols:
        return []
    k = max(1, min(int(k), len(numeric_cols)))

    X = train_df[numeric_cols].to_numpy()
    y = train_df[target_col].astype(str).to_numpy()
    groups = train_df[groups_col].astype(str).to_numpy()

    # Estimator simples e robusto para seleção: LR com padronização.
    base_est = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        ),
    )
    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )
    # IMPORTANTE (compatibilidade sklearn):
    # Em algumas versões do scikit-learn, o SequentialFeatureSelector.fit()
    # não aceita `groups=`. Para manter a restrição por paciente (ID_PT),
    # pré-computamos os splits do CV interno e passamos a lista de índices
    # como `cv=...` (isso impede vazamento de grupos sem precisar de groups=).
    inner_splits_idx = list(inner_cv.split(X, y, groups))

    sfs = SequentialFeatureSelector(
        estimator=base_est,
        n_features_to_select=k,
        direction=direction,
        scoring="accuracy",
        cv=inner_splits_idx,
        # Evita excesso de processos/joblib (pode causar warnings e instabilidade
        # em ambientes compartilhados). Ajuste se quiser paralelizar.
        n_jobs=1,
    )
    sfs.fit(X, y)
    mask = sfs.get_support()
    return [c for c, keep in zip(numeric_cols, mask) if bool(keep)]


def _select_two_stage_features(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    groups_col: str,
    numeric_cols: list[str],
    k_pre: int,
    k_final: int,
    inner_splits: int,
    seed: int,
    direction: str,
) -> tuple[list[str], list[str]]:
    """
    Estágio 1 (rápido): SelectKBest reduz o espaço de busca.
    Estágio 2 (caro, multivariado): SequentialFeatureSelector no pool reduzido.

    Retorna (kbest_cols, final_cols).
    """
    if not numeric_cols:
        return [], []

    kbest_cols = _select_kbest_features(
        train_df, target_col=target_col, numeric_cols=numeric_cols, k=k_pre
    )
    if not kbest_cols:
        return [], []

    final_cols = _select_sfs_features(
        train_df,
        target_col=target_col,
        groups_col=groups_col,
        numeric_cols=kbest_cols,
        k=k_final,
        inner_splits=inner_splits,
        seed=seed,
        direction=direction,
    )
    return kbest_cols, final_cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Testa modelos (PyCaret) com StratifiedGroupKFold usando estratificação"
            " em GROUP+SEX e grupos por ID_PT."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["pycaret", "shap", "both"],
        default="pycaret",
        help=(
            "Controla o que executar neste script. "
            "'pycaret' roda apenas a avaliação/modelagem com PyCaret. "
            "'shap' roda apenas SHAP baseline (sem depender do PyCaret). "
            "'both' roda SHAP baseline e também PyCaret."
        ),
    )
    parser.add_argument(
        "--csv",
        default="/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv",
        help="Caminho para o CSV de features.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="Lista de ROIs (coluna roi), separadas por vírgula. Ex.: hippocampus,amygdala",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Lista de labels (coluna label), separados por vírgula. Ex.: 17,53",
    )
    parser.add_argument(
        "--roi-label",
        type=str,
        default="",
        help="Lista de pares roi:label. Ex.: hippocampus:17,hippocampus:53",
    )
    parser.add_argument(
        "--balance",
        choices=["none", "downsample"],
        default="none",
        help=(
            "Balanceamento aplicado SOMENTE no treino do fold externo. "
            "'none' não balanceia; "
            "'downsample' faz downsampling por paciente (ID_PT) balanceando GROUP+SEX."
        ),
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Gera SHAP baseline (LogReg em features numéricas) e salva ranking de atributos/ROIs.",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=400,
        help="Quantas amostras usar na explicação SHAP (baseline).",
    )
    parser.add_argument(
        "--shap-background",
        type=int,
        default=250,
        help="Quantas amostras usar como background no SHAP (baseline).",
    )
    parser.add_argument(
        "--shap-pycaret-topk",
        type=int,
        default=0,
        help=(
            "Calcula SHAP também para os top-K modelos retornados pelo PyCaret em cada fold externo. "
            "0 desliga. Ex.: 2 para explicar os 2 melhores modelos por fold."
        ),
    )
    parser.add_argument(
        "--shap-pycaret-samples",
        type=int,
        default=300,
        help="Quantas amostras do teste (por fold) usar no SHAP dos modelos PyCaret.",
    )
    parser.add_argument(
        "--shap-pycaret-background",
        type=int,
        default=200,
        help="Quantas amostras do treino (por fold) usar como background no SHAP dos modelos PyCaret.",
    )
    parser.add_argument(
        "--n-splits",
        "--n-split",
        dest="n_splits",
        type=int,
        default=10,
        help="Número de folds (alias aceito: --n-split).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed de reprodutibilidade.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Quantos modelos selecionar no compare_models.",
    )
    parser.add_argument(
        "--inner-fold",
        type=int,
        default=5,
        help="CV interno do PyCaret dentro de cada fold externo.",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "kbest", "sfs", "two_stage"],
        default="none",
        help=(
            "Método de seleção de atributos aplicado dentro de cada fold externo (no treino). "
            "'kbest' é univariado; 'sfs' é multivariado (wrapper); "
            "'two_stage' aplica kbest -> sfs (recomendado quando há muitas features)."
        ),
    )
    parser.add_argument(
        "--fs-k",
        type=int,
        default=30,
        help="Número de atributos numéricos a selecionar (para kbest/sfs).",
    )
    parser.add_argument(
        "--fs-k-pre",
        type=int,
        default=120,
        help=(
            "Estágio 1 (two_stage): quantas features numéricas manter após SelectKBest "
            "(pool para o SFS)."
        ),
    )
    parser.add_argument(
        "--fs-k-final",
        type=int,
        default=30,
        help="Estágio 2 (two_stage): quantas features numéricas selecionar com SFS.",
    )
    parser.add_argument(
        "--sfs-direction",
        choices=["forward", "backward"],
        default="forward",
        help="Direção do SequentialFeatureSelector (para --feature-selection sfs ou two_stage).",
    )
    args = parser.parse_args()

    if args.feature_selection == "two_stage":
        if args.fs_k_pre < 1:
            raise SystemExit("--fs-k-pre deve ser >= 1")
        if args.fs_k_final < 1:
            raise SystemExit("--fs-k-final deve ser >= 1")
        if args.fs_k_final > args.fs_k_pre:
            raise SystemExit(
                "Em --feature-selection two_stage, exija --fs-k-final <= --fs-k-pre "
                "(o SFS seleciona dentro do pool do KBest)."
            )

    df = pd.read_csv(args.csv)
    rois = _parse_csv_list(args.roi)
    labels = [int(x) for x in _parse_csv_list(args.label)]
    roi_label = _parse_csv_list(args.roi_label)
    if rois or labels or roi_label:
        before = df.shape
        df = filter_by_roi_label(df, rois=rois, labels=labels, roi_label=roi_label)
        print(f"[ROI] filtro aplicado: {before} -> {df.shape}")

    want_shap = (args.mode in ("shap", "both")) or bool(args.shap)
    want_pycaret = args.mode in ("pycaret", "both")

    if want_shap:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        shap_dir = os.path.join(os.path.dirname(__file__), "outputs", f"shap_models_teste_{run_id}")
        _run_shap_baseline(
            df,
            out_dir=shap_dir,
            seed=int(args.seed),
            samples=int(args.shap_samples),
            background=int(args.shap_background),
        )

    # Se o modo for apenas SHAP, encerramos aqui (evita depender do PyCaret neste ambiente)
    if not want_pycaret:
        return 0

    try:
        from pycaret.classification import ClassificationExperiment
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PyCaret não está instalado neste ambiente.\n"
            "Instale com: pip install pycaret\n"
            f"Erro original: {type(e).__name__}: {e}"
        )

    required = {"ID_PT", "GROUP", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    # Estratificação multi-nível: classe + sexo
    df = df.copy()
    df["strat_col"] = df["GROUP"].astype(str) + "_" + df["SEX"].astype(str)

    # Detecta colunas categóricas automaticamente (strings/categoricals),
    # exceto target e colunas que serão ignoradas.
    ignore_cols = {"strat_col", "ID_PT"}
    cat_feats = [
        c
        for c in df.columns
        if c not in ({"GROUP"} | ignore_cols)
        and (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
        )
    ]
    # Garante que SEX (se existir) esteja incluída.
    if "SEX" in df.columns and "SEX" not in cat_feats and "SEX" not in ignore_cols:
        cat_feats.append("SEX")

    classes = sorted(df["GROUP"].astype(str).unique().tolist())
    if len(classes) != 2:
        raise SystemExit(
            f"Esperado problema binário em GROUP, mas encontrei {len(classes)} classes: {classes}"
        )

    positive_class = _pick_positive_class(classes)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", f"models_teste_{run_id}")
    _ensure_dir(out_dir)

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )

    fold_rows: list[dict] = []
    model_rows: list[dict] = []
    leaderboard_rows: list[pd.DataFrame] = []
    selected_rows: list[dict] = []

    # Importante: o StratifiedGroupKFold estratifica em y (aqui y=strat_col),
    # e garante que nenhum ID_PT vaze entre treino e teste.
    X_dummy = np.zeros((len(df), 1), dtype=np.int8)
    y_strat = df["strat_col"].astype(str).to_numpy()
    groups = df["ID_PT"].astype(str).to_numpy()

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_dummy, y_strat, groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Seleção de atributos (somente numéricos), fitada APENAS no treino do fold externo.
        # As colunas categóricas continuam sendo passadas ao PyCaret via categorical_features.
        base_ignore = {"GROUP", "strat_col", "ID_PT"}
        numeric_cols = _get_numeric_feature_cols(train_df, ignore=base_ignore | set(cat_feats))
        selected_numeric_cols: list[str] = numeric_cols
        kbest_pool_cols: list[str] | None = None
        if args.feature_selection != "none":
            if args.feature_selection == "kbest":
                selected_numeric_cols = _select_kbest_features(
                    train_df, target_col="GROUP", numeric_cols=numeric_cols, k=args.fs_k
                )
            elif args.feature_selection == "sfs":
                selected_numeric_cols = _select_sfs_features(
                    train_df,
                    target_col="GROUP",
                    groups_col="ID_PT",
                    numeric_cols=numeric_cols,
                    k=args.fs_k,
                    inner_splits=args.inner_fold,
                    seed=args.seed,
                    direction=args.sfs_direction,
                )
            elif args.feature_selection == "two_stage":
                kbest_pool_cols, selected_numeric_cols = _select_two_stage_features(
                    train_df,
                    target_col="GROUP",
                    groups_col="ID_PT",
                    numeric_cols=numeric_cols,
                    k_pre=args.fs_k_pre,
                    k_final=args.fs_k_final,
                    inner_splits=args.inner_fold,
                    seed=args.seed,
                    direction=args.sfs_direction,
                )

        keep_cols = ["GROUP", "strat_col", "ID_PT"] + cat_feats + selected_numeric_cols
        # remove duplicatas preservando ordem
        keep_cols = list(dict.fromkeys(keep_cols))
        train_df = train_df[keep_cols].copy()
        test_df = test_df[keep_cols].copy()

        # z-score SEM vazamento: fit no treino do fold externo e aplica no teste externo
        if selected_numeric_cols:
            scaler = StandardScaler()
            scaler.fit(train_df[selected_numeric_cols].to_numpy())
            train_df.loc[:, selected_numeric_cols] = scaler.transform(
                train_df[selected_numeric_cols].to_numpy()
            )
            test_df.loc[:, selected_numeric_cols] = scaler.transform(
                test_df[selected_numeric_cols].to_numpy()
            )

        # Balanceamento por downsample (somente treino) respeitando paciente (ID_PT)
        if args.balance == "downsample":
            td = train_df.copy()
            td["strat__"] = td["GROUP"].astype(str) + "_" + td["SEX"].astype(str)
            pt_strat = td.groupby("ID_PT", sort=False)["strat__"].first()
            rng = np.random.RandomState(int(args.seed) + int(fold_idx))
            pts_by_strat: dict[str, list[str]] = {}
            for pt, st in pt_strat.items():
                pts_by_strat.setdefault(str(st), []).append(str(pt))
            min_n = min(len(v) for v in pts_by_strat.values())
            selected_pts: set[str] = set()
            for st, pts in pts_by_strat.items():
                pts = list(pts)
                rng.shuffle(pts)
                selected_pts.update(pts[:min_n])
            train_df = train_df[train_df["ID_PT"].astype(str).isin(selected_pts)].copy()
            print(
                f"[BALANCE] downsample fold={fold_idx}: "
                f"train rows -> {len(train_df)} | "
                f"ID_PT -> {train_df['ID_PT'].astype(str).nunique()} | "
                f"strata -> {train_df['GROUP'].astype(str).value_counts().to_dict()} / "
                f"{train_df['SEX'].astype(str).value_counts().to_dict()}"
            )

        selected_rows.append(
            {
                "fold": fold_idx,
                "feature_selection": args.feature_selection,
                "fs_k": int(args.fs_k),
                "fs_k_pre": int(args.fs_k_pre) if args.feature_selection == "two_stage" else "",
                "fs_k_final": int(args.fs_k_final) if args.feature_selection == "two_stage" else "",
                "n_numeric_candidates": int(len(numeric_cols)),
                "n_kbest_pool": int(len(kbest_pool_cols)) if kbest_pool_cols is not None else "",
                "n_numeric_selected": int(len(selected_numeric_cols)),
                "kbest_pool_numeric_features": (
                    json.dumps(kbest_pool_cols, ensure_ascii=False)
                    if kbest_pool_cols is not None
                    else ""
                ),
                "selected_numeric_features": json.dumps(selected_numeric_cols, ensure_ascii=False),
            }
        )

        # CV interno (PyCaret) também deve respeitar grupos por paciente (ID_PT)
        inner_splitter = StratifiedGroupKFold(
            n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
        )
        X_dummy_tr = np.zeros((len(train_df), 1), dtype=np.int8)
        y_strat_tr = train_df["strat_col"].astype(str).to_numpy()
        groups_tr = train_df["ID_PT"].astype(str).to_numpy()
        inner_splits_idx = list(inner_splitter.split(X_dummy_tr, y_strat_tr, groups_tr))

        exp = ClassificationExperiment()
        exp.setup(
            data=train_df,
            target="GROUP",
            test_data=test_df,
            index=False,
            ignore_features=["strat_col", "ID_PT"],
            categorical_features=cat_feats,
            session_id=args.seed,
            normalize=False,
            fix_imbalance=False,
            fold=inner_splits_idx,
            verbose=False,
        )

        best_models = exp.compare_models(n_select=args.top_k)
        # Leaderboard completa (CV) do fold (melhor -> pior) para TODOS os modelos avaliados
        try:
            lb = exp.pull()
            if isinstance(lb, pd.DataFrame) and not lb.empty:
                lb = lb.copy()
                lb.insert(0, "fold", fold_idx)
                leaderboard_rows.append(lb)
                lb.to_csv(os.path.join(out_dir, f"fold_{fold_idx:02d}_leaderboard_cv.csv"), index=False)
        except Exception:
            # Se por algum motivo o pull falhar, seguimos sem a leaderboard.
            pass

        if not isinstance(best_models, list):
            best_models = [best_models]

        if int(args.shap_pycaret_topk) > 0:
            k = max(1, min(int(args.shap_pycaret_topk), len(best_models)))
            shap_dir = os.path.join(out_dir, "shap_pycaret")
            _run_shap_pycaret_models(
                exp,
                best_models[:k],
                out_dir=shap_dir,
                fold_idx=int(fold_idx),
                seed=int(args.seed),
                samples=int(args.shap_pycaret_samples),
                background=int(args.shap_pycaret_background),
            )

        # Avalia e salva resultados para cada modelo selecionado (top-k).
        proba_col = f"prediction_score_{positive_class}"
        best_model_str = str(best_models[0])

        for rank, model in enumerate(best_models, start=1):
            model_str = str(model)
            model_tag = f"fold_{fold_idx:02d}_rank_{rank:02d}"

            preds = exp.predict_model(model, data=test_df, raw_score=True)

            y_true = preds["GROUP"].astype(str).to_numpy()
            y_pred = preds["prediction_label"].astype(str).to_numpy()

            # Nem todo modelo retorna probabilidade (ex.: RidgeClassifier pode não expor predict_proba).
            # Quando não houver score, calculamos métricas baseadas em rótulo e deixamos AUC como NaN.
            y_proba_pos = None
            if proba_col in preds.columns:
                y_proba_pos = preds[proba_col].astype(float).to_numpy()
            else:
                # Fallback: se existir prediction_score_<classe>, mas a classe positiva escolhida não estiver
                # (ou se raw_score não estiver disponível), não forçamos erro.
                available_score_cols = [c for c in preds.columns if c.startswith("prediction_score")]
                if not available_score_cols:
                    y_proba_pos = None
                elif len(available_score_cols) == 1 and available_score_cols[0] == "prediction_score":
                    # Esse score é da classe prevista; não é adequado para AUC da classe positiva.
                    y_proba_pos = None

            acc = accuracy_score(y_true, y_pred)
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, pos_label=positive_class)
            auc = float("nan")
            if y_proba_pos is not None:
                auc = roc_auc_score((y_true == positive_class).astype(int), y_proba_pos)
            cm = confusion_matrix(y_true, y_pred, labels=classes)

            model_rows.append(
                {
                    "fold": fold_idx,
                    "rank": rank,
                    "model": model_str,
                    "is_best": model_str == best_model_str,
                    "positive_class": positive_class,
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bacc),
                    "f1_pos": float(f1),
                    "roc_auc_pos": float(auc),
                }
            )

            preds_path = os.path.join(out_dir, f"{model_tag}_preds.csv")
            preds.to_csv(preds_path, index=False)

            report_path = os.path.join(out_dir, f"{model_tag}_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Fold {fold_idx}\n")
                f.write(f"Rank: {rank}\n")
                f.write(f"Positive class: {positive_class}\n")
                f.write(f"Model: {model_str}\n\n")
                f.write(
                    classification_report(
                        y_true, y_pred, labels=classes, target_names=classes, digits=4
                    )
                )
                f.write("\n\nConfusion matrix (labels in order): " + json.dumps(classes) + "\n")
                f.write(np.array2string(cm) + "\n")

        # Resumo do fold (mantém compatibilidade com summary.csv)
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_groups_train": int(train_df["ID_PT"].nunique()),
                "n_groups_test": int(test_df["ID_PT"].nunique()),
                "best_model": best_model_str,
                "positive_class": positive_class,
            }
        )

    summary = pd.DataFrame(fold_rows).sort_values("fold")
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    models_summary = pd.DataFrame(model_rows).sort_values(["fold", "rank"])
    models_summary_path = os.path.join(out_dir, "models_summary.csv")
    models_summary.to_csv(models_summary_path, index=False)

    selected_df = pd.DataFrame(selected_rows).sort_values("fold")
    selected_df.to_csv(os.path.join(out_dir, "selected_features_by_fold.csv"), index=False)

    if leaderboard_rows:
        leaderboard_all = pd.concat(leaderboard_rows, ignore_index=True)
        leaderboard_all_path = os.path.join(out_dir, "leaderboard_cv_all_folds.csv")
        # Se existir a coluna Accuracy, salvamos ordenado por fold e Accuracy (desc).
        if "Accuracy" in leaderboard_all.columns:
            leaderboard_all = leaderboard_all.sort_values(
                ["fold", "Accuracy"], ascending=[True, False]
            )
        else:
            leaderboard_all = leaderboard_all.sort_values(["fold"])
        leaderboard_all.to_csv(leaderboard_all_path, index=False)

        # Agregado: média/std das métricas CV por modelo ao longo dos folds
        metric_cols = [c for c in leaderboard_all.columns if c not in {"fold", "Model"}]
        metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(leaderboard_all[c])]
        if metric_cols:
            agg_cv = (
                leaderboard_all.groupby("Model")[metric_cols]
                .agg(["mean", "std"])
                .reset_index()
            )
            # achata colunas MultiIndex
            agg_cv.columns = [
                ("Model" if c[0] == "Model" else f"{c[0]}_{c[1]}")
                if isinstance(c, tuple)
                else str(c)
                for c in agg_cv.columns
            ]
            agg_cv_path = os.path.join(out_dir, "leaderboard_cv_agg_by_model.csv")
            agg_cv.to_csv(agg_cv_path, index=False)

            # Versão ranqueada por acurácia (média) se disponível
            if "Accuracy_mean" in agg_cv.columns:
                agg_cv_ranked = agg_cv.sort_values("Accuracy_mean", ascending=False)
                agg_cv_ranked_path = os.path.join(
                    out_dir, "leaderboard_cv_agg_by_model_ranked_by_accuracy.csv"
                )
                agg_cv_ranked.to_csv(agg_cv_ranked_path, index=False)

    metrics = ["accuracy", "balanced_accuracy", "f1_pos", "roc_auc_pos"]
    # Agrega métricas do melhor modelo (rank 1) em cada fold
    best_only = models_summary[models_summary["rank"] == 1].copy()
    agg = best_only[metrics].agg(["mean", "std"]).T
    agg_path = os.path.join(out_dir, "summary_agg.csv")
    agg.to_csv(agg_path)

    print(f"Concluído. Resultados em: {out_dir}")
    print("\nResumo (média ± std):")
    for m in metrics:
        print(f"- {m}: {best_only[m].mean():.4f} ± {best_only[m].std(ddof=1):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

