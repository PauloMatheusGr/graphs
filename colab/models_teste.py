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
        "--csv",
        default="/mnt/study-data/pgirardi/graphs/csvs/abordagem_4_teste/features_all_abordagem_4_teste.csv",
        help="Caminho para o CSV de features.",
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

    try:
        from pycaret.classification import ClassificationExperiment
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PyCaret não está instalado neste ambiente.\n"
            "Instale com: pip install pycaret\n"
            f"Erro original: {type(e).__name__}: {e}"
        )

    df = pd.read_csv(args.csv)

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

        exp = ClassificationExperiment()
        exp.setup(
            data=train_df,
            target="GROUP",
            test_data=test_df,
            index=False,
            ignore_features=["strat_col", "ID_PT"],
            categorical_features=cat_feats,
            session_id=args.seed,
            normalize=True,
            normalize_method="zscore",
            fix_imbalance=True,
            fold=args.inner_fold,
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

