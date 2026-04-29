from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models


DEFAULT_CSV = (
    "/mnt/study-data/pgirardi/graphs/csvs/abordagem_4_teste/"
    "features_all_abordagem_4_teste.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina um CNN 1D simples a partir de um CSV de features."
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Caminho do CSV.")
    parser.add_argument("--kbest", type=int, default=100, help="Número de features (KBest).")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas de treino.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção do teste.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do split.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    pd_feat = pd.read_csv(csv_path)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)

    if "GROUP" not in pd_feat.columns:
        raise KeyError(
            "Coluna target 'GROUP' não encontrada no CSV. "
            f"Colunas disponíveis: {list(pd_feat.columns)[:30]}..."
        )

    # Features numéricas (float) e target
    X_df = pd_feat.select_dtypes(include=["float"])
    # for col in ("t12", "t13"):
    #     if col in X_df.columns:
    #         X_df = X_df.drop([col], axis=1)
    y = pd_feat["GROUP"]

    selector = SelectKBest(score_func=f_classif, k=args.kbest)
    selector.fit(X_df, y)
    selected_features = X_df.columns[selector.get_support()]

    X_selected = pd_feat[selected_features].to_numpy()
    y_target = pd_feat["GROUP"].to_numpy()

    # Encode se não for binário numérico
    if y_target.dtype.kind in {"U", "S", "O"} or len(np.unique(y_target)) > 2:
        y_target = LabelEncoder().fit_transform(y_target)

    # (n_amostras, n_features, 1)
    X_cnn = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_target, test_size=args.test_size, random_state=args.seed
    )

    is_binary = len(np.unique(y_target)) == 2
    if not is_binary:
        raise ValueError(
            "Esse script está configurado para classificação binária no pós-processamento "
            "(threshold 0.5). Seu target parece ter mais de 2 classes."
        )

    model = models.Sequential(
        [
            tf.keras.Input(shape=(X_cnn.shape[1], 1)),
            layers.Conv1D(32, 2, activation="relu"),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    y_pred_proba = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\n--- Metricas ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2-Score: {r2:.4f}")


if __name__ == "__main__":
    main()
