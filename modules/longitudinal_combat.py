"""Longitudinal ComBat (Beer et al., NeuroImage 2020).

Implementação REML/empirical Bayes com intercepto aleatório por sujeito.
O fit usa apenas o treino do fold. A transformação de sujeitos novos usa os
parâmetros de batch congelados e estima o BLUP somente das visitas daquele
sujeito, evitando estimar efeitos de scanner com o conjunto de teste.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


@dataclass(frozen=True)
class LongitudinalCombatModel:
    """Parâmetros necessários para aplicar Longitudinal ComBat fora do treino."""

    feature_names: tuple[str, ...]
    batch_levels: tuple[str, ...]
    bio_beta: np.ndarray
    batch_effects: np.ndarray
    batch_effects_adjusted: np.ndarray
    sigma: np.ndarray
    random_intercept_var: np.ndarray
    gamma_star: np.ndarray
    delta2_star: np.ndarray
    baseline_age_mean: float
    time_mean: float

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Harmoniza batches conhecidos; mantém batches novos sem alteração."""
        required = {"ID_PT", "batch", "baseline_age", "time_years", "SEX"}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Covariáveis Longitudinal ComBat ausentes: {missing}")

        out = data.copy()
        known = out["batch"].astype(str).isin(self.batch_levels).to_numpy()
        if not known.any():
            return out

        work = out.loc[known]
        y = work.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Longitudinal ComBat não aceita features ausentes/não finitas.")

        x_bio = _biological_design(
            work,
            baseline_age_mean=self.baseline_age_mean,
            time_mean=self.time_mean,
        )
        fixed_bio = x_bio @ self.bio_beta
        batch_idx = np.array(
            [self.batch_levels.index(str(v)) for v in work["batch"]],
            dtype=int,
        )
        batch = self.batch_effects[batch_idx]
        batch_adjusted = self.batch_effects_adjusted[batch_idx]
        eta = _subject_blup(
            y - fixed_bio - batch,
            work["ID_PT"].astype(str).to_numpy(),
            self.sigma,
            self.random_intercept_var,
        )
        predicted = fixed_bio + batch + eta
        standardized = (y - predicted + batch_adjusted) / self.sigma
        harmonized = (
            self.sigma
            / np.sqrt(self.delta2_star[batch_idx])
            * (standardized - self.gamma_star[batch_idx])
            + predicted
            - batch_adjusted
        )
        out.loc[known, list(self.feature_names)] = harmonized
        return out


def fit_longitudinal_combat(
    data: pd.DataFrame,
    feature_names: list[str],
    *,
    n_iter: int = 30,
) -> LongitudinalCombatModel:
    """Ajusta Beer et al. com REML, efeitos AGE basal + tempo + SEX e (1|ID_PT)."""
    required = {
        "ID_PT",
        "batch",
        "baseline_age",
        "time_years",
        "SEX",
        *feature_names,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Colunas Longitudinal ComBat ausentes: {missing}")
    if len(feature_names) < 2:
        raise ValueError("Longitudinal ComBat EB exige pelo menos 2 features.")
    if data[list(required)].isna().any().any():
        raise ValueError("Longitudinal ComBat não aceita valores ausentes.")

    work = data.copy()
    work["batch"] = work["batch"].astype(str)
    work["ID_PT"] = work["ID_PT"].astype(str)
    levels = tuple(sorted(work["batch"].unique()))
    if len(levels) < 2:
        raise ValueError("Longitudinal ComBat exige pelo menos 2 batches.")

    counts = work["batch"].value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Longitudinal ComBat exige >=2 observações por batch: "
            f"{too_small.to_dict()}"
        )

    baseline_age_mean = float(work["baseline_age"].mean())
    time_mean = float(work["time_years"].mean())
    x_bio = _biological_design(
        work,
        baseline_age_mean=baseline_age_mean,
        time_mean=time_mean,
    )
    batch_codes = pd.Categorical(work["batch"], categories=levels).codes
    x_batch = np.column_stack(
        [(batch_codes == i).astype(float) for i in range(1, len(levels))]
    )
    exog = np.column_stack([x_bio, x_batch])
    groups = work["ID_PT"].to_numpy()
    y = work[feature_names].to_numpy(dtype=float)
    n_obs, n_features = y.shape

    bio_beta = np.empty((x_bio.shape[1], n_features))
    raw_batch = np.empty((len(levels), n_features))
    sigma = np.empty(n_features)
    random_var = np.empty(n_features)
    predicted = np.empty_like(y)

    for feature_idx, feature in enumerate(feature_names):
        fit = _fit_mixed_model(y[:, feature_idx], exog, groups, feature)
        fixed = np.asarray(fit.fe_params, dtype=float)
        bio_beta[:, feature_idx] = fixed[: x_bio.shape[1]]
        raw_batch[:, feature_idx] = np.r_[0.0, fixed[x_bio.shape[1] :]]
        sigma[feature_idx] = np.sqrt(float(fit.scale))
        random_var[feature_idx] = max(float(np.asarray(fit.cov_re)[0, 0]), 0.0)

        fixed_prediction = exog @ fixed
        eta = _subject_blup(
            (y[:, [feature_idx]] - fixed_prediction[:, None]),
            groups,
            sigma[[feature_idx]],
            random_var[[feature_idx]],
        )
        predicted[:, feature_idx] = fixed_prediction + eta[:, 0]

    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("Variância residual inválida no Longitudinal ComBat.")

    batch_n = np.bincount(batch_codes, minlength=len(levels)).astype(float)
    gamma_ref = -(batch_n[1:] @ raw_batch[1:]) / n_obs
    batch_adjusted = raw_batch + gamma_ref
    expanded_batch = batch_adjusted[batch_codes]
    standardized = (y - predicted + expanded_batch) / sigma

    gamma_hat = np.empty((len(levels), n_features))
    delta2_hat = np.empty_like(gamma_hat)
    batch_rows: list[np.ndarray] = []
    for batch_idx in range(len(levels)):
        rows = np.flatnonzero(batch_codes == batch_idx)
        batch_rows.append(rows)
        gamma_hat[batch_idx] = standardized[rows].mean(axis=0)
        delta2_hat[batch_idx] = standardized[rows].var(axis=0, ddof=1)

    gamma_star, delta2_star = _empirical_bayes(
        standardized,
        batch_rows,
        batch_n,
        gamma_hat,
        delta2_hat,
        n_iter=n_iter,
    )
    return LongitudinalCombatModel(
        feature_names=tuple(feature_names),
        batch_levels=levels,
        bio_beta=bio_beta,
        batch_effects=raw_batch,
        batch_effects_adjusted=batch_adjusted,
        sigma=sigma,
        random_intercept_var=random_var,
        gamma_star=gamma_star,
        delta2_star=delta2_star,
        baseline_age_mean=baseline_age_mean,
        time_mean=time_mean,
    )


def _biological_design(
    data: pd.DataFrame,
    *,
    baseline_age_mean: float,
    time_mean: float,
) -> np.ndarray:
    """Intercepto + idade basal + tempo desde T1 + sexo; sem rótulo diagnóstico."""
    return np.column_stack(
        [
            np.ones(len(data)),
            data["baseline_age"].to_numpy(dtype=float) - baseline_age_mean,
            data["time_years"].to_numpy(dtype=float) - time_mean,
            data["SEX"].to_numpy(dtype=float),
        ]
    )


def _fit_mixed_model(
    endog: np.ndarray,
    exog: np.ndarray,
    groups: np.ndarray,
    feature: str,
):
    model = MixedLM(
        endog=endog,
        exog=exog,
        groups=groups,
        exog_re=np.ones((len(endog), 1)),
    )
    last_error: Exception | None = None
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(reml=True, method=method, disp=False)
            if not fit.converged:
                warnings.warn(
                    f"MixedLM não convergiu para {feature!r}; resultado pode ser instável."
                )
            return fit
        except (np.linalg.LinAlgError, ValueError) as error:
            last_error = error
    raise RuntimeError(f"MixedLM falhou para {feature!r}.") from last_error


def _subject_blup(
    residual: np.ndarray,
    subjects: np.ndarray,
    sigma: np.ndarray,
    random_var: np.ndarray,
) -> np.ndarray:
    """BLUP de intercepto por sujeito usando somente suas próprias visitas."""
    out = np.empty_like(residual)
    sigma2 = sigma**2
    for subject in pd.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        n = len(rows)
        summed = residual[rows].sum(axis=0)
        eta = random_var * summed / (sigma2 + n * random_var)
        out[rows] = eta
    return out


def _empirical_bayes(
    standardized: np.ndarray,
    batch_rows: list[np.ndarray],
    batch_n: np.ndarray,
    gamma_hat: np.ndarray,
    delta2_hat: np.ndarray,
    *,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrinkage paramétrico de Beer et al.; iteração Jacobi igual ao pacote R."""
    gamma_bar = gamma_hat.mean(axis=1)
    tau2 = gamma_hat.var(axis=1, ddof=1)
    d_bar = delta2_hat.mean(axis=1)
    s2 = delta2_hat.var(axis=1, ddof=1)
    if (
        not np.isfinite([*tau2, *s2]).all()
        or (tau2 <= 0).any()
        or (s2 <= 0).any()
    ):
        raise ValueError(
            "Hiperparâmetros EB degenerados; harmonize mais features do mesmo tipo."
        )

    lambda_bar = (d_bar**2 + 2.0 * s2) / s2
    theta_bar = (d_bar**3 + d_bar * s2) / s2
    n_col = batch_n[:, None]
    tau_col = tau2[:, None]
    gamma_bar_col = gamma_bar[:, None]

    gamma_star = (
        n_col * tau_col * gamma_hat + delta2_hat * gamma_bar_col
    ) / (n_col * tau_col + delta2_hat)
    delta2_star = _update_delta(
        standardized, batch_rows, gamma_star, batch_n, lambda_bar, theta_bar
    )
    for _ in range(n_iter):
        gamma_new = (
            n_col * tau_col * gamma_hat + delta2_star * gamma_bar_col
        ) / (n_col * tau_col + delta2_star)
        delta_new = _update_delta(
            standardized, batch_rows, gamma_star, batch_n, lambda_bar, theta_bar
        )
        gamma_star, delta2_star = gamma_new, delta_new

    if not np.isfinite(gamma_star).all() or not np.isfinite(delta2_star).all():
        raise ValueError("Estimativas EB não finitas no Longitudinal ComBat.")
    if (delta2_star <= 0).any():
        raise ValueError("Escala EB não positiva no Longitudinal ComBat.")
    return gamma_star, delta2_star


def _update_delta(
    standardized: np.ndarray,
    batch_rows: list[np.ndarray],
    gamma: np.ndarray,
    batch_n: np.ndarray,
    lambda_bar: np.ndarray,
    theta_bar: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(gamma)
    for batch_idx, rows in enumerate(batch_rows):
        squared = ((standardized[rows] - gamma[batch_idx]) ** 2).sum(axis=0)
        out[batch_idx] = (
            theta_bar[batch_idx] + 0.5 * squared
        ) / (batch_n[batch_idx] / 2.0 + lambda_bar[batch_idx] - 1.0)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    rows: list[dict[str, float | str]] = []
    for subject_idx in range(12):
        batch = "scanner_a" if subject_idx < 6 else "scanner_b"
        batch_shift = 3.0 if batch == "scanner_b" else 0.0
        subject_effect = rng.normal()
        for visit in range(3):
            rows.append(
                {
                    "ID_PT": f"P{subject_idx:02d}",
                    "batch": batch,
                    "baseline_age": 65.0 + subject_idx,
                    "time_years": visit * 0.5,
                    "SEX": float(subject_idx % 2),
                    "f1": subject_effect + visit * 0.2 + batch_shift + rng.normal(0, 0.2),
                    "f2": subject_effect - visit * 0.1 + 0.5 * batch_shift + rng.normal(0, 0.2),
                    "f3": -subject_effect + visit * 0.1 + 1.5 * batch_shift + rng.normal(0, 0.2),
                }
            )
    demo = pd.DataFrame(rows)
    train = demo[~demo["ID_PT"].isin(["P05", "P11"])].copy()
    test = demo[demo["ID_PT"].isin(["P05", "P11"])].copy()
    model = fit_longitudinal_combat(train, ["f1", "f2", "f3"])
    train_h = model.transform(train)
    test_h = model.transform(test)
    assert np.isfinite(train_h[["f1", "f2", "f3"]]).all().all()
    assert np.isfinite(test_h[["f1", "f2", "f3"]]).all().all()
    assert not np.allclose(test_h[["f1", "f2", "f3"]], test[["f1", "f2", "f3"]])
    print("longitudinal_combat self-check OK")
