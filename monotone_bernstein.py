from dataclasses import dataclass
import math

import joblib
import numpy as np
from scipy.optimize import lsq_linear


def normalize_to_unit_interval(x: np.ndarray, a: float, b: float, clip: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    den = max(float(b - a), 1e-12)
    t = (x - float(a)) / den
    if clip:
        t = np.clip(t, 0.0, 1.0)
    return t


def bernstein_basis_at_t(t: np.ndarray, degree: int) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    n = int(degree)
    basis = np.zeros((t.size, n + 1), dtype=float)
    for k in range(n + 1):
        c = float(math.comb(n, k))
        basis[:, k] = c * np.power(t, k) * np.power(1.0 - t, n - k)
    return basis


def bernstein_design_matrix(
    x: np.ndarray,
    degree: int,
    a: float | None = None,
    b: float | None = None,
    clip: bool = True,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("x must not be empty")

    x_min = float(np.min(x)) if a is None else float(a)
    x_max = float(np.max(x)) if b is None else float(b)
    if x_max <= x_min:
        x_max = x_min + 1e-6

    t = normalize_to_unit_interval(x, x_min, x_max, clip=clip)
    bmat = bernstein_basis_at_t(t, degree=degree)
    return bmat, x_min, x_max, t


def order_transform_matrix(degree: int, monotone: str) -> np.ndarray:
    n = int(degree)
    if monotone not in {"increasing", "decreasing"}:
        raise ValueError("monotone must be 'increasing' or 'decreasing'")

    sign = 1.0 if monotone == "increasing" else -1.0
    tmat = np.zeros((n + 1, n + 1), dtype=float)
    tmat[:, 0] = 1.0
    for k in range(1, n + 1):
        tmat[k:, k] = sign
    return tmat


def second_difference_matrix(n_coef: int) -> np.ndarray:
    m = int(n_coef) - 2
    if m <= 0:
        return np.zeros((0, int(n_coef)), dtype=float)

    d2 = np.zeros((m, int(n_coef)), dtype=float)
    for i in range(m):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    return d2


@dataclass
class BernsteinMonotoneConfig:
    degree: int = 7
    monotone: str = "increasing"
    lambda_smooth: float = 0.0
    clip: bool = True


@dataclass
class BernsteinMonotoneState:
    degree: int
    monotone: str
    lambda_smooth: float
    x_min: float
    x_max: float
    coef: np.ndarray
    clip: bool


class BernsteinMonotoneRegressor:
    def __init__(self, config: BernsteinMonotoneConfig):
        self.config = config
        self.state: BernsteinMonotoneState | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size < 2:
            raise ValueError("At least 2 samples are required.")

        degree = int(self.config.degree)
        monotone = str(self.config.monotone)
        lambda_smooth = float(self.config.lambda_smooth)

        bmat, x_min, x_max, _ = bernstein_design_matrix(
            x,
            degree=degree,
            clip=self.config.clip,
        )
        tmat = order_transform_matrix(degree=degree, monotone=monotone)
        a = bmat @ tmat
        y_aug = y.copy()

        if lambda_smooth > 0.0:
            d2 = second_difference_matrix(degree + 1)
            reg = np.sqrt(lambda_smooth) * (d2 @ tmat)
            a = np.vstack([a, reg])
            y_aug = np.concatenate([y_aug, np.zeros(reg.shape[0], dtype=float)])

        lb = np.concatenate([[-np.inf], np.zeros(degree, dtype=float)])
        ub = np.full(degree + 1, np.inf, dtype=float)

        res = lsq_linear(a, y_aug, bounds=(lb, ub), method="bvls", max_iter=2000)
        if not res.success:
            raise RuntimeError(f"Constrained least squares failed: {res.message}")

        z = np.asarray(res.x, dtype=float)
        coef = tmat @ z

        self.state = BernsteinMonotoneState(
            degree=degree,
            monotone=monotone,
            lambda_smooth=lambda_smooth,
            x_min=float(x_min),
            x_max=float(x_max),
            coef=coef,
            clip=bool(self.config.clip),
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Model is not fitted.")

        bmat, _, _, _ = bernstein_design_matrix(
            np.asarray(x, dtype=float).reshape(-1),
            degree=self.state.degree,
            a=self.state.x_min,
            b=self.state.x_max,
            clip=self.state.clip,
        )
        return np.asarray(bmat @ self.state.coef, dtype=float).reshape(-1)

    def predict_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Model is not fitted.")

        x = np.asarray(x, dtype=float).reshape(-1)
        if self.state.degree <= 0:
            return np.zeros_like(x)

        n = int(self.state.degree)
        scale = 1.0 / max(self.state.x_max - self.state.x_min, 1e-12)

        coef_diff = self.state.coef[1:] - self.state.coef[:-1]
        bmat_n1, _, _, _ = bernstein_design_matrix(
            x,
            degree=n - 1,
            a=self.state.x_min,
            b=self.state.x_max,
            clip=self.state.clip,
        )
        deriv = n * (bmat_n1 @ coef_diff) * scale
        return np.asarray(deriv, dtype=float).reshape(-1)

    def monotonicity_metrics(self, x_grid: np.ndarray) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Model is not fitted.")

        d = self.predict_derivative(x_grid)
        if self.state.monotone == "increasing":
            violation = d < 0.0
            max_violation = float(np.max(np.maximum(-d, 0.0)))
        else:
            violation = d > 0.0
            max_violation = float(np.max(np.maximum(d, 0.0)))

        return {
            "violation_rate": float(np.mean(violation)),
            "max_violation_derivative": max_violation,
            "mean_derivative": float(np.mean(d)),
        }

    def to_dict(self) -> dict:
        if self.state is None:
            raise RuntimeError("Model is not fitted.")

        return {
            "state": {
                "degree": self.state.degree,
                "monotone": self.state.monotone,
                "lambda_smooth": self.state.lambda_smooth,
                "x_min": self.state.x_min,
                "x_max": self.state.x_max,
                "coef": self.state.coef,
                "clip": self.state.clip,
            }
        }

    @classmethod
    def from_dict(cls, data: dict):
        st = data["state"]
        cfg = BernsteinMonotoneConfig(
            degree=int(st["degree"]),
            monotone=str(st["monotone"]),
            lambda_smooth=float(st.get("lambda_smooth", 0.0)),
            clip=bool(st.get("clip", True)),
        )
        reg = cls(cfg)
        reg.state = BernsteinMonotoneState(
            degree=int(st["degree"]),
            monotone=str(st["monotone"]),
            lambda_smooth=float(st.get("lambda_smooth", 0.0)),
            x_min=float(st["x_min"]),
            x_max=float(st["x_max"]),
            coef=np.asarray(st["coef"], dtype=float),
            clip=bool(st.get("clip", True)),
        )
        return reg


def save_model(path: str, model: BernsteinMonotoneRegressor, extra: dict | None = None):
    payload = model.to_dict()
    payload["extra"] = {} if extra is None else extra
    joblib.dump(payload, path)


def load_model(path: str) -> tuple[BernsteinMonotoneRegressor, dict]:
    payload = joblib.load(path)
    model = BernsteinMonotoneRegressor.from_dict(payload)
    return model, payload.get("extra", {})


def threshold_from_target_d50(model: BernsteinMonotoneRegressor, d50_target: float) -> float:
    return float(model.predict(np.array([float(d50_target)], dtype=float))[0])


def invert_monotone_binary_search(
    model: BernsteinMonotoneRegressor,
    y_target: float,
    x_lo: float | None = None,
    x_hi: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if model.state is None:
        raise RuntimeError("Model is not fitted.")

    lo = model.state.x_min if x_lo is None else float(x_lo)
    hi = model.state.x_max if x_hi is None else float(x_hi)

    f_lo = float(model.predict(np.array([lo]))[0])
    f_hi = float(model.predict(np.array([hi]))[0])
    y_t = float(y_target)

    increasing = model.state.monotone == "increasing"
    if increasing:
        if y_t <= f_lo:
            return lo
        if y_t >= f_hi:
            return hi
    else:
        if y_t >= f_lo:
            return lo
        if y_t <= f_hi:
            return hi

    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = float(model.predict(np.array([mid]))[0])

        if abs(f_mid - y_t) < tol or abs(right - left) < tol:
            return mid

        if increasing:
            if f_mid < y_t:
                left = mid
            else:
                right = mid
        else:
            if f_mid > y_t:
                left = mid
            else:
                right = mid

    return 0.5 * (left + right)
