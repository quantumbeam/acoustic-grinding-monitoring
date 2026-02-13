from dataclasses import dataclass

import joblib
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear


@dataclass
class ISplineConfig:
    degree: int = 3
    n_internal_knots: int = 6
    monotone: str = "increasing"  # increasing or decreasing


@dataclass
class ISplineModelState:
    intercept: float
    coef: np.ndarray
    knot_vector: np.ndarray
    degree: int
    x_min: float
    x_max: float
    monotone: str


class MonotoneISplineRegressor:
    def __init__(self, config: ISplineConfig):
        self.config = config
        self.state: ISplineModelState | None = None
        self._basis_splines: list[BSpline] = []
        self._antiderivatives: list[BSpline] = []
        self._norm_factors: np.ndarray | None = None

    def _build_knot_vector(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max <= x_min:
            x_max = x_min + 1e-6

        n_int = max(int(self.config.n_internal_knots), 0)
        if n_int > 0:
            q = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
            internal = np.quantile(x, q)
            internal = np.unique(internal)
            internal = internal[(internal > x_min) & (internal < x_max)]
        else:
            internal = np.array([], dtype=float)

        k = int(self.config.degree)
        t = np.concatenate([
            np.repeat(x_min, k + 1),
            np.asarray(internal, dtype=float),
            np.repeat(x_max, k + 1),
        ])
        return t

    def _prepare_basis(self, knot_vector: np.ndarray):
        k = int(self.config.degree)
        n_basis = len(knot_vector) - k - 1
        self._basis_splines = []
        self._antiderivatives = []
        for j in range(n_basis):
            c = np.zeros(n_basis, dtype=float)
            c[j] = 1.0
            b = BSpline(knot_vector, c, k, extrapolate=False)
            self._basis_splines.append(b)
            self._antiderivatives.append(b.antiderivative())

    def _ispline_design(self, x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
        assert self._antiderivatives
        x = np.asarray(x, dtype=float).reshape(-1)
        x_clip = np.clip(x, x_min, x_max)

        n = x_clip.shape[0]
        p = len(self._antiderivatives)
        design = np.zeros((n, p), dtype=float)

        if self._norm_factors is None:
            norms = []
            for a in self._antiderivatives:
                total = float(a(x_max) - a(x_min))
                norms.append(total if abs(total) > 1e-12 else 1.0)
            self._norm_factors = np.array(norms, dtype=float)

        for j, a in enumerate(self._antiderivatives):
            vals = a(x_clip) - a(x_min)
            design[:, j] = np.asarray(vals, dtype=float) / self._norm_factors[j]

        return design

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size < 2:
            raise ValueError("At least 2 samples are required.")

        sign = 1.0 if self.config.monotone == "increasing" else -1.0
        y_fit = sign * y

        knot_vector = self._build_knot_vector(x)
        self._prepare_basis(knot_vector)

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        self._norm_factors = None
        i_design = self._ispline_design(x, x_min=x_min, x_max=x_max)

        a = np.column_stack([np.ones_like(x), i_design])
        lb = np.concatenate([[-np.inf], np.zeros(i_design.shape[1], dtype=float)])
        ub = np.full(i_design.shape[1] + 1, np.inf, dtype=float)

        res = lsq_linear(a, y_fit, bounds=(lb, ub), method="trf")
        if not res.success:
            raise RuntimeError(f"Constrained least squares failed: {res.message}")

        intercept = float(res.x[0])
        coef = np.asarray(res.x[1:], dtype=float)

        self.state = ISplineModelState(
            intercept=intercept,
            coef=coef,
            knot_vector=np.asarray(knot_vector, dtype=float),
            degree=int(self.config.degree),
            x_min=x_min,
            x_max=x_max,
            monotone=self.config.monotone,
        )
        return self

    def _predict_increasing(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None
        design = self._ispline_design(x, x_min=self.state.x_min, x_max=self.state.x_max)
        return self.state.intercept + design @ self.state.coef

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None
        sign = 1.0 if self.state.monotone == "increasing" else -1.0
        return sign * self._predict_increasing(x)

    def predict_derivative(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None
        x = np.asarray(x, dtype=float).reshape(-1)
        x_clip = np.clip(x, self.state.x_min, self.state.x_max)

        deriv = np.zeros_like(x_clip, dtype=float)
        for j, b in enumerate(self._basis_splines):
            vals = np.asarray(b(x_clip), dtype=float)
            deriv += self.state.coef[j] * vals / self._norm_factors[j]

        sign = 1.0 if self.state.monotone == "increasing" else -1.0
        return sign * deriv

    def monotonicity_metrics(self, x_grid: np.ndarray) -> dict[str, float]:
        d = self.predict_derivative(x_grid)
        if self.state is None:
            raise RuntimeError("Model is not fitted.")

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
        assert self.state is not None
        return {
            "state": {
                "intercept": self.state.intercept,
                "coef": self.state.coef,
                "knot_vector": self.state.knot_vector,
                "degree": self.state.degree,
                "x_min": self.state.x_min,
                "x_max": self.state.x_max,
                "monotone": self.state.monotone,
            }
        }

    @classmethod
    def from_dict(cls, data: dict):
        state_d = data["state"]
        cfg = ISplineConfig(
            degree=int(state_d["degree"]),
            n_internal_knots=max(int(len(state_d["knot_vector"]) - 2 * (int(state_d["degree"]) + 1)), 0),
            monotone=str(state_d["monotone"]),
        )
        reg = cls(cfg)
        reg.state = ISplineModelState(
            intercept=float(state_d["intercept"]),
            coef=np.asarray(state_d["coef"], dtype=float),
            knot_vector=np.asarray(state_d["knot_vector"], dtype=float),
            degree=int(state_d["degree"]),
            x_min=float(state_d["x_min"]),
            x_max=float(state_d["x_max"]),
            monotone=str(state_d["monotone"]),
        )
        reg._prepare_basis(reg.state.knot_vector)
        reg._norm_factors = None
        _ = reg._ispline_design(np.array([reg.state.x_min, reg.state.x_max]), reg.state.x_min, reg.state.x_max)
        return reg


def save_model(path: str, model: MonotoneISplineRegressor, extra: dict | None = None):
    payload = model.to_dict()
    payload["extra"] = {} if extra is None else extra
    joblib.dump(payload, path)


def load_model(path: str) -> tuple[MonotoneISplineRegressor, dict]:
    payload = joblib.load(path)
    model = MonotoneISplineRegressor.from_dict(payload)
    return model, payload.get("extra", {})
