import json
from dataclasses import dataclass

import numpy as np

try:
    import gpflow
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "monotone_svgp.py requires optional dependencies 'tensorflow' and 'gpflow'. "
        "Install them in your runtime environment before running monotone GP scripts."
    ) from exc


TF_DTYPE = tf.float64
NP_DTYPE = np.float64


@dataclass
class Standardizer:
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


@dataclass
class MonotoneConfig:
    num_derivative_points: int = 50
    derivative_softness: float = 0.05
    inducing_points: int = 50
    monotone_weight: float = 1.0
    learning_rate: float = 0.01
    steps: int = 2500
    seed: int = 42
    derivative_sign: int = -1  # -1: decreasing (df<=0), +1: increasing (df>=0)


def _safe_std(x: np.ndarray, eps: float = 1e-9) -> float:
    s = float(np.std(x))
    return s if s > eps else 1.0


def build_quantile_points(x: np.ndarray, m: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, m)
    return np.quantile(x.reshape(-1), q).reshape(-1, 1)


def _standard_normal_cdf(x: tf.Tensor) -> tf.Tensor:
    return 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(tf.constant(2.0, dtype=TF_DTYPE))))


class MonotoneSVGPRegressor:
    def __init__(self, config: MonotoneConfig):
        self.config = config
        self.standardizer: Standardizer | None = None
        self.model: gpflow.models.SVGP | None = None
        self.x_deriv_norm: tf.Tensor | None = None
        self.training_summary: dict[str, float] = {}

    def _build_model(self, x_norm: np.ndarray, y_norm: np.ndarray):
        m = min(self.config.inducing_points, x_norm.shape[0])
        z = build_quantile_points(x_norm, m)

        kernel = gpflow.kernels.SquaredExponential(
            variance=tf.constant(1.0, dtype=TF_DTYPE),
            lengthscales=tf.constant(1.0, dtype=TF_DTYPE),
        )
        likelihood = gpflow.likelihoods.Gaussian(variance=tf.constant(0.05, dtype=TF_DTYPE))

        self.model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=z,
            num_latent_gps=1,
        )

        self.x_deriv_norm = tf.convert_to_tensor(
            build_quantile_points(x_norm, self.config.num_derivative_points),
            dtype=TF_DTYPE,
        )

    def _constraint_penalty(self) -> tf.Tensor:
        assert self.model is not None
        assert self.x_deriv_norm is not None

        with tf.GradientTape() as tape:
            tape.watch(self.x_deriv_norm)
            mu, _ = self.model.predict_f(self.x_deriv_norm)
        dmu = tape.gradient(mu, self.x_deriv_norm)
        if dmu is None:
            return tf.constant(0.0, dtype=TF_DTYPE)

        nu = tf.constant(self.config.derivative_softness, dtype=TF_DTYPE)
        sign = tf.constant(float(self.config.derivative_sign), dtype=TF_DTYPE)
        cdf = _standard_normal_cdf((sign * dmu) / nu)
        cdf = tf.clip_by_value(cdf, 1e-10, 1.0)
        return -tf.reduce_mean(tf.math.log(cdf))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: bool = False):
        x_train = np.asarray(x_train, dtype=NP_DTYPE).reshape(-1, 1)
        y_train = np.asarray(y_train, dtype=NP_DTYPE).reshape(-1, 1)

        x_mean = float(np.mean(x_train))
        x_std = _safe_std(x_train)
        y_mean = float(np.mean(y_train))
        y_std = _safe_std(y_train)
        self.standardizer = Standardizer(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

        x_norm = self.standardizer.transform_x(x_train)
        y_norm = self.standardizer.transform_y(y_train)

        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._build_model(x_norm, y_norm)
        assert self.model is not None

        x_tf = tf.convert_to_tensor(x_norm, dtype=TF_DTYPE)
        y_tf = tf.convert_to_tensor(y_norm, dtype=TF_DTYPE)
        data = (x_tf, y_tf)

        optimizer = tf.optimizers.Adam(learning_rate=self.config.learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                elbo = self.model.elbo(data)
                monotone_loss = self._constraint_penalty()
                total_loss = -elbo + self.config.monotone_weight * monotone_loss
            variables = self.model.trainable_variables
            grads = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return elbo, monotone_loss, total_loss

        last_elbo = np.nan
        last_mono = np.nan
        for step in range(self.config.steps):
            elbo_t, mono_t, total_t = train_step()
            last_elbo = float(elbo_t.numpy())
            last_mono = float(mono_t.numpy())
            if verbose and ((step + 1) % 500 == 0 or step == 0):
                print(
                    f"step={step+1:4d}  elbo={float(elbo_t.numpy()):.4f}  "
                    f"mono={float(mono_t.numpy()):.4f}  loss={float(total_t.numpy()):.4f}"
                )

        mu_d, _ = self.predict_df(x_train, n_samples=128)
        violation_mask = self._violation_mask(mu_d)
        violation_rate = float(np.mean(violation_mask))
        max_violation = float(np.max(self._violation_amount(mu_d)))

        self.training_summary = {
            "final_elbo": last_elbo,
            "final_monotone_penalty": last_mono,
            "violation_rate_mu_df": violation_rate,
            "max_violation_df": max_violation,
        }
        return self

    def predict_f(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        assert self.standardizer is not None

        x_query = np.asarray(x_query, dtype=NP_DTYPE).reshape(-1, 1)
        x_norm = self.standardizer.transform_x(x_query)
        mu_n, var_n = self.model.predict_f(tf.convert_to_tensor(x_norm, dtype=TF_DTYPE))
        mu_n = mu_n.numpy().reshape(-1, 1)
        var_n = var_n.numpy().reshape(-1, 1)

        mu = self.standardizer.inverse_y(mu_n)
        var = var_n * (self.standardizer.y_std ** 2)
        return mu.reshape(-1), var.reshape(-1)

    def predict_df(self, x_query: np.ndarray, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        assert self.standardizer is not None

        x_query = np.asarray(x_query, dtype=NP_DTYPE).reshape(-1)
        order = np.argsort(x_query)
        inv_order = np.argsort(order)
        x_sorted = x_query[order]

        x_sorted_tf = tf.convert_to_tensor(
            self.standardizer.transform_x(x_sorted.reshape(-1, 1)),
            dtype=TF_DTYPE,
        )

        with tf.GradientTape() as tape:
            tape.watch(x_sorted_tf)
            mu_n, _ = self.model.predict_f(x_sorted_tf)
        dmu_n = tape.gradient(mu_n, x_sorted_tf)
        if dmu_n is None:
            dmu_n_np = np.zeros_like(x_sorted, dtype=NP_DTYPE)
        else:
            dmu_n_np = dmu_n.numpy().reshape(-1)

        f_samples = self.model.predict_f_samples(x_sorted_tf, num_samples=n_samples).numpy()[:, :, 0]
        df_samples = np.vstack([
            np.gradient(f_samples[i], x_sorted) for i in range(f_samples.shape[0])
        ])
        dvar_n = np.var(df_samples, axis=0, ddof=1)

        scale = self.standardizer.y_std / self.standardizer.x_std
        dmu = dmu_n_np * scale
        dvar = dvar_n * (scale ** 2)

        return dmu[inv_order], dvar[inv_order]

    def evaluate_monotonicity(self, x_grid: np.ndarray) -> dict[str, float]:
        dmu, _ = self.predict_df(x_grid)
        violation_mask = self._violation_mask(dmu)
        violation_amount = self._violation_amount(dmu)
        return {
            "violation_rate": float(np.mean(violation_mask)),
            "max_violation_derivative": float(np.max(violation_amount)),
            "mean_derivative": float(np.mean(dmu)),
        }

    def _violation_mask(self, derivative: np.ndarray) -> np.ndarray:
        if self.config.derivative_sign < 0:
            return derivative > 0.0
        return derivative < 0.0

    def _violation_amount(self, derivative: np.ndarray) -> np.ndarray:
        if self.config.derivative_sign < 0:
            return np.maximum(derivative, 0.0)
        return np.maximum(-derivative, 0.0)


def save_model_npz(path: str, model: MonotoneSVGPRegressor, extra_config: dict):
    assert model.model is not None
    assert model.standardizer is not None

    params = gpflow.utilities.parameter_dict(model.model)
    serializable_params: dict[str, np.ndarray] = {}
    for name, value in params.items():
        serializable_params[name] = np.asarray(value.numpy(), dtype=NP_DTYPE)

    meta = {
        "monotone_config": model.config.__dict__,
        "standardizer": model.standardizer.__dict__,
        "training_summary": model.training_summary,
        "extra_config": extra_config,
    }

    np.savez_compressed(
        path,
        params_json=np.array([json.dumps(meta)]),
        **serializable_params,
    )


def load_model_npz(path: str) -> tuple[MonotoneSVGPRegressor, dict]:
    bundle = np.load(path, allow_pickle=False)
    meta = json.loads(str(bundle["params_json"][0]))

    cfg = MonotoneConfig(**meta["monotone_config"])
    reg = MonotoneSVGPRegressor(config=cfg)

    z_key = next((k for k in bundle.keys() if "inducing_variable.Z" in k), None)
    if z_key is None:
        raise KeyError("Saved model is missing inducing variable parameters.")
    inducing_rows = int(bundle[z_key].shape[0])
    n_dummy = max(inducing_rows, 2)

    dummy_x = np.linspace(0.0, 1.0, n_dummy, dtype=NP_DTYPE).reshape(-1, 1)
    dummy_y = np.zeros((n_dummy, 1), dtype=NP_DTYPE)
    reg.standardizer = Standardizer(**meta["standardizer"])
    reg._build_model(dummy_x, dummy_y)

    assert reg.model is not None
    param_dict = gpflow.utilities.parameter_dict(reg.model)
    for name, var in param_dict.items():
        if name in bundle:
            var.assign(bundle[name])

    reg.training_summary = meta.get("training_summary", {})
    return reg, meta.get("extra_config", {})
