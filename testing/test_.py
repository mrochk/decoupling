import jax
import jax.numpy as jnp
import pytest

from decoupling import Algorithm
from decoupling.scaler import JacobianScaler
from decoupling.utils import collect_information_from_random, function_error

DEGREE = 3
RANK = 4
N = 100

def target(x):
    a, b = x
    return jnp.array([
        1000 * (a ** 3 + b ** 2 + a * b),   # deliberately ~1000x larger gradient
        b ** 3 + a ** 2 + a * b,
    ])

# --------------------------------------------------------------------------- #
# Fixtures: nothing heavy runs at import/collection time.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    key = jax.random.key(0)
    X, Y, J = collect_information_from_random(target, N, key)
    return {"key": key, "X": X, "Y": Y, "J": J}

@pytest.fixture(scope="module")
def scaled(data):
    scaler = JacobianScaler(data["J"])
    Js, Ys = scaler.scale(data["J"], data["Y"])
    return {"scaler": scaler, "Js": Js, "Ys": Ys}

@pytest.fixture(scope="module")
def fitted(data, scaled):
    algo = Algorithm(
        rank=RANK,
        niters=100,
        splines_dof=int(jnp.sqrt(N)),
        splines_degree=DEGREE,
        key=data["key"],
        ninits=3,
    )
    dec = algo.run(data["X"], scaled["Ys"], scaled["Js"])
    return {"algo": algo, "decoupling": dec}

# --------------------------------------------------------------------------- #
# Sanity of the test setup itself.
# --------------------------------------------------------------------------- #
def test_dof_above_floor():
    # the minimum we established: dof >= degree + 1, else the basis is degenerate
    assert int(jnp.sqrt(N)) >= DEGREE + 1

def test_scaler_shapes_and_values(data, scaled):
    n, m, Nn = data["J"].shape
    sf = scaled["scaler"].scaling_factors
    assert sf.shape == (n,)
    assert jnp.all(sf > 0)                       # norms of nonzero jacobians
    # scaling should bring per-output magnitudes to the same ballpark
    Js = scaled["Js"]
    per_output_norm = jnp.linalg.norm(Js.reshape(n, -1), axis=1)
    ratio = per_output_norm.max() / per_output_norm.min()
    assert ratio < 2.0, "scaling did not equalize output magnitudes"

def test_scale_is_invertible(data, scaled):
    # dividing then multiplying back recovers the original jacobian
    scaler = scaled["scaler"]
    J_round = scaled["Js"] * scaler.scaling_factors[:, None, None]
    assert jnp.allclose(J_round, data["J"], rtol=1e-5, atol=1e-6)

# --------------------------------------------------------------------------- #
# Core algorithm behaviour.
# --------------------------------------------------------------------------- #
def test_run_returns_finite_factors(fitted):
    dec = fitted["decoupling"]
    for name in ("W", "V", "H"):
        arr = getattr(dec, name)
        assert jnp.all(jnp.isfinite(arr)), f"{name} contains non-finite values"

def test_error_decreases(fitted):
    # best-so-far error should be monotone non-increasing by construction,
    # and the final best should be well below the first iterate.
    errs = fitted["algo"].info.errors
    running_best = jax.lax.associative_scan(jnp.minimum, errs)
    assert jnp.all(jnp.diff(running_best) <= 1e-8)
    assert errs.min() < errs[0]

def test_no_nan_in_error_history(fitted):
    errs = fitted["algo"].info.errors
    assert jnp.all(jnp.isfinite(errs)), "NaN/inf in error history -- seed blew up"

# --------------------------------------------------------------------------- #
# The unscale aliasing bug we discussed: .W and .factors[0] must agree,
# and unscaling must not mutate-then-desync.
# --------------------------------------------------------------------------- #
def test_unscale_keeps_W_and_factors_consistent(fitted, scaled):
    dec = scaled["scaler"].unscale(fitted["decoupling"])
    assert jnp.allclose(dec.W, dec.factors[0]), (
        ".W and .factors[0] disagree after unscale -- the aliasing bug is back"
    )

def test_unscale_is_idempotent(fitted, scaled):
    scaler = scaled["scaler"]
    once = scaler.unscale(fitted["decoupling"])
    W_once = once.W
    twice = scaler.unscale(once)
    assert jnp.allclose(twice.W, W_once), "unscale applied twice changed W"

# --------------------------------------------------------------------------- #
# End-to-end: the loose functional bound (kept last, kept loose on purpose).
# --------------------------------------------------------------------------- #
def test_end_to_end_function_error(fitted, scaled, data):
    dec = scaled["scaler"].unscale(fitted["decoupling"])
    errors = jnp.asarray(function_error(target, dec, data["X"]))
    # Loose sanity bound: this is the WEAKEST assertion in the suite. It catches
    # gross failure, not subtle regressions -- those are covered by the property
    # tests above. If you want a tighter bound, anchor it to the SCALED error
    # (what the algorithm actually minimizes), not this unscaled function error.
    assert jnp.all(errors < 10.0)
