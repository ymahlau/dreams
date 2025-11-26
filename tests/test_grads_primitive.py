import jax
import jax.numpy as jnp
import numpy as onp
import scipy
from jax.test_util import check_grads
from dreams.jax_primitive import spherical_jn, spherical_hankel1, lpmv 

jax.config.update("jax_enable_x64", True)

def test_grad_spherical_jn_z():
    test_points = [
        0.5 + 0.2j,
        1.0 + 0.0j,
        -0.3 + 0.7j,
        1e-10 + 0.0j,   # hits |z| < 1e-8 branch
        0.0 + 0.0j,     # exact zero
    ]

    
    v_cases = [
        0, 1, 2, 5,
        jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
    ]

    for v in v_cases:
        def f_xy(xy):
            x, y = xy
            z = x + 1j * y
            val = spherical_jn(v, z)   
            return jnp.real(val).sum()        # scalarize for grad checking
        g = jax.grad(f_xy)
        g_jit = jax.jit(g)

        for z0 in test_points:
            xy0 = jnp.array([z0.real, z0.imag], dtype=jnp.float64)
            check_grads(
                f_xy,
                (xy0,),
                modes=["fwd", "rev"],
                order=1,
                atol=1e-8,
                rtol=1e-8,
            )

            ge = g(xy0)
            gj = g_jit(xy0)
            onp.testing.assert_allclose(onp.asarray(ge), onp.asarray(gj),
                                        rtol=1e-12, atol=5e-12)


def test_grad_spherical_hankel():
    # Note at z=0 Hankel is singular. .
    test_points = [
        0.5 + 0.2j,
        1.0 + 0.0j,
        -0.3 + 0.7j,
        1e-4 + 2e-4j,   # small but not too small
    ]

    v_cases = [
        0, 1, 2, 5,
        jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
    ]

    for v in v_cases:
        def f_xy(xy):
            x, y = xy
            z = x + 1j * y
            val = spherical_hankel1(v, z)  
            return jnp.real(val).sum()       # scalar output required by check_grads

        g = jax.grad(f_xy)
        g_jit = jax.jit(g)

        for z0 in test_points:
            xy0 = jnp.array([z0.real, z0.imag], dtype=jnp.float64)

            # 1) AD vs finite-diff style checks
            check_grads(
                f_xy,
                (xy0,),
                modes=["fwd", "rev"],
                order=1,
                eps=1e-8,
                atol=1e-5,
                rtol=1e-5,
            )

            ge = g(xy0)
            gj = g_jit(xy0)

            onp.testing.assert_allclose(onp.asarray(ge), onp.asarray(gj),
                                        rtol=1e-12, atol=5e-12)


def test_lpmv_parity():
    rng = onp.random.default_rng(1234)

    ls = onp.arange(-6, 6)  # l = 0..5
    xs = rng.uniform(-0.9, 0.9, size=20)  # avoid |x|=1 where things can blow up

    atol = 1e-12
    rtol = 1e-12

    for l in ls:
        for m in range(-l, l+1):
            for x in xs:
                # SciPy's lpmv: lpmv(m, v=l, x)
                Plm_pos = scipy.special.lpmv(m, l, x)
                Plm_neg = scipy.special.lpmv(m, l, -x)

                # predicted by parity formula
                predicted = (-1)**(l + m) * Plm_pos

                assert onp.allclose(
                    Plm_neg, predicted, rtol=rtol, atol=atol
                ), (
                    f"Parity failed for l={l}, m={m}, x={x}\n"
                    f"  lpmv(m,l,-x) = {Plm_neg}\n"
                    f"  (-1)**(l+m) * lpmv(m,l,x) = {predicted}"
                )

def test_grad_lpmv():
    #  discontinuity points at x=0 and x=pi ( primal flips sign outside [0, pi])
    x_cases = [
        jnp.array(0.7, dtype=jnp.float64),
        jnp.array(-0.6, dtype=jnp.float64),
        jnp.array(3.5, dtype=jnp.float64),  # > pi
        jnp.array([0.6, 1.4, 2.7], dtype=jnp.float64),
        jnp.array([-0.6, 0.8, 3.5], dtype=jnp.float64),
    ]

    mv_cases = [
        (7, 6),
        (2, 4),
        (0, 3),
     ]

    for m, v in mv_cases:
        def f_x(x):
            return jnp.real(lpmv(m, v, x)).sum()  # scalarize for grad checking

        g = jax.grad(f_x)
        g_jit = jax.jit(g)

        for x0 in x_cases:
            # 1) AD vs finite-difference style check
            check_grads(
                f_x,
                (x0,),
                modes=["fwd", "rev"],
                order=1,
                eps=1e-6,     
                atol=1e-5,
                rtol=1e-5,
            )

     
            ge = g(x0)
            gj = g_jit(x0)
            onp.testing.assert_allclose(onp.asarray(ge), onp.asarray(gj),
                                        rtol=1e-12, atol=1e-10)


if __name__ == "__main__":
    #test_grad_lpmv()
    test_lpmv_parity()