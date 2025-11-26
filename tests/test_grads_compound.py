import jax
import jax.numpy as jnp
import numpy as onp
from jax.test_util import check_grads
from dreams.jax_primitive import spherical_jn, spherical_hankel1, lpmv 
from dreams.jax_tmat import global_tmat, sphere, core_shell_sphere
from dreams.jax_tr import smat_spheres
from dreams.jax_coord import car2pol as car2pol_jax
from dreams.jax_recursive import incgamma
from dreams.jax_la_bounded import lsumsw2d_shift_vmap, _recsw2d
from test_cfg import config, prepare, check_dist, limit_box

jax.config.update("jax_enable_x64", True)

def test_grad_tmatrix_radius():
    cfg = config(); cfg.lmax = 3; cfg.helicity = False
    r0 = jnp.asarray(cfg.r, jnp.float64)
    eps_layers = jnp.asarray([cfg.eps_fg, cfg.eps_bg])
    def f(r):
        T = sphere(cfg.lmax, cfg.k0, r, eps_layers, helicity=cfg.helicity)
        return jnp.real(jnp.vdot(T, T))  
    
    check_grads(f, (r0,), order=1, modes=['fwd','rev'], atol=1e-6, rtol=1e-6)
    g_nojit = jax.grad(f)(r0)
    g_jit   = jax.jit(jax.grad(f))(r0)
    assert onp.allclose(onp.asarray(g_nojit), onp.asarray(g_jit), rtol=1e-12, atol=1e-14)

def test_grad_core_shell_radii_eps():
    cfg = config(); cfg.lmax = 3; cfg.helicity = False
    rc0 = cfg.r_core
    rs0 = cfg.r_shell
    x0 = jnp.array([rc0, rs0], dtype=jnp.float64)

    eps0 = jnp.array([cfg.eps_fg, cfg.eps_sh, cfg.eps_bg], dtype=jnp.complex128)

    def f(x, eps_layers):
        T = core_shell_sphere(
            cfg.lmax, cfg.k0,
            jnp.asarray(x, jnp.float64),
            eps_layers,
            poltype=cfg.polarization,
        )
        return jnp.real(jnp.vdot(T, T))

    check_grads(
        f,
        (x0, eps0),
        order=1,
        modes=['fwd', 'rev'],
        atol=2e-6,
        rtol=2e-6,
    )

    g_nojit = jax.grad(f, argnums=(0, 1))(x0, eps0)
    g_jit   = jax.jit(jax.grad(f, argnums=(0, 1)))(x0, eps0)

    for gn, gj in zip(g_nojit, g_jit):
        onp.testing.assert_allclose(onp.asarray(gn), onp.asarray(gj),
                                    rtol=1e-12, atol=1e-14)
        
def test_grad_global_pos_and_radii():
    cfg = config(); cfg.lmax = 3; cfg.helicity = False
    pos0, r0, eps0 = prepare(cfg)
    pos0 = jnp.asarray(pos0, jnp.float64)
    r0   = jnp.asarray(r0,   jnp.float64)
    eps0 = jnp.asarray(eps0)

    def f(pos, radii):
        T, _, _ = global_tmat(pos, radii, eps0, cfg.lmax, cfg.k0, cfg.helicity)
        return jnp.real(jnp.vdot(T, T))   # scalar loss

    check_grads(f, (pos0, r0), order=1, modes=['rev'], atol=3e-6, rtol=3e-6)
    
    g = jax.grad(f, argnums=(0, 1))
    g_e_pos, g_e_rad = g(pos0, r0)
    g_j_pos, g_j_rad = jax.jit(g)(pos0, r0)
    onp.testing.assert_allclose(onp.asarray(g_e_pos), onp.asarray(g_j_pos), rtol=1e-12, atol=1e-14)
    onp.testing.assert_allclose(onp.asarray(g_e_rad), onp.asarray(g_j_rad), rtol=1e-12, atol=1e-14)
    
    
def test_grad_smatrix_pos_and_radii():
    cfg = config() 
    cfg.lmax = 3 
    cfg.helicity = False 

    pos0, radii0, eps0 = prepare(cfg)
    pos0   = jnp.asarray(pos0,   jnp.float64)       
    radii0 = jnp.asarray(radii0, jnp.float64)
    eps0   = jnp.asarray(eps0)                  

    ok, gap, pair = check_dist(onp.asarray(pos0), onp.asarray(radii0), min_gap=0.0)
    assert ok, f"prepare(cfg) has touching/overlap at {pair}, gap={gap:.3e}"
    viol = limit_box(onp.asarray(pos0), onp.asarray(radii0), pitch=cfg.lat, min_gap=0.0)
    assert viol <= 0.0, f"crosses unit-cell boundary; violation {viol:.3e}"

    def f(pos, radii):
        S, _ = smat_spheres(
            radii, eps0, cfg.eps_bg, cfg.lmax, cfg.k0, pos,
            cfg.helicity, cfg.kx, cfg.ky, cfg.lat,
            origin=None, rmax_coef=cfg.rmax_coef, local=False
        )
        return jnp.real(jnp.vdot(S, S))


    check_grads(f, (pos0, radii0), order=1, modes=['rev'], atol=3e-6, rtol=3e-6)
    g = jax.grad(f, argnums=(0, 1))        
    g_e_pos, g_e_rad = g(pos0, radii0)
    g_j_pos, g_j_rad = jax.jit(g)(pos0, radii0)

    onp.testing.assert_allclose(onp.asarray(g_e_pos), onp.asarray(g_j_pos), rtol=1e-10, atol=1e-10)
    onp.testing.assert_allclose(onp.asarray(g_e_rad), onp.asarray(g_j_rad), rtol=1e-10, atol=1e-10)



def test_grad_car2pol():
    """Check gradients"""

    # Points where the mapping is smooth (no origin)
    pts = jnp.array([
        [1.0, 0.5],
        [1., 0.],
        [-0.3, 0.7],
        [2.0, -1.5],
        [-1.2, -0.8],
    ])

    def rph_fun(xy):
        ans = car2pol_jax(xy)
        return jnp.sum(ans)

    rph_fun_jit   = jax.jit(rph_fun)


    check_grads(
        rph_fun,
        (pts,),
        order=2,
        modes=["fwd", "rev"],
        atol=1e-6,
        rtol=1e-6)
        
    for p in pts:
        p = jnp.asarray(p)

        # Forward & reverse mode consistency for r
        check_grads(
            rph_fun,
            (p,),
            order=2,
            modes=["fwd", "rev"],
            atol=1e-6,
            rtol=1e-6,
        )

 
        # JIT vs non-JIT gradient equality (r)
        g_r      = jax.grad(rph_fun)(p)
        g_r_jit  = jax.grad(rph_fun_jit)(p)
        onp.testing.assert_allclose(
            onp.asarray(g_r),
            onp.asarray(g_r_jit),
            rtol=1e-12,
            atol=1e-12,
        )

def test_grad_incgamma_z():
    size_loop = 20
    n = 0.5  

    # Real-valued wrapper: R^2 -> R
    # xy is a 2-vector: [Re(z), Im(z)]
    def f_xy(xy):
        x, y = xy
        z = x + 1j * y
        val = incgamma(n, z, size_loop)
        # pick a real scalar to differentiate (here, real part)
        return jnp.real(val)

    # Choose some z values *off* the negative real axis
    test_points = [
        0.5 + 0.2j,
        1.0 + 0.0j,      # pure positive real is fine
        -0.3 + 0.7j,     # left half-plane but not on the cut
    ]

    for z0 in test_points:
        xy0 = jnp.array([z0.real, z0.imag], dtype=jnp.float64)
        check_grads(
            f_xy,
            (xy0,),
            modes=["fwd", "rev"],
            order=1,
            atol=1e-5,
            rtol=1e-5,
        )


def test_grad_recsw2d():
    size_loop = 20
    l, m = 2, 0
    phi = 0.3
    eta = 0.5 + 0.2j
    def f_beta(beta):
        beta = jnp.asarray(beta, dtype=jnp.float64)
        val = _recsw2d(
            l,
            m,
            beta,
            jnp.array(phi, dtype=jnp.float64),
            jnp.array(eta, dtype=jnp.complex128),
            size_loop,
        )
        return jnp.real(val)

    beta0 = jnp.array(0.7, dtype=jnp.float64)

    check_grads(
        f_beta,
        (beta0,),
        modes=["fwd", "rev"],
        order=1,
        atol=1e-5,
        rtol=1e-5,
    )

def test_grad_lsumsw2d_shift_vmap_rs():
    """Check reverse-mode grads of lsumsw2d_shift_vmap w.r.t. rs."""
    # simple 2D square lattice
    a = jnp.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=jnp.float64)

    # one position, shape (n_r1, n_r2, 3)
    rs0 = jnp.array([[[0.1, -0.05, 0.0]]], dtype=jnp.float64)

    k = 2 * onp.pi / 1.5
    ks = jnp.array([[k]], dtype=jnp.float64)          # (n_k, 1)
    kpar = jnp.array([0.1 * k, 0.0 * k], dtype=jnp.float64)
    eta = 0.7

    # small set of modes
    ls = jnp.array([1, 2], dtype=int)
    ms = jnp.array([-1, 0], dtype=int)

    # scalar objective: ||dlms||^2
    def f(rs):
        dlms = lsumsw2d_shift_vmap(ls, ms, ks, kpar, a, rs, eta)
        return jnp.real(jnp.vdot(dlms, dlms))

    # 1) finite-diff vs reverse-mode check
    check_grads(f, (rs0,), order=1, modes=["rev"], eps=1e-6, atol=3e-6, rtol=3e-6)

    # 2) non-jitted vs jitted grad consistency
    g = jax.grad(f)
    g_e = g(rs0)
    g_j = jax.jit(g)(rs0)

    onp.testing.assert_allclose(
        onp.asarray(g_e),
        onp.asarray(g_j),
        rtol=1e-12,
        atol=1e-14,
    )

if __name__ == "__main__":
    test_grad_lsumsw2d_shift_vmap_rs()
    #test_grad_recsw2d_beta()
    #test_car2pol_grads()
    #test_grad_smatrix_pos_and_radii()
