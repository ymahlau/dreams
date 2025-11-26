import jax.numpy as np
import jax
from dreams.jax_tmat import global_tmat
from dreams.jax_op import rotate
from dreams.jax_tr import smat_spheres_full
from test_cfg import config

cfg = config()


def prepare():
    """Prepares the quantities needed for t-matrix and s-matrix calculation"""
    pos = np.array([[0, 0, 0], [2, 0, 0]])
    radii = np.ones([2])
    ones = radii
    epsilons = np.vstack((2 * ones, 1 * ones)).T
    return pos, radii, epsilons


def test_regression_rotate_helicity(ndarrays_regression):
    pos, radii, epsilons = prepare()
    t_matrix, _, _ = global_tmat(pos, radii, epsilons, cfg.lmax, cfg.k0, True)
    rotated = rotate(t_matrix, 0.1, 0.1, 0.1)
    result = rotated
    ndarrays_regression.check({"result": result})


def test_regression_rotate_parity(ndarrays_regression):
    pos, radii, epsilons = prepare()
    t_matrix, _, _ = global_tmat(pos, radii, epsilons, cfg.lmax, cfg.k0, False)
    rotated = rotate(t_matrix, 0.1, 0.1, 0.1)
    result = np.abs(np.sum(rotated))
    ndarrays_regression.check({"result": result})


def test_regression_tmatrix_grad(ndarrays_regression):
    pos, radii, epsilons = prepare()
    helicity = True

    def calc_tmatrix_sum(radii):
        t_matrix, _, _ = global_tmat(pos, radii, epsilons, cfg.lmax, cfg.k0, helicity)
        return np.abs(np.sum(t_matrix))

    grad = jax.grad(calc_tmatrix_sum)
    result = np.sum(grad(radii))
    ndarrays_regression.check({"result": result})


def test_regression_smatrix_grad(ndarrays_regression):
    pos, radii, epsilons = prepare()

    def calc_smatrix_sum(radii):
        s_matrix, modesq = smat_spheres_full(
            radii,
            epsilons,
            cfg.eps_bg,
            cfg.lmax,
            cfg.k0,
            pos,
            cfg.helicity,
            cfg.kx,
            cfg.ky,
            cfg.lat,
            eps_below=cfg.eps_below,
            eps_above=cfg.eps_above,
            origin=None,
            eta=0.,
            mu=1,
            kappa=0,
            rmax_coef=cfg.rmax_coef,
            local=False
        )
        return np.sum(np.abs(s_matrix))

    grad = jax.grad(calc_smatrix_sum)
    result = np.sum(grad(radii))
    ndarrays_regression.check({"result": result})
