from dataclasses import dataclass
from dreams.jax_tmat import defaultmodes
import numpy as onp
import jax.numpy as jnp

def _outer_radius(radii):
        r = onp.asarray(radii)
        return r if r.ndim == 1 else r.max(axis=-1)  # (N,)
    
def check_dist(pos, radii, min_gap=0.0, eps=1e-12):
    """
    check that every pair satisfies:
        ||x_i - x_j||  >  r_i + r_j + min_gap
    Returns (ok: bool, min_clearance: float, worst_pair: (i,j)|None)

    - pos:   (N,3)
    - radii: (N,) or (N,L)  -> outer radius is used
    - min_gap: required *strict* extra spacing (> 0 forbids 'just touching')
    - eps: small numeric slack to avoid misclassification
    """
    pos = onp.asarray(pos, float).reshape(-1, 3)
    r   = _outer_radius(radii).astype(float).reshape(-1)
    N   = pos.shape[0]
    if N <= 1:
        return True, onp.inf, None

    dvec = pos[:, None, :] - pos[None, :, :]             # (N,N,3)
    d    = onp.sqrt(onp.einsum("ijk,ijk->ij", dvec, dvec)) # (N,N)
    need = r[:, None] + r[None, :] + (min_gap)      # (N,N)

    iu, ju = onp.triu_indices(N, 1)
    sep = d[iu, ju] - need[iu, ju] - eps  # >0 okay, <=0 violation

    if sep.size == 0:
        return True, onp.inf, None

    k = int(onp.argmin(sep))         # most negative = worst
    min_cl = sep[k]          # can be negative or small
    ok = bool(onp.all(sep > 0.0))
    worst = (int(iu[k]), int(ju[k]))
    return ok, min_cl, worst

def limit_box(pos, radii, pitch, min_gap=0.0, tol=1e-12):
    """
    Max violation of box boundary.
    pos: (N,3), radii: (N,) or (N,L), pitch: scalar side length.
    """
    pos = onp.asarray(pos, float).reshape(-1, 3)
    r = _outer_radius(radii).astype(float).reshape(-1, 1)  # (N,1)
    need = onp.abs(pos) + r + min_gap             # (N,3)
    bound = pitch/2. - tol
    viol = need - bound                                  # (N,3)
    return onp.max(viol)


def positions(config):
    """TODO: clean up"""
    #xs = np.linspace(config.gap, config.lat - config.gap, config.num)
    xs = onp.linspace(-config.lat/2 + config.gap, config.lat/2 - config.gap, config.num)
    X = xs[:, None, None]          
    Y = xs[None, :, None]   
    Z = onp.zeros_like(xs)[None, None, :]      
    #Z = xs[None, None, :]        
    return onp.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def prepare(cfg):
    """Prepares the quantities needed for t-matrix and s-matrix calculation"""
    pos = positions(cfg)
    radii = onp.ones_like(pos[:, 0]) * cfg.r
    ok, min_cl, pair = check_dist(pos, radii, min_gap=1e-3)  # can be 1e-3 units gap
    if not ok:
        i, j = pair
        raise ValueError(f"Overlap or touching: min distance = {min_cl:.3e} between {i} and {j}")
    ones = onp.ones_like(radii.flatten())
    epsilons = onp.vstack((cfg.eps_fg * ones, cfg.eps_bg * ones)).T
    return pos, radii, epsilons

@dataclass
class config:
    lmax = 10
    rmax_coef = 10
    helicity = False
    @property
    def polarization(self):
        return "helicity" if self.helicity else "parity"

    kx = 0.0
    ky = 0.0

    wl = 400.0

    eps_fg = 12.
    eps_sh = 4.
    eps_bg = 2.
    eps_above = 1
    eps_below = 1

    @property
    def k0(self):
        return 2 * onp.pi / self.wl

    @property
    def modes(self):
        return defaultmodes(self.lmax)

    lat = 100

    @property
    def a(self):
        return onp.array([[self.lat, 0], [0, self.lat]])

    r = 20.0
    num = 2  #  ^3 = 8 spheres

    r_core = 100.
    r_shell = 130.
    @property
    def gap(self):
        return self.r + 5.0
