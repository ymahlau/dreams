import jax.numpy as np
import numpy as anp
import matplotlib.pyplot as plt
import nlopt
import scipy
import scipy.io
import h5py
import treams
import jax
from jax import value_and_grad, grad, jit
from jax.test_util import check_grads

from dreams.jax_smat import _check_modes
from dreams.jax_tmat import (
    defaultmodes,
    refractive_index,
    tmats_interact,
    tmats_no_int,
    sphere_parity,
)
from dreams.jax_op import _sw_pw_expand
from dreams.jax_tr import smat_spheres, tr
from dreams.jax_waves import efield
from treams._core import SphericalWaveBasis as SWB
from func_helper import fob, xs_treams, fob_treams, field_wl_treams, multipole_xs_treams

# constraint should be differentiable
def overlap(pos, radii):
    d = pos[:, None, :] - pos
    rd = radii[:, None] + radii
    nonzero = np.triu_indices(len(pos), k=1)
    dmat = np.linalg.norm(d[nonzero], axis=-1)
    rd = rd[np.triu_indices(len(radii), k=1)]
    # if distances dmat between are smaller than allowed rd, this is positive,
    # when not, it's negative — none should be positive thus, so max value should always be negative.
    answer = np.max(rd - dmat)
    return answer

def nlopt_constraint(params, gd):
    v, g = value_and_grad(
        lambda params: overlap(
            params[: 3 * len(params)//4].reshape(-1, 3),
            params[3 * len(params)//4:],
        )
    )(params)
    if gd.size > 0:
        gd[:] = g.flatten()
    return v.item()


def optimizer(p, n_steps):
    global rl
    opt = nlopt.opt(
        nlopt.LD_MMA,
        p.flatten().shape[0],
    )  # or e.g. nlopt.LD_LBFGS
    #v_g = jit(value_and_grad(fob)) not worth it
    v_g = value_and_grad(fob)
    index = 0
    def nlopt_objective(params, gd):
        nonlocal index
        print(f"evaluating objective function: {index}")
        index += 1
        v, g = v_g(params, cfg)
        if gd.size > 0:
            gd[:] = g.ravel()
        v = anp.array(v)
        va.append(v)
        pas.append(params)
        return v.item()

    opt.set_max_objective(nlopt_objective)  
    opt.add_inequality_constraint(nlopt_constraint, 1e-8)

    bound = [-float("inf")] * p.shape[0]
    ln = p.shape[0] // 4
    bound[3 * p.shape[0] // 4 :] = [rl] * ln
    opt.set_lower_bounds(bound)
    opt.set_maxeval(n_steps)  # maximum number of function evaluations
    p_opt = opt.optimize(p)
    pas.append(p_opt)

# parameter definitions

helicity = False
if helicity:
    poltype = "helicity"
else:
    poltype = "parity"

treams.config.POLTYPE = poltype

n_steps = 100
wavelengths = np.arange(400.0, 1001, 5.0) #range includes exact point of evaliation
k0s = 2 * np.pi / wavelengths
num = 6
eps_emb = 1
eps_obj = 2.5 ** 2
epsilons = np.tile(np.array([eps_obj, eps_emb]), (num, 1))
pol = [0, 1, 0]
Sgn = 1

lmax = 3
lmax_glob = 6
radius = 80

if Sgn == 1:
    tetamin = 0
    tetamax = np.pi / 2
else:
    tetamin = np.pi / 2
    tetamax = np.pi

phimin = 0
phimax = 2 * np.pi
Nteta = 20
Nphi = 36
dteta = (tetamax - tetamin) / float(Nteta - 1)
dphi = (phimax - phimin) / float(Nphi)
r = 100000

tetalist = anp.ones((int(Nteta), int(Nphi))) * anp.linspace(
    tetamin, tetamax, int(Nteta)
)[:, None]
philist = anp.ones((int(Nteta), int(Nphi))) * anp.linspace(
    phimin, phimax, int(Nphi), endpoint=False
)[None, :]

xff = (r * anp.sin(tetalist) * anp.cos(philist)).flatten()
yff = (r * anp.sin(tetalist) * anp.sin(philist)).flatten()
zff = (r * anp.cos(tetalist)).flatten()
d_solid_surf = (r ** 2 * anp.sin(tetalist) * dteta * dphi)

grid_f = anp.transpose(anp.array([xff, yff, zff]))
grid_b = -grid_f

R = 400.0
ni = np.linspace(0, 360.0, num + 1)[:-1]
x = R * np.cos(ni * np.pi / 180)
y = R * np.sin(ni * np.pi / 180)
z = np.zeros_like(x)
positions = np.array([x, y, z]).T
radii = np.ones(num) * radius
rl = 2.0
va = []
pas = []
pos = positions.flatten()
print("OVERLAP", overlap(positions, radii))
wl1 = 800.0
k0 = 2 * np.pi / wl1

cfg = {
    "num": num,
    "lmax": lmax,
    "k0": k0,
    "lmax_glob": lmax_glob,
    "eps_emb": eps_emb,
    "eps_obj": eps_obj,
    "epsilons": epsilons,   
    "wavelengths": wavelengths,
    "pol": tuple(pol),
    "Sgn": Sgn,
    "grid_f": grid_f,
    "grid_b": grid_b,
    "d_solid_surf": d_solid_surf,
    "poltype": poltype,
}


# optimization
param = np.append(pos, radii)
name = (
    f"paper_results/num-{num}-parity-R-{R}-local-pos-rad-{wl1}-lmax-{lmax}"
    f"-pol-{pol[0]}-nsteps-{n_steps}-rad-{radius}-nph-{Nphi}-nth-{Nteta}"
)

vfi, vbi = field_wl_treams(anp.array(param), cfg)

fob_init_treams, sum_init_treams, tmats_init = fob_treams(anp.array(param), cfg)
xss_init = multipole_xs_treams(tmats_init, cfg)

optimizer(param, n_steps)
va.append(fob(pas[-1], cfg))

# postprocess
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 17))

ax3.scatter(wavelengths, fob_init_treams, color="g", label="Initial f/b treams")

fob_final_treams, sum_final_treams, tmats_final = fob_treams(anp.array(pas[-1]), cfg)
xss_final = multipole_xs_treams(tmats_final, cfg)

ax3.scatter(wavelengths, fob_final_treams, color="r", label="Final f/b")
ax3.set_ylabel("f/b")
ax3.set_xlabel(r"$\lambda$ (nm)")

ax4.set_xlabel(r"$\lambda$ (nm)")
ax4.scatter(wavelengths, sum_init_treams, color="g", label="Initial f+b treams")
ax4.scatter(wavelengths, sum_final_treams, color="r", label="Final f+b")
ax4.set_ylabel("f+b")

ax3.legend()
ax1.scatter(range(len(va)), va)

posf = pas[-1][: int(3.0 / 4.0 * len(pas[-1]))].reshape((-1, 3))
radf = pas[-1][int(3.0 / 4.0 * len(pas[-1])) :]

vff, vbf = field_wl_treams(pas[-1], cfg)

with open(__file__, "r") as file:
    script = file.read()

with h5py.File(f"{name}.h5", "w") as f:
    for key, value in xss_init.items():
        f[f"{key}_init"] = value
    for key, value in xss_final.items():
        f[f"{key}_final"] = value
    f["fwd_field_init"] = vfi
    f["bwd_field_init"] = vbi
    f["fwd_field_final"] = vff
    f["bwd_field_final"] = vbf
    f["xff"] = xff
    f["yff"] = yff
    f["zff"] = zff
    f["values"] = va
    f["radii_final"] = radf
    f["pos_final"] = posf
    f["radii_init"] = radii
    f["pos_init"] = positions
    f["R"] = R
    f["eps_emb"] = eps_emb
    f["eps_obj"] = eps_obj
    f["fob_init"] = fob_init_treams
    f["fob_final"] = fob_final_treams
    f["sum_init"] = sum_init_treams
    f["sum_final"] = sum_final_treams
    f["tmats_init"] = tmats_init
    f["tmats_final"] = tmats_final
    f["wls"] = wavelengths
    f["script"] = script
    f["wl_at"] = wl1
    f["lmax"] = lmax
    f["lmax_glob"] = lmax_glob
    f["nteta"] = Nteta
    f["nphi"] = Nphi
    f["r"] = r