# %%
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
from dreams.jax_tmat import tmats_interact, tmats_no_int
from dreams.jax_op import _sw_pw_expand
from dreams.jax_misc import defaultmodes
from dreams.jax_waves import efield
from treams._core import SphericalWaveBasis as SWB
from docs.examples.paper.plot_helper import plot_opt

# constraint should be differentiable
def overlap(pos, radii):
    d = pos[:, None, :] - pos
    rd = radii[:, None] + radii
    nonzero = np.triu_indices(len(pos), k=1)
    dmat = np.linalg.norm(d[nonzero], axis=-1)
    rd = rd[np.triu_indices(len(radii), k=1)]
    answer = np.max(
        rd - dmat
    )  # if distances dmat between are smaller than allowed rd, this is positive, when not, its negative, none should be positive thus, so max value should always be negative.
    return answer



def nlopt_constraint(params, gd):
    v, g = value_and_grad(
        lambda params: overlap(params[: 3 * len(radii)].reshape(-1, 3), params[3 * len(radii) :])
    )(params)
    # v, g = value_and_grad(
    #     lambda params: overlap(params)
    # )(params)
    if gd.size > 0:
        gd[:] = g.flatten()
    return v.item()

def optimizer(p, n_steps, verbose=True):
    it = 0
    opt = nlopt.opt(
        nlopt.LD_MMA, p.flatten().shape[0]
    )  # or e.g. nlopt.LD_LBFGS (see available algorithms here: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
    #v_g = jit(value_and_grad(func))
    v_g = value_and_grad(func)
    def nlopt_objective(params, gd):
        nonlocal it
        it += 1
        v, g = v_g(params)
        if gd.size > 0:
            gd[:] = g.ravel()
        v = anp.array(v)
        va.append(v)
        pas.append(params)
        if verbose:
            print(f"[iter {it:4d}] f = {float(v): .6e}")
        return v.item()
    opt.set_max_objective(
        nlopt_objective
    )  # or opt.set_min_objective() for minimization
    opt.add_inequality_constraint(nlopt_constraint, 1e-8)
    bound=[-float('inf')]*p.shape[0]
    ln = p.shape[0]//4
    bound[3*p.shape[0]//4:] = [rl]*ln 
    print("bound", bound) 
    opt.set_lower_bounds(bound)
    opt.set_maxeval(
        n_steps
    )  # maximum number of function evaluations (similar to iterations)
    p_opt = opt.optimize(p)
    pas.append(p_opt)

@jax.jit
def func(params):
    """
    Calculate the ratio of intensity in the far field in forward and backward direction (0 and 180 degrees).
    The computation is performed using local basis.
    """
    radii = params[int(len(params) * 3 / 4) :]
    positions = params[: int(len(params) * 3 / 4)].reshape((-1, 3))
    #gls = global_tmat(positions, radii, epsilons, lmax, k0, helicity=False)[0] 
    #tm = gls
    modes = defaultmodes(lmax, num)
    ts = tmats_no_int(radii, epsilons, lmax, k0, helicity)
    tm = tmats_interact(ts, positions, modes, k0, helicity, eps_emb)
    inc = treams.plane_wave([0, 0, Sgn*k0], pol, k0=k0, material=eps_emb)
    plane_basis = inc.basis
    pidx, l, m, pols = defaultmodes(lmax, nmax=num)
    modetype = "regular" 
    inc_expand = _sw_pw_expand(plane_basis, pidx, l, m, pols, positions, k0, eps_emb, poltype, modetype = modetype)  @ np.array(inc)
    sca_dr = tm @ inc_expand    
    efl_fdr = efield(point, pidx, l, m, pols, positions, k0, eps_emb,  modetype="singular", poltype=poltype)
    fld_f = np.array(efl_fdr @ sca_dr)
    efl_bdr = efield(-point, pidx, l, m, pols, positions, k0, eps_emb, modetype="singular", poltype=poltype)
    fld_b = np.array(efl_bdr @ sca_dr)
    ans = np.sum(np.abs(fld_f) ** 2)/np.sum(np.abs(fld_b) ** 2)
    return ans



def func_treams(params):
    materials = [treams.Material(eps_obj), treams.Material(eps_emb)]
    radii = params[int(len(params) * 3 / 4) :]
    positions = params[: int(len(params) * 3 / 4)].reshape((-1, 3))
    tms = [treams.TMatrix.sphere(lmax, k0, radius, materials) for radius in radii ]
    tm = treams.TMatrix.cluster(tms, positions).interaction.solve()
    inc = treams.plane_wave([0, 0, Sgn*k0], pol, k0=tm.k0, material=tm.material)
    sca_tr = tm @ inc.expand(tm.basis)
    fw =  anp.sum(anp.abs(sca_tr.efield(point)) ** 2)
    bw =  anp.sum(anp.abs(sca_tr.efield(-point)) ** 2)
    return fw/bw


### Parameter definitions

helicity = False
if helicity:
    poltype = "helicity"
else:
    poltype = "parity"

treams.config.POLTYPE = poltype
num = 3
wl1 = 800.
k0 = 2*np.pi/wl1
eps_emb = 1
eps_obj = 2.5**2
epsilons = np.tile(np.array([eps_obj, eps_emb]), (num, 1))
lmax = 4

# Field calculation parameters
r = 10000 # distance to the point of interest
point = anp.array([0., 0., r])[:, None]
pol = [0, 1, 0] # y-polarization
Sgn = 1 # direction of light propagation


# Set up arrangement of spheres
radius = 80
R = 150.
ni = np.linspace(0, 360., num + 1)[:-1]
x = R * np.cos(ni*np.pi/180)
y = R * np.sin(ni*np.pi/180)
z  = np.zeros_like(x)
positions = np.array([x, y, z]).T
radii = np.ones(num)*radius
pos = positions.flatten()

# Optimization parameters
index = 0
n_steps = 200
rl = 2. #minimal radius permitted
va = []
pas = []

print( "INITIAL OVERLAP", overlap(positions, radii))

# Optimization
param = np.append(pos, radii) # optimize both positions and radii
treams_obj = func_treams(param)
dreams_obj = func(param)
print("CLOSE Initial", dreams_obj, treams_obj, np.isclose(dreams_obj, treams_obj))

optimizer(param, n_steps)

v = func(pas[-1])
treams_obj = func_treams(pas[-1])
print("CLOSE FINAL", v, treams_obj, np.isclose(v, treams_obj))

va.append(v)

# Postprocess
posf = pas[-1][: int(3.0 / 4.0 * len(pas[-1]))].reshape((-1, 3))
radf = pas[-1][int(3.0 / 4.0 * len(pas[-1])) :]
print("FINAL OVERLAP", overlap(posf, radf))
print("FINAL RADII", radf)
name = f"result_lmax_{lmax}"
plot_opt(positions, radii, posf, radf, 2*wl1, ax=None)
plt.savefig(f"{name}_structures")
plt.figure()
# Plot optimization curve
plt.scatter(np.arange(len(va)), va)
plt.ylabel(f"Intensity at point ({point[0]}, {point[1]}, {point[2]})")
plt.xlabel("Steps")
plt.savefig(f"{name}_optcurve.png")
plt.close()

# Save results
with open(__file__, "r") as file:
    script = file.read()

with h5py.File(f"{name}.h5", "w") as f:
    f["values"] = va
    f["radii_final"] = radf
    f["pos_final"] = posf
    f["radii_init"] = radii
    f["pos_init"] = positions
    f["R"] = R
    f["eps_emb"] = eps_emb
    f["eps_obj"] = eps_obj
    f["script"] = script
    f["wl_at"] = wl1
    f["lmax"] = lmax
    f["r"] = r
 
# %%
