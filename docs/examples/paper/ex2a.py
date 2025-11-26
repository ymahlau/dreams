from functools import partial
import jax
import jax.numpy as np
import numpy as anp
import matplotlib.pyplot as plt
import nlopt
import h5py
import treams

from jax import value_and_grad
from jax.test_util import check_grads

from refractiveindex import RefractiveIndexMaterial

from func_helper import dreams_rcd, treams_rcd, treams_rcd_parallel
from dreams.jax_tmat import defaultmodes                     # noqa: F401

jax.config.update("jax_enable_x64", True)

def overlap(pos, radii):
    """Overlap margin; <= 0 means no overlaps (with safespace)."""
    pos = np.reshape(pos, (-1, 3))
    d = pos[:, None, :] - pos[None, :, :]
    rd = radii[:, None] + radii[None, :]
    I, J = np.triu_indices(pos.shape[0], k=1)
    dmat = np.linalg.norm(d[I, J, :], axis=-1)
    rd_ij = rd[I, J]
    return np.max(rd_ij - dmat) + safespace


def limit(pos, radii):
    """No sphere crosses circumscribing sphere of radius pitch/2 at origin."""
    pos = np.reshape(pos, (-1, 3))
    r2 = np.sum(pos**2, axis=1)
    mask0 = np.isclose(r2, 0.0)
    rs = np.sqrt(np.where(mask0, 1.0, r2))
    rs = np.where(mask0, 0.0, rs)
    dist = rs + radii
    return np.max(dist - pitch / 2.0)


def limit_unit(pos, radii):
    """Axis-aligned unit-cell boundary (AABB) constraint."""
    pos = np.reshape(pos, (-1, 3))
    a = np.max(np.abs(pos[:, 0]) - pitch / 2.0)
    b = np.max(np.abs(pos[:, 1]) - pitch / 2.0)
    c = np.max(np.abs(pos[:, 2]) - pitch / 2.0)
    return np.max(np.array([a, b, c]))


# ---------------------------
# NLopt wrappers (read globals radii, pitch, safespace)
# ---------------------------

def nlopt_constraint(params, gd):
    v, g = value_and_grad(
        lambda p: overlap(p[: 3 * len(radii)], p[3 * len(radii):])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return float(v)


def nlopt_constraint2(params, gd):
    v, g = value_and_grad(
        lambda p: limit(p[: 3 * len(radii)], p[3 * len(radii):])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return float(v)


def nlopt_constraint2l(params, gd):
    v, g = value_and_grad(
        lambda p: limit_unit(p[: 3 * len(radii)], p[3 * len(radii):])
    )(params)
    if gd.size > 0:
        gd[:] = anp.asarray(g).ravel()
    return float(v)


def optimizer(p, n_steps):
    """Run NLopt with your objective and constraints (uses global cfg)."""
    opt = nlopt.opt(nlopt.LD_MMA, p.size)

    v_g = value_and_grad(dreams_rcd_func)
    index = 0
    def nlopt_objective(params, gd):
        nonlocal index
        print(f"evaluating objective function: {index}")
        index += 1
        v, g = v_g(params)
        if gd.size > 0:
            gd[:] = anp.asarray(g).ravel()
        v = anp.array(v)
        va.append(v)
        pas.append(params.copy())
        return float(v)

    opt.set_max_objective(nlopt_objective)

    # constraints
    opt.add_inequality_constraint(nlopt_constraint, 1e-8)
    opt.add_inequality_constraint(nlopt_constraint2, 1e-8)

    # lower bounds: last N entries (radii) >= rl
    bound = [-float("inf")] * p.size
    ln = p.size // 4
    bound[3 * ln:] = [rl] * ln
    opt.set_lower_bounds(bound)

    opt.set_maxeval(n_steps)
    p_opt = opt.optimize(p)
    pas.append(p_opt.copy())


# ---------------------------
# treams reference function for a single wavelength (for spot checks)
# ---------------------------

def treams_func(params):
    """Single-λ treams comparison (|Ry - Rx|) used before/after optimization."""
    radii_local = params[int(len(params) * 3 / 4):]
    positions_local = params[: int(len(params) * 3 / 4)].reshape((-1, 3))

    lattice = treams.Lattice.square(pitch)
    treams.config.POLTYPE = 'parity'
    mats = [treams.Material(eps_object), treams.Material(eps_medium)]

    spheres = [treams.TMatrix.sphere(lmax, k0, r, mats) for r in radii_local]
    tm = treams.TMatrix.cluster(spheres, positions_local).interaction.solve()
    tm_global = tm.expand(treams.SphericalWaveBasis.default(lmax))
    tm_lat = tm_global.latticeinteraction.solve(lattice, kpars)
    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpars, lattice, rmax_coef * k0)

    # x-illum
    plw = treams.plane_wave(kpars, [1, 0, 0], k0=k0, basis=pwb, material=mats[1])
    array = treams.SMatrices.from_array(tm_lat, pwb)
    tr_x = array.tr(plw)[1]

    # y-illum
    plw = treams.plane_wave(kpars, [0, 1, 0], k0=k0, basis=pwb, material=mats[1])
    tr_y = array.tr(plw)[1] 

    return np.abs(tr_y - tr_x)




# ---------------------------
# parameters & run
# ---------------------------

lmax = 7
lmax_glob = 7
rmax_coef = 2
wls = anp.array([1050.0, 950.0])
eps_medium = 1.5**2

pitch = 600.0
r_init = 10.0
num = 5
R = 170.0
zmax = 0.0
shift = 0.0
shift2 = 0.0
safespace = 5.0

ni = np.linspace(0, 360.0, num + 1)[:-1]
x = R * np.cos(ni * np.pi / 180.0)
y = R * np.sin(ni * np.pi / 180.0)
z = np.linspace(-zmax, zmax, x.shape[0])

positions = np.array([x, y, z]).T
rand = anp.random.uniform(-shift, shift, size=positions.shape)
positions = positions + rand

radii = np.ones_like(positions[:, 0]) * r_init
rand2 = anp.random.uniform(-shift2, shift2, size=positions.shape[0])
radii = radii + rand2

modes = defaultmodes(lmax)  
helicity = False
kx = 0.0
ky = 0.0
kpars = anp.array([kx, ky])
rl = 5.0
n_steps = 200
pos = positions.flatten()

# sanity checks
ov = overlap(pos, radii)
lim = limit(pos, radii)
print("overlap", ov, "limit", lim)

param = np.append(pos, radii)

for wl in wls:
    va = []
    pas = []

    k0 = 2 * anp.pi / wl
    si = RefractiveIndexMaterial("main", "Si", "Schinke")
    eps_object = si.get_epsilon(wl)

    cfg = {
        "lmax": lmax,
        "lmax_glob": lmax_glob,
        "pitch": pitch,
        "eps_medium": eps_medium,
        "eps_object": eps_object,
        "rmax_coef": rmax_coef,
        "helicity": helicity,
        "kx": kx,
        "ky": ky,
        "k0": k0,
    }
    dreams_rcd_func = jax.jit(partial(dreams_rcd, eps_object=eps_object, k0=k0, cfg=cfg))

    # single evaluation (treams vs dreams)
    r_treams = treams_func(param)
    r_dreams = dreams_rcd_func(param)
    check_grads(dreams_rcd_func, (param,), order=1, modes=['rev'])
    print("PASS CHECK GRADS")
    print("CLOSE", r_treams, r_dreams,
          np.allclose(anp.asarray(r_treams), anp.asarray(r_dreams)))

    # optimize
    optimizer(param, n_steps)

    tr_f = treams_func(pas[-1])
    print('va last', va[-1] if len(va) else None)
    print("tr vs last val", tr_f, va[-1],
          np.allclose(anp.asarray(tr_f), anp.asarray(va[-1])))

    # postprocess / sweep
    posf = pas[-1][: int(3.0 / 4.0 * len(pas[-1]))].reshape((-1, 3))
    radf = pas[-1][int(3.0 / 4.0 * len(pas[-1])):]

    params_final = anp.concatenate((anp.array(posf).flatten(), anp.array(radf)))

    r_treams2, _, _ = treams_rcd(params_final, cfg)
    r_dreams2 = dreams_rcd_func(params_final)
    print("CLOSE final", r_treams2, r_dreams2,
          np.allclose(anp.asarray(r_treams2), anp.asarray(r_dreams2)))

    wls_range = anp.arange(900.0, 1200.0, 2.0)
    eps_objects_range = si.get_epsilon(wls_range)
    cfg_sweep = dict(cfg)
    cfg_sweep["k0s"] = 2 * anp.pi / wls_range
    cfg_sweep["eps_objects"] = eps_objects_range
    cfg_sweep["lmax"] = 15  # higher order for finalspectra

    resfinal = treams_rcd_parallel(params_final, cfg_sweep)
    print("resfinal", resfinal.shape)
    rcdfinal_15 = resfinal[0, :]
    r1_final = resfinal[1, :]
    r2_final = resfinal[2, :]

    name = (
        f"notnorm-si-safe-{safespace}-rcd-circle-R-{R}-zshift-{zmax}-"
        f"nosub-num-{len(radii)}-randr-{shift2}-randpos-{shift}-rinit-{r_init}-"
        f"lmax-{lmax}-{lmax_glob}-nsteps-{n_steps}-wl-{wl}-pitch-{pitch}-"
        f"rmax-{rmax_coef}-rl-{rl}-with-limit.h5"
    )

    try:
        with open(__file__, "r") as file:
            script = file.read()
    except NameError:
        script = "# __file__ not availabl"

    out_path = "paper_results/" + name
    with h5py.File(out_path, "w") as f:
        f["script"] = script
        f["values"] = anp.asarray(va)
        f["R"] = R
        f["safety"] = safespace
        f["radii_final"] = anp.asarray(radf)
        f["pos_final"] = anp.asarray(posf)
        f["radii_init"] = anp.asarray(radii)
        f["pos_init"] = anp.asarray(positions)
        f["pitch"] = pitch
        f["lmax"] = lmax
        f["wl"] = wl
        f["eps_obj"] = anp.asarray(eps_object)
        f["eps_emb"] = eps_medium
        f["rcd_final_wls_150_lm15"] = anp.asarray(rcdfinal_15)
        f["r1_final_wls_150_lm15"] = anp.asarray(r1_final)
        f["r2_final_wls_150_lm15"] = anp.asarray(r2_final)
        f["wls_150_range"] = anp.asarray(wls_range)

    print("saved:", out_path)