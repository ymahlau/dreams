import jax.numpy as np
import numpy as anp
import h5py
import treams
from refractiveindex import RefractiveIndexMaterial
from functools import partial
import nlopt
import jax
from jax import value_and_grad
from dreams.jax_tmat import defaultmodes
from func_helper import dreams_rcd, treams_rcd, treams_rcd_parallel

jax.config.update("jax_enable_x64", True)

def overlap(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    d = pos[:, None, :] - pos
    rd = radii[:, None] + radii
    nonzero = np.triu_indices(len(pos), k=1)
    dmat = np.linalg.norm(d[nonzero], axis=-1)
    rd = rd[np.triu_indices(len(radii), k=1)]
    return np.max(rd - dmat) + safety

def limit(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    r2 = np.sum(pos**2, axis=1)
    mask0 = np.isclose(r2, 0.0)
    rs = np.sqrt(np.where(mask0, 1.0, r2))
    rs = np.where(mask0, 0.0, rs)
    dist = rs + radii
    return np.max(dist - pitch/2)

def limit_unit(pos, radii):
    pos = np.reshape(pos, (-1, 3))
    ans1 = np.max(np.abs(pos[:, 0]) - pitch/2)
    ans2 = np.max(np.abs(pos[:, 1]) - pitch/2)
    ans3 = np.max(np.abs(pos[:, 2]) - pitch/2)
    return np.max(np.array([ans1, ans2, ans3]))

def nlopt_constraint(params, gd):
    v, g = value_and_grad(lambda p: overlap(p[: 3 * len(radii)], p[3 * len(radii):]))(params)
    if gd.size > 0:
        gd[:] = g.flatten()
    return v.item()

def nlopt_constraint2(params, gd):
    v, g = value_and_grad(lambda p: limit(p[: 3 * len(radii)], p[3 * len(radii):]))(params)
    if gd.size > 0:
        gd[:] = g.flatten()
    return v.item()

def nlopt_constraint2l(params, gd):
    v, g = value_and_grad(lambda p: limit_unit(p[: 3 * len(radii)], p[3 * len(radii):]))(params)
    if gd.size > 0:
        gd[:] = g.flatten()
    return v.item()

def optimizer(p, n_steps):
    opt = nlopt.opt(nlopt.LD_MMA, p.flatten().shape[0])
    v_g = value_and_grad(partial(opt_func_global))
    index = 0
    def nlopt_objective(params, gd):
        nonlocal index
        print(f"evaluating objective function: {index}")
        index += 1
        v, g = v_g(params)
        if gd.size > 0:
            gd[:] = g.ravel()
        v = anp.array(v)
        va.append(v)
        pas.append(params)
        return v.item()
    opt.set_max_objective(nlopt_objective)
    opt.add_inequality_constraint(nlopt_constraint, 1e-8)
    opt.add_inequality_constraint(nlopt_constraint2, 1e-8)
    ln = p.shape[0] // 4
    bound = [-float('inf')] * p.shape[0]
    bound[3*p.shape[0]//4:] = [rl]*ln
    opt.set_lower_bounds(bound)
    opt.set_maxeval(n_steps)
    p_opt = opt.optimize(p)
    pas.append(p_opt)

def opt_func_global(params):
    drcds = []
    for i, wl in enumerate(wls):
        k0 = 2*anp.pi/wl
        eps_object = eps_objects[i]
        drcds.append(dreams_rcd_func(params, eps_object, k0))
    drcds = np.array(drcds)
    x, y = drcds[0], drcds[1]
    eps = 1e-8
    return (2 * x * y + eps * np.min(drcds)) / (x + y + eps)

def compute_r_for_params(params, wls_nm, eps_objects, pitch_nm, rmax_coef, lmax, eps_medium):
    radii_loc = params[int(len(params) * 3 / 4):]
    positions_loc = params[: int(len(params) * 3 / 4)].reshape((-1, 3))
    rs_x, rs_y = [], []
    for i, wl in enumerate(wls_nm):
        k0_loc = 2*np.pi/wl
        mats = [treams.Material(eps_objects[i]), treams.Material(eps_medium)]
        lattice = treams.Lattice.square(pitch_nm)
        treams.config.POLTYPE = 'parity'
        spheres = [treams.TMatrix.sphere(lmax, k0_loc, r, mats) for r in radii_loc]
        tm = treams.TMatrix.cluster(spheres, positions_loc).interaction.solve()
        tm_global = tm.expand(treams.SphericalWaveBasis.default(lmax))
        metasurf_t = tm_global.latticeinteraction.solve(lattice, [0.0, 0.0])
        pwb = treams.PlaneWaveBasisByComp.diffr_orders([0.0, 0.0], lattice, rmax_coef * k0_loc)
        metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
        illu_x = treams.plane_wave([0.0, 0.0], 0, k0=k0_loc, basis=pwb, material=mats[1])
        illu_y = treams.plane_wave([0.0, 0.0], 1, k0=k0_loc, basis=pwb, material=mats[1])
        rs_x.append(np.asarray(metasurf_s.tr(illu_x)[0], dtype=np.complex128))
        rs_y.append(np.asarray(metasurf_s.tr(illu_y)[0], dtype=np.complex128))
    return rs_x, rs_y

lmax = 7
lmax_glob = 8
rmax_coef = 2
wls = anp.array([950., 1050.])
k0s = 2 * np.pi / wls
si = RefractiveIndexMaterial("main", "Si", "Schinke")
eps_objects = si.get_epsilon(wls)
eps_medium = 1.5**2
pitch = 600.
r_init = 10.
num = 5
R = 170.
zmax = 0.

ni = np.linspace(0, 360., num + 1)[:-1]
x = R * np.cos(ni*np.pi/180)
y = R * np.sin(ni*np.pi/180)
z = np.linspace(-zmax, zmax, x.shape[0])

shift = 0.
shift2 = 0.
loops = 1 if shift == 0 else 2

for _ in range(loops):
    seed = anp.random.randint(0, 2**32 - 1)
    anp.random.seed(seed)
    positions = np.array([x, y, z]).T
    positions = positions + anp.random.uniform(-shift, shift, size=positions.shape)
    radii = np.ones_like(positions[:, 0]) * r_init
    radii = radii + anp.random.uniform(-shift2, shift2, size=positions.shape[0])
    modes = defaultmodes(lmax)
    helicity = False
    kx = 0.0
    ky = 0.0
    rl = 5.
    safety = 5.
    kpars = anp.array([kx, ky])
    va = []
    pas = []
    pos = positions.flatten()
    cfg = {
        "lmax": lmax,
        "lmax_glob": lmax_glob,
        "pitch": pitch,
        "eps_medium": eps_medium,
        "eps_objects": eps_objects,
        "rmax_coef": rmax_coef,
        "helicity": helicity,
        "kx": kx,
        "ky": ky,
        "k0s": 2*anp.pi/wls,
        "wls": wls
    }

    param = np.append(pos, radii)
    _ = overlap(pos, radii)
    _ = limit(pos, radii)
    dreams_rcd_func = partial(dreams_rcd, cfg=cfg) 
    rcd_init = []
    for i_wl, wl in enumerate(wls):
        cfg["k0"] = 2*anp.pi/wl
        cfg["eps_object"]  = eps_objects[i_wl]
        r_treams, _, _ = treams_rcd(param, cfg)
        rcd_init.append(r_treams)
    rcd_init = np.array(rcd_init)
    optimizer(param, 200)
    posf = pas[-1][: int(3.0 / 4.0 * len(pas[-1]))].reshape((-1, 3))
    radf = pas[-1][int(3.0 / 4.0 * len(pas[-1])):]
    params_final = anp.concatenate((anp.array(posf).flatten(), anp.array(radf)))

    rcd_final = []
    for i_wl, wl in enumerate(wls):
        cfg["k0"] = 2*anp.pi/wl
        cfg["eps_object"]  = eps_objects[i_wl]
        r_treams, _, _ = treams_rcd(params_final, cfg)
        rcd_final.append(r_treams)

    wls_range = np.arange(900., 1200., 2)
    eps_objects_range = si.get_epsilon(wls_range)
    cfg["k0s"] = 2*anp.pi/wls_range
    cfg["eps_objects"] = eps_objects_range
    cfg["lmax"] = 15
    resfinal = treams_rcd_parallel(params_final, cfg)
    rcdfinal_15 = resfinal[0, :]
    r1_final = resfinal[1, :]
    r2_final = resfinal[2, :]

    rsx_init, rsy_init   = compute_r_for_params(param, wls, eps_objects, pitch, rmax_coef, lmax, eps_medium)
    rsx_final, rsy_final = compute_r_for_params(params_final, wls, eps_objects, pitch, rmax_coef, lmax, eps_medium)

    name = (
        f"cortr-1r-balanced-epsminxy-zshift-{zmax}-numwls-{len(wls)}-safecor-{safety}-"
        f"notnorm-si-rcd-circle-adapt-R-{R}-nosub-num-{len(radii)}-randr-{shift2}-"
        f"randpos-{shift}-rinit-{r_init}-lmax-{lmax}-{lmax_glob}-nsteps-200-"
        f"wls-{wls[0]}-{wls[-1]}-pitch-{pitch}-rmax-{rmax_coef}-rl-{rl}-"
        f"with-limit-seed-{seed}.h5"
    )

    try:
        with open(__file__, "r") as file:
            script = file.read()
    except NameError:
        script = ""

    with h5py.File("paper_results/" + name, "w") as f:
        f["script"] = script
        f["values"] = anp.asarray(va)
        f["R"] = R
        f["radii_final"] = anp.asarray(radf)
        f["pos_final"] = anp.asarray(posf)
        f["radii_init"] = anp.asarray(radii)
        f["pos_init"] = anp.asarray(positions)
        f["safety"] = safety
        f["pitch"]  = pitch
        f["lmax"] = lmax
        f["lmax_glob"] = lmax_glob
        f["wls"] = anp.asarray(wls)
        f["eps_objs"] = anp.asarray(eps_objects)
        f["eps_emb"] = eps_medium
        f["rcd_final_wls_150_lm15"] = anp.asarray(rcdfinal_15)
        f["r1_final_wls_150_lm15"] = anp.asarray(r1_final)
        f["r2_final_wls_150_lm15"] = anp.asarray(r2_final)
        f["wls_150_range"] = anp.asarray(wls_range)
        f["rcd_final"] = anp.asarray(rcd_final)
        f["r_init_x"]  = anp.asarray(rsx_init)
        f["r_init_y"]  = anp.asarray(rsy_init)
        f["r_final_x"]  = anp.asarray(rsx_final)
        f["r_final_y"]  = anp.asarray(rsy_final)
        f["r_wavelength_nm"] = anp.asarray(wls, float)
        f["r_k0"] = 2*np.pi/anp.asarray(wls, float)