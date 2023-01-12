#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
import numpy as np

#change this depending on the optimization iteration you want surfaces for
#can pass this on on the cluster via array id
optimiter = 1000

ncoils = 4
R0 = 1.0
R1 = 0.5
order = 8

base_curves = create_equally_spaced_curves(ncoils, 2, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, 2, True)
bs = BiotSavart(coils)
bs.x = np.loadtxt(f"LPQA2022-optimization-iterations/iteration_{optimiter:04}.txt")
curves = [c.curve for c in coils]
curves_to_vtk(curves,  f"/tmp/curves_{optimiter:04}")

filename = "input.LandremanPaul2022_QA"

coils_bmn = coils
nfp = 2
stellsym = True
mpol = 10
ntor = 16
nphi = ntor + 1
ntheta = 2*mpol + 1
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)

#     coils_bmn = coils_fil_pert[sampleidx]
#     nfp = 1
#     stellsym = False

#     mpol = 10
#     ntor = 32
#     nphi = int(1.5*(2*ntor + 1))
#     ntheta = int(1.5*(2*mpol + 1))
#     phis = np.linspace(0, 1., nphi, endpoint=False)
#     thetas = np.linspace(0, 1., ntheta, endpoint=False)

s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas,
)

s.least_squares_fit(
    SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=-phis, quadpoints_theta=-thetas).gamma()
)

bs_tf = BiotSavart(coils_bmn)
bs = BiotSavart(coils_bmn)

current_sum = sum(abs(c.current.get_value()) for c in coils_bmn)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
iota = 0.416

tf = ToroidalFlux(s, bs_tf)
tf_target = tf.J()
tf_ratios = np.linspace(0.001, 1.0, 50, endpoint=True)[::-1]
tf_targets = [ratio*tf_target for ratio in tf_ratios]

boozer_surface = BoozerSurface(bs, s, tf, tf_target)
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
    tol=1e-10, maxiter=200, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

for i, tf_target in enumerate(tf_targets):
    boozer_surface = BoozerSurface(bs, s, tf, tf_target)
    if i > 0:
        s.scale(tf_target/tf_targets[i-1])
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=20, constraint_weight=100., iota=res['iota'], G=res['G'])
        print(
            f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(
        tol=1e-9, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    # s.to_vtk(f"qa_surfaces/temp/surf_{i}")
    np.savetxt(f"qa_surfaces/surf_{optimiter:04}/optimiter_{optimiter:04}_surface_{i:04}.txt", s.x)
