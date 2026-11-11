"""Symbolic and numerical checks of the equations of the manuscript: the thermofield double, the stationary point and its
stability (Eq. stat), the feedback linearization (Eq. lin) and its delayed form (Eq. delay), the capture threshold,
the uncoupled span, the finite-quanta bounds (Eq. nq and the kick bound), the single-mode stability limit and the
binding per channel, the change of variables of the null QEI (Eq. qei), the conformal time-slice bound, the
electromagnetic-like multipole sum, the near-extremal Reissner-Nordstrom relation and the rate bound (Eq. theta)."""
import math, sys
import sympy as sp

fails = []
def ok(name, cond):
    print(("ok    " if cond else "FAIL  ") + name)
    if not cond:
        fails.append(name)

u, k, A, eta, D, th, gp, gd, z, ud, w = sp.symbols("u k A eta Delta theta gamma_p gamma_d z u_d w", positive=True)
# thermofield double: phi = ln(k/cosh(k u)) solves phi'' = -e^{2 phi} with E = A k^2
phi = sp.log(k / sp.cosh(k * u))
ok("TFD solves the uncoupled equation", sp.simplify(sp.diff(phi, u, 2) + sp.exp(2 * phi)) == 0)
ok("TFD energy A k^2", sp.simplify(A * (sp.diff(phi, u) ** 2 + sp.exp(2 * phi)) - A * k ** 2) == 0)
# stationary point of phi'' = -e^{2 phi} + (D eta/A) e^{2 D phi}
p = sp.symbols("p", real=True)
F = -sp.exp(2 * p) + D * eta / A * sp.exp(2 * D * p)
eta_star = A * th ** (2 * (1 - D)) / D
ok("stationary point theta^{2(1-D)} = D eta/A", sp.simplify(F.subs({eta: eta_star, p: sp.log(th)})) == 0)
E_star = (A * sp.exp(2 * p) - eta * sp.exp(2 * D * p)).subs({eta: eta_star, p: sp.log(th)})
ok("stationary energy -A theta^2 (1-D)/D", sp.simplify(E_star + A * th ** 2 * (1 - D) / D) == 0)
dF = sp.diff(F, p).subs({eta: eta_star, p: sp.log(th)})
ok("oscillation frequency omega^2 = 2(1-D) theta^2", sp.simplify(-dF - 2 * (1 - D) * th ** 2) == 0)
# feedback: eta = eta* s(x), x = 1 - gp dphi - gd dphi', s(1) = s'(1) = 1 -> Eq. (lin)
x1, x2 = sp.symbols("x1 x2", real=True)
for s in (lambda x: x, lambda x: x + (x - 1) ** 2 - (x - 1) ** 3):      # two saturations with s(1) = 1 and s'(1) = 1
    Ffb = -sp.exp(2 * (sp.log(th) + x1)) + D * eta_star / A * s(1 - gp * x1 - gd * x2) * sp.exp(2 * D * (sp.log(th) + x1))
    lin = sum(sp.diff(Ffb, v).subs({x1: 0, x2: 0}) * v for v in (x1, x2))
    target = th ** 2 * ((2 * (D - 1) - gp) * x1 - gd * x2)
    ok("feedback linearization, Eq. (lin)", sp.simplify(sp.expand(lin - target)) == 0)
# delayed loop: delta'' = lambda^2 delta - theta^2 (gp delta(u-ud) + gd delta'(u-ud)) with delta = e^{z u}
lam2 = 2 * (D - 1) * th ** 2
char = sp.simplify((z ** 2 - lam2 + th ** 2 * (gp + gd * z) * sp.exp(-z * ud)))
expr = sp.diff(sp.exp(z * u), u, 2) - lam2 * sp.exp(z * u) + th ** 2 * (gp * sp.exp(z * (u - ud)) + gd * z * sp.exp(z * (u - ud)))
ok("delay characteristic equation, Eq. (delay)", sp.simplify(expr / sp.exp(z * u) - char) == 0)
# exact delay margin at the reference loop (Delta = 2, theta = 0.5, gp = 4, gd = 3)
thv, gpv, gdv, l2 = 0.5, 4.0, 3.0, 2 * 0.25
b_, c_ = 2 * l2 - gdv ** 2 * thv ** 4, l2 ** 2 - gpv ** 2 * thv ** 4
wc = math.sqrt(0.5 * (-b_ + math.sqrt(b_ * b_ - 4 * c_)))
tau = math.atan2(gdv * wc, gpv) / wc
res = complex(-wc ** 2 - l2 + thv ** 2 * (gpv + gdv * 1j * wc) * complex(math.cos(wc * tau), -math.sin(wc * tau)))
ok("delay margin 0.336/theta* is a root on the imaginary axis", abs(res) < 1e-12 and abs(tau * thv - 0.336) < 5e-4)
# capture threshold: A k^2 - eta k^{2D} < 0  <=>  k < theta Delta^{-1/(2(1-Delta))}
for Dv in (0.25, 0.5, 0.75):
    thr = 0.35 * Dv ** (-1 / (2 * (1 - Dv)))
    etav = thr ** 0 * 0.35 ** (2 * (1 - Dv)) / Dv
    ok("capture threshold at Delta = %g" % Dv, abs(thr ** 2 - etav * thr ** (2 * Dv)) < 1e-12)
# uncoupled span: arcsin(sech w) = pi/2 - gd(w) for w > 0
ok("uncoupled span arcsin(e^phi sqrt(A/E))", all(abs(math.asin(1 / math.cosh(v)) - (math.pi / 2 - math.atan(math.sinh(v)))) < 1e-14
                                               for v in (0.1, 0.7, 2.0, 5.0)))
# finite quanta: N hbar D theta < |E*| with A theta = hbar N kappa
Nq, hb, Nh, kap, eps = sp.symbols("N_q hbar N_h kappa epsilon", positive=True)
Ath = hb * Nh * kap / th
bound = sp.solve(sp.Eq(Nq * hb * D * th, (Ath * th ** 2 * (1 - D) / D)), Nq)[0]
ok("finite quanta at the passive ground state, Eq. (nq)", sp.simplify(bound - (1 - D) * Nh * kap / D ** 2) == 0)
bound_f = sp.solve(sp.Eq(Nq * hb * D * th, eps * Ath * th ** 2), Nq)[0]
ok("finite quanta from the kick tolerance", sp.simplify(bound_f - eps * Nh * kap / D) == 0)
# single-mode stability: H = D b^dag b - g Dn (b + b^dag)^2 -> (D/2)(p^2 + x^2) - 2 g Dn x^2
g, Dn = sp.symbols("g D_n", positive=True)
curv = D / 2 - 2 * g * Dn
ok("single-mode stability limit g = Delta/(4 D_Delta)", sp.solve(sp.Eq(curv, 0), g)[0] == D / (4 * Dn))
kap_cap = D * (D / (4 * Dn)) * Dn / 4 ** D
ok("binding at the stability limit Delta^2/4^{Delta+1}, 1/16 at Delta = 1",
   sp.simplify(kap_cap - D ** 2 / 4 ** (D + 1)) == 0 and kap_cap.subs(D, 1) == sp.Rational(1, 16))
# QEI change of variables: x+ = t_in + 2 sigma, h(x+) = H(sigma)
sig, xp = sp.symbols("sigma x_plus", real=True)
H = sp.sin(3 * sig) * sp.sin(sig)            # test sampler on (0, pi/2)
lhs_h2 = sp.integrate((H.subs(sig, xp / 2)) ** 2, (xp, 0, sp.pi))
ok("int h^2 dx+ = 2 int H^2 dsigma", sp.simplify(lhs_h2 - 2 * sp.integrate(H ** 2, (sig, 0, sp.pi / 2))) == 0)
lhs_hp = sp.integrate(sp.diff(H.subs(sig, xp / 2), xp) ** 2, (xp, 0, sp.pi))
ok("int h'^2 dx+ = (1/2) int H_sigma^2 dsigma", sp.simplify(lhs_hp - sp.integrate(sp.diff(H, sig) ** 2, (sig, 0, sp.pi / 2)) / 2) == 0)
# conformal time-slice bound with h = sqrt(2) sin(sigma) on half the doubled circle (L = 2 pi)
hc = sp.sqrt(2) * sp.sin(sig)
bnd = -(sp.integrate(sp.diff(hc, sig) ** 2, (sig, 0, sp.pi)) - sp.Rational(1, 4) * sp.integrate(hc ** 2, (sig, 0, sp.pi))) / (12 * sp.pi)
ok("conformal time-slice bound -1/16 per unit central charge", sp.simplify(bnd + sp.Rational(1, 16)) == 0)
# electromagnetic-like multipoles at the stability estimate: sum_{l>=1} 2(2l+1)(l+2)^2/4^{l+3} = 14/27
l = sp.symbols("l", integer=True, positive=True)
Sem = sp.summation(2 * (2 * l + 1) * (l + 2) ** 2 / 4 ** (l + 3), (l, 1, sp.oo))
ok("electromagnetic-like multipole sum 14/27 = 0.52", sp.simplify(Sem - sp.Rational(14, 27)) == 0)
# near-extremal RN (G = c = hbar = k_B = 1): M - Q -> 2 pi^2 Q^3 T^2
Q, d = sp.symbols("Q delta", positive=True)
M = sp.sqrt(Q ** 2 + d ** 2)
rp, rm = M + d, M - d
T = (rp - rm) / (4 * sp.pi * rp ** 2)
ratio = sp.limit((M - Q) / (2 * sp.pi ** 2 * Q ** 3 * T ** 2), d, 0)
ok("near-extremal Reissner-Nordstrom: M - M_ext -> 2 pi^2 r_e^3 T^2", ratio == 1)
# rate bound: theta <= kappa_max N/(C hbar), C hbar = r_e^3/(l_P^2 c)
re_, lP, c, N = sp.symbols("r_e l_P c N", positive=True)
ok("rate bound N c l_P^2/(16 r_e^3), Eq. (theta)",
   sp.simplify(sp.Rational(1, 16) * N / (re_ ** 3 / (lP ** 2 * c)) - N * c * lP ** 2 / (16 * re_ ** 3)) == 0)
print("equation checks done, %d failures" % len(fails))
sys.exit(1 if fails else 0)
