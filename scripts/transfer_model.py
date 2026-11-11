"""Opening and maintenance of a traversable wormhole by two distinct boundary couplings, with the transfer of
gravitational support between them, in the near-horizon region of a near-extremal charged black-hole pair.

Framework (Section S1). Four-dimensional Einstein-Maxwell theory reduces near the horizon of a near-extremal
Reissner-Nordstrom hole to Jackiw-Teitelboim (JT) gravity on AdS2 x S2 (radius r_e): the boundary dynamics is the
Schwarzian with coefficient C = r_e^3/(l_P^2 hbar c) per side, M - M_ext = 2 pi^2 C (k T)^2. Bulk fields are the
multipoles of the throat; each is a generalized free field of conformal dimension Delta in AdS2. The coupled
gravitational-electromagnetic multipoles have Delta = l (l >= 2) and Delta = l + 2 (l >= 1), so every photon or
graviton channel has Delta >= 2. Electrically charged throats discharge (charged fields violate the AdS2
Breitenlohner-Freedman bound), so the throat is magnetic, as in Maldacena-Milekhin-Popov (MMP).

Hamiltonian (two sides L, R; sectors k = opening "o", holding "h"; distinct multipole channels):
    H_total = H_gravity + H_quantum + H_open + H_hold + H_interaction + H_drive,
    H_open = -eta_o(u) sum_j O^j_L O^j_R   (Delta_o; mercury-switched line),
    H_hold = -eta_h(u) sum_j P^j_L P^j_R   (Delta_h; hydrogen-referenced carrier or a reference sector),
H_interaction: at leading order in 1/N the sectors interact only through the shared boundary mode (gravity).
Semiclassical equations (symmetric configuration, t_L = t_R = t(u), t' = e^phi, units A = C hbar^2 = 1 with
the initial thermofield-double rate 2 pi/beta = 1):
    phi'' = -e^{2 phi} + sum_k Delta_k eta_k e^{2 Delta_k phi},
    E = phi'^2 + e^{2 phi} - sum_k eta_k e^{2 Delta_k phi},   dE/du = -sum_k (d eta_k/du) e^{2 Delta_k phi}.
Bulk side: the first-order null stress tensor of each sector in the global AdS2 vacuum, and the JT ray identity
    int_ray sin^2(sigma) T_++ dx^+ = -(C hbar^2/2)(t'_in + t'_out)      (= -(t'_in + t'_out)/2 for A = 1),
which the boundary equation must reproduce.

Checks:
  S1  reduction: exact near-extremal RN mass excess -> 2 pi^2 r_e^3 T^2/G; near-horizon coefficients of the coupled
      gravitational-electromagnetic potentials (Chandrasekhar) give Delta = l + 2 and l; Q -> 0 limits; test field
      Delta = l + 1; electron instability in electric throats (scalar estimate and Dirac threshold).
  S2  boundary dynamics: thermofield-double solution, energy balance, stationary point and frequency (Maldacena-Qi),
      capture threshold, runaway of Delta > 1 couplings, reference protocols and the five determinations, the
      period of the bound passive orbit, the kick tolerance of the stabilized loop and its exact delay margin.
  S3  stress tensor: static ray identity int 2 sin^2 tau_0 = -Delta <O_L O_R> for several Delta, conservation, L-R
      symmetry, mode-sum versus Euclidean evaluation, exact first-order null energy along rays through the
      transition (both sectors) against the JT ray identity, sector shares, null quantum energy inequality with
      optimal samplers (lowest Dirichlet mode of -d^2 - 1 + 48 pi g tau on every maximal qualifying segment),
      the null-line two-point function of every channel against the massless one (which carries the conformal
      QEI to any Delta) and the boundary limit of the propagator used there.
  S4  hydrogen interaction classes: resonant intermediary (no resonance in the ground-state correlation),
      replenished correlations (bounded by the coupled ground state, nothing without a coupling), hydrogen matter
      (Delta = m_H c r_e/hbar).
  S5  physical scales: binding per channel (single-mode stability estimate and null QEI, scanned over Delta),
      the Einstein-Maxwell multipole sum, channel numbers, finite-quanta capacities, MMP comparison,
      field-exclusion radii (superconducting mercury, hydrogen maser), light times, electroweak scale.
"""
import os, sys, json, math, functools
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import make_interp_spline
from scipy.special import gammaln, eval_gegenbauer
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import dpttrf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# CODATA 2018 values
HBAR, C, G, KB = 1.054571817e-34, 2.99792458e8, 6.67430e-11, 1.380649e-23
QE, ME, AMU, EPS0 = 1.602176634e-19, 9.1093837015e-31, 1.66053906660e-27, 8.8541878128e-12
TC_HG = 4.15                                       # K, mercury critical temperature
LP = math.sqrt(G * HBAR / C ** 3)
ALPHA = QE ** 2 / (4 * math.pi * EPS0 * HBAR * C)
M_H = 1.00782503223 * AMU
BC_HG = 0.0412                                     # T, mercury critical field
M_W = 80.369e9 * QE / C ** 2                       # kg
T_UNIVERSE = 13.79e9 * 3.15576e7                   # s
BR = C ** 2 / math.sqrt(4 * math.pi * EPS0 * G) / C   # T m: horizon field times r_e of an extremal magnetic hole
NU_HFS = 1420.405751768e6                          # Hz, hydrogen ground-state hyperfine line
MU_B = QE * HBAR / (2 * ME)
G_J, G_I_H = 2.00231930436, -2 * 2.79284734 * ME / 1.67262192e-27   # electron; proton in Bohr-magneton units
B_BR = 2 * math.pi * HBAR * NU_HFS / ((G_J - G_I_H) * MU_B)          # T, Breit-Rabi decoupling field of hydrogen
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


# =====================================================================================================  S1
def rn_excess_ratio(T):
    """(M - Q)/(2 pi^2 Q^3 T^2) for an RN hole of charge Q = 1 at temperature T (G = c = hbar = k = 1);
    x = r_+ - 1 solves T = x (x + 2)/(4 pi (1 + x)^3) and M - Q = x^2/(2 (1 + x))."""
    x = brentq(lambda y: y * (y + 2) / (4 * math.pi * (1 + y) ** 3) - T, 0.0, 0.7, xtol=1e-18, rtol=1e-15)
    return x * x / (2 * (1 + x)) / (2 * math.pi ** 2 * T ** 2)


def rn_potentials(l, r, Q, Mass=1.0):
    """Chandrasekhar's decoupled axial potentials of RN (Z_1: electromagnetic as Q -> 0, Z_2: gravitational).
    V_i = (Delta/r^5)[(mu^2 + 2) r - q_j + 4 Q^2/r],  q_1 + q_2 = 6M,  q_1 q_2 = -4 Q^2 mu^2,  mu^2 = (l-1)(l+2)."""
    mu2 = (l - 1) * (l + 2)
    s = math.sqrt(9 * Mass ** 2 + 4 * Q ** 2 * mu2)
    q1, q2 = 3 * Mass + s, 3 * Mass - s
    rp, rm = Mass + math.sqrt(max(Mass * Mass - Q * Q, 0.0)), Mass - math.sqrt(max(Mass * Mass - Q * Q, 0.0))
    dRN = (r - rp) * (r - rm)
    V1 = dRN / r ** 5 * ((mu2 + 2) * r - q2 + 4 * Q * Q / r)
    V2 = dRN / r ** 5 * ((mu2 + 2) * r - q1 + 4 * Q * Q / r)
    return V1, V2


def rn_polar_potentials(l, r, Q, Mass=1.0):
    """Polar (even) partners through Chandrasekhar's relation V_i^(+-) = +- q_j df_i/dr_* + q_j^2 f_i^2 + kappa f_i,
    f_i = Delta/(r^3 (mu^2 r + q_j)), kappa = mu^2 (mu^2 + 2); the minus sign reproduces rn_potentials."""
    mu2 = (l - 1) * (l + 2)
    s = math.sqrt(9 * Mass ** 2 + 4 * Q ** 2 * mu2)
    q1, q2 = 3 * Mass + s, 3 * Mass - s
    rp, rm = Mass + math.sqrt(max(Mass * Mass - Q * Q, 0.0)), Mass - math.sqrt(max(Mass * Mass - Q * Q, 0.0))
    out = []
    for qj in (q2, q1):
        dRN = lambda x: (x - rp) * (x - rm)
        f = lambda x: dRN(x) / (x ** 3 * (mu2 * x + qj))
        dfdr = (mu2 * r + qj) ** -1 * ((2 * r - rp - rm) / r ** 3 - 3 * dRN(r) / r ** 4) - dRN(r) * mu2 / (
            r ** 3 * (mu2 * r + qj) ** 2)
        out.append(qj * dfdr * dRN(r) / r ** 2 + qj ** 2 * f(r) ** 2 + mu2 * (mu2 + 2) * f(r))
    return out


def nh_coefficient(Vfun, eps=1e-7):
    """lim V z^2 with the Poincare coordinate z = M^2/(r - M) of the extremal throat (M = Q = 1)."""
    r = 1.0 + eps
    return Vfun(r) / eps ** 2


def delta_of(cval):
    return 0.5 + math.sqrt(0.25 + cval)


def s1():
    out = {}
    Ts = [1e-4, 5e-5, 2.5e-5]
    rat = [rn_excess_ratio(T) for T in Ts]
    extrap = (8 * rat[2] - 6 * rat[1] + rat[0]) / 3          # Richardson, error O(T^3)
    out["rn_ratio"] = rat
    out["rn_ratio_T0"] = extrap
    check("S1 near-extremal RN: (M - Q) / (2 pi^2 r_e^3 T^2) -> 1", abs(extrap - 1) < 1e-6, "%.8f" % extrap)
    rows = []
    ok_em, ok_gr, ok_tf = True, True, True
    for l in range(1, 7):
        c_em = nh_coefficient(lambda r: rn_potentials(l, r, 1.0)[0])
        D_em = delta_of(c_em)
        ok_em &= abs(D_em - (l + 2)) < 1e-5
        row = dict(l=l, c_em=c_em, Delta_em=D_em)
        if l >= 2:
            c_gr = nh_coefficient(lambda r: rn_potentials(l, r, 1.0)[1])
            D_gr = delta_of(c_gr)
            ok_gr &= abs(D_gr - l) < 1e-5
            row.update(c_gr=c_gr, Delta_gr=D_gr)
        c_tf = nh_coefficient(lambda r: (r - 1) ** 2 / r ** 4 * l * (l + 1))
        ok_tf &= abs(delta_of(c_tf) - (l + 1)) < 1e-5
        row["Delta_test"] = delta_of(c_tf)
        rows.append(row)
    out["spectrum"] = rows
    check("S1 coupled EM-like multipoles: Delta = l + 2 (l = 1..6)", ok_em)
    check("S1 coupled gravitational multipoles: Delta = l (l = 2..6)", ok_gr)
    # polar partners: same near-horizon coefficients (the df/dr_* and f^2 terms are subleading at the horizon)
    okp = True
    for l in range(2, 7):
        cp = [nh_coefficient(lambda r, k=k: rn_polar_potentials(l, r, 1.0)[k], eps=1e-5) for k in (0, 1)]
        okp &= abs(delta_of(cp[0]) - (l + 2)) < 2e-3 and abs(delta_of(cp[1]) - l) < 2e-3
    # minus-sign relation reproduces the axial potentials away from extremality
    V1, V2 = rn_potentials(3, 2.4, 0.9)
    Vm = []
    for k, qsign in ((0, -1), (1, -1)):
        mu2 = 10
        s_ = math.sqrt(9 + 4 * 0.81 * mu2)
        qj = (3 - s_) if k == 0 else (3 + s_)
        rp, rm = 1 + math.sqrt(1 - 0.81), 1 - math.sqrt(1 - 0.81)
        dRN = lambda x: (x - rp) * (x - rm)
        f = lambda x: dRN(x) / (x ** 3 * (mu2 * x + qj))
        h = 1e-6
        dfs = (f(2.4 + h) - f(2.4 - h)) / (2 * h) * dRN(2.4) / 2.4 ** 2
        Vm.append(-qj * dfs + qj ** 2 * f(2.4) ** 2 + mu2 * (mu2 + 2) * f(2.4))
    okrel = abs(Vm[0] - V1) < 1e-9 and abs(Vm[1] - V2) < 1e-9
    check("S1 polar multipoles (Chandrasekhar partners): same Delta = l + 2 and l; the relation reproduces the axial "
          "potentials", okp and okrel)
    check("S1 test Maxwell field: Delta = l + 1", ok_tf)
    # Q -> 0: Regge-Wheeler potentials (sanity of the labels)
    r = 3.7
    V1, V2 = rn_potentials(3, r, 0.0)
    f = 1 - 2 / r
    check("S1 Q -> 0: Z_1 = spin-1, Z_2 = spin-2 Regge-Wheeler",
          abs(V1 - f * 12 / r ** 2) < 1e-12 and abs(V2 - f * (12 / r ** 2 - 6 / r ** 3)) < 1e-12)
    out["Delta_min_EM_sector"] = 2
    # electric throats: Breitenlohner-Freedman violation for electrons, ratio (q E l^2)/(m l) = e/(m_e sqrt(4 pi eps0 G))
    ratio = QE / (ME * math.sqrt(4 * math.pi * EPS0 * G))
    out["electron_charge_to_mass_grav"] = ratio
    # charge parameter q_eff = ratio x with x = r_e m c/hbar; scalar (l = 0): nu^2 = x^2 + 1/4 - q_eff^2;
    # Dirac field (lowest S2 eigenvalue j + 1/2 = 1): nu^2 = x^2 + 1 - q_eff^2
    r_bf = (HBAR / (ME * C)) / (2 * math.sqrt(ratio ** 2 - 1))
    r_bf_dirac = (HBAR / (ME * C)) / math.sqrt(ratio ** 2 - 1)
    out["electric_bf_radius"] = r_bf
    out["electric_dirac_radius"] = r_bf_dirac
    check("S1 electric throats: a scalar of the electron's charge and mass violates the BF bound for r_e > %.1e m, "
          "and the electron (Dirac, j = 1/2) has complex dimension for r_e > %.1e m" % (r_bf, r_bf_dirac),
          ratio > 1e20 and r_bf < 1e-30 and abs(r_bf_dirac / r_bf - 2) < 1e-12)
    out["BR"] = BR
    out["B_EW"] = M_W ** 2 * C ** 2 / (QE * HBAR)
    out["r_e_EW"] = BR / out["B_EW"]
    return out


# =====================================================================================================  S2
def ramp(u, a, w):
    return 0.5 * (1.0 + np.tanh((u - a) / w))


def window(u, a, b, w):
    return ramp(u, a, w) * (1.0 - ramp(u, b, w))


class Sector:
    """A boundary coupling -eta(u, phi, phi') sum O_L O_R of dimension D (effective strength, units A = 1)."""
    def __init__(self, D, eta, label):
        self.D, self.eta, self.label = D, eta, label


# reference parameters (model units: A = 1, initial 2 pi / beta = 1)
P = dict(u0=-10.0, u1=70.0,
         D_o=2.0, eta_o=0.40, a_o=0.0, b_o=5.0, w=0.4,       # opening photon channels (proposed mercury switch)
         D_p=0.25, th_p=0.35,                                   # passive holding sector (reference, Delta < 1)
         D_f=2.0, th_f=0.5, kp=4.0, kd=3.0, emax=2.0,          # stabilized photon channels (proposed maser phase reference)
         u_c=1.5)


def opening(eta0=None, a=None, b=None, w=None):
    eta0 = P["eta_o"] if eta0 is None else eta0
    a = P["a_o"] if a is None else a
    b = P["b_o"] if b is None else b
    w = P["w"] if w is None else w
    return Sector(P["D_o"], lambda u, p, dp: eta0 * window(u, a, b, w), "opening")


def stationary(D, theta):
    """Maldacena-Qi stationary point: eta, energy, small-oscillation frequency (A = 1)."""
    eta = theta ** (2 * (1 - D)) / D
    E0 = -theta ** 2 * (1 - D) / D
    om2 = 2 * (1 - D) * theta ** 2
    return eta, E0, om2


def holding_passive(uc=None, theta=None, D=None, w=None):
    D = P["D_p"] if D is None else D
    theta = P["th_p"] if theta is None else theta
    uc = P["u_c"] if uc is None else uc
    w = P["w"] if w is None else w
    eta = stationary(D, theta)[0]
    return Sector(D, lambda u, p, dp: eta * ramp(u, uc, w), "holding")


def holding_stabilized(uc=None, emax=None):
    D, th = P["D_f"], P["th_f"]
    uc = P["u_c"] if uc is None else uc
    emax = P["emax"] if emax is None else emax
    eta_s = stationary(D, th)[0]
    ph_s = math.log(th)

    a = emax / (emax - 1.0)

    def eta(u, p, dp):
        x = 1.0 - P["kp"] * (p - ph_s) - P["kd"] * dp          # linear control law (x = 1 at the target)
        sat = emax / (1.0 + (emax - 1.0) * np.exp(-a * (x - 1.0)))   # smooth saturation: sat(1) = 1, sat'(1) = 1
        return ramp(u, uc, P["w"]) * eta_s * sat
    return Sector(D, eta, "holding")


def tfd(u):
    return np.array([-np.log(np.cosh(u)), -np.tanh(u), np.arcsin(np.tanh(u))])


def rhs(u, y, secs):
    phi, dphi = y[0], y[1]
    F = -np.exp(2 * phi)
    for s in secs:
        F = F + s.D * s.eta(u, phi, dphi) * np.exp(2 * s.D * phi)
    return [dphi, F, np.exp(phi)]


def run(secs, u0=None, u1=None, du=0.002):
    u0 = P["u0"] if u0 is None else u0
    u1 = P["u1"] if u1 is None else u1
    blow = lambda u, y, *a: y[0] - 4.0
    blow.terminal = True
    sol = solve_ivp(rhs, (u0, u1), tfd(u0), args=(secs,), method="DOP853", rtol=1e-12, atol=1e-14,
                    dense_output=True, events=blow, max_step=0.02)
    uu = np.arange(u0, sol.t[-1] + 1e-12, du)
    Y = sol.sol(uu)
    phi, dphi, t = Y
    etas = [np.array([s.eta(a, p, q) for a, p, q in zip(uu, phi, dphi)]) for s in secs]
    E = dphi ** 2 + np.exp(2 * phi)
    for s, e in zip(secs, etas):
        E = E - e * np.exp(2 * s.D * phi)
    return dict(u=uu, phi=phi, dphi=dphi, t=t, E=E, etas=etas, secs=secs, sol=sol,
                runaway=bool(sol.t[-1] < u1 - 1e-9), u_end=float(sol.t[-1]))


def remaining_free(phi, dphi, E):
    """Global time still available to a free (uncoupled) boundary at (phi, phi') with energy E > 0."""
    x = min(1.0, math.exp(phi) / math.sqrt(E))
    return math.asin(x) if dphi < 0 else math.pi - math.asin(x)


def protocols():
    """Reference configurations: no coupling, opening only, holding only, opening followed by holding."""
    return {
        "none": [],
        "open": [opening()],
        "hold_p": [holding_passive()],
        "open_hold_p": [opening(), holding_passive()],
        "hold_f": [holding_stabilized()],
        "open_hold_f": [opening(), holding_stabilized()],
    }


def global_span(R):
    """t(+inf) - t(-inf) for a run that ends uncoupled (couplings off) or the simulated span otherwise."""
    E_end = R["E"][-1]
    t_m = -math.pi / 2
    coupled_end = any(abs(e[-1]) > 1e-12 for e in R["etas"])
    if coupled_end:
        return R["t"][-1] - t_m, False
    return R["t"][-1] + remaining_free(R["phi"][-1], R["dphi"][-1], E_end) - t_m, True


def arrival(R, u_launch):
    """Boundary time at which a null ray launched at u_launch reaches the other boundary (nan if it does not)."""
    t_in = np.interp(u_launch, R["u"], R["t"])
    target = t_in + math.pi
    if target > R["t"][-1]:
        return float("nan")
    return float(np.interp(target, R["t"], R["u"]))


def s2():
    out = {}
    # TFD solution and energy
    R0 = run([], u1=10.0)
    tf = tfd(R0["u"])
    check("S2 thermofield double: phi = -log cosh u, E = 1", np.max(abs(R0["phi"] - tf[0])) < 1e-9 and
          np.max(abs(R0["E"] - 1.0)) < 1e-9)
    # stationary point (Maldacena-Qi) and small oscillations
    for D, th in ((P["D_p"], P["th_p"]), (0.5, 0.4)):
        eta, E0, om2 = stationary(D, th)
        sec = Sector(D, lambda u, p, dp, e=eta: e, "x")
        y0 = [math.log(th) + 1e-4, 0.0, 0.0]
        sol = solve_ivp(rhs, (0, 200), y0, args=([sec],), method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        uu = np.linspace(0, 200, 40001)
        ph = sol.sol(uu)[0] - math.log(th)
        zc = uu[1:][(ph[1:] > 0) & (ph[:-1] <= 0)]
        om_num = 2 * math.pi / np.mean(np.diff(zc))
        out["omega_err_%g" % D] = abs(om_num / math.sqrt(om2) - 1)
        check("S2 stationary point D=%.2f: omega = sqrt(2(1-D)) t'" % D, abs(om_num / math.sqrt(om2) - 1) < 1e-4,
              "%.6f vs %.6f" % (om_num, math.sqrt(om2)))
    out["stationary_p"] = dict(zip(("eta", "E0", "omega2"), stationary(P["D_p"], P["th_p"])))
    out["stationary_f"] = dict(zip(("eta", "E0", "omega2"), stationary(P["D_f"], P["th_f"])))
    out["E_f_positive"] = P["th_f"] ** 2 * (P["D_f"] - 1) / P["D_f"]
    # capture threshold of a sudden holding coupling at the moment of time symmetry
    D, th = P["D_p"], P["th_p"]
    k_thr = th * D ** (-1.0 / (2 * (1 - D)))
    out["capture_threshold"] = k_thr
    eta = stationary(D, th)[0]
    E_after = 1.0 - eta                       # k = 1, phi = 0 at u = 0
    check("S2 capture threshold: k = 1 > %.3f, a sudden holding coupling at u = 0 leaves E = %.3f > 0" % (k_thr, E_after),
          k_thr < 1 and E_after > 0)
    out["E_after_sudden_hold"] = E_after
    # runaway of a Delta > 1 coupling held on (no stabilization)
    Rr = run([Sector(2.0, lambda u, p, dp: 0.6 * ramp(u, -1.0, 0.4), "x")], u1=30.0)
    check("S2 Delta = 2 coupling above the stationary strength runs away (finite-time blow-up)", Rr["runaway"],
          "blow-up at u = %.2f" % Rr["u_end"])
    out["runaway_u"] = Rr["u_end"]
    # reference protocols
    runs = {k: run(v) for k, v in protocols().items()}
    summ = {}
    for k, R in runs.items():
        span, closed = global_span(R)
        if R["runaway"]:
            summ[k] = dict(runaway=True)
            continue
        s = dict(runaway=False, E_end=float(R["E"][-1]), tp_end=float(math.exp(R["phi"][-1])),
                 span=float(span), span_closed=closed)
        summ[k] = s
    out["protocols"] = summ
    # energy balance along the transition (dE/du = -sum eta_k' e^{2 D_k phi})
    for key in ("open_hold_p", "open_hold_f"):
        R = runs[key]
        u = R["u"]
        dE = np.gradient(R["E"], u)
        rhs_E = np.zeros_like(u)
        for s, e in zip(R["secs"], R["etas"]):
            rhs_E -= np.gradient(e, u) * np.exp(2 * s.D * R["phi"])
        err = np.max(abs(dE - rhs_E)[50:-50]) / max(1e-12, np.max(abs(rhs_E)))
        out["energy_balance_err_" + key] = err
        check("S2 energy balance dE/du = -sum eta_k' e^(2 D_k phi) (%s)" % key, err < 2e-4, "rel %.1e" % err)
    # (1) opening window: opening alone, probe limit
    Ro = runs["open"]
    span_o, _ = global_span(Ro)
    out["open_span_excess"] = span_o - math.pi
    t_inf = -math.pi / 2 + span_o
    u_w = float(np.interp(t_inf - math.pi, Ro["t"], Ro["u"]))
    out["open_window_end_u"] = u_w
    out["open_E_after"] = float(Ro["E"][-1])
    check("S2 (1) opening alone: global span exceeds pi (probe window for launches before u = %.2f)" % u_w,
          span_o > math.pi + 0.5 and not Ro["runaway"])
    check("S2 no coupling: span = pi (no window)", abs(global_span(runs["none"])[0] - math.pi) < 1e-6)
    weak = []
    for e0 in (1e-2, 1e-3, 1e-4):
        Rw = run([opening(eta0=e0)], u1=40.0)
        sp, _ = global_span(Rw)
        dlt = sp - math.pi
        t_inf = -math.pi / 2 + sp
        uw = float(np.interp(t_inf - math.pi, Rw["t"], Rw["u"]))
        weak.append((e0, dlt, uw, math.log(dlt / 2)))
    out["weak_windows"] = weak
    lin = [w_[1] / w_[0] for w_ in weak]
    check("S2 weak opening: span excess proportional to the coupling, launch deadline u_w = ln(excess/2)",
          abs(lin[2] / lin[1] - 1) < 2e-3 and all(abs(w_[2] - w_[3]) < 0.02 for w_ in weak[1:]),
          "excess/eta = %.4f, %.4f, %.4f" % tuple(lin))
    check("S2 opening alone ends uncoupled with E > 0 (window closes)", Ro["E"][-1] > 0)
    # (2) support share of the holding sector at the start of the opening ramp-down (coupling-force share)
    for key in ("open_hold_p", "open_hold_f"):
        R = runs[key]
        i = np.argmin(abs(R["u"] - (P["b_o"] - 2 * P["w"])))
        forces = [s.D * e[i] * math.exp(2 * s.D * R["phi"][i]) for s, e in zip(R["secs"], R["etas"])]
        out["force_share_hold_" + key] = forces[1] / sum(forces)
        out["force_share_hold_end_" + key] = 1.0
    check("S2 (2) holding sector carries most of the coupling force before the opening ramp-down (passive)",
          out["force_share_hold_open_hold_p"] > 0.5, "%.3f" % out["force_share_hold_open_hold_p"])
    check("S2 (2) holding sector carries most of the coupling force before the opening ramp-down (stabilized)",
          out["force_share_hold_open_hold_f"] > 0.5, "%.3f" % out["force_share_hold_open_hold_f"])
    # (3) traversable through the transition: every launch from the first arriving one onwards arrives
    for key in ("open_hold_p", "open_hold_f"):
        R = runs[key]
        launches = np.arange(-6.0, R["u"][-1] - 25.0, 0.01)
        arr = np.array([arrival(R, x) for x in launches])
        ok = np.isfinite(arr)
        first = launches[np.argmax(ok)]
        cont = bool(np.all(ok[np.argmax(ok):]))
        out["first_launch_" + key] = float(first)
        out["continuous_" + key] = cont
        check("S2 (3) %s: every launch after u = %.2f arrives (no gap through the transition)" % (key, first), cont)
    # (4) maintenance after the opening coupling is off
    Rp = runs["open_hold_p"]
    E_p = float(Rp["E"][-1])
    E0p = stationary(P["D_p"], P["th_p"])[1]
    out["E_final_p"] = E_p
    check("S2 (4) passive holding after opening off: E0 < E < 0 (bound, traversable)", E0p < E_p < 0,
          "E = %.4f, E0 = %.4f" % (E_p, E0p))
    Rf = runs["open_hold_f"]
    dev = abs(Rf["phi"][-1] - math.log(P["th_f"]))
    out["phi_dev_end_f"] = dev
    check("S2 (4) stabilized holding after opening off: t' -> t'* (|dphi| < 1e-8)", dev < 1e-8, "%.1e" % dev)
    # holding alone fails in both cases (the opening is needed)
    check("S2 holding alone (passive) does not capture: E stays > 0", runs["hold_p"]["E"][-1] > 0)
    check("S2 holding alone (stabilized, same strength limit) does not capture", runs["hold_f"]["phi"][-1] < -10)
    # passive holding alone at any turn-on time: best energy
    best = min(run([holding_passive(uc=x)], u1=40.0)["E"][-1] for x in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5))
    out["E_best_hold_alone_p"] = float(best)
    check("S2 passive holding alone at its best turn-on time still leaves E > 0", best > 0, "%.3f" % best)
    # (5) stability: passive -- orbit range and response to energy kicks; stabilized -- closed-loop roots, delay
    ph = Rp["phi"][Rp["u"] > 30]
    out["phi_range_p"] = (float(ph.min()), float(ph.max()))
    kicks = []
    for frac in (0.25, 0.5, 0.9, 1.1):
        dE = frac * abs(E_p)
        y = [Rp["phi"][-1], -math.sqrt(Rp["dphi"][-1] ** 2 + dE), Rp["t"][-1]]
        sol = solve_ivp(rhs, (0, 400), y, args=([holding_passive(uc=-100.0)],), method="DOP853", rtol=1e-11,
                        atol=1e-13, dense_output=True)
        pm = sol.sol(np.linspace(0, 400, 20001))[0]
        kicks.append((frac, float(pm.min()), bool(pm.min() > -40)))
    out["kicks_p"] = kicks
    check("S2 (5) passive: kicks below the binding energy stay bound, above it escape",
          all(k[2] for k in kicks[:3]) and not kicks[3][2])
    # period of the bound passive orbit: T = 2 int dphi / sqrt(E + eta e^{2 D phi} - e^{2 phi}) between the turning
    # points (substitution phi = c + d sin x removes the endpoint singularities), checked by continued integration
    eta_p, Dp = stationary(P["D_p"], P["th_p"])[0], P["D_p"]
    rad = lambda x: E_p + eta_p * math.exp(2 * Dp * x) - math.exp(2 * x)
    ph_s = math.log(P["th_p"])
    lo, hi = brentq(rad, ph_s - 30.0, ph_s), brentq(rad, ph_s, ph_s + 5.0)
    c0, d0 = 0.5 * (lo + hi), 0.5 * (hi - lo)
    per = 2 * quad(lambda x: d0 * math.cos(x) / math.sqrt(max(rad(c0 + d0 * math.sin(x)), 1e-300)),
                   -math.pi / 2, math.pi / 2, limit=400, epsabs=0, epsrel=1e-11)[0]
    y = [Rp["phi"][-1], Rp["dphi"][-1], Rp["t"][-1]]
    sol = solve_ivp(rhs, (0, 400), y, args=([holding_passive(uc=-100.0)],), method="DOP853", rtol=1e-12,
                    atol=1e-14, dense_output=True)
    uu = np.linspace(0, 400, 400001)
    pp = sol.sol(uu)[0]
    pk = uu[1:-1][(pp[1:-1] > pp[:-2]) & (pp[1:-1] >= pp[2:])]
    per_num = float(np.mean(np.diff(pk)))
    out["orbit_period_p"] = per
    out["orbit_tp_range_p"] = (math.exp(lo), math.exp(hi))
    out["omega_small_p"] = math.sqrt(stationary(P["D_p"], P["th_p"])[2])
    check("S2 (5) passive orbit after the transition: period %.2f (frequency %.3f) by quadrature = continued "
          "integration; small-oscillation frequency %.3f" % (per, 2 * math.pi / per, out["omega_small_p"]),
          abs(per_num / per - 1) < 1e-4, "%.6f vs %.6f" % (per_num, per))
    th = P["th_f"]
    lam2 = 2 * (P["D_f"] - 1) * th ** 2
    roots = np.roots([1.0, P["kd"] * th ** 2, (P["kp"] - 2 * (P["D_f"] - 1)) * th ** 2])
    out["closed_loop_roots"] = [complex(z) for z in roots]
    out["open_loop_rate"] = math.sqrt(lam2)
    check("S2 (5) stabilized: open-loop growth rate sqrt(2(D-1)) t'* = %.3f, closed-loop roots in the left half-plane"
          % math.sqrt(lam2), np.all(roots.real < 0))
    # largest loop delay: s^2 + kd th^2 e^{-s tau} s + kp th^2 e^{-s tau} - lam2 = 0 reaches s = i w:
    # th^2 |kp + i kd w| = w^2 + lam2 and w tau = arg(kp + i kd w)
    th4 = th ** 4
    bq, cq = 2 * lam2 - P["kd"] ** 2 * th4, lam2 ** 2 - P["kp"] ** 2 * th4
    w_c = math.sqrt(0.5 * (-bq + math.sqrt(bq * bq - 4 * cq)))
    tau_max = math.atan2(P["kd"] * w_c, P["kp"]) / w_c

    def max_real(tau):
        from numpy.polynomial import polynomial as Pn
        # Pade(4,4) approximation of e^{-s tau}
        c = [1.0, -0.5, 3.0 / 28, -1.0 / 84, 1.0 / 1680]
        num = np.array([c[k] * tau ** k for k in range(5)])
        den = np.array([abs(c[k]) * tau ** k for k in range(5)])
        poly = Pn.polyadd(Pn.polymul([-lam2, 0.0, 1.0], den),
                          Pn.polymul([P["kp"] * th ** 2, P["kd"] * th ** 2], num))
        return float(np.max(np.roots(poly[::-1]).real))
    taus = np.linspace(0.0, 6.0, 6001)
    mr = np.array([max_real(x) for x in taus])
    tau_pade = float(taus[np.argmax(mr >= 0)])
    out["max_loop_delay"] = tau_max
    out["max_loop_delay_theta"] = tau_max * th
    check("S2 (5) stabilized: exact delay margin tau t'* = %.3f (crossing frequency %.3f t'*), Pade(4,4) scan agrees"
          % (tau_max * th, w_c / th), abs(tau_pade / tau_max - 1) < 5e-3, "Pade %.4f" % (tau_pade * th))
    # kick tolerance of the stabilized loop: largest energy kick (units A t'*^2) at the stationary point from which
    # the controlled boundary returns to t'*, for either sign of phi'
    sec_f = holding_stabilized(uc=-100.0)
    blow_f = lambda u, y, *a: y[0] - 4.0
    blow_f.terminal = True

    def recovers(eps, sgn):
        y0 = [math.log(th), sgn * th * math.sqrt(eps), 0.0]
        so = solve_ivp(rhs, (0, 300), y0, args=([sec_f],), method="DOP853", rtol=1e-10, atol=1e-12, events=blow_f)
        return so.t[-1] >= 300 - 1e-9 and abs(so.y[0, -1] - math.log(th)) < 1e-6
    KICK_MAX = 1e3
    up_all = all(recovers(e_, 1) for e_ in np.geomspace(1e-3, KICK_MAX, 25))
    a_, b_ = 0.0, 1.0
    while recovers(b_, -1):
        a_, b_ = b_, 2 * b_
    for _ in range(40):
        m_ = 0.5 * (a_ + b_)
        a_, b_ = (m_, b_) if recovers(m_, -1) else (a_, m_)
    out["kick_tol_f_up_all_to"] = KICK_MAX if up_all else float("nan")
    out["kick_tol_f"] = a_
    check("S2 (5) stabilized: the loop recovers from every kick that raises the rate (tested to %.0e A t'*^2) and "
          "from kicks that lower it up to %.3f A t'*^2" % (KICK_MAX, a_), up_all and 0 < a_ < 1)
    # holding alone: span beyond pi (probe window) while the pair stays unbound
    for key in ("hold_p", "hold_f"):
        R = runs[key]
        out["span_excess_" + key] = summ[key]["span"] - math.pi
        out["phi_end_" + key] = float(R["phi"][-1])
    check("S2 holding alone lengthens the span (probe window %.3f, %.3f) with the rate decaying (phi_end < -15)"
          % (out["span_excess_hold_p"], out["span_excess_hold_f"]),
          out["span_excess_hold_p"] > 0 and out["span_excess_hold_f"] > 0 and
          max(out["phi_end_hold_p"], out["phi_end_hold_f"]) < -15)
    # an early holding coupling raises t' before the opening pulse acts, and the Delta_o = 2 coupling (force ~ t'^4)
    # then runs away at a lower strength
    early = {}
    for uc in (-1.5, P["u_c"]):
        R = run([opening(eta0=0.30), holding_passive(uc=uc)], u1=45.0, du=0.01)
        early[str(uc)] = dict(runaway=R["runaway"], tp0=float(math.exp(np.interp(0.0, R["u"], R["phi"]))))
    out["early_hold"] = early
    check("S2 opening strength 0.30: holding on at u_c = -1.5 raises t'(0) to %.2f and runs away; at u_c = %.1f, "
          "t'(0) = %.2f and no runaway" % (early["-1.5"]["tp0"], P["u_c"], early[str(P["u_c"])]["tp0"]),
          early["-1.5"]["runaway"] and not early[str(P["u_c"])]["runaway"])
    # rate during and after the opening pulse alone
    Ro = runs["open"]
    out["tp_open"] = {str(x): float(math.exp(np.interp(x, Ro["u"], Ro["phi"]))) for x in (0.0, 1.0, 2.0, 4.0, 5.0)}
    out["tp_tfd"] = {str(x): 1 / math.cosh(x) for x in (1.0, 2.0, 4.0, 5.0)}
    for key in ("open_hold_p", "open_hold_f"):
        R = runs[key]
        out["tp_" + key] = {str(x): float(math.exp(np.interp(x, R["u"], R["phi"]))) for x in (1.0, 2.0, 3.0, 4.0, 5.0)}
    return out, runs


# =====================================================================================================  S3
def Dnorm(D):
    return math.exp(2 * gammaln(D) + 2 * D * math.log(2) - gammaln(2 * D)) / (4 * math.pi)


def OOeq(D):
    """<O_L(t) O_R(t)> in the global vacuum, O = lim sin(sigma)^-Delta phi."""
    return Dnorm(D) / 4 ** D


def _kern(D, s, tau, which):
    sn, cs = math.sin(s), math.cos(s)
    ch, sh = math.cosh(tau), math.sinh(tau)
    XL = 2 * (ch - cs) / sn
    XR = 2 * (ch + cs) / sn
    Dn = Dnorm(D)
    if which == "phi2":
        return Dn * XL ** (-D) * Dn * XR ** (-D)
    WL = -D * Dn * XL ** (-D - 1) * (1j * sh / sn + (1 - ch * cs) / sn ** 2)
    WR = -D * Dn * XR ** (-D - 1) * (1j * sh / sn - (1 + ch * cs) / sn ** 2)
    return WL * WR


@functools.lru_cache(maxsize=None)
def tau_moment(D, s, j):
    """tau_j(sigma) = (4/j!) Re[i^j int_0^inf tau^j F_E dtau]: first-order T_++ per unit coupling is
    sum_j g^(j)(t) tau_j(sigma) (exact for analytic g, asymptotic otherwise)."""
    up = min(600.0, 60.0 / D + 40.0)
    s0 = min(s, math.pi - s)
    f = lambda x: (1j ** j * x ** j * _kern(D, s, x, "T")).real
    pts = [0.0, min(3 * s0, up * 0.5), up]
    val = sum(quad(f, pts[i], pts[i + 1], limit=400, epsabs=0, epsrel=1e-11)[0] for i in range(2))
    return 4.0 / math.factorial(j) * val


def phi2_static(D, s):
    up = min(600.0, 60.0 / D + 40.0)
    s0 = min(s, math.pi - s)
    f = lambda x: _kern(D, s, x, "phi2")
    pts = [0.0, min(3 * s0, up * 0.5), up]
    return 4.0 * sum(quad(f, pts[i], pts[i + 1], limit=400, epsabs=0, epsrel=1e-11)[0] for i in range(2))


def filon_weights(Om, h):
    """int_0^h f(t_m + x) e^{-i Om (h - x)} dx for f linear between f_m (x = 0) and f_{m+1} (x = h):
    returns the weights of f_m and f_{m+1}."""
    z = -1j * Om * h
    small = abs(z) < 1e-6
    zs = np.where(small, 1.0, z)
    ez = np.exp(zs)
    w_old = np.where(small, 0.5 * h, h * (ez * (zs - 1) + 1) / zs ** 2)
    w_new = np.where(small, 0.5 * h, h * (ez - 1 - zs) / zs ** 2)
    return w_old, w_new


class ModeSums:
    """Global-AdS2 mode functions of a dimension-D field: a_n(sigma) = <d_+ phi O_L> mode weights and
    S_K(sigma) = sum_{n+m=K} a_n a_m (-1)^m on a set of sigma nodes."""
    def __init__(self, D, sig, nmax):
        self.D, self.sig, self.nmax = D, sig, nmax
        n = np.arange(nmax + 1)
        lam = D
        logC1 = gammaln(n + 2 * lam) - gammaln(n + 1) - gammaln(2 * lam)
        P2 = Dnorm(D) * np.exp(gammaln(2 * D + n) - gammaln(2 * D) - gammaln(n + 1))
        x = np.cos(sig)[:, None]
        sn = np.sin(sig)[:, None]
        cn = eval_gegenbauer(n[None, :], lam, x) / np.exp(logC1)[None, :]
        dC = np.zeros_like(cn)
        dC[:, 1:] = 2 * lam * eval_gegenbauer(n[None, 1:] - 1, lam + 1, x) / np.exp(logC1[1:])[None, :]
        f = sn ** D * cn
        df = D * sn ** (D - 1) * np.cos(sig)[:, None] * cn - sn ** (D + 1) * dC
        om = (D + n)[None, :]
        self.a = 0.5 * P2[None, :] * (-1j * om * f + df)
        K = np.arange(nmax + 1)
        S = np.zeros((len(sig), nmax + 1), dtype=complex)
        sgn = (-1.0) ** n
        for i in range(len(sig)):
            S[i] = np.convolve(self.a[i], self.a[i] * sgn)[: nmax + 1]
        self.S = S
        self.Om = 2 * D + K


def s3_static(out):
    for D in (0.25, 0.5, 1.0, 2.0, 3.0):
        I = quad(lambda s: 2 * math.sin(s) ** 2 * tau_moment(D, s, 0), 0, math.pi, limit=200, epsrel=1e-10)[0]
        out["ray_identity_%g" % D] = I / OOeq(D)
        check("S3 static ray identity D=%g: int 2 sin^2 tau_0 dsigma = -D <O_L O_R>" % D,
              abs(I / OOeq(D) + D) < 1e-7, "%.10f" % (I / OOeq(D)))
    # conservation and L-R symmetry (static): T_++' = (sin^2 T_+-)'/sin^2, T_+- = m^2 phi2/(4 sin^2)
    for D in (0.25, 2.0):
        m2 = D * (D - 1)
        h = 1e-3
        errs = []
        for s in (0.3, 0.8, 1.3, 2.1, 2.7):
            st = lambda f: (-f(s + 2 * h) + 8 * f(s + h) - 8 * f(s - h) + f(s - 2 * h)) / (12 * h)
            dT = st(lambda x: tau_moment(D, x, 0))
            q = lambda x: math.sin(x) ** 2 * m2 * phi2_static(D, x) / (4 * math.sin(x) ** 2)
            dq = st(q) / math.sin(s) ** 2
            errs.append(abs(dT - dq) / max(abs(dT), 1e-12))
        out["conservation_err_%g" % D] = max(errs)
        check("S3 static conservation D=%g: d T_++ = sin^-2 d(sin^2 T_+-)" % D, max(errs) < 1e-5, "%.1e" % max(errs))
        sym = max(abs(tau_moment(D, s, 0) - tau_moment(D, math.pi - s, 0)) for s in (0.2, 0.7, 1.2))
        check("S3 static L-R symmetry D=%g: T_++(sigma) = T_++(pi - sigma)" % D, sym < 1e-10)
    # mode sums (Abel) reproduce the Euclidean static profile
    for D in (0.25, 2.0):
        sig = np.array([0.6, 1.1, 1.9])
        ms = ModeSums(D, sig, 6000)
        vals = []
        for eps in (0.02, 0.01, 0.005):
            vals.append(np.real(4 * np.sum(ms.S * np.exp(-eps * np.arange(6001))[None, :] / ms.Om[None, :], axis=1)))
        rich = (8 * vals[2] - 6 * vals[1] + vals[0]) / 3
        ref = np.array([tau_moment(D, s, 0) for s in sig])
        err = np.max(abs(rich - ref) / abs(ref))
        out["modesum_static_err_%g" % D] = float(err)
        check("S3 mode sums (Abel, extrapolated) = Euclidean static profile D=%g" % D, err < 1e-3, "%.1e" % err)


def g_series(R, sec_index, nder):
    """Effective global coupling g(t) = eta e^{(2D-1) phi} / <O_L O_R> of one sector and its t-derivatives on the
    run's grid (chain rule d/dt = e^-phi d/du with quintic splines in u)."""
    s = R["secs"][sec_index]
    u, phi = R["u"], R["phi"]
    Gu = R["etas"][sec_index] * np.exp((2 * s.D - 1) * phi) / OOeq(s.D)
    spG = make_interp_spline(u, Gu, k=7)
    spP = make_interp_spline(u, phi, k=7)
    Gd = [spG.derivative(k)(u) if k else Gu for k in range(nder + 1)]
    Pd = [spP.derivative(k)(u) if k else phi for k in range(5)]
    # recursive: X_{j+1} = e^{-phi} d/du X_j, with X_j expressed through u-derivatives
    out = [Gu]
    cur = [Gd[k] for k in range(nder + 1)]          # derivatives of X_0 w.r.t. u
    for j in range(nder):
        # X_{j+1} = e^{-phi} X_j'; its u-derivatives: product rule with e^{-phi}
        e = np.exp(-phi)
        new = []
        # derivatives of e^{-phi}: d(e^{-phi}) = -phi' e^{-phi}, second: (phi'^2 - phi'') e^{-phi}, ...
        ed = [e, -Pd[1] * e, (Pd[1] ** 2 - Pd[2]) * e,
              (-Pd[1] ** 3 + 3 * Pd[1] * Pd[2] - Pd[3]) * e,
              (Pd[1] ** 4 - 6 * Pd[1] ** 2 * Pd[2] + 3 * Pd[2] ** 2 + 4 * Pd[1] * Pd[3] - Pd[4]) * e]
        for m in range(nder - j):
            val = 0.0
            for k in range(m + 1):
                val = val + math.comb(m, k) * ed[k] * cur[m - k + 1]
            new.append(val)
        cur = new
        out.append(cur[0])
    return np.array(out)


class RayStress:
    """Exact first-order null stress tensor of one sector along null rays, T = sum_{j<=J} g^(j) tau_j + remainder,
    remainder = -4 Im sum_K S_K rho_K, rho_K = (-1)^{J+1} (i Om_K)^-(J+1) int g^(J+1)(t') e^{-i Om_K (t-t')} dt'."""
    def __init__(self, R, sec_index, J, nmax, nsig=600):
        s = R["secs"][sec_index]
        self.D, self.J = s.D, J
        x, wq = np.polynomial.legendre.leggauss(nsig)
        self.sig = 0.5 * math.pi * (x + 1)
        self.wsig = 0.5 * math.pi * wq
        self.tau = np.array([[tau_moment(s.D, sg, j) for sg in self.sig] for j in range(J + 1)])
        self.ms = ModeSums(s.D, self.sig, nmax)
        gd = g_series(R, sec_index, J + 1)
        t = R["t"]
        # uniform global-time grid for the history integrals
        self.tg = np.arange(t[0], t[-1], getattr(RayStress, "H_T", 0.002))
        self.gd = np.array([np.interp(self.tg, t, gd[j]) for j in range(J + 2)])
        Om = self.ms.Om
        h = self.tg[1] - self.tg[0]
        f = self.gd[J + 1]
        # Filon-trapezoid recursion for I_K(t) = int_{-inf}^t f(t') e^{-i Om (t - t')} dt'
        z = -1j * Om * h
        ez = np.exp(z)
        w_old, w_new = filon_weights(Om, h)
        self.I = np.zeros((len(self.tg), len(Om)), dtype=complex)
        cur = np.zeros(len(Om), dtype=complex)
        for m in range(len(self.tg) - 1):
            cur = ez * cur + w_old * f[m] + w_new * f[m + 1]
            self.I[m + 1] = cur
        self.pref = (-1.0) ** (J + 1) / (1j * Om) ** (J + 1)

    def T(self, t_points, i_sig):
        """T_++ per field (unit normalization absorbed in g) at sigma node i_sig and global times t_points."""
        tp = np.atleast_1d(t_points)
        val = np.zeros(len(tp))
        for j in range(self.J + 1):
            val += np.interp(tp, self.tg, self.gd[j]) * self.tau[j, i_sig]
        # remainder: interpolate I_K with exact phase transport from the grid point below
        m = np.clip(np.searchsorted(self.tg, tp) - 1, 0, len(self.tg) - 2)
        dt = tp - self.tg[m]
        Om = self.ms.Om
        fJ = self.gd[self.J + 1]
        for q in range(len(tp)):
            ph = np.exp(-1j * Om * dt[q])
            wo, wn = filon_weights(Om, dt[q])
            Iq = ph * self.I[m[q]] + wo * fJ[m[q]] + wn * np.interp(tp[q], self.tg, fJ)
            val[q] += -4 * np.imag(np.sum(self.ms.S[i_sig] * self.pref * Iq))
        return val

    def profile(self, t_in):
        """T_++ at every sigma node of the ray x^- = t_in (node i at global time t_in + sigma_i)."""
        tp = t_in + self.sig
        val = np.zeros(len(tp))
        for j in range(self.J + 1):
            val += np.interp(tp, self.tg, self.gd[j]) * self.tau[j]
        m = np.clip(np.searchsorted(self.tg, tp) - 1, 0, len(self.tg) - 2)
        dt = tp - self.tg[m]
        Om = self.ms.Om
        fJ = self.gd[self.J + 1]
        ph = np.exp(-1j * np.outer(dt, Om))
        wo, wn = filon_weights(Om[None, :], dt[:, None])
        Iq = ph * self.I[m] + wo * fJ[m][:, None] + wn * np.interp(tp, self.tg, fJ)[:, None]
        val += -4 * np.imag(np.sum(self.ms.S * self.pref[None, :] * Iq, axis=1))
        return val

    def anec(self, t_in):
        """int_ray sin^2(sigma) T_++ dx^+ along x^- = t_in (x^+ = t_in + 2 sigma)."""
        return float(np.sum(self.wsig * 2 * np.sin(self.sig) ** 2 * self.profile(t_in)))


NREM = 150          # modes kept in the remainder (terms fall as K^(Delta - J - 2)); J = 3 for Delta = 2 and
                    # J = 1 for Delta = 1/4 (higher J is ill-conditioned where t' is small and g varies fast in t)


def s3_flight_only(out):
    """The ray integral of every two-particle mode vanishes: the null energy along a ray depends only on the
    couplings acting while the ray is in flight (the history before launch drops out)."""
    worst = 0.0
    for D in (0.25, 2.0):
        x, w = np.polynomial.legendre.leggauss(3000)
        sig = 0.5 * math.pi * (x + 1)
        ws = 0.5 * math.pi * w
        ms = ModeSums(D, sig, 200)
        for K in range(0, 201, 10):
            J = abs(np.sum(ws * 2 * np.sin(sig) ** 2 * ms.S[:, K] * np.exp(-1j * ms.Om[K] * sig)))
            Jabs = np.sum(ws * 2 * np.sin(sig) ** 2 * abs(ms.S[:, K]))
            worst = max(worst, J / Jabs)
    out["flight_only_worst"] = worst
    check("S3 ray integral of every two-particle mode vanishes (null energy on a ray depends only on in-flight "
          "couplings)", worst < 1e-9, "max relative %.1e (K <= 200, D = 1/4, 2)" % worst)


def ray_row(R, rs, t_in):
    """(t_in, u_in, u_out, opening-sector ray integral, holding-sector ray integral, -(t'_in + t'_out)/2)."""
    t = R["t"]
    u_in = float(np.interp(t_in, t, R["u"]))
    u_out = float(np.interp(t_in + math.pi, t, R["u"]))
    rhs_id = -0.5 * (math.exp(np.interp(u_in, R["u"], R["phi"])) + math.exp(np.interp(u_out, R["u"], R["phi"])))
    return (float(t_in), u_in, u_out, rs[0].anec(t_in), rs[1].anec(t_in), rhs_id)


FIG_RAYS = (-2.5, 16.0, 90)       # launch window (u) and number of rays of the support figure


def s3_dynamic(out, runs):
    res, fig, objs = {}, {}, {}
    for key, J_h in (("open_hold_f", 3), ("open_hold_p", 1)):
        R = runs[key]
        rs = [RayStress(R, 0, 3, NREM), RayStress(R, 1, J_h, NREM)]
        objs[key] = (R, rs)
        t = R["t"]
        # 26 rays spanning the run and the 90 rays of the support figure
        rows = np.array([ray_row(R, rs, x) for x in np.linspace(-1.2, t[-1] - math.pi - 0.2, 26)])
        u_hi = min(FIG_RAYS[1], float(np.interp(t[-1] - math.pi - 0.1, t, R["u"])))
        rows_f = np.array([ray_row(R, rs, float(np.interp(x, R["u"], t)))
                           for x in np.linspace(FIG_RAYS[0], u_hi, FIG_RAYS[2])])
        err = np.max(abs(rows[:, 3] + rows[:, 4] - rows[:, 5]) / abs(rows[:, 5]))
        err_f = np.max(abs(rows_f[:, 3] + rows_f[:, 4] - rows_f[:, 5]) / abs(rows_f[:, 5]))
        res[key], fig[key] = rows, rows_f
        out["ray_identity_err_" + key] = float(max(err, err_f))
        out["ray_identity_err_fig_" + key] = float(err_f)
        out["ray_last_u_in_" + key] = float(rows[-1, 1])
        check("S3 dynamic ray identity through the transition (%s): sum of sector null energies = "
              "-(t'_in + t'_out)/2" % key, max(err, err_f) < 2e-4,
              "max rel %.1e over the %d figure rays, %.1e over %d rays to u = %.0f"
              % (err_f, len(rows_f), err, len(rows), rows[-1, 1]))
        share = rows[:, 4] / (rows[:, 3] + rows[:, 4])
        out["anec_share_hold_" + key] = share.tolist()
        out["anec_tin_" + key] = rows[:, 0].tolist()
        # ray launched at the start of the opening ramp-down
        u_rd = P["b_o"] - 2 * P["w"]
        t_rd = float(np.interp(u_rd, R["u"], t))
        a_rd = [r.anec(t_rd) for r in rs]
        out["ray_share_hold_rampdown_" + key] = a_rd[1] / (a_rd[0] + a_rd[1])
        out["ray_total_rampdown_" + key] = a_rd[0] + a_rd[1]
        # opening sector: largest ray integral and its value per unit of the sector's largest per-channel coupling
        allrows = np.vstack([rows, rows_f])
        a_o = float(np.max(abs(allrows[:, 3])))
        out["anec_open_max_" + key] = a_o
        out["anec_open_per_g_" + key] = a_o / float(np.max(abs(rs[0].gd[0])))
    return res, fig, objs


# --------------------------------------------------------------------------- null QEI with optimal samplers
QEI_M = 2400            # uniform sigma grid (intervals on (0, pi)) of the sampler eigenproblem


def _posdef(T, dx, g):
    """-d^2/dsigma^2 - 1 + 48 pi g T positive definite on the interior nodes (three-point Laplacian, Dirichlet ends)."""
    d = 2.0 / dx ** 2 - 1.0 + 48 * math.pi * g * T
    return dpttrf(d, np.full(len(T) - 1, -1.0 / dx ** 2))[2] == 0


def gstar(T, dx, rtol=1e-11):
    """Largest scale g of the profile T (per unit coupling, interior nodes of a segment) for which the null QEI
    int T h^2 dx^+ >= -B(h) holds for every sampler h vanishing at the ends. With x^+ = t_in + 2 sigma,
    B(h) = (1/24 pi) int (h_sigma^2 - h^2) dsigma, so the QEI holds for all h iff -d^2 - 1 + 48 pi g T >= 0;
    the lowest eigenfunction at the limit is the optimal sampler. The lowest eigenvalue is concave in g, so the
    admissible g >= 0 form an interval and bisection on positive definiteness finds its end."""
    if not _posdef(T, dx, 0.0):
        return 0.0
    lo, hi = 0.0, 1.0
    while _posdef(T, dx, hi):
        lo, hi = hi, 2.0 * hi
        if hi > 1e8:
            return float("inf")
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _posdef(T, dx, mid) else (lo, mid)
    return 0.5 * (lo + hi)


def gstar_segments(Tg, M, families=(1, 2), stride=None):
    """Smallest gstar over the maximal qualifying segments of a profile Tg on the uniform grid i pi/M.
    Family 1: (a, b) with 2b - a < pi (past cones from the slice at the segment's start), maximal b for each a;
    family 2: 2a - b > 0 (future cones from the slice at its end), minimal a for each b. A segment inside a larger
    qualifying one admits fewer samplers, so the maximal segments give the limit. Coarse search, then refinement."""
    h = math.pi / M
    stride = max(1, M // 100) if stride is None else stride
    best = (float("inf"), None)

    def seg(fam, i):
        return (i, (M + i - 1) // 2) if fam == 1 else ((i // 2) + 1, i)

    def f(fam, i):
        ia, ib = seg(fam, i)
        return gstar(Tg[ia + 1:ib], h) if ib - ia > 8 else float("inf")

    for fam in families:
        idx = list(range(1, M - 3)) if fam == 1 else list(range(4, M))
        coarse = sorted(set(idx[::stride]) | {idx[-1]})
        vals = [f(fam, i) for i in coarse]
        k = int(np.argmin(vals))
        for i in range(coarse[max(k - 1, 0)], coarse[min(k + 1, len(coarse) - 1)] + 1):
            v = vals[k] if i == coarse[k] else f(fam, i)
            if v < best[0]:
                best = (v, (fam,) + tuple(x * h for x in seg(fam, i)))
    return best


@functools.lru_cache(maxsize=None)
def static_profile_grid(D, M):
    """tau_0 on the uniform grid i pi/M (symmetric under sigma -> pi - sigma)."""
    half = np.array([tau_moment(D, i * math.pi / M, 0) for i in range(1, M // 2 + 1)])
    T = np.full(M + 1, np.nan)
    T[1:M // 2 + 1] = half
    T[M // 2 + 1:M] = half[:-1][::-1]
    return T


def gstar_extrapolated(profile_on, M):
    """Limit of gstar_segments on grids M and 2M: the segment ends sit within one grid step of the edges of the
    qualifying region, an error linear in pi/M, so g = 2 g(2M) - g(M)."""
    r1 = gstar_segments(*profile_on(M))
    r2 = gstar_segments(*profile_on(2 * M))
    return 2 * r2[0] - r1[0], abs(r2[0] - r1[0]), r2[1]


KAPPA_DS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


NULL_DS = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0)        # dimensions tested on the null line
NULL_LINES = ((1.2, 0.4, 0.0), (2.6, 1.1, 0.7))  # (x^+, x^+', x^-) of the two pairs tested


def null_line_limit(D, xp1, xp2, xm1, nb=9, nterm=6, dps=90):
    """lim_{B -> 0} d_+ d_+' <d_+ chi(x) d_+ chi(x')> for a channel of dimension D, at points of the null line
    x^- = xm1 separated by A = xp1 - xp2, from the global AdS2 two-point function Q_{D-1}(zeta)/(2 pi) with
    zeta - 1 = -2 sin(A/2) sin(B/2)/(sin sigma sin sigma').  The limit is taken by fitting the b^k and b^k log b
    terms that the Hadamard series produces, from a geometric sequence of spacelike separations B = -b."""
    import mpmath as mp
    with mp.workdps(dps):
        D, xp1, xp2, xm1 = (mp.mpf(repr(v)) for v in (D, xp1, xp2, xm1))

        def deriv(b):
            xm2 = xm1 + b

            def f(u, v):
                z = 1 - 2 * mp.sin((u - v) / 2) * mp.sin((xm1 - xm2) / 2) / (mp.sin((u - xm1) / 2)
                                                                             * mp.sin((v - xm2) / 2))
                return mp.legenq(D - 1, 0, z, type=3) / (2 * mp.pi)
            return mp.re(mp.diff(f, (xp1, xp2), (1, 1)))

        bs = [mp.mpf(2) ** (-k) / 64 for k in range(nb)]
        y = mp.matrix([deriv(b) for b in bs])
        M = mp.matrix(nb, nb)
        for i, b in enumerate(bs):
            basis = [mp.mpf(1)]
            for k in range(1, nterm):
                basis += [b ** k, b ** k * mp.log(b)]
            for j in range(nb):
                M[i, j] = basis[j]
        return float(mp.lu_solve(M, y)[0])


def null_boundary_ratio(D, t=0.3, sig=1.1, sigp=1e-7, dps=40):
    """Q_{D-1}(zeta)(2 zeta)^D/(2 pi D_Delta) as one point approaches a boundary: 1 if the invariant two-point
    function used on the null line is the one whose boundary limit is the bulk-to-boundary function D_Delta
    [2(cos(t - t_1) -+ cos sigma)/sin sigma]^-D of the first-order stress tensor."""
    import mpmath as mp
    with mp.workdps(dps):
        D, t, sig, sigp = (mp.mpf(repr(v)) for v in (D, t, sig, sigp))
        zeta = (mp.cos(t) - mp.cos(sig) * mp.cos(sigp)) / (mp.sin(sig) * mp.sin(sigp))
        W = mp.re(mp.legenq(D - 1, 0, zeta, type=3) / (2 * mp.pi))
        DD = 4 ** D * mp.gamma(D) ** 2 / (4 * mp.pi * mp.gamma(2 * D))
        return float(W * (2 * zeta) ** D / DD)


def s3_nullline(out):
    """The null QEI of Eq. (qei) is a conformal-field-theory bound; it applies to a channel of any dimension
    because the two-point function of d_+ chi restricted to a null line carries no dimension dependence."""
    bworst = max(abs(null_boundary_ratio(D) - 1) for D in NULL_DS)
    out["nullline_boundary_err"] = bworst
    check("S3 the invariant two-point function Q_{Delta-1}(zeta)/(2 pi) has the bulk-to-boundary limit "
          "D_Delta (2 zeta)^-Delta used in the first-order stress tensor", bworst < 1e-7,
          "worst relative deviation %.1e over Delta = %s" % (bworst, ", ".join("%g" % D for D in NULL_DS)))
    worst = 0.0
    for xp1, xp2, xm1 in NULL_LINES:
        massless = -1.0 / (16 * math.pi * math.sin((xp1 - xp2) / 2) ** 2)
        for D in NULL_DS:
            lim = null_line_limit(D, xp1, xp2, xm1)
            worst = max(worst, abs(lim / massless - 1))
    out["nullline_worst"] = worst
    out["nullline_Dmax"] = max(NULL_DS)
    check("S3 null-line two-point function of d_+ chi is the massless one for every channel dimension "
          "(so the conformal null QEI applies at any Delta)", worst < 1e-7,
          "worst relative deviation %.1e over Delta = %s on %d null lines"
          % (worst, ", ".join("%g" % D for D in NULL_DS), len(NULL_LINES)))

QEI_RAYS = 15


def s3_qnei(out, dyn_objects):
    # static first-order profiles (symmetric, so family 1 suffices)
    for D in KAPPA_DS:
        g, dg, segm = gstar_extrapolated(lambda M, D=D: (static_profile_grid(D, M), M, (1,)), QEI_M)
        out["g_qnei_%g" % D] = g
        out["g_qnei_step_%g" % D] = dg
        out["g_qnei_seg_%g" % D] = segm[1:]
        out["g_cap_%g" % D] = D / (4 * Dnorm(D))
    check("S3 null QEI with optimal samplers, static first-order profiles: per-channel limits %s at Delta = 1/4, 1/2, "
          "1, 2" % ", ".join("%.3f" % out["g_qnei_%g" % D] for D in (0.25, 0.5, 1.0, 2.0)),
          all(0 < out["g_qnei_%g" % D] < 10 and out["g_qnei_step_%g" % D] < 5e-3 * out["g_qnei_%g" % D]
              for D in (0.25, 0.5, 1.0, 2.0)),
          "grid change %s" % ", ".join("%.1e" % out["g_qnei_step_%g" % D] for D in (0.25, 0.5, 1.0, 2.0)))
    # Delta = 1: tau_0 = -1/(4 pi^2), so the limit on a segment of length l is (pi/12)((pi/l)^2 - 1), pi/4 on the
    # half strip (0, pi/2)
    T1 = static_profile_grid(1.0, QEI_M)
    flat = np.nanmax(abs(T1[1:-1] + 1 / (4 * math.pi ** 2)))
    check("S3 Delta = 1: tau_0 = -1/(4 pi^2) and the QEI limit is pi/4 (optimal segment (0, pi/2))",
          flat < 1e-9 and abs(out["g_qnei_1"] / (math.pi / 4) - 1) < 2e-4,
          "%.6f vs %.6f" % (out["g_qnei_1"], math.pi / 4))
    # conformal sector, stationary state: half-circle slice bound with h = sqrt(2) sin(sigma)
    xs = np.linspace(0, math.pi, 20001)
    hh = math.sqrt(2) * np.sin(xs)
    dh = math.sqrt(2) * np.cos(xs)
    L = 2 * math.pi
    bound = -(1 / (12 * math.pi)) * (np.trapezoid(dh ** 2, xs) - (math.pi / L) ** 2 * np.trapezoid(hh ** 2, xs))
    out["conformal_stationary_bound"] = bound
    check("S3 stationary conformal sector: support per unit central charge >= -1/16 (null QEI on half the "
          "doubled circle), equal to the MMP value at zero mouth separation", abs(bound + 1 / 16) < 1e-8,
          "%.8f" % bound)
    # the transition: per-sector first-order profile along rays, both segment families, per unit of the sector's
    # largest per-channel coupling; profiles interpolated in sin^2(sigma) T (regular at the ends)
    Md = QEI_M // 2
    for key, objs in dyn_objects.items():
        R, rs = objs
        t = R["t"]
        for k, r_s in enumerate(rs):
            gmax = float(np.max(abs(r_s.gd[0])))
            worst, worst_step = float("inf"), 0.0
            for tin in np.linspace(-1.2, min(t[-1] - math.pi - 0.2, 12.0), QEI_RAYS):
                w = np.sin(r_s.sig) ** 2 * r_s.profile(tin)

                def on(M, w=w):
                    grid = np.arange(M + 1) * math.pi / M
                    Tg = np.full(M + 1, np.nan)
                    Tg[1:M] = np.interp(grid[1:M], r_s.sig, w) / np.sin(grid[1:M]) ** 2
                    return Tg, M
                g, dg, _ = gstar_extrapolated(on, Md)
                if g * gmax < worst:
                    worst, worst_step = g * gmax, dg * gmax
            out["g_qnei_dyn_%s_%d" % (key, k)] = worst
            out["g_qnei_dyn_step_%s_%d" % (key, k)] = worst_step
            check("S3 null QEI along rays through the transition (%s, %s sector, %d rays, optimal samplers): holds "
                  "for per-channel couplings below %.3f" % (key, ["opening", "holding"][k], QEI_RAYS, worst),
                  0 < worst < float("inf") and worst_step < 1e-2 * worst, "grid change %.1e" % worst_step)


# =====================================================================================================  S4
def resonant_intermediary(lam=0.15, wm=1.0, w0s=None):
    """Two throat modes (frequency wm) coupled through a resonator (frequency w0, the hydrogen line):
    H = sum p^2/2 + wm^2 (x_L^2 + x_R^2)/2 + w0^2 X^2/2 + lam (x_L + x_R) X. Ground-state <x_L x_R> and energy."""
    w0s = np.linspace(0.6, 1.4, 801) if w0s is None else w0s
    corr, E = [], []
    for w0 in w0s:
        Kmat = np.array([[wm ** 2, 0, lam], [0, wm ** 2, lam], [lam, lam, w0 ** 2]])
        ev, U = np.linalg.eigh(Kmat)
        om = np.sqrt(ev)
        cov = U @ np.diag(1 / (2 * om)) @ U.T          # <x x^T> in the ground state
        corr.append(cov[0, 1])
        E.append(0.5 * om.sum() - 0.5 * (2 * wm + w0))
    return w0s, np.array(corr), np.array(E)


def replenished(mu, w=1.0):
    """Two modes with coupling -mu x_L x_R: minimum energy over injected two-mode squeezed states versus the
    exact coupled ground state (both relative to the uncoupled vacuum)."""
    f = lambda r: 2 * w * math.sinh(r) ** 2 - mu * math.sinh(r) * math.cosh(r) / w
    rmin = minimize_scalar(f, bounds=(0, 5), method="bounded", options=dict(xatol=1e-12))
    exact = 0.5 * (math.sqrt(w * w - mu) + math.sqrt(w * w + mu)) - w
    return rmin.fun, exact, rmin.x


def s4(out):
    w0s, corr, E = resonant_intermediary()
    i0 = np.argmin(abs(w0s - 1.0))
    smooth = 0.5 * (corr[i0 - 1] + corr[i0 + 1])
    curv = np.max(abs(np.diff(corr, 2))) / (w0s[1] - w0s[0]) ** 2
    out["res_corr_at_res"] = float(corr[i0])
    out["res_corr_curv"] = float(curv)
    # second-order (counter-rotating) estimate of <x_L x_R>: lam^2 (wm + 2 w0) ... compare sign and size
    check("S4 resonant intermediary: ground-state <x_L x_R> smooth across w0 = wm (no resonant enhancement)",
          abs(corr[i0] - smooth) < 1e-6 and curv < 1.0, "corr %.4e, max curvature %.2e" % (corr[i0], curv))
    out["res_corr_range"] = (float(corr.min()), float(corr.max()))
    rows = []
    for mu in (0.0, 0.1, 0.4, 0.8):
        tm, ex, r = replenished(mu)
        rows.append((mu, tm, ex, r))
    out["replenished"] = rows
    check("S4 replenished correlations: no coupling -> minimum at r = 0, energy 0", abs(rows[0][1]) < 1e-12 and rows[0][3] < 1e-5)
    check("S4 replenished correlations: injected-state energy >= coupled ground state",
          all(r[1] >= r[2] - 1e-12 for r in rows))
    # hydrogen as matter in the throat
    for r_e in (1e-19, 1e-3, 1.0, 1.5e13):
        out["Delta_H_%g" % r_e] = 0.5 + math.sqrt(0.25 + (M_H * C * r_e / HBAR) ** 2)
    check("S4 hydrogen matter: Delta = m_H c r_e/hbar >> 1 for r_e >= 1e-15 m",
          0.5 + math.sqrt(0.25 + (M_H * C * 1e-15 / HBAR) ** 2) > 4)


# =====================================================================================================  S5
def kappa_cap(D):
    """Single-mode stability estimate of the binding per channel: Delta g <O_L O_R> at g = Delta/(4 D_norm), where the
    lowest mode of the strip (weight D_norm in both O_L and O_R) turns unstable."""
    return D * D / (4 * 4 ** D)


def C_hbar(r_e):
    """Schwarzian coefficient times hbar (s): C hbar = r_e^3 / (l_P^2 c)."""
    return r_e ** 3 / (LP ** 2 * C)


def s5(out, o3, o2):
    # binding per channel: first-order value within the single-mode stability estimate and the null QEI
    kap = {D: D * OOeq(D) * min(o3["g_cap_%g" % D], o3["g_qnei_%g" % D]) for D in KAPPA_DS}
    check("S5 stability estimate of the binding per channel: Delta g_cap <O_L O_R> = Delta^2/(4 4^Delta)",
          all(abs(D * OOeq(D) * o3["g_cap_%g" % D] / kappa_cap(D) - 1) < 1e-12 for D in KAPPA_DS))
    kmax_D = max(kap, key=kap.get)
    kmax = kap[kmax_D]
    out["kappa_channel"] = {str(k): v for k, v in kap.items()}
    out["kappa_qnei"] = {str(D): D * OOeq(D) * o3["g_qnei_%g" % D] for D in KAPPA_DS}
    out["kappa_binding_limit"] = {str(D): ("stability" if o3["g_cap_%g" % D] <= o3["g_qnei_%g" % D] else "QEI")
                                  for D in KAPPA_DS}
    out["kappa_max"] = kmax
    out["kappa_max_Delta"] = kmax_D
    check("S5 binding per channel over Delta = 1/4..6 (first order, stability estimate and null QEI): largest at "
          "Delta = 1, kappa_max = 1/16, the conformal bound per unit central charge",
          kmax_D == 1.0 and abs(16 * kmax - 1) < 1e-5 and abs(o3["conformal_stationary_bound"] + 1 / 16) < 1e-8,
          ", ".join("%g: %.4f" % (D, kap[D]) for D in KAPPA_DS))
    check("S5 the stability estimate binds for Delta = 3..6 (it falls as Delta^2 4^-Delta), so it bounds the higher "
          "multipoles", all(o3["g_cap_%g" % D] < o3["g_qnei_%g" % D] for D in (3.0, 4.0, 5.0, 6.0)))
    # Einstein-Maxwell multipoles of the throat: both parities and 2l + 1 orientations per l; gravitational-like
    # family Delta = l (l >= 2), electromagnetic-like family Delta = l + 2 (l >= 1)
    kof = lambda D: kap[D] if D in kap else kappa_cap(D)
    K_g = sum(2 * (2 * l + 1) * kof(float(l)) for l in range(2, 80))
    K_e = sum(2 * (2 * l + 1) * kof(float(l + 2)) for l in range(1, 80))
    out["K_EM_grav"], out["K_EM_em"], out["K_EM"] = K_g, K_e, K_g + K_e
    out["K_EM_channels_equiv"] = (K_g + K_e) / kmax
    out["r_e_cross_tU_EM"] = ((K_g + K_e) * LP ** 2 * C * T_UNIVERSE / math.pi) ** (1 / 3)
    out["B_cross_tU_EM"] = BR / out["r_e_cross_tU_EM"]
    check("S5 Einstein-Maxwell multipoles of the throat: summed binding %.2f (gravitational-like %.2f, "
          "electromagnetic-like %.2f; %.0f channels at kappa_max); crossing within the age of the universe needs "
          "r_e <= %.1e m" % (K_g + K_e, K_g, K_e, out["K_EM_channels_equiv"], out["r_e_cross_tU_EM"]),
          1 < K_g + K_e < 4)
    # channels needed: theta = pi / T_cross,  N >= C hbar theta / kappa
    out["N_needed_1cm_1s"] = C_hbar(0.01) * math.pi / kmax
    out["N_needed_1cm_tU"] = C_hbar(0.01) * math.pi / (kmax * T_UNIVERSE)
    for N, tag in ((1e4, "1e4"), (1e8, "1e8")):
        r = (kmax * N * LP ** 2 * C * T_UNIVERSE / math.pi) ** (1 / 3)
        out["r_e_cross_tU_N" + tag] = r
        out["B_cross_tU_N" + tag] = BR / r
    out["B_EW"] = M_W ** 2 * C ** 2 / (QE * HBAR)
    out["r_e_EW"] = BR / out["B_EW"]
    check("S5 a throat held by 1e4 channels and crossed within the age of the universe has r_e = %.1e m, horizon "
          "field %.1e T, above the electroweak field %.1e T" % (out["r_e_cross_tU_N1e4"], out["B_cross_tU_N1e4"],
                                                               out["B_EW"]),
          out["B_cross_tU_N1e4"] > out["B_EW"])
    out["Chbar_1cm"] = C_hbar(0.01)
    out["Chbar_EW"] = C_hbar(out["r_e_EW"])
    # field-exclusion radii for devices, B(r) = BR r_e / r^2: superconducting mercury (B_c) and the hydrogen maser
    # (well below the Breit-Rabi decoupling field)
    for r_e in (1e-19, 0.01, 1.0):
        out["r_dev_Hg_%g" % r_e] = math.sqrt(BR * r_e / BC_HG)
        out["r_dev_maser_%g" % r_e] = math.sqrt(BR * r_e / B_BR)
    out["B_BR"] = B_BR
    check("S5 superconducting mercury needs r >= 9.2e9 sqrt(r_e/m) m from an extremal magnetic throat",
          abs(out["r_dev_Hg_1"] / 9.19e9 - 1) < 2e-3, "%.3e" % out["r_dev_Hg_1"])
    check("S5 hydrogen maser: Breit-Rabi field %.4f T (h nu_HFS / ((g_J - g_I) mu_B)); B < B_BR needs r >= %.2e "
          "sqrt(r_e/m) m" % (B_BR, out["r_dev_maser_1"]), abs(B_BR / 0.0506 - 1) < 2e-3)
    # light time to the devices against the delay margin of the stabilized holding loop
    out["light_round_trip_Hg_1cm"] = 2 * out["r_dev_Hg_0.01"] / C
    out["light_round_trip_maser_1cm"] = 2 * out["r_dev_maser_0.01"] / C
    out["loop_delay_1s_crossing"] = o2["max_loop_delay_theta"] / math.pi
    out["T_cross_min_maser_1cm"] = math.pi * out["light_round_trip_maser_1cm"] / o2["max_loop_delay_theta"]
    check("S5 stabilization from the maser radius of a 1 cm throat: round-trip light time %.2f s against a delay margin "
          "of %.3f s for a 1 s crossing (crossing times above %.0f s fit)" % (out["light_round_trip_maser_1cm"],
                                                                             out["loop_delay_1s_crossing"],
                                                                             out["T_cross_min_maser_1cm"]),
          out["light_round_trip_maser_1cm"] > out["loop_delay_1s_crossing"])
    # MMP: c = 54 q fermions, q = 2 sqrt(alpha) r_e / l_P; conformal bound c/16 per unit central charge
    q_per_m = 2 * math.sqrt(ALPHA) / LP
    out["q_per_m"] = q_per_m
    out["q_1cm"] = q_per_m * 0.01
    out["N_MMP_1cm"] = 54 * q_per_m * 0.01
    # finite quanta in transit per holding channel: passive (ground state, Eq. nq) and stabilized (kick tolerance:
    # N_q hbar Delta t'* < eps_max A t'*^2 with A t'* = hbar N kappa)
    Dp = P["D_p"]
    out["quanta_per_channel_hold_p"] = (1 - Dp) * kap[0.25] / Dp ** 2
    out["quanta_per_channel_hold_f"] = o2["kick_tol_f"] * kap[2.0] / P["D_f"]
    # opening: support per channel along the transition = ray integral per unit coupling x dynamic QEI limit
    per = {k: o3["anec_open_per_g_" + k] * o3["g_qnei_dyn_%s_0" % k] for k in ("open_hold_p", "open_hold_f")}
    key_o = min(per, key=per.get)
    kap_o, a_o = per[key_o], o3["anec_open_max_" + key_o]
    dE = 1.0 - o2["open_E_after"]                          # energy the opening pulse extracts (units C hbar^2 k^2)
    out["kappa_open_dyn"] = kap_o
    out["kappa_open_dyn_run"] = key_o
    out["anec_open_max"] = a_o
    out["anec_open_per_g"] = o3["anec_open_per_g_" + key_o]
    out["open_extracted"] = dE
    out["quanta_per_open_channel"] = dE * kap_o / (P["D_o"] * a_o)
    out["N_open_one_quantum"] = math.ceil(1 / out["quanta_per_open_channel"])
    out["quanta_per_open_channel_static"] = dE * kap[2.0] / (P["D_o"] * a_o)
    out["N_open_one_quantum_static"] = math.ceil(1 / out["quanta_per_open_channel_static"])
    out["k_max_open_1cm_N1e4"] = 1e4 * kap_o / (a_o * C_hbar(0.01))
    check("S5 opening: about %.4f quanta per opening channel fit in the extracted energy (support per channel %.4f "
          "along the transition; %.4f with the static kappa(2))" % (out["quanta_per_open_channel"], kap_o,
                                                                    out["quanta_per_open_channel_static"]),
          out["quanta_per_open_channel"] < out["quanta_per_open_channel_static"] < 0.1)
    return out


# =====================================================================================================
def results():
    CHECKS.clear()
    res = {}
    res["S1"] = s1()
    o2, runs = s2()
    res["S2"] = o2
    o3 = {}
    s3_static(o3)
    s3_flight_only(o3)
    dyn, dyn_fig, objs = s3_dynamic(o3, runs)
    s3_qnei(o3, objs)
    s3_nullline(o3)
    o3["rays"] = {k: v.tolist() for k, v in dyn.items()}
    o3["rays_fig"] = {k: v.tolist() for k, v in dyn_fig.items()}
    res["S3"] = o3
    o4 = {}
    s4(o4)
    res["S4"] = o4
    res["S5"] = s5({}, o3, o2)
    res["P"] = P
    return res, list(CHECKS), runs, objs


if __name__ == "__main__":
    res, chk, runs, objs = results()
    nf = 0
    for n, ok, d in chk:
        print(("PASS " if ok else "FAIL ") + n + (" | " + d if d else ""))
        nf += (not ok)
    print("%d checks, %d failures" % (len(chk), nf))
    res["checks"] = {"total": len(chk), "failures": nf}
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    with open(os.path.join(HERE, "results_transfer.json"), "w") as f:
        json.dump(res, f, indent=1, default=conv)
