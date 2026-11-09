"""How much torque can GEM's default PMSM make at a given speed, with i_d = 0 vs with field weakening?
Steady-state voltage ellipse (L_q i_q)^2 + (L_d i_d + psi)^2 <= (U / w_el)^2 and current circle |i| <= I_max."""
import numpy as np
Ld, Lq, psi, p, U, Imax = 0.37e-3, 1.2e-3, 0.066, 3, 300 / np.sqrt(3), 400.0
def iq_max(w_m, id_=0.0):
    w = p * w_m; room = (U / w) ** 2 - (Ld * id_ + psi) ** 2
    return np.sqrt(max(room, 0)) / Lq
print("voltage ceiling U =", round(U, 1), "V; characteristic current psi/Ld =", round(psi / Ld, 1), "A")
for wm in [100, 195, 314, 418]:
    best = 0
    for idv in np.linspace(0, -300, 601):
        iq = min(iq_max(wm, idv), np.sqrt(max(Imax ** 2 - idv ** 2, 0)))
        best = max(best, 1.5 * p * (psi + (Ld - Lq) * idv) * iq)
    T0 = 1.5 * p * psi * min(iq_max(wm, 0), Imax)
    print(f"omega_m={wm:3d} rad/s ({wm*9.549:5.0f} rpm): max torque i_d=0: {T0:6.1f} N.m | with field weakening: {best:6.1f} N.m")
print("i_d=0 base speed at 240 A:", round(U / np.hypot(Lq * 240, psi) / p), "rad/s")
