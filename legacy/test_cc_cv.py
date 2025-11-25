import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def obtain_Iave_CCCV(Icc, I_cut, soc0, R, Q_nom, Vcv, dt=0.1):

    I_aveCC, t_endCC = obtain_IaveCC(Icc, soc0, R, Q_nom, Vcv, dt)
    I_aveCV, t_endCV = obtain_IaveCV(Icc, R, Q_nom, I_cut)

    return (I_aveCC * t_endCC + I_aveCV * t_endCV) / (t_endCC + t_endCV), t_endCC + t_endCV

def obtain_IaveCC(Icc, soc0, R, Q_nom, Vcv, dt=0.1):

    I_ave = Icc

    if np.sign(Icc) == 1:
        soc_array = np.arange(soc0, 1, step=dt * Icc / Q_nom)
    else:
        soc_array = np.arange(soc0, 0, step=dt * Icc / Q_nom)
    V_ocv = OCV(soc_array)
    V_tar = Vcv - Icc * R

    if np.sign(Icc) == 1:
        idx_cross = np.where(V_ocv > V_tar)[0][0]
    else:
        idx_cross = np.where(V_ocv < V_tar)[0][0]

    if idx_cross == 0:
        t_end = dt
    else:

        t0 = (idx_cross - 1) * dt
        # Interpolate to get t_end
        t_end = t0  + dt / (V_ocv[idx_cross] - V_ocv[idx_cross - 1]) * (V_tar - V_ocv[idx_cross - 1])

    return I_ave, t_end


def obtain_IaveCV(Icc, R, Q_nom, I_cut):

    t_end = -np.log(I_cut / Icc) * (R * Q_nom) / 1.2

    I_ave = I_cc / t_end * (R * Q_nom) / 1.2 * (1 - np.exp(-1.2 * t_end / (R * Q_nom)))
    return I_ave, t_end


def simulate_cc_cv(t, soc0, R, Q_nom, I_cc, I_cut, V_cv, OCV_func):

    dt = t[1] - t[0]
    N = len(t)

    soc = np.zeros(N)
    I = np.zeros(N)
    V = np.zeros(N)

    soc[0] = soc0
    cv_mode = False

    I_cv = []

    for k in range(N - 1):

        if not cv_mode:
            I[k] = I_cc
            V[k] = OCV_func(soc[k]) + I[k] * R

            if (V[k] >= V_cv and np.sign(I_cc) == 1) or (V[k] <= V_cv and np.sign(I_cc) == -1):
                cv_mode = True
                I[k] = (V_cv - OCV_func(soc[k])) / R
                V[k] = V_cv
                I_cv.append(I[k])

        else:
            I[k] = (V_cv - OCV_func(soc[k])) / R
            V[k] = V_cv
            I_cv.append(I[k])

        if np.abs(I[k]) < np.abs(I_cut):
            break

        soc[k+1] = soc[k] + (I[k] * dt) / Q_nom
        soc[k+1] = np.clip(soc[k+1], 0, 1)

    I[-1] = I[-2]
    V[-1] = V[-2]

    I = I[:k]
    soc = soc[:k]
    t = t[:k]
    V = V[:k]

    return t, I, V, soc, I_cv

def OCV(soc):
    """
    Example nonlinear OCV vs SOC curve.
    Replace this with your own model.
    """
    return 3.0 + 1.2*soc

if __name__ == "__main__":
    # Parameters
    R = 0.05           # ohm
    Q_nom = 3.0*3600   # 3 Ah cell → Coulombs
    I_cc = -5         # A
    V_cv = 3.0         # V
    soc0 = 1.0       # initial SOC = 10%
    I_cut = -0.5

    t = np.linspace(0, 4000, 4001)  # time in seconds

    t, I, V, soc, I_cv = simulate_cc_cv(
        t=t,
        soc0=soc0,
        R=R,
        Q_nom=Q_nom,
        I_cc=I_cc,
        I_cut=I_cut,
        V_cv=V_cv,
        OCV_func=OCV
    )

    plt.figure(figsize=(10,6))

    plt.subplot(3,1,1)
    plt.plot(t, V)
    plt.ylabel("Voltage (V)")

    plt.subplot(3,1,2)
    plt.plot(t, I)
    plt.ylabel("Current (A)")

    plt.subplot(3,1,3)
    plt.plot(t, soc)
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")

    plt.tight_layout()
    plt.show()

    print(np.mean(I_cv), obtain_IaveCV(I_cc, R, Q_nom, I_cut))
    print(obtain_IaveCC(I_cc, soc0, R, Q_nom, V_cv))

    print(np.mean(I), obtain_Iave_CCCV(I_cc, I_cut, soc0, R, Q_nom, V_cv))
