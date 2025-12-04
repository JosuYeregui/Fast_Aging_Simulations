import numpy as np
import bisect

from matplotlib import pyplot as plt


class Step:
    def __init__(self, I, V_end, Cutoff_I=None):
        self.I = I
        self.V_end = V_end
        self.soc0 = None
        self.socEND = None
        self.I_cut = Cutoff_I
        self.t_duration = None
        self.I_avg = None

    def __call__(self, t):
        if self.socEND is None:
            raise Exception("Step not yet simulated")

        return self.I_avg, self.soc0 + (self.socEND - self.soc0) * t / self.t_duration

    def simulate_step(self, *args, **kwargs):
        pass

    def update_step(self, *args, **kwargs):
        self.simulate_step(*args, **kwargs)

    def _obtain_IaveCC(self, soc0, R, Q_nom, OCV, dt=0.1):

        I_avg = self.I

        if np.sign(I_avg) == 1:
            soc_array = np.arange(soc0, 1.01, step=dt * I_avg / Q_nom)
        else:
            soc_array = np.arange(soc0, -0.01, step=dt * I_avg / Q_nom)
        V_ocv = OCV(soc_array)
        V_tar = self.V_end - I_avg * R

        try:
            if np.sign(I_avg) == 1:
                idx_cross = np.where(V_ocv > V_tar)[0][0]
            else:
                idx_cross = np.where(V_ocv < V_tar)[0][0]
        except:
            idx_cross = 0

        if idx_cross == 0:
            t_end = dt

        else:
            t0 = (idx_cross - 1) * dt
            # Interpolate to get t_end
            t_end = t0 + dt / (V_ocv[idx_cross] - V_ocv[idx_cross - 1]) * (V_tar - V_ocv[idx_cross - 1])

        return t_end, I_avg, soc0 + I_avg * t_end / Q_nom

    def _obtain_IaveCV(self, soc0, R, Q_nom, OCV, I_cut):

        soc_est = 1. if self.I > 0 else 0.
        b = (self.V_end - OCV(soc0)) / (soc_est - soc0)

        t_end = -np.log(I_cut / self.I) * (R * Q_nom) / b

        I_ave = self.I / t_end * (R * Q_nom) / b * (1 - np.exp(-b * t_end / (R * Q_nom)))
        return I_ave, t_end, soc0 + I_ave * t_end / Q_nom


class CCStep(Step):
    def __init__(self, I, V_end):
        super().__init__(I, V_end, Cutoff_I=None)

    def simulate_step(self, soc0, R, Q_nom, OCV, dt=0.001):

        self.soc0 = soc0
        t_end, I_avg, soc = self._obtain_IaveCC(soc0, R, Q_nom, OCV, dt)

        self.I_avg = I_avg
        self.t_duration = t_end
        self.socEND = soc


class CCCVStep(Step):
    def __init__(self, I, V_end, Cutoff_I):
        super().__init__(I, V_end, Cutoff_I=Cutoff_I)

    def simulate_step(self, soc0, R, Q_nom, OCV, dt=0.001):

        self.soc0 = soc0
        t_end_cc, I_avg_cc, soc_cc = self._obtain_IaveCC(soc0, R, Q_nom, OCV, dt)
        t_end_cv, I_avg_cv, soc_cv = self._obtain_IaveCV(soc_cc, R, Q_nom, OCV, self.I_cut)

        self.I_avg = (I_avg_cc * t_end_cc + I_avg_cv * t_end_cv) / (t_end_cc + t_end_cv)
        self.t_duration = t_end_cc + t_end_cv
        self.socEND = soc_cv


class Cycler:
    def __init__(self, steps, SOC0, R, Q, OCV_fn):
        self.steps = steps
        self.soc0 = SOC0
        self.R = R
        self.Q = Q
        self.OCV = OCV_fn
        self.times = []
        self.total_time = 0.0
        self._build()

    def _build(self):

        soc = self.soc0
        total_time = 0.0
        times = [0.]
        for step in self.steps:
            step.simulate_step(soc, self.R, self.Q, self.OCV)

            soc = step.socEND
            total_time += step.t_duration
            times.append(total_time)

        self.total_time = total_time
        self.times = times

    def __call__(self, t):
        mod_t = t % self.total_time
        idx = bisect.bisect_left(self.times[1:], mod_t)
        t_in = mod_t - self.times[idx]

        return self.steps[idx](t_in)

    def update_cycle(self, R, Q, OCV_fn):
        # self.times = [t * Q / self.Q for t in self.times]
        # self.total_time *= Q / self.Q
        self.R = R
        self.Q = Q
        self.OCV = OCV_fn

        self._build()

if __name__ == "__main__":
    # Dummy OCV function
    def OCV_fn(soc):
        return 2.5 + 1.7 * soc  # simple linear OCV for demonstration

    steps = [
        CCStep(I=1.5, V_end=3.5),
        CCCVStep(I=3, V_end=4.2, Cutoff_I=0.2),
        CCStep(I=-5, V_end=3.)
    ]

    cycl = Cycler(
        steps=steps,
        SOC0=0.1,
        R=0.049,
        Q=5.15,
        OCV_fn=OCV_fn
    )

    t = np.linspace(0, 100000, 100000) / 3600
    I = []
    soc = []

    for t_ in t:
        I_avg, soc_ = cycl(t_)
        I.append(I_avg)
        soc.append(soc_)

    fig, ax = plt.subplots()
    ax.plot(t, I)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Current (A)")
    plt.show()

    plt.plot(t, soc)
    plt.xlabel("Time (h)")
    plt.ylabel("SOC")
    plt.show()

    print(cycl.total_time)

    cycl.update_cycle(R=cycl.R, Q=4.78579286553214, OCV_fn=cycl.OCV)

    I = []
    soc = []

    for t_ in t:
        I_avg, soc_ = cycl(t_)
        I.append(I_avg)
        soc.append(soc_)

    fig, ax = plt.subplots()
    ax.plot(t, I)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Current (A)")
    plt.show()

    plt.plot(t, soc)
    plt.xlabel("Time (h)")
    plt.ylabel("SOC")
    plt.show()

    print(cycl.total_time)