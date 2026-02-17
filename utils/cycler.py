import numpy as np
import pandas as pd
import bisect

from matplotlib import pyplot as plt


class Step:
    """
    Base class to handle a step in battery cycling.
    A step can be a constant current (CC) charge/discharge, constant current-constant
    voltage (CCCV) charge/discharge, rest, or dynamic profile.

    :param I: Current during the step [A]
    :param V_end: Target voltage at the end of the step [V]
    :param Cutoff_I: Cutoff current for CV step [A]
    :param Vlim: Voltage limits for the battery operation, [Vmin, Vmax] [V]
    """
    def __init__(self, I, V_end, Cutoff_I=None, Vlim=None):
        self.I = I
        self.V_end = V_end
        self.soc0 = None
        self.socEND = None
        self.I_cut = Cutoff_I
        self.Vlim = Vlim
        self.t_duration = None
        self.I_avg = None

        self.I_avg_dyn = {"p": 0., "n": 0.}

    def __call__(self, t):
        if self.socEND is None:
            raise Exception("Step not yet simulated")

        return self.I_avg, self.soc0 + (self.socEND - self.soc0) * t / self.t_duration

    def simulate_step(self, *args, **kwargs):
        pass

    def update_step(self, *args, **kwargs):
        self.simulate_step(*args, **kwargs)

    def _obtain_IaveCC(self, soc0, R, Q_nom, OCV, dt=0.1):
        """
        Obtain average current and duration for a CC step.

        :param soc0: Starting state of charge [0-1]
        :param R: Internal resistance [Ohm]
        :param Q_nom: Nominal capacity [Ah]
        :param OCV: Open-circuit voltage function
        :param dt: Discretization time step [s]
        :return: t_end, I_avg, soc_end: Duration of the step [s], average current [A],
                 and state of charge at the end of the step [0-1]
        """

        I_avg = self.I  # The average current is equal to the applied current in CC

        # Create SOC array to evaluate OCV
        if np.sign(I_avg) == 1:
            soc_array = np.arange(soc0, 1.01, step=dt * I_avg / Q_nom)
        else:
            soc_array = np.arange(soc0, -0.01, step=dt * I_avg / Q_nom)
        V_ocv = OCV(soc_array)
        # Compute the target OCV value. When will the terminal voltage reach V_end?
        V_tar = self.V_end - I_avg * R

        # Find the crossing index
        try:
            if np.sign(I_avg) == 1:
                idx_cross = np.where(V_ocv > V_tar)[0][0]
            else:
                idx_cross = np.where(V_ocv < V_tar)[0][0]
        except:
            idx_cross = 0

        # If none found, set to end
        if idx_cross == 0:
            t_end = dt
        # Else, interpolate to get t_end
        else:
            t0 = (idx_cross - 1) * dt  # time at previous step
            t_end = t0 + dt / (V_ocv[idx_cross] - V_ocv[idx_cross - 1]) * (V_tar - V_ocv[idx_cross - 1])

        return t_end, I_avg, soc0 + I_avg * t_end / Q_nom

    def _obtain_IaveCV(self, soc0, R, Q_nom, OCV, I_cut):
        """
        Obtain average current and duration for a CV step.

        :param soc0: Starting state of charge [0-1]
        :param R: Internal resistance [Ohm]
        :param Q_nom: Nominal capacity [Ah]
        :param OCV: Open-circuit voltage function
        :param I_cut: Cutoff current [A]
        :return: I_ave, t_end, soc_end: Average current [A], duration of the step [s],
                 and state of charge at the end of the step [0-1]
        """

        # Estimate time to reach cutoff current
        soc_est = 1. if self.I > 0 else 0.
        Vlim = self.Vlim[1] if self.I > 0 else self.Vlim[0]

        # With a OCV-R model we can derive an analytical expression (linear)
        b = (Vlim - OCV(soc0)) / (soc_est - soc0)
        if b <= 0:
            return 0, 0, soc0

        t_end = -np.log(I_cut / self.I) * (R * Q_nom) / b

        I_ave = self.I / t_end * (R * Q_nom) / b * (1 - np.exp(-b * t_end / (R * Q_nom)))
        return I_ave, t_end, soc0 + I_ave * t_end / Q_nom

class Rest(Step):
    """
    Class to handle rest steps in battery cycling.
    During a rest step, the battery is neither charged nor discharged,
    and the current is set to zero for a specified duration.

    :param t: Duration of the rest step [s]
    """
    def __init__(self, t):
        super().__init__(0., None)

        self.t_duration = t

    def simulate_step(self, soc0, R, Q_nom, OCV, dt=0.001):

        self.I_avg = 0.
        self.t_duration = self.t_duration
        self.soc0 = soc0
        self.socEND = soc0

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
    def __init__(self, I, V_end, Cutoff_I, Vlim):
        super().__init__(I, V_end, Cutoff_I=Cutoff_I, Vlim=Vlim)

    def simulate_step(self, soc0, R, Q_nom, OCV, dt=0.001):

        self.soc0 = soc0
        t_end_cc, I_avg_cc, soc_cc = self._obtain_IaveCC(soc0, R, Q_nom, OCV, dt)
        t_end_cv, I_avg_cv, soc_cv = self._obtain_IaveCV(soc_cc, R, Q_nom, OCV, self.I_cut)

        self.I_avg = (I_avg_cc * t_end_cc + I_avg_cv * t_end_cv) / (t_end_cc + t_end_cv)
        self.t_duration = t_end_cc + t_end_cv
        self.socEND = soc_cv

import time
class DynamicProfile(Step):
    """
    Class to handle dynamic current profiles for battery cycling.
    It can load current profiles from a file or accept them directly as input vectors.
    The class simulates the battery behavior under the given current profile,
    considering voltage limits and cycling options.

    :param V_lim: Voltage limits for the battery operation, [Vmin, Vmax] [V]
    :param cycle: Boolean indicating whether to repeat the profile cyclically [bool]
    :param dir: File path to load the current profile from [str, optional]
    :param I_vec: Current profile vector [array-like, optional]
    :param t_vec: Time vector corresponding to the current profile [array-like, optional]
    """
    def __init__(self, V_lim, cycle, dir=None, I_vec=None, t_vec=None):
        self.cycle = cycle

        # Check if a directory is provided to load the profile, else expect vectors
        if dir is not None:
            I, t = self._load_profile(dir)
            I = I
            t = t
        elif I_vec is not None and t_vec is not None:
            I = I_vec
            t = t_vec
        else:
            raise Exception("No valid input")

        self.t_cycle= t

        super().__init__(I, None, Cutoff_I=None, Vlim=V_lim)

    def simulate_step(self, soc0, R, Q_nom, OCV):

        self.soc0 = soc0

        self.I_avg, self.t_duration, self.socEND, self.I_avg_dyn = self._obtain_IaveDynamic(soc0, R, Q_nom, OCV)

    def _load_profile(self, filename):
        data = np.loadtxt(filename, skiprows=1)
        t = data[:, 0] / 3600.
        I = data[:, 1]
        return I, t

    def _obtain_IaveDynamic(self, soc0, R, Q_nom, OCV):

        def perform_cycle(soc0, I, t, Q, R, OCV):
            dt = np.diff(t)
            soc_minus_ini = soc0 + np.cumsum(I[:-1] * dt / Q)
            V = OCV(soc_minus_ini) + I[-1] * R

            out_of_bounds = np.where((V < self.Vlim[0]) | (V > self.Vlim[1]))[0]

            if len(out_of_bounds) > 0:
                idx = out_of_bounds[0]  # first occurrence
                # Remember idx can go as high as len(soc_minus_ini)-1 because of dt, we want to give the next step
                soc = soc_minus_ini[idx+1]
                t = t[idx+1]
                terminal = True
            else:
                # If voltage never goes out of limits
                soc = soc_minus_ini[-1]
                t = t[-1]
                terminal = False

            return soc, t, terminal

        terminal = False
        t_end = 0
        soc_cycle = soc0

        I_comp = {"p": 0., "n": 0.}

        while not terminal or t_end > 1e6:

            soc_cycle, t, terminal = perform_cycle(soc_cycle, self.I, self.t_cycle, Q_nom, R, OCV)
            t_end += t

            if not self.cycle:
                terminal = True

            t_new = self.t_cycle[self.t_cycle <= t_end]
            I_new = self.I[self.t_cycle <= t_end]
            I_comp_p, I_comp_n = self._obtain_IaveCompensation(I_new, t_new)
            I_comp["p"] += I_comp_p
            I_comp["n"] += I_comp_n

        I_ave = (soc_cycle - soc0) * Q_nom / t_end

        I_comp["p"] /= t_end
        I_comp["n"] /= t_end

        return I_ave, t_end, soc_cycle, I_comp

    @staticmethod
    def _obtain_IaveCompensation(I, t):

        I_p = I.copy()
        I_p[I_p < 0] = 0
        I_comp_p = np.trapz(I_p, t)
        # Get Iave for positive current profile taking t into account

        I_n = I.copy()
        I_n[I_n > 0] = 0
        I_comp_n = np.trapz(I_n, t)
        return I_comp_p, I_comp_n

class PowerProfile(Step):
    def __init__(self, V_lim, P, t):

        self.t_cycle= t

        super().__init__(P, None, Cutoff_I=None, Vlim=V_lim)
        self.P = self.I

    def simulate_step(self, soc0, R, Q_nom, OCV):

        self.soc0 = soc0

        self.I_avg, self.t_duration, self.socEND = self._obtain_IavePower(soc0, R, Q_nom, OCV)

    def _obtain_IavePower(self, soc0, R, Q_nom, OCV):
        """
        Obtain average current and duration for a power profile step.
        :param soc0: Starting state of charge [0-1]
        :param R:
        :param Q_nom:
        :param OCV:
        :return:
        """

        P = self.P
        t = self.t_cycle
        dt = np.diff(t)

        soc = soc0
        t_end = 0.0

        for k in range(len(dt)):

            Voc = OCV(soc)
            Pk = P[k]

            # Solve R*I^2 - Voc*I + P = 0
            disc = Voc**2 - 4 * R * Pk

            if disc < 0:
                raise ValueError("Requested power exceeds battery capability")

            # Discharge: use smaller root
            I = (Voc - np.sqrt(disc)) / (2 * R)

            # Terminal voltage
            V = Voc + I * R

            # Voltage limits
            if (V < self.Vlim[0]) or (V > self.Vlim[1]):
                break

            # SOC update
            soc += I * dt[k] / Q_nom
            if soc > 1.0:
                soc = 1.0
                break
            elif soc < 0.0:
                soc = 0.0
                break
            t_end += dt[k]

        # Average current is computed from SOC change
        I_ave = (soc - soc0) * Q_nom / t_end if t_end > 0 else 0.0

        return I_ave, t_end, soc


class Cycler:
    """
    Class to manage a sequence of battery operation steps (charging, discharging, resting).
    It tracks the state of charge (SoC) and time progression through the defined steps,
    allowing for simulation of battery cycling behavior.

    :param steps: List of Step objects defining the sequence of operations [Step list]
    :param SOC0: Initial state of charge (SoC) at the start of the cycle [0-1]
    :param R: Internal resistance of the battery [Ohm]
    :param Q: Battery capacity [Ah]
    :param OCV_fn: Function to compute open-circuit voltage (OCV) based on SoC
    """

    def __init__(self, steps, SOC0, R, Q, OCV_fn):
        self.steps = steps
        self.soc0 = SOC0
        self.soc = []
        self.R = R
        self.Q = Q
        self.OCV = OCV_fn
        self.times = []
        self.total_time = 0.0
        self._build()

    def _build(self):
        """
        Builds the sequence of states of charge (SoC) and time progression for a series of simulation
        steps. Tracks the resulting total time and state changes after each step.

        :return: None
        """

        # Initialize the first state of charge
        soc = self.soc0
        self.soc = [soc]
        total_time = 0.0
        times = [0.]

        # Loop through each step in the sequence
        for step in self.steps:
            step.simulate_step(soc, self.R, self.Q, self.OCV)

            soc = step.socEND
            self.soc.append(soc)
            total_time += step.t_duration
            times.append(total_time)

        self.total_time = total_time
        self.times = times

    def add_step(self, step):
        """
        Adds a new simulation step to the cycler, updating the cycler variables

        :param step: New simulation step to be added
        :return: None
        """

        soc = self.soc[-1]
        step.simulate_step(soc, self.R, self.Q, self.OCV)

        self.steps.append(step)
        self.soc.append(step.socEND)
        self.total_time += step.t_duration
        self.times.append(self.total_time)

    def __call__(self, t):
        """
        Given a time t, returns the average current and state of charge (SoC)
        at that time by determining the appropriate step in the sequence.

        :param t: Time at which to evaluate the cycler
        :return state: Tuple of average current and SoC at time t"""

        # If the total time is zero, the first step function is called with an input of 0.
        if self.total_time == 0:
            return self.steps[0](0.)

        # Find the step index corresponding to the given time
        mod_t = t % self.total_time
        idx = bisect.bisect_left(self.times[1:], mod_t)
        t_in = mod_t - self.times[idx]

        return self.steps[idx](t_in)

    def update_cycle(self, R, Q, OCV_fn):
        """
        Updates the cycle parameters such as resistance (R), capacity (Q), and
        open-circuit voltage function (OCV_fn). Rebuilds the cycle with the new
        parameters.
        :param R: Internal resistance
        :param Q: Battery capacity
        :param OCV_fn: Open-circuit voltage function
        :return: None
        """
        # self.times = [t * Q / self.Q for t in self.times]
        # self.total_time *= Q / self.Q
        self.R = R
        self.Q = Q
        self.OCV = OCV_fn

        self._build()

    def get_soc_ave(self):
        """
        Computes the average state of charge (SoC) over the entire cycle duration.
        :return: soc_ave: Average state of charge over the cycle duration
        """

        # In some cases, total_time can be zero (e.g., Initial conditions not feasible)
        if self.total_time == 0:
            return self.soc0

        # Calculate time-weighted average SoC
        soc_ave = np.zeros(len(self.soc) - 1)
        t_steps = np.diff(self.times)
        for i in range(len(self.soc)-1):
            soc_ave[i] = (self.soc[i] + self.soc[i+1])/2 * t_steps[i]

        return np.sum(soc_ave) / self.total_time

    def get_n_cycles(self, t, t_prev):
        """
        Computes the number of cycles completed between two time points t_prev and t.
        :param t: The final time point
        :param t_prev: Previous execution time point
        :return: n_cycles: Number of cycles completed between t_prev and t
        """

        if self.total_time == 0:
            return 0.

        dt = t - t_prev
        return dt / self.total_time

if __name__ == "__main__":
    # Dummy OCV function
    def OCV_fn(soc):
        return 2.5 + 1.7 * soc  # simple linear OCV for demonstration

    P = np.random.uniform(low=-20, high=20, size=(1000,))
    t = np.arange(0, len(P), 1) / 3600

    steps = [
        CCStep(I=1.5, V_end=3.5),
        PowerProfile(V_lim=[2.5, 4.2], P=P, t=t)
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