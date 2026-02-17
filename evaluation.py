from matplotlib import pyplot as plt

from main import scipy_rk23_integrate
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *


if __name__ == "__main__":

    tot_cycles = 10000
    initial_soc = 0.1


    # -------------- Load parameters ----------------

    input_vars = load_params("./input/M50_okane_modified_plating.yaml")
    input_vars["D_ec"] /= 200
    input_vars["Beta_p"] /= 600
    input_vars["Beta_n"] *= 20


    # ---------- Define simulation steps -------------

    steps = [
        CCCVStep(I=1., V_end=4.2, Cutoff_I=0.25, Vlim=[2.5, 4.2]),
        CCStep(I=-1.5, V_end=2.5),
    ]

    # --------------- Define cycler ------------------

    OCV_fn = get_OVC_func(input_vars)

    cycl = Cycler(
        steps=steps,
        SOC0=initial_soc,
        R=input_vars["R_int"],
        Q=input_vars["Q"],
        OCV_fn=OCV_fn
    )

    input_vars = set_c_ini(cycl, input_vars)

    # -------------- Run simulation ---------------

    y0_vec = [input_vars["L_0"], input_vars["eps_0_n"], input_vars["eps_0_p"], 0.]  # L, eps_n, eps_p, Plating

    start_ode_time = time.time()
    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        input_vars=input_vars,
        tot_cycles=tot_cycles,
        cycler=cycl,
        stop_Q=0.6,
        rtol=[1e-3, 1e-3, 1e-3, 1e-3],
        atol=[1e-9 ,1e-4, 1e-4, 1e-7],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)


    # -------------- Plotting ---------------

    #t_ode = [h[0] for h in history]
    t_ode = history["n_cycles"]
    L_ode = history["y"][:, 0]
    eps_n_ode = history["y"][:, 1]
    eps_p_ode = history["y"][:, 2]
    plating_ode = history["y"][:, 3]
    t_final_evol = history["times"]
    Q_ode = history["Q"]

    fig, ax = plt.subplots()
    ax.scatter(np.asarray(t_ode), L_ode, label="ODE",
               c="#b8bb26", s=80)
    ax.plot(np.asarray(t_ode), L_ode, label="ODE",
               c="#b8bb26", linestyle="dashed")
    ax.set_xlabel("cycles")
    ax.set_ylabel("L SEI [m]")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(np.asarray(t_ode), eps_n_ode, label="\\epsilon_n ODE",
               c="#63a5df", s=80)
    ax.plot(np.asarray(t_ode), eps_n_ode, label="\\epsilon_n ODE",
               c="#63a5df", linestyle="dashed")
    ax.scatter(np.asarray(t_ode), eps_p_ode, label="\\epsilon_p ODE",
               c="#fe8019", s=80)
    ax.plot(np.asarray(t_ode), eps_p_ode, label="\\epsilon_p ODE",
               c="#fe8019", linestyle="dashed")
    # ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("cycles")
    ax.set_ylabel("\\epsilon")
    plt.show()

    # Plot Plating
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(t_ode), plating_ode, label="Plating ODE", color="green", s=80)
    ax.plot(np.asarray(t_ode), plating_ode, label="Plating ODE", color="green", linestyle="dashed")
    ax.set_xlabel("cycles")
    ax.set_ylabel("Plating [m]")
    plt.show()

    # Plot Q
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(t_ode), Q_ode, label="ODE", color="black", s=80)
    ax.plot(np.asarray(t_ode), Q_ode, label="ODE", color="black", linestyle="dashed")
    ax.set_xlabel("cycles")
    ax.set_ylabel("Q [A.h]")
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(np.asarray(t_ode), t_final_evol, label="ODE", color="blue")
    # ax.legend()
    # ax.set_xlabel("cycles")
    # plt.show()