"""
Plot the results of the fleet study (fleet_study.py).

Usage (from this folder):
    python plot_fleet_results.py [--dir fleet_results_v2_ref]

Reads <dir>/fleet_results.csv (default: fleet_results_v1_baseline, the legacy run
with the new metrics) and shows:
    1. Median cycles-to-EOL heatmap (archetype x climate)
    2. Cycles-to-EOL distributions per archetype and climate
    3. Cycles-to-EOL vs temperature (P50 with P10-P90 whiskers)
    4. Dominant degradation mode share per slice (counterfactual capacity attribution)
    5. Capacity loss attributed to each mechanism per slice
    6. Manufacturing sensitivity (Spearman rho) heatmap
    7. Final SEI thickness vs final plating thickness
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the 'science' style)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # project root, needed by fleet_study -> main
sys.path.insert(0, str(HERE))

from fleet_study import (ARCHETYPES, FAST_ARCHETYPES, CLIMATES_V2, MODES, drop_truncated, summarize,
                         sensitivity_spearman)

plt.style.use(['science'])

# Fixed colors (entity -> color, never by rank)
ALL_ARCH = [a.name for a in ARCHETYPES + FAST_ARCHETYPES]
ARCH_COLORS = dict(zip(ALL_ARCH, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                  "#7a5cc4", "#8c6d31", "#4d4d4d"]))
ARCH_MARKERS = dict(zip(ALL_ARCH, ["o", "s", "^", "D", "v", "P", "X", "*"]))
CLIM_COLORS = dict(zip([c.name for c in CLIMATES_V2], ["#1c5cab", "#5598e7", "#eda100", "#c9361a", "#7a1a0a"]))
# Same colours as the degradation-mode curves in tests/Validation/comparative_tests_solv_T.py
MODE_COLORS = {"LLI_SEI": "#7a8b2e", "plating": "#9a4468", "LAM_n": "#2b5f8e", "LAM_p": "#c26a1c"}
#MODE_COLORS = {"LLI_SEI": "#62ca15", "plating": "#db5f81", "LAM_n": "#337dbe", "LAM_p": "#fe8019"}
MODE_COLORS = {"LLI_SEI": "#5F7F45", "plating": "#BC8FB0", "LAM_n": "#34557F", "LAM_p": "#E3833B"}


MODE_LABELS = {"LLI_SEI": "SEI", "plating": "Plating", "LAM_n": r"LAM$_n$", "LAM_p": r"LAM$_p$", "dc_fast": "DC fast"}


# Manufacturing parameters (keys of MFG_PARAMS_*) in LaTeX notation
PARAM_LABELS = {
    "eps_0_n": r"$\varepsilon_{0,n}$", "eps_0_p": r"$\varepsilon_{0,p}$",
    "L_0": r"$L_{0}$",
    "k_sei": r"$k_{\mathrm{SEI}}$", "k_SEI": r"$k_{\mathrm{SEI}}$",
    "D_ec": r"$D_{\mathrm{ec}}$",
    "Beta_n": r"$\beta_{n}$", "Beta_p": r"$\beta_{p}$",
    "k_plating": r"$k_{\mathrm{pl}}$",
    "R_int": r"$R_{\mathrm{int}}$", "Q": r"$Q$",
}


def param_label(name):
    return PARAM_LABELS.get(name, label(name))


def label(name):
    """Display name: no underscores, first letter uppercase."""
    name = MODE_LABELS.get(name, name.replace("_", " "))
    return name[:1].upper() + name[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="fleet_results_v4_low_plating_peak", help="Results folder (relative to this file)")
    parser.add_argument("--no-show", action="store_true", help="Only save the figures, don't open windows")
    args = parser.parse_args()
    show = plt.close if args.no_show else plt.show

    RESULTS_DIR = HERE / args.dir
    FIG_DIR = RESULTS_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_DIR / "fleet_results.csv")
    n_fail = (~df["success"]).sum()
    print(f"Loaded {len(df)} simulations ({n_fail} failed, excluded)")
    df = drop_truncated(df[df["success"]]).copy()
    if "dQ_LAM_n" not in df.columns:
        raise SystemExit("No capacity attribution columns: rerun the scenario with the current fleet_study.py")

    ARCH_ORDER = [a for a in ALL_ARCH if a in set(df["archetype"])]
    CLIM_T_C = df.groupby("climate")["T_kelvin"].first().sub(273.15).sort_values().to_dict()
    CLIM_ORDER = list(CLIM_T_C)

    summary = summarize(df)
    print(summary)

    # -------------------------------------------------------------------------
    # 1. Median cycles-to-EOL heatmap
    # -------------------------------------------------------------------------
    p50 = summary["eol_p50"].unstack("climate").reindex(index=ARCH_ORDER, columns=CLIM_ORDER)

    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(CLIM_ORDER), 1.0 + 0.4 * len(ARCH_ORDER)))
    im = ax.imshow(p50.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(CLIM_ORDER)))
    ax.set_xticklabels([f"{label(c)}\n({CLIM_T_C[c]:.0f} °C)" for c in CLIM_ORDER])
    ax.set_yticks(range(len(ARCH_ORDER)))
    ax.set_yticklabels([label(a) for a in ARCH_ORDER])
    ax.minorticks_off()
    vmid = np.nanmean([np.nanmin(p50.values), np.nanmax(p50.values)])
    for i in range(p50.shape[0]):
        for j in range(p50.shape[1]):
            v = p50.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize="small",
                    color="white" if v > vmid else "black")
    fig.colorbar(im, ax=ax, label="Median cycles to EOL")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'eol_heatmap.pdf', dpi=300, bbox_inches="tight")
    show()

    # -------------------------------------------------------------------------
    # 2. Cycles-to-EOL distributions
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(1.2 * len(ARCH_ORDER), 3.0))
    n_c = len(CLIM_ORDER)
    width = 0.8 / n_c
    for j, clim in enumerate(CLIM_ORDER):
        data = [df.loc[(df["archetype"] == a) & (df["climate"] == clim), "cycles_to_eol"].dropna().values
                for a in ARCH_ORDER]
        pos = np.arange(len(ARCH_ORDER)) + (j - (n_c - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=True, manage_ticks=False,
                        flierprops=dict(marker="o", markersize=2, alpha=0.6),
                        medianprops=dict(color="black", linewidth=1))
        for box in bp["boxes"]:
            box.set_facecolor(CLIM_COLORS[clim])
            box.set_edgecolor("black")
            box.set_linewidth(0.5)
        ax.plot([], [], "s", color=CLIM_COLORS[clim], label=f"{label(clim)} ({CLIM_T_C[clim]:.0f} °C)")
    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels([label(a).replace(" ", "\n") for a in ARCH_ORDER])
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel("Cycles to EOL")
    ax.legend(fontsize="x-small", ncol=n_c, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'eol_boxplot.pdf', dpi=300, bbox_inches="tight")
    show()

    # -------------------------------------------------------------------------
    # 3. Cycles-to-EOL vs temperature
    # -------------------------------------------------------------------------
    # P50 markers with P10-P90 whiskers, slightly offset in T so they don't overlap
    fig, ax = plt.subplots(figsize=(3.5 + (1.4 if len(ARCH_ORDER) > 5 else 0), 2.8))
    T = np.array([CLIM_T_C[c] for c in CLIM_ORDER])
    for k, arch in enumerate(ARCH_ORDER):
        s = summary.loc[arch].reindex(CLIM_ORDER)
        dT = (k - (len(ARCH_ORDER) - 1) / 2) * 0.6
        yerr = [s["eol_p50"] - s["eol_p10"], s["eol_p90"] - s["eol_p50"]]
        ax.errorbar(T + dT, s["eol_p50"], yerr=yerr, color=ARCH_COLORS[arch], marker=ARCH_MARKERS[arch],
                    markersize=4, linewidth=1.5, elinewidth=0.8, capsize=1.5, label=label(arch))
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Cycles to EOL")
    if len(ARCH_ORDER) > 5:  # too many series to fit inside the axes
        ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        ax.legend(fontsize="x-small")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'eol_vs_T.pdf', dpi=300, bbox_inches="tight")
    show()

    # -------------------------------------------------------------------------
    # 4. Dominant degradation mode share
    # -------------------------------------------------------------------------
    modes = [m for m in MODE_COLORS if m in df["dominant_mode"].unique()]
    share = (df.groupby(["archetype", "climate"])["dominant_mode"]
               .value_counts(normalize=True).unstack(fill_value=0.0)
               .reindex(columns=modes, fill_value=0.0))

    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(1.4 * len(ARCH_ORDER), 2.6), sharey=True)
    for ax, arch in zip(axes, ARCH_ORDER):
        sub = share.loc[arch].reindex(CLIM_ORDER, fill_value=0.0)
        bottom = np.zeros(len(CLIM_ORDER))
        for mode in modes:
            ax.bar(range(len(CLIM_ORDER)), sub[mode], bottom=bottom, width=0.75,
                   color=MODE_COLORS[mode], edgecolor="white", linewidth=1, label=label(mode))
            bottom += sub[mode].values
        ax.set_title(label(arch), fontsize="small")
        ax.set_xticks(range(len(CLIM_ORDER)))
        ax.set_xticklabels([label(c) for c in CLIM_ORDER], rotation=45, ha="right", fontsize="x-small")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of cells")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize="x-small", ncol=len(modes),
               loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(FIG_DIR / 'dominant_mode.pdf', dpi=300, bbox_inches="tight")
    show()

    # -------------------------------------------------------------------------
    # 5. Capacity loss attributed to each mechanism (counterfactual, mean per slice)
    # -------------------------------------------------------------------------
    dq = df.groupby(["archetype", "climate"])[[f"dQ_{m}" for m in MODES]].mean().clip(lower=0)

    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(1.4 * len(ARCH_ORDER), 2.6), sharey=True)
    for ax, arch in zip(np.atleast_1d(axes), ARCH_ORDER):
        sub = dq.loc[arch].reindex(CLIM_ORDER, fill_value=0.0)
        bottom = np.zeros(len(CLIM_ORDER))
        for mode in MODES:
            ax.bar(range(len(CLIM_ORDER)), sub[f"dQ_{mode}"], bottom=bottom, width=0.75,
                   color=MODE_COLORS[mode], edgecolor="white", linewidth=1, label=label(mode))
            bottom += sub[f"dQ_{mode}"].values
        ax.set_title(label(arch), fontsize="small")
        ax.set_xticks(range(len(CLIM_ORDER)))
        ax.set_xticklabels([label(c) for c in CLIM_ORDER], rotation=45, ha="right", fontsize="x-small")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    np.atleast_1d(axes)[0].set_ylabel(r"Counterfactual $\Delta Q$ [Ah]")
    handles, labels_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize="x-small", ncol=len(MODES),
               loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(FIG_DIR / 'capacity_attribution.pdf', dpi=300, bbox_inches="tight")
    show()

    # -------------------------------------------------------------------------
    # 6. Manufacturing sensitivity (Spearman rho vs cycles to EOL)
    # -------------------------------------------------------------------------
    sens = sensitivity_spearman(df)
    if sens.empty:
        print("Not enough samples per slice for sensitivity analysis (need >= 10)")
    else:
        sens["slice"] = sens["archetype"].map(label) + " / " + sens["climate"].map(label)
        slice_order = [f"{label(a)} / {label(c)}" for a in ARCH_ORDER for c in CLIM_ORDER]
        rho = (sens.pivot(index="param", columns="slice", values="rho")
                   .reindex(columns=[s for s in slice_order if s in sens["slice"].values]))

        fig, ax = plt.subplots(figsize=(max(7.0, 2.0 + 0.17 * rho.shape[1]), 3.0))
        im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(rho.shape[1]))
        ax.set_xticklabels(rho.columns, rotation=90, fontsize="x-small")
        ax.set_yticks(range(rho.shape[0]))
        ax.set_yticklabels([param_label(p) for p in rho.index])
        ax.minorticks_off()
        # Separate archetype groups
        for k in range(1, len(ARCH_ORDER)):
            ax.axvline(k * len(CLIM_ORDER) - 0.5, color="white", linewidth=2)
        fig.colorbar(im, ax=ax, label=r"Spearman $\rho$ (cycles to EOL)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'sensitivity_heatmap.pdf', dpi=300, bbox_inches="tight")
        show()

    # -------------------------------------------------------------------------
    # 7. Final degradation state: SEI vs plating
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for arch in ARCH_ORDER:
        for clim in CLIM_ORDER:
            sub = df[(df["archetype"] == arch) & (df["climate"] == clim)]
            ax.scatter(sub["final_L_SEI"] * 1e9, sub["final_plating"] * 1e9,
                       color=CLIM_COLORS[clim], marker=ARCH_MARKERS[arch],
                       s=12, edgecolors="white", linewidths=0.3, alpha=0.85)
    clim_handles = [plt.Line2D([], [], linestyle="", marker="o", color=CLIM_COLORS[c],
                               label=f"{label(c)} ({CLIM_T_C[c]:.0f} °C)") for c in CLIM_ORDER]
    arch_handles = [plt.Line2D([], [], linestyle="", marker=ARCH_MARKERS[a], color="gray",
                               label=label(a)) for a in ARCH_ORDER]
    fig.legend(handles=clim_handles, fontsize="x-small", loc="upper left", bbox_to_anchor=(0.70, 0.95))
    fig.legend(handles=arch_handles, fontsize="x-small", loc="lower left", bbox_to_anchor=(0.70, 0.15))
    ax.set_xlabel(r"Final $L_{SEI}$ [nm]")
    ax.set_ylabel(r"Final $L_{pl}$ [nm]")
    plt.tight_layout(rect=(0, 0, 0.70, 1))
    plt.savefig(FIG_DIR / 'final_state.pdf', dpi=300, bbox_inches="tight")
    show()
