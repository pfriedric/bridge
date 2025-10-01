"""
On 1 machine: run and plot N_offline_trajs ablation

Alternatively: do runs with separate main_mujoco calls and use ablation_N_offline.py to collate & plot (see doc of ablation_N_offline.py on HowTo)
"""

# %%
import sys
import os
import time
from datetime import datetime
import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import numpy as np
import re

from continuous.experiment_runner_continuous import (
    run_experiment_mp,
    preprocess_params_dict,
    pad_metrics_mujoco,
    Tee,
)

from common_utils import (
    parse_args_mujoco,
    dirs_and_loads,
    save_metrics,
)


def main():
    main_time = time.time()
    if len(sys.argv) > 1:
        params_dict = parse_args_mujoco(args=None)
    else:
        raise ValueError("Must provide config with -cn [config name without the .yaml]")

    vals = dirs_and_loads(params_dict)
    (
        run_dir,
        params_dict,
        multi_metrics,
        _,
        _,
        _,
        metrics_path,
    ) = vals
    params_dict["run_dir"] = run_dir
    # logging
    pid = os.getpid()
    log_file_path = os.path.join(run_dir, f"log_{pid}_{datetime.now().strftime('%y%m%d-%H%M')}.txt")
    cli_call = " ".join(sys.argv)
    tee_output = Tee(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    os.environ["EXPERIMENT_LOG_FILE"] = log_file_path

    print(f"cli call:\n{cli_call}")
    print(f"logging to: {log_file_path}")
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        params, params_dict = preprocess_params_dict(params_dict)

        # ==== EXPERIMENT RUNS ====
        noffline_trajs = [10, 15, 20, 25, 30]
        noffline_trajs_strs = [str(noffline_trajs) for noffline_trajs in noffline_trajs]
        radii = [1.1, 0.9, 0.65, 0.45, 0.4]
        radius_strs = [str(radius).replace(".", "_") for radius in radii]
        for noffline_trajs, noffline_trajs_str, radius, radius_str in zip(
            noffline_trajs, noffline_trajs_strs, radii, radius_strs
        ):
            exp_str = "bridge_N" + noffline_trajs_str + "_r" + radius_str
            if exp_str not in multi_metrics:
                params.noffline_trajs = noffline_trajs
                params.radius = radius
                params.baseline_or_bridge = "bridge"
                metrics_per_seed_bridge, _, _ = run_experiment_mp(params)
                multi_metrics[exp_str] = pad_metrics_mujoco(metrics_per_seed_bridge, params_dict)
                save_metrics(multi_metrics, metrics_path, exp_str, params_dict)
                plt.close("all")
                gc.collect()

        plot_N_offline_ablation(multi_metrics, max_N_offline=30)

        save_dir = "exps/ablations/" + run_dir[run_dir.index("/") + 1 :]

        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, "plot.pdf")
        png_path = os.path.join(save_dir, "plot.png")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print("Plots saved to:")
        print(f"  {pdf_path}")
        print(f"  {png_path}")
        print(f"Time taken: {time.time() - main_time:.2f} seconds")

    finally:
        sys.stdout = original_stdout
        tee_output.close()
        print(f"log file saved to {log_file_path}")
        # sys.exit()


def plot_N_offline_ablation(multi_metrics, max_N_offline=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["text.usetex"] = True

    bridge_base_color = "#2A9D8F"

    def extract_N_offline(exp_name):
        """extract N_offline from names like 'bridge_N10', 'bridge_N100', 'bridge_N50'"""
        match = re.search(r"N(\d+)", exp_name)
        if match:
            return int(match.group(1))
        return 0

    bridge_experiments = []

    for name in multi_metrics.keys():
        if name.startswith("bridge_"):
            if max_N_offline is not None:
                n_offline = extract_N_offline(name)
                if n_offline <= max_N_offline:
                    bridge_experiments.append(name)
                    print(f"Including BRIDGE experiment: {name} (N_offline={n_offline})")
                else:
                    print(
                        f"Excluding BRIDGE experiment: {name} (N_offline={n_offline} > {max_N_offline})"
                    )
            else:
                bridge_experiments.append(name)

    bridge_experiments.sort(key=extract_N_offline)

    n_bridge = len(bridge_experiments)
    if n_bridge > 0:
        # generate BRIDGE color variations (lightest to darkest)
        base_rgb = mcolors.hex2color(bridge_base_color)
        base_hsv = mcolors.rgb_to_hsv(base_rgb)

        green_colors = []
        if n_bridge == 1:
            green_colors = [bridge_base_color]
        else:
            # create variations by adjusting brightness (value)
            for i in range(n_bridge):
                # i=0 (lowest N_offline) -> lightest, i=n_bridge-1 (highest N_offline) -> darkest
                brightness_factor = 1.4 - 0.8 * i / (n_bridge - 1)  # range from 1.4 to 0.6
                new_hsv = base_hsv.copy()
                new_hsv[2] = min(1.0, max(0.3, base_hsv[2] * brightness_factor))
                new_rgb = mcolors.hsv_to_rgb(new_hsv)
                green_colors.append(mcolors.rgb2hex(new_rgb))

    colors = {}
    labels = {}

    for i, exp_name in enumerate(bridge_experiments):
        colors[exp_name] = green_colors[i]
        n_offline = extract_N_offline(exp_name)
        labels[exp_name] = f"BRIDGE (N={n_offline})"

    max_x = 0
    if multi_metrics:
        exp_lengths = []
        for name, expt in multi_metrics.items():
            if expt and isinstance(expt, list) and len(expt) > 0:
                if "regrets" in expt[0]:
                    exp_lengths.append(len(expt[0]["regrets"]))
        if exp_lengths:
            max_x = max(exp_lengths)

    all_experiments = bridge_experiments

    for name in all_experiments:
        expt = multi_metrics[name]
        if not expt or not isinstance(expt, list) or len(expt) == 0:
            continue

        num_iterations = len(expt[0]["regrets"]) if "regrets" in expt[0] else 0
        if num_iterations == 0:
            continue

        x_axis = np.arange(1, num_iterations + 1)
        cumulative_regrets = []

        for seed in range(len(expt)):
            try:
                regrets_per_it = np.array(expt[seed]["regrets"])
            except KeyError:
                regrets_per_it = np.array([-1 * x for x in expt[seed]["suboptimality_gaps"]])
            cumulative_regret_per_seed = np.cumsum(regrets_per_it)
            cumulative_regrets.append(cumulative_regret_per_seed)

        cumulative_regrets = np.array(cumulative_regrets)
        avg_cumulative_regret = np.mean(cumulative_regrets, axis=0)

        plt.plot(x_axis, avg_cumulative_regret, label=labels[name], color=colors[name], linewidth=2)

        final_cumulative_regret = cumulative_regrets[:, -1]
        print(f"Avg cumulative regret R_T of {name}: {np.mean(final_cumulative_regret):.2f}")

    if max_x > 0:
        plt.plot([1, max_x], [0, 0], color="r", linestyle=":", alpha=0.7)

    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative regret")
    # ax.set_title("N_offline Ablation: Cumulative Regret Comparison")

    plt.tight_layout()


if __name__ == "__main__":
    main()
