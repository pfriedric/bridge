"""
On 1 machine: run and plot radius ablation

Alternatively: do runs with separate main_mujoco calls and use ablation_radii.py to collate & plot (see doc of ablation_radii.py on HowTo)
"""

import sys
import os
import time
from datetime import datetime
import gc


import matplotlib

import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        # baseline
        if "baseline" not in multi_metrics and params.run_baseline:
            print("Running baseline experiment")
            params.baseline_or_bridge = "baseline"
            metrics_per_seed_baseline, avg_expert_reward, avg_bc_reward = run_experiment_mp(params)
            multi_metrics["avg_expert_reward"] = avg_expert_reward
            multi_metrics["avg_bc_reward"] = avg_bc_reward
            multi_metrics["baseline"] = pad_metrics_mujoco(metrics_per_seed_baseline, params_dict)
            save_metrics(multi_metrics, metrics_path, "baseline", params_dict)
            plt.close("all")
            gc.collect()

        radii = [0.3, 0.65, 1.5, 3, 5, 10]
        radius_strs = [str(radius).replace(".", "_") for radius in radii]
        for radius, radius_str in zip(radii, radius_strs):
            exp_str = "bridge_" + radius_str
            if exp_str not in multi_metrics and params.run_bridge:
                print(f"Running BRIDGE experiment {exp_str}")
                params.radius = radius
                params.baseline_or_bridge = "bridge"
                metrics_per_seed_bridge, _, _ = run_experiment_mp(params)
                multi_metrics[exp_str] = pad_metrics_mujoco(metrics_per_seed_bridge, params_dict)
                save_metrics(multi_metrics, metrics_path, exp_str, params_dict)
                plt.close("all")
                gc.collect()

        plot_radius_ablation(multi_metrics, max_radius=10, only_bridge=False)

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


def plot_radius_ablation(multi_metrics, max_radius=None, only_bridge=False):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["text.usetex"] = True

    baseline_color = "#7851A9"  # baseline: purple
    bridge_base_color = "#2A9D8F"  # bridge: green

    def extract_radius(exp_name):
        """extract radius from names like 'bridge_3', 'bridge_0_5', 'bridge_10'"""
        radius_str = exp_name.replace("bridge_", "")
        radius_str = radius_str.replace("_", ".")
        try:
            return float(radius_str)
        except ValueError:
            return 0

    bridge_experiments = []
    baseline_experiments = []

    for name in multi_metrics.keys():
        if name.startswith("bridge_"):
            if max_radius is not None:
                radius = extract_radius(name)
                if radius <= max_radius:
                    bridge_experiments.append(name)
                    print(f"Including BRIDGE experiment: {name} (radius={radius:.1f})")
                else:
                    print(
                        f"Excluding BRIDGE experiment: {name} (radius={radius:.1f} >= {max_radius})"
                    )
            else:
                bridge_experiments.append(name)
        elif name in ["baseline", "purely_online"]:
            baseline_experiments.append(name)

    bridge_experiments.sort(key=extract_radius)

    # generate BRIDGE color variations (lightest to darkest)
    n_bridge = len(bridge_experiments)
    if n_bridge > 0:
        base_rgb = mcolors.hex2color(bridge_base_color)
        base_hsv = mcolors.rgb_to_hsv(base_rgb)

        green_colors = []
        if n_bridge == 1:
            green_colors = [bridge_base_color]
        else:
            for i in range(n_bridge):
                # create variations by adjusting brightness (value)
                # i=0 (lowest radius) -> lightest, i=n_bridge-1 (highest radius) -> darkest
                brightness_factor = 1.4 - 0.8 * i / (n_bridge - 1)  # range from 1.4 to 0.6
                new_hsv = base_hsv.copy()
                new_hsv[2] = min(1.0, max(0.3, base_hsv[2] * brightness_factor))
                new_rgb = mcolors.hsv_to_rgb(new_hsv)
                green_colors.append(mcolors.rgb2hex(new_rgb))

    colors = {}
    labels = {}
    for exp_name in baseline_experiments:
        colors[exp_name] = baseline_color
        if exp_name == "baseline":
            labels[exp_name] = "Baseline"
        elif exp_name == "purely_online":
            labels[exp_name] = "Online PbRL"
    for i, exp_name in enumerate(bridge_experiments):
        colors[exp_name] = green_colors[i]
        radius = extract_radius(exp_name)
        labels[exp_name] = f"BRIDGE (r={radius})"

    max_x = 0
    if multi_metrics:
        exp_lengths = []
        for name, expt in multi_metrics.items():
            if expt and isinstance(expt, list) and len(expt) > 0:
                if "regrets" in expt[0]:
                    exp_lengths.append(len(expt[0]["regrets"]))
        if exp_lengths:
            max_x = max(exp_lengths)

    if only_bridge:
        all_experiments = bridge_experiments
    else:
        all_experiments = baseline_experiments + bridge_experiments

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

        ## Plot confidence interval
        # confidence_interval = (
        #     1.96 * np.std(cumulative_regrets, axis=0) / np.sqrt(len(cumulative_regrets))
        # )
        # plt.fill_between(
        #     x_axis,
        #     avg_cumulative_regret - confidence_interval,
        #     avg_cumulative_regret + confidence_interval,
        #     alpha=0.2,
        #     color=colors[name],
        # )

        final_cumulative_regret = cumulative_regrets[:, -1]
        print(f"Avg cumulative regret R_T of {name}: {np.mean(final_cumulative_regret):.2f}")

    if max_x > 0:
        plt.plot([1, max_x], [0, 0], color="r", linestyle=":", alpha=0.7)

    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative regret")
    # ax.set_title("Radius Ablation: Cumulative Regret Comparison")

    plt.tight_layout()


if __name__ == "__main__":
    main()
