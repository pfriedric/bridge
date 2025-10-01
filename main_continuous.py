"""
Main entry point for preference-based reinforcement learning experiments.

Paul Friedrich, paul.friedrich@uzh.ch, 2025.
"""

import sys
import os
import yaml
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime
import gc

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
    plot_suboptimalities_multimetrics,
    plot_pi_set_sizes_multimetrics,
    plot_mujoco_multimetrics,
)

matplotlib.use("Agg")


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
        fig_paths_subopt,
        fig_path_pi_set_sizes,
        fig_path_mujoco,
        metrics_path,
    ) = vals
    params_dict["run_dir"] = run_dir

    # set up logging for subprocesses
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
        if "bridge" not in multi_metrics and params.run_bridge:
            print("=== BRIDGE ===")
            params.baseline_or_bridge = "bridge"
            metrics_per_seed_bridge, avg_expert_reward, avg_bc_reward = run_experiment_mp(params)
            multi_metrics["avg_expert_reward"] = avg_expert_reward
            multi_metrics["avg_bc_reward"] = avg_bc_reward
            multi_metrics["bridge"] = pad_metrics_mujoco(metrics_per_seed_bridge, params_dict)
            save_metrics(multi_metrics, metrics_path, "bridge", params_dict)
            plt.close("all")
            gc.collect()
            # this saves avg expert & bc rewards b/c when no metrics exist, we save the entire dict to a fresh file (so either for baseline or bridge)

        if "baseline" not in multi_metrics and params.run_baseline:
            print("\n=== BASELINE ===")
            params.baseline_or_bridge = "baseline"
            metrics_per_seed_baseline, avg_expert_reward, avg_bc_reward = run_experiment_mp(params)
            multi_metrics["avg_expert_reward"] = avg_expert_reward
            multi_metrics["avg_bc_reward"] = avg_bc_reward
            multi_metrics["baseline"] = pad_metrics_mujoco(metrics_per_seed_baseline, params_dict)
            save_metrics(multi_metrics, metrics_path, "baseline", params_dict)
            plt.close("all")
            gc.collect()

        # ==== PLOTTING ====
        plot_types = (
            [params.which_plot_subopt]
            if isinstance(params.which_plot_subopt, str)
            else params.which_plot_subopt
        )
        figsize = (6, 3) if params.plot_slim else None
        for plot_type, fig_path_subopt in zip(plot_types, fig_paths_subopt):
            plt.figure()
            plot_suboptimalities_multimetrics(
                multi_metrics,
                fig_path_subopt,
                paper_style=True,
                mle_policy_avg_reward=multi_metrics["avg_bc_reward"],
                opt_policy_avg_reward=multi_metrics["avg_expert_reward"],
                which_plot=plot_type,  # "suboptimality_percent" or "regret" or "cumulative_regret",
                exclude_outliers=params.exclude_outliers,
                figsize=figsize,
            )
            plt.close()
        plt.close("all")

        plt.figure()
        plot_pi_set_sizes_multimetrics(
            multi_metrics,
            params_dict,
            fig_path_pi_set_sizes,
            paper_style=True,
            exclude_outliers=params.exclude_outliers,
            figsize=figsize,
        )
        plt.close()

        plt.figure()
        plot_mujoco_multimetrics(
            multi_metrics,
            fig_path_mujoco,
            paper_style=True,
            exclude_outliers=params.exclude_outliers,
        )
        plt.close()

        # ==== SAVING PARAMS ====
        if params.save_results:
            # if running notebook-style: save params to run_dir/params.yaml
            params_path = os.path.join(run_dir, "params.yaml")
            with open(params_path, "w") as f:
                yaml.dump(params_dict, f, default_flow_style=False)
            # for running from CLI: save params as yaml to configs/exps/{env}/{run_ID}.yaml
            run_id = run_dir.split("/")[-1]
            config_path = os.path.join("configs", "exps", f"{params.env_id}", f"{run_id}.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(params_dict, f, default_flow_style=False)

        print(
            f"run '{run_id}' ended successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Time taken: {time.time() - main_time:.2f} seconds")

    finally:
        sys.stdout = original_stdout
        tee_output.close()
        print(f"log file saved to {log_file_path}")


if __name__ == "__main__":
    main()
