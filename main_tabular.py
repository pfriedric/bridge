"""
Main entry point for preference-based reinforcement learning experiments.

Paul Friedrich, paul.friedrich@uzh.ch, 2025. Heavily (!) adapted from https://openreview.net/forum?id=2pJpFtdVNe.
"""

import os
import sys
import matplotlib.pyplot as plt
import yaml

import time

from discrete.experiment_runner_discrete import (
    run_experiment_multiprocessing,
    pad_metrics,
)
from common_utils import (
    parse_args_discrete,
    dirs_and_loads,
    save_metrics,
    plot_suboptimalities_multimetrics,
    plot_pi_set_sizes_multimetrics,
)


def main():
    """Main function for running experiments."""
    if len(sys.argv) > 1:
        params = parse_args_discrete()
    else:
        raise ValueError("Must provide config with -cn [config name without the .yaml]")

    vals = dirs_and_loads(params)
    (
        run_dir,
        params,
        multi_metrics,
        fig_paths_subopt,
        fig_path_pi_set_sizes,
        _,  # fig_path_mujoco
        metrics_path,
    ) = vals
    ## experiment runs. checks if run already in metrics (e.g. when they're loaded). if not, runs and optionally saves.
    # BRIDGE
    if "bridge" not in multi_metrics and params["run_bridge"]:
        print("%%%%% EXP: BRIDGE %%%%%")
        params["do_offline_BC"] = True
        metrics_per_seed_bridge, _, _, bc_reward, opt_reward = run_experiment_multiprocessing(
            params
        )
        multi_metrics["avg_bc_reward"] = bc_reward
        multi_metrics["avg_expert_reward"] = opt_reward
        multi_metrics["bridge"] = pad_metrics(metrics_per_seed_bridge, params)
        save_metrics(multi_metrics, metrics_path, "bridge", params)
    else:
        print("%%%%% EXP: BRIDGE already in metrics, or run_bridge is False. skipping %%%%%")

    # baseline
    if "purely_online" not in multi_metrics and params["run_baseline"]:
        print("%%%%% EXP: PURELY ONLINE %%%%%")
        params["do_offline_BC"] = False
        metrics_per_seed_purely_online, _, _, _, _ = run_experiment_multiprocessing(params)
        multi_metrics["purely_online"] = pad_metrics(metrics_per_seed_purely_online, params)
        save_metrics(multi_metrics, metrics_path, "purely_online", params)
    else:
        print(
            "%%%%% EXP: PURELY ONLINE already in metrics, or run_baseline is False. skipping %%%%%"
        )

    ## backwards compat: look for BC and opt rewards in multi_metrics. if not present, use old code
    try:
        bc_policy_avg_reward = multi_metrics["avg_bc_reward"]
        opt_policy_avg_reward = multi_metrics["avg_expert_reward"]
    except Exception as e:
        raise ValueError(f"Error getting BC and expert rewards: {e}")

    # figure 1: suboptimalities
    plot_types = (
        [params["which_plot_subopt"]]
        if isinstance(params["which_plot_subopt"], str)
        else params["which_plot_subopt"]
    )
    figsize = (6, 3) if params["plot_slim"] else None
    for plot_type, fig_path_subopt in zip(plot_types, fig_paths_subopt):
        plt.figure()
        plot_suboptimalities_multimetrics(
            multi_metrics,
            fig_path_subopt,
            paper_style=True,
            mle_policy_avg_reward=bc_policy_avg_reward,
            opt_policy_avg_reward=opt_policy_avg_reward,
            which_plot=plot_type,
            figsize=figsize,
        )
        plt.close()

    # figure 2: pi set sizes
    plt.figure()
    plot_pi_set_sizes_multimetrics(
        multi_metrics,
        params,
        fig_path_pi_set_sizes,
        paper_style=True,
        figsize=figsize,
    )
    plt.close()

    if params["save_results"]:
        # saving to 2 dirs: run_dir/params.yaml and configs/exps/[run_ID].yaml
        params_path = os.path.join(run_dir, "params.yaml")
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        run_id = run_dir.split("/")[-1]
        config_path = os.path.join("configs", "exps", f"{run_id}.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)

    print("run ended successfully")
    return 0


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
