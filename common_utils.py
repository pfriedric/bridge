"""
Contains utils used by both discrete & continuous implementation
"""

import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import copy
import yaml
import numpy as np
import argparse

SHORT_ENV_NAMES = {
    "StarMDP_with_random_flinging": "StarMDP",
    "StarMDP_with_random_staying": "StarMDPStaying",
    "Gridworld": "Gridworld",
    "HalfCheetah-v5": "HalfCheetah",
    "halfcheetah": "HalfCheetah",
    "Reacher-v5": "Reacher",
    "reacher": "Reacher",
    "Ant-v5": "Ant",
    "ant": "Ant",
    "Walker2d-v5": "Walker",
    "walker": "Walker",
    "Hopper-v5": "Hopper",
    "hopper": "Hopper",
    "Humanoid-v5": "Humanoid",
    "humanoid": "Humanoid",
}

ENV_N_STATES_ACTIONS = {
    "StarMDP": (5, 4),
    "StarMDP_staying": (5, 4),
    "Gridworld": (16, 4),
}


# ==== Directories, saving, loading ==== #
def get_run_dir(params):
    """
    looks at params and returns the directory where the run will be saved. does NOT create dir.
    if run_ID is None: return path of dir with next higher available 3-digit ID
    if run_ID is specified: return path of dir with run_ID
    """
    env_name = SHORT_ENV_NAMES[params.get("env") or params.get("env_id")]
    os.makedirs(f"exps/{env_name}", exist_ok=True)  # ensure dir for env exists
    # if user specified a run_ID, check if dir for run exists -- if yes: return it
    if params.get("run_ID") is not None:
        run_dir = f"exps/{env_name}/{params.get('run_ID')}"

    # if not, find the next available ID (find highest 3-digit number in folder names of the save directory, increment by 1)
    else:
        # get all folders in figs/{env_name}
        folders = [
            f for f in os.listdir(f"exps/{env_name}") if os.path.isdir(f"exps/{env_name}/{f}")
        ]
        # get the highest three-digit number
        highest_number = (
            max([int(f.split("_")[0]) for f in folders if f.isdigit()]) if len(folders) > 0 else 0
        )
        next_id = f"{highest_number + 1:03d}"  # id+1
        run_dir = f"exps/{env_name}/{next_id}"
    return run_dir


def get_fig_names(params):
    """
    create the names of the figs that will be saved. names contain some run characteristics, and a timestamp.
    """
    env_name = SHORT_ENV_NAMES[params.get("env") or params.get("env_id")]

    if env_name in ["StarMDP", "StarMDPStaying", "Gridworld"]:
        if params["offlineradius_formula"] == "full":
            radius = "full"
        elif params["offlineradius_formula"] == "ignore_bracket":
            radius = "IB"
        elif params["offlineradius_formula"] == "only_alpha":
            radius = "OA"
        elif params["offlineradius_formula"] == "hardcode_radius_scaled":
            radius = "HCscaled-" + str(params["offlineradius_override_value"])
        elif params["offlineradius_formula"] == "hardcode_radius":
            radius = "HC-" + str(params["offlineradius_override_value"])
        else:
            raise ValueError(f"Unknown offlineradius formula: {params['offlineradius_formula']}")

        if params["gamma_t_hardcoded_value"]:
            gamma_t = params["gamma_t_hardcoded_value"]
        else:
            gamma_t = "calc"

        bonus_multiplier = params["online_confset_bonus_multiplier"]

        env_move_prob_str = str(params["env_move_prob"]).replace(".", "-")

        if params["which_confset_construction_method"] == "rejection-sampling-from-all":
            confsize = "RSall"
        elif params["which_confset_construction_method"] == "rejection-sampling-from-sample":
            confsize = "RSsamp"
        else:
            raise ValueError(
                f"Unknown confset construction method: {params['which_confset_construction_method']}"
            )

        stem = (
            f"{env_name}_mp{env_move_prob_str}_"
            f"F{params['phi_name']}_"
            f"T{params['N_offline_trajs']}_"
            f"H{params['episode_length']}_"
            f"rad{radius}_"
            f"exp{params['N_experiments']}_"
            f"its{params['N_iterations']}_"
            f"gamma{gamma_t}_"
            f"B{bonus_multiplier}_"
            f"confset{confsize}"
        )
    elif env_name in ["HalfCheetah", "Reacher", "Ant", "Hopper", "Walker"]:
        if params["which_policy_selection"] == "random":
            polsel = "Rand"
        elif params["which_policy_selection"] == "ucb":
            polsel = "UCB"
        elif params["which_policy_selection"] == "max_uncertainty":
            polsel = "MaxUnc"
        else:
            raise ValueError(f"Unknown policy selection method: {params['which_policy_selection']}")
        stem = (
            f"{env_name}_"
            f"F{params['embedding_name']}_"
            f"T{params['N_offline_trajs']}_"
            f"H{params['episode_length']}_"
            f"exp{params['N_experiments']}_"
            f"its{params['N_iterations']}_"
            f"confset{params['N_confset_size']}_"
            f"PolSel{polsel}_"
        )

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    subopt_type_mapping = {
        "suboptimality_percent": "subopt_percent",
        "regret": "iter_regret",
        "cumulative_regret": "cum_regret",
        "raw_reward": "raw_reward",
        "regret_indiv": "iter_regret_indiv",
    }

    fig_names_subopt = []
    if isinstance(params["which_plot_subopt"], str):
        params["which_plot_subopt"] = [params["which_plot_subopt"]]
    for plot_type in params["which_plot_subopt"]:
        if plot_type not in subopt_type_mapping:
            raise ValueError(f"Unknown subopt type: {plot_type}")
        subopt_type = subopt_type_mapping[plot_type]
        fig_names_subopt.append(f"{stem}_{subopt_type}_{timestamp}.png")

    fig_name_pi_set_sizes = f"{stem}_pi_set_sizes_{timestamp}.png"
    fig_name_mujoco = f"{stem}_mujocometrics_{timestamp}.png"

    return fig_names_subopt, fig_name_pi_set_sizes, fig_name_mujoco


def dirs_and_loads(params):
    """
    Initialize experiment by setting & creating run directory and loading params and/or metrics.
    """

    run_dir = get_run_dir(params)
    return_params = params.copy()
    return_metrics = {}

    # if user specified a run_ID: see if we can load metrics and/or params
    if params["run_ID"] is not None:
        # if 'run_ID' directory exists: 'load' branch
        # check what user wants to do when loading (continue previous run, redo loaded run, or overwrite with current params)
        if os.path.exists(run_dir):
            # if continue: load metrics (-> sim only what's missing). None defaults to continue as it's not destructive
            if (
                params["loaded_run_behaviour"] == "continue"
                or params["loaded_run_behaviour"] is None
            ):
                pkl_files = [f for f in os.listdir(run_dir) if f.endswith(".pkl")]
                if len(pkl_files) == 0:
                    print(
                        "no pkl found even though run_ID specified and dir exists, so doing 'continue'"
                    )
                    pass
                if len(pkl_files) > 1:
                    raise RuntimeError(f"More than one .pkl file found in {run_dir}: {pkl_files}")
                if len(pkl_files) == 1:
                    with open(os.path.join(run_dir, pkl_files[0]), "rb") as f:
                        return_metrics = pickle.load(f)
                    print(f"Continuing loaded run: Loaded metrics from {run_dir}/{pkl_files[0]}")
            # if redo: load params (-> re-sim with loaded params)
            elif params["loaded_run_behaviour"] == "redo":
                with open(os.path.join(run_dir, "params.yaml"), "r") as f:
                    return_params = yaml.load(f, Loader=yaml.FullLoader)
                print(f"Redoing loaded run: Loaded params from {run_dir}/params.yaml")
                return_params["loaded_run_behaviour"] = "redo"  # relevant when saving metrics later
            # if overwrite: no loading (-> re-sim with current params)
            elif params["loaded_run_behaviour"] == "overwrite":
                print("Overwriting existing run: Not loading anything")
                pass
            else:
                print("Run already exists. Specify 'loaded_run_behaviour'.")
    # if no run_ID specified, or specified but dir doesn't exist yet: fresh run, create new run dir
    os.makedirs(run_dir, exist_ok=True)

    fig_names_subopt, fig_name_pi_set_sizes, fig_name_mujoco = get_fig_names(params)
    fig_paths_subopt = [f"{run_dir}/{fig_name}" for fig_name in fig_names_subopt]
    fig_path_pi_set_sizes = f"{run_dir}/{fig_name_pi_set_sizes}"
    fig_path_mujoco = f"{run_dir}/{fig_name_mujoco}"
    # if not saving results, don't save figs
    if not params["save_results"]:
        fig_paths_subopt = []
        fig_path_pi_set_sizes = None
        fig_path_mujoco = None
    metrics_path = f"{run_dir}/multi_metrics.pkl"
    return (
        run_dir,
        return_params,
        return_metrics,
        fig_paths_subopt,
        fig_path_pi_set_sizes,
        fig_path_mujoco,
        metrics_path,
    )


def save_metrics(multi_metrics, metrics_path, key, params):
    """this should save multi_metrics to metrics_path. saves run to existing dict if key not present, or saves to fresh file if no existing dict.
    This allows two separate processes that each do different runs of the same params to save to one metrics dict, for later plotting.

    Args:
        multi_metrics: dict of metrics, with keys being the run identifier (e.g. 'BRIDGE' or `purely_online') and values being a list of dicts (one dict per seed)
        i.e.: {'bridge': [{'metric1': [t1, t2, ...], 'metric2': [t1, t2, ...]}, {...}, ..., {...}], 'purely_online': [{..}, ..., {..}]}
        metrics_path: path to the metrics pkl file
        key: the run identifier (e.g. 'BRIDGE' or `purely_online')
        save_results: whether to save or skip.
    """
    if not params["save_results"]:
        return
    # if metrics file already exists, load it
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            loaded_metrics = pickle.load(f)
        # if loaded metrics already contain the run you're trying to save, and you're not overwriting or redoing the run, abort
        if (
            key in loaded_metrics
            and params["loaded_run_behaviour"] != "overwrite"
            and params["loaded_run_behaviour"] != "redo"
        ):
            print(f"Key {key} already exists in {metrics_path}. Aborting save.")
            return
        # if it doesn't, add it and save the dict
        loaded_metrics[key] = multi_metrics[key]
        with open(metrics_path, "wb") as f:
            pickle.dump(loaded_metrics, f)
        print(f"Saved {key} to {metrics_path} (added to existing file)")
        return
    # if metrics file doesn't exist, save yours
    else:
        with open(metrics_path, "wb") as f:
            pickle.dump(multi_metrics, f)
        print(f"Saved {key} to {metrics_path} (fresh file)")
        return


# ===================================== #


# ==== Plotting ==== #
def plot_suboptimalities_multimetrics(
    multi_metrics,
    save_name=None,
    paper_style=False,
    mle_policy_avg_reward=None,
    opt_policy_avg_reward=None,
    which_plot="suboptimality_percent",  # or "regret" or "cumulative_regret"
    exclude_outliers=False,
    figsize=None,  # (6,4)?,
):
    """Plots suboptimalities of the best iteration policy vs the true optimal policy

    multi_metrics is a dict with keys being the names of the experiments, and values being lists of dicts, each dict being a run of the experiment
    i.e.: {
        "purely_online": [
            {"regret": [value1, value2, ...], "pi_set_sizes": [value1, value2, ...]},  # seed0
            {"regret": [value1, value2, ...], "pi_set_sizes": [value1, value2, ...]},  # seed1
            ...
        ],
    }

    Styles:
    - "suboptimality_percent": plot suboptimality in percent (best iteration policy / true optimal policy)
    - "suboptimality_gap": plot suboptimality in absolute terms (best iteration policy - true optimal policy)

    Use 'exclude_outliers' to exclude certain BRIDGE runs from the plot:
      "worst_cumregret": the worst run
      "95conf_cumregret": any run outside the approximate 95% confidence interval w.r.t. cumulative regret
    """
    if paper_style:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["text.usetex"] = True  # IF ERRORS THROWN: COMMENT THIS OUT

    if figsize is not None:
        plt.figure(figsize=figsize)

    colors = {
        "purely_online": "#7851A9",  # "#007BA7",
        "baseline": "#7851A9",  # cont. experiments use different key
        # "traj50": "#2A9D8F",
        "bridge": "#2A9D8F",
    }
    labels = {
        "purely_online": "Online PbRL",
        "baseline": "Online PbRL",
        "bridge": "BRIDGE",
    }

    multi_metrics_ = copy.deepcopy(multi_metrics)

    # Find the min and max x-values across all experiments
    max_x = 0
    if multi_metrics_:
        # Check if there are any experiments to prevent errors with max() on empty sequence
        exp_lengths = [
            len(expt[0]["avg_rewards_best_iteration_policy"])
            for name, expt in multi_metrics_.items()
            if expt and name not in ["avg_expert_reward", "avg_bc_reward"]
        ]
        if exp_lengths:
            max_x = max(exp_lengths)

    # exclude outliers from plotting
    if exclude_outliers in [
        "worst_bcexpertdist",
        "worst_cumregret",
        "95conf_bcexpertdist",
        "95conf_cumregret",
    ]:
        # print(
        #     f"%%% TMP %%%: no. of seeds for bridge before filtering: {len(multi_metrics_['bridge'])}"
        # )
        bridge_metrics = multi_metrics_["bridge"]
        filtermetric = []
        try:
            for seed_idx in range(len(bridge_metrics)):
                if exclude_outliers in ["worst_bcexpertdist", "95conf_bcexpertdist"]:
                    filtermetric.append(bridge_metrics[seed_idx]["dist_bc_expert"][0])
                    filtermetric_name = "dist_bc_expert"
                elif exclude_outliers in ["worst_cumregret", "95conf_cumregret"]:
                    filtermetric.append(np.sum(bridge_metrics[seed_idx]["regrets"]))
                    filtermetric_name = "cumregret"

            # build index mask of which seeds to keep
            exclude_mask = np.ones(len(bridge_metrics), dtype=bool)
            if exclude_outliers in ["worst_bcexpertdist", "worst_cumregret"]:
                worst_seed_idx = np.argmax(filtermetric)
                exclude_mask[worst_seed_idx] = False
                n_excluded_runs = len(exclude_mask) - np.sum(exclude_mask)
                print(
                    f"Plotting: excluding {n_excluded_runs} run, seed {worst_seed_idx} (worst {filtermetric_name})"
                )
            elif exclude_outliers in [
                "95conf_bcexpertdist",
                "95conf_cumregret",
            ]:  # exclude if value > "mean + 1.96 * stdev" (0.95%conf)
                # exclude_mask[np.argsort(cumregrets)[:int(len(bridge_metrics) * 0.05)]] = False  # alternative exclusion
                approx_95_conf = np.mean(filtermetric) + 1.96 * np.std(filtermetric)
                exclude_mask[filtermetric > approx_95_conf] = False
                n_excluded_runs = len(exclude_mask) - np.sum(exclude_mask)
                print(
                    f"Plotting: excluding {int(n_excluded_runs)} runs, ones with {filtermetric_name} > {approx_95_conf:.2f} (95% conf)"
                )

            # print(f"keeping seeds (exclusion mask): {exclude_mask}")
            multi_metrics_["bridge"] = [
                expt
                for expt, seed_idx in zip(bridge_metrics, range(len(bridge_metrics)))
                if exclude_mask[seed_idx]
            ]
            # print(
            #     f"%%% TMP %%%: no. of seeds for bridge after filtering: {len(multi_metrics_['bridge'])}"
            # )
        except Exception as e:
            print(f"Plotting: error during filtering, skipping. Error: {e}")

    # for each seed, calculate the suboptimality in percent at that iteration (reward of best iteration policy / reward of true optimal policy)

    for name, expt in multi_metrics_.items():
        if name in ["avg_expert_reward", "avg_bc_reward"]:
            continue  # these are just scalars and don't belong to a single run
        num_iterations = len(expt[0]["avg_rewards_best_iteration_policy"])

        # print(f"%%% TMP %%%: no. of seeds for expt {name}: {len(expt)} after filtering")

        x_axis = np.arange(1, num_iterations + 1)
        subopt = []
        if which_plot == "suboptimality_percent":
            upper_clip = 100
            for seed in range(len(expt)):
                subopt.append(
                    np.array(expt[seed]["avg_rewards_best_iteration_policy"])
                    / np.array(expt[seed]["avg_rewards_true_opt"])
                    * 100
                )
            subopt = np.array(subopt)
        elif which_plot in ["regret", "regret_indiv"]:
            upper_clip = None
            for seed in range(len(expt)):
                subopt.append(np.array(expt[seed]["regrets"]))
            subopt = np.array(subopt)
            cumulative_regret_per_seed_at_T = np.sum(subopt, axis=1)  # ax1 is iterations
            print(
                f"Avg cumulative regret R_T of {name}: {np.mean(cumulative_regret_per_seed_at_T):.2f}"
            )
        elif which_plot == "cumulative_regret":
            # idea is to plot cumulative regret (sum of regret over iterations so far)
            upper_clip = None
            for seed in range(len(expt)):
                regrets_per_it = np.array(expt[seed]["regrets"])
                cumulative_regret_per_seed = np.cumsum(regrets_per_it)
                subopt.append(cumulative_regret_per_seed)
            subopt = np.array(subopt)
            cumulative_regret_per_seed_at_T = subopt[:, -1]  # ax1 is iterations
            print(
                f"Avg cumulative regret R_T of {name}: {np.mean(cumulative_regret_per_seed_at_T):.2f}"
            )
        elif which_plot == "raw_reward":
            # idea is to plot the raw (expected) reward of best iteration policy, and an hline for expert reward
            upper_clip = None
            for seed in range(len(expt)):
                subopt.append(np.array(expt[seed]["avg_rewards_best_iteration_policy"]))
            subopt = np.array(subopt)

        avg_subopt = np.mean(subopt, axis=0)  # ax0 is seeds
        confidence_interval = 1.96 * np.std(subopt, axis=0) / np.sqrt(len(subopt))

        if which_plot in ["suboptimality_percent", "regret", "cumulative_regret"]:
            plt.plot(
                x_axis, np.clip(avg_subopt, 0, upper_clip), label=labels[name], color=colors[name]
            )
            plt.fill_between(
                x_axis,
                np.clip(avg_subopt - confidence_interval, 0, upper_clip),
                np.clip(avg_subopt + confidence_interval, 0, upper_clip),
                alpha=0.2,
                color=colors[name],
            )
            algorithm_ylim = plt.gca().get_ylim()
        elif which_plot in ["raw_reward", "regret_indiv"]:
            # Plot individual seed lines in the background with transparency
            for seed_idx in range(len(subopt)):
                plt.plot(x_axis, subopt[seed_idx], color=colors[name], alpha=0.3)
            # Plot the mean line on top with the label
            plt.plot(x_axis, avg_subopt, label=labels[name], color=colors[name])

    if which_plot == "suboptimality_percent":
        if max_x > 0:
            plt.plot([1, max_x], [100, 100], color="r", linestyle=":")
    elif which_plot in ["regret", "regeret_indiv", "cumulative_regret"]:
        if max_x > 0:
            plt.plot([1, max_x], [0, 0], color="r", linestyle=":")
    elif which_plot == "raw_reward":
        if max_x > 0:
            plt.plot(
                [1, max_x], [opt_policy_avg_reward, opt_policy_avg_reward], color="r", linestyle=":"
            )

    if mle_policy_avg_reward is not None:
        if which_plot == "suboptimality_percent":
            mle_subopt = mle_policy_avg_reward / opt_policy_avg_reward * 100
            if max_x > 0:
                plt.plot([1, max_x], [mle_subopt, mle_subopt], color="g", linestyle=":")
        elif which_plot in ["regret", "regret_indiv"]:
            mle_subopt = opt_policy_avg_reward - mle_policy_avg_reward
            if max_x > 0:
                print(f"MLE regret R_T: {mle_subopt * max_x:.2f}")
                plt.plot([1, max_x], [mle_subopt, mle_subopt], color="g", linestyle=":")
        elif which_plot == "cumulative_regret":
            mle_subopt = opt_policy_avg_reward - mle_policy_avg_reward
            if max_x > 0:
                mle_x_axis = np.arange(1, max_x + 1)
                mle_cumulative_regret = np.cumsum(np.ones(max_x) * mle_subopt)
                plt.plot(mle_x_axis, mle_cumulative_regret, color="g", linestyle=":")
                print(
                    f"MLE regret R_T: {mle_cumulative_regret[-1]:.2f} (={max_x} * {mle_subopt:.2f})"
                )
                plt.gca().set_ylim(algorithm_ylim)  # MLE regret >> BRIDGE/PbRL regret => is cut off
        elif which_plot == "raw_reward":
            if max_x > 0:
                plt.plot(
                    [1, max_x],
                    [mle_policy_avg_reward, mle_policy_avg_reward],
                    color="g",
                    linestyle=":",
                )

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort to put 'Online PbRL' first, then others
    order = []
    for i, label in enumerate(labels):
        if "Online PbRL" in label:
            order.insert(0, i)
        else:
            order.append(i)

    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left")
    # plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    if which_plot == "suboptimality_percent":
        ax.set_ylabel("\% of optimal reward")
    elif which_plot in ["regret", "regret_indiv"]:
        ax.set_ylabel("Regret")
    elif which_plot == "cumulative_regret":
        ax.set_ylabel("Cumulative regret")
    elif which_plot == "raw_reward":
        ax.set_ylabel("Best policy reward")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    else:
        plt.show()


def plot_pi_set_sizes_multimetrics(
    multi_metrics, params, save_name=None, paper_style=False, exclude_outliers=False, figsize=None
):
    """Plots the number of policies in the online confidence set Pi_t at each iteration
    Use 'exclude_outliers' to exclude certain BRIDGE runs from the plot:
      "worst_cumregret": the worst run
      "95conf_cumregret": any run outside the approximate 95% confidence interval w.r.t. cumulative regret"""
    multi_metrics_ = copy.deepcopy(multi_metrics)

    if figsize is not None:
        plt.figure(figsize=figsize)

    if paper_style:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["axes.edgecolor"] = "#333333"

        # IF ERRORS THROWN: COMMENT THIS OUT
        plt.rcParams["text.usetex"] = True

    colors = {
        "purely_online": "#7851A9",  # "#007BA7",
        "baseline": "#7851A9",  # cont. experiments use different key
        # "traj50": "#2A9D8F",
        "bridge": "#2A9D8F",
    }
    labels = {
        "purely_online": "Online PbRL",
        "baseline": "Online PbRL",
        "bridge": "BRIDGE",
    }
    baseline_search_space = params.get("baseline_search_space")
    big_search_space = (
        baseline_search_space in ["all_policies", "augmented_ball"]
        or params.get("plot_logy") is not None
    )
    if baseline_search_space in ["all_policies", "augmented_ball"]:  # StarMDP, Gridworld
        if (
            params["env"] == "StarMDP_with_random_flinging"
            or params["env"] == "StarMDP_with_random_staying"
        ):
            upper_clip = 4**5
        else:  # case env==Gridworld (can't do 'all' so do 'augmented_ball')
            upper_clip = params["N_confset_size"]
    elif params.get("confset_dilution") is not None:  # MuJoCo envs with dilution
        upper_clip = params["N_confset_size"] + params["N_confset_dilution"]
    else:  # MuJoCo envs without dilution, discrete envs without big search space
        upper_clip = params["N_confset_size"]

    # Find the min and max x-values across all experiments
    max_x = 0
    if multi_metrics_:
        # Check if there are any experiments to prevent errors with max() on empty sequence
        exp_lengths = [
            len(expt[0]["avg_rewards_best_iteration_policy"])
            for name, expt in multi_metrics_.items()
            if expt and name not in ["avg_expert_reward", "avg_bc_reward"]
        ]
        if exp_lengths:
            max_x = max(exp_lengths)

    # exclude outlier runs
    if exclude_outliers in [
        "worst_bcexpertdist",
        "worst_cumregret",
        "95conf_bcexpertdist",
        "95conf_cumregret",
    ]:
        try:
            bridge_metrics = multi_metrics_["bridge"]
            filtermetric = []
            for seed_idx in range(len(bridge_metrics)):
                if exclude_outliers in ["worst_bcexpertdist", "95conf_bcexpertdist"]:
                    filtermetric.append(bridge_metrics[seed_idx]["dist_bc_expert"][0])
                if exclude_outliers in ["worst_cumregret", "95conf_cumregret"]:
                    filtermetric.append(np.sum(bridge_metrics[seed_idx]["regrets"]))

            # build index mask of which seeds to keep
            exclude_mask = np.ones(len(bridge_metrics), dtype=bool)
            if exclude_outliers in ["worst_bcexpertdist", "worst_cumregret"]:
                worst_seed_idx = np.argmax(filtermetric)
                exclude_mask[worst_seed_idx] = False
            elif exclude_outliers in [
                "95conf_bcexpertdist",
                "95conf_cumregret",
            ]:  # exclude if value > "mean + 1.96 * stdev" (0.95%conf)
                approx_95_conf = np.mean(filtermetric) + 1.96 * np.std(filtermetric)
                exclude_mask[filtermetric > approx_95_conf] = False

            multi_metrics_["bridge"] = [
                expt
                for expt, seed_idx in zip(bridge_metrics, range(len(bridge_metrics)))
                if exclude_mask[seed_idx]
            ]
        except Exception as e:
            print(f"Plotting: error during filtering, skipping. Error: {e}")

    # for each seed, calculate the number of policies in Pi_t at that iteration
    for name, expt in multi_metrics_.items():
        if name in ["avg_expert_reward", "avg_bc_reward"]:
            continue  # these are just scalars and don't belong to a single run
        x_axis = np.arange(1, len(expt[0]["pi_set_sizes"]) + 1)
        pi_set_sizes = []
        for seed in range(len(expt)):
            pi_set_sizes.append(np.array(expt[seed]["pi_set_sizes"]))
        pi_set_sizes = np.array(pi_set_sizes)
        avg_pi_set_sizes = np.mean(pi_set_sizes, axis=0)
        confidence_interval = 1.96 * np.std(pi_set_sizes, axis=0) / np.sqrt(len(pi_set_sizes))
        if big_search_space:
            plt.semilogy(x_axis, avg_pi_set_sizes, label=labels[name], color=colors[name])
            plt.fill_between(
                x_axis,
                np.clip(avg_pi_set_sizes - confidence_interval, 1, upper_clip),
                np.clip(avg_pi_set_sizes + confidence_interval, 1, upper_clip),
                alpha=0.2,
                color=colors[name],
            )
            if max_x > 0:
                plt.plot([1, max_x], [1, 1], color="black", linestyle=":")
        else:
            plt.plot(x_axis, avg_pi_set_sizes, label=labels[name], color=colors[name])
            plt.fill_between(
                x_axis,
                np.clip(avg_pi_set_sizes - confidence_interval, 0, upper_clip),
                np.clip(avg_pi_set_sizes + confidence_interval, 0, upper_clip),
                alpha=0.2,
                color=colors[name],
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort to put 'Online PbRL' first, then others
    order = []
    for i, label in enumerate(labels):
        if "Online PbRL" in label:
            order.insert(0, i)
        else:
            order.append(i)

    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right")

    # plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Number of policies in $\Pi_t$")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    else:
        plt.show()


def plot_mujoco_multimetrics(
    multi_metrics, save_name=None, paper_style=False, exclude_outliers=False
):
    """Plots the metrics of the mujoco experiments
    Use 'exclude_outliers' to exclude certain BRIDGE runs from the plot:
      "worst_cumregret": the worst run
      "95conf_cumregret": any run outside the approximate 95% confidence interval w.r.t. cumulative regret"""
    if paper_style:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["axes.edgecolor"] = "#333333"

        # IF ERRORS THROWN: COMMENT THIS OUT
        plt.rcParams["text.usetex"] = True

    colors = {
        "purely_online": "#7851A9",  # "#007BA7",
        "baseline": "#7851A9",  # cont. experiments use different key
        # "traj50": "#2A9D8F",
        "bridge": "#2A9D8F",
    }
    labels = {
        "purely_online": "Online PbRL",
        "baseline": "Online PbRL",
        "bridge": "BRIDGE",
    }
    multi_metrics_ = copy.deepcopy(multi_metrics)

    # exclude the worst run from plotting
    if exclude_outliers in [
        "worst_bcexpertdist",
        "worst_cumregret",
        "95conf_bcexpertdist",
        "95conf_cumregret",
    ]:
        try:
            bridge_metrics = multi_metrics_["bridge"]
            filtermetric = []
            for seed_idx in range(len(bridge_metrics)):
                if exclude_outliers in ["worst_bcexpertdist", "95conf_bcexpertdist"]:
                    filtermetric.append(bridge_metrics[seed_idx]["dist_bc_expert"][0])
                if exclude_outliers in ["worst_cumregret", "95conf_cumregret"]:
                    filtermetric.append(np.sum(bridge_metrics[seed_idx]["regrets"]))

            # build index mask of which seeds to keep
            exclude_mask = np.ones(len(bridge_metrics), dtype=bool)
            if exclude_outliers in ["worst_bcexpertdist", "worst_cumregret"]:
                worst_seed_idx = np.argmax(filtermetric)
                exclude_mask[worst_seed_idx] = False
            elif exclude_outliers in [
                "95conf_bcexpertdist",
                "95conf_cumregret",
            ]:  # exclude if value > "mean + 1.96 * stdev" (0.95%conf)
                approx_95_conf = np.mean(filtermetric) + 1.96 * np.std(filtermetric)
                exclude_mask[filtermetric > approx_95_conf] = False

            multi_metrics_["bridge"] = [
                expt
                for expt, seed_idx in zip(bridge_metrics, range(len(bridge_metrics)))
                if exclude_mask[seed_idx]
            ]
        except Exception as e:
            print(f"Plotting: error during filtering, skipping. Error: {e}")
    fig, axes = plt.subplots(6, 2, figsize=(12, 20))
    axes = axes.flatten()  # flatten to make indexing easier

    # for each seed: plot all metrics
    for name, expt in multi_metrics_.items():
        if name in ["avg_expert_reward", "avg_bc_reward"]:
            continue  # these are just scalars and don't belong to a single run
        x_axis = np.arange(1, len(expt[0]["pi_set_sizes"]) + 1)
        norm_delta_ws = []
        norm_ws = []
        uncertainties = []
        var_exp_return_k_topscorers = []
        logdet_Vs = []
        trace_V_invs = []
        spearman_corrs_all = []
        spearman_corrs_eval = []
        expert_reward_ranks = []
        expert_score_ranks = []
        highest_rew_score_ranks = []
        loop_times = []

        for seed in range(len(expt)):
            norm_delta_ws.append(np.array(expt[seed]["norm_delta_w"]))
            norm_ws.append(np.array(expt[seed]["norm_w"]))
            uncertainties.append(np.array(expt[seed]["uncertainty"]))
            var_exp_return_k_topscorers.append(np.array(expt[seed]["var_exp_return_k_topscorers"]))
            logdet_Vs.append(np.array(expt[seed]["logdet_V"]))
            trace_V_invs.append(np.array(expt[seed]["trace_V_inv"]))
            spearman_corrs_all.append(np.array(expt[seed]["spearman_corr_all"]))
            spearman_corrs_eval.append(np.array(expt[seed]["spearman_corr_eval"]))
            expert_reward_ranks.append(np.array(expt[seed]["expert_reward_rank"]))
            expert_score_ranks.append(np.array(expt[seed]["expert_score_rank"]))
            highest_rew_score_ranks.append(np.array(expt[seed]["highest_rew_score_rank"]))
            loop_times.append(np.array(expt[seed]["loop_time"]))

        # Convert to numpy arrays
        norm_delta_ws = np.array(norm_delta_ws)
        norm_ws = np.array(norm_ws)
        uncertainties = np.array(uncertainties)
        var_exp_return_k_topscorers = np.array(var_exp_return_k_topscorers)
        logdet_Vs = np.array(logdet_Vs)
        trace_V_invs = np.array(trace_V_invs)
        spearman_corrs_all = np.array(spearman_corrs_all)
        spearman_corrs_eval = np.array(spearman_corrs_eval)
        loop_times = np.array(loop_times)
        placeholder = np.zeros((2, len(x_axis)))
        expert_reward_ranks = np.array(expert_reward_ranks)
        expert_score_ranks = np.array(expert_score_ranks)
        expert_rankings_mismatch = np.array(expert_reward_ranks - expert_score_ranks)
        highest_rew_score_ranks = np.array(highest_rew_score_ranks)

        # Calculate means and confidence intervals for each metric
        metrics_data = [
            (norm_delta_ws, r"$\|\Delta w\| (\rightarrow 0?)$", "norm_delta_ws"),
            (norm_ws, r"$\|w\|$ (stabilizes?)", "norm_ws"),
            (uncertainties, r"Uncertainty ($\rightarrow 0$?)", "uncertainties"),
            (
                var_exp_return_k_topscorers,
                r"$\mathrm{Var}(\mathrm{Exp.\ return\ of\ top\ 10\ scorers})$ (stabilizes?)",
                "var_exp_return_k_topscorers",
            ),
            (logdet_Vs, r"$\log\det(V)$ (mon. incr. \& stabilizes?)", "logdet_Vs"),
            (trace_V_invs, r"$\mathrm{tr}(V^{-1})$ ($\rightarrow 0$?)", "trace_V_invs"),
            (loop_times, "Online loop time", "loop_times"),
            (placeholder, "PLACEHOLDER", "placeholder"),
            (
                spearman_corrs_all,
                r"Spearman correlation all candidates ($\rightarrow 1$?)",
                "spearman_corrs_all",
            ),
            (
                spearman_corrs_eval,
                r"Spearman correlation eval set ($\rightarrow 1$?)",
                "spearman_corrs_eval",
            ),
            (
                expert_rankings_mismatch,
                r"Expert ranking, 'true ($E[R]$) - model' ($\rightarrow 0$?)",
                "expert_rankings",
            ),
            (
                -highest_rew_score_ranks,
                r"Empirical best $\pi$ ranking by model ($\rightarrow 0$?)",
                "highest_rew_score_ranks",
            ),
        ]

        for i, (metric_data, ylabel, metric_name) in enumerate(metrics_data):
            avg_metric = np.mean(metric_data, axis=0)
            confidence_interval = 1.96 * np.std(metric_data, axis=0) / np.sqrt(len(metric_data))

            if metric_name == "placeholder":
                pass

            if metric_name == "uncertainties":
                # Plot invisible point at x_axis[0] to force xlim
                axes[i].plot(x_axis[0], avg_metric[1], alpha=0, markersize=0)
                axes[i].plot(x_axis[1:], avg_metric[1:], label=labels[name], color=colors[name])
                axes[i].fill_between(
                    x_axis[1:],
                    avg_metric[1:] - confidence_interval[1:],
                    avg_metric[1:] + confidence_interval[1:],
                    alpha=0.2,
                    color=colors[name],
                )
            else:
                axes[i].plot(x_axis, avg_metric, label=labels[name], color=colors[name])
                axes[i].fill_between(
                    x_axis,
                    avg_metric - confidence_interval,
                    avg_metric + confidence_interval,
                    alpha=0.2,
                    color=colors[name],
                )
            if metric_name in ["spearman_corrs_all", "spearman_corrs_eval"]:
                axes[i].axhline(y=1, color="r", linestyle="--")
                axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            if metric_name == "expert_rankings":
                axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            if metric_name == "highest_rew_score_ranks":
                axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel(ylabel)
            axes[i].legend()

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    else:
        plt.show()


# ==================================== #


# === Argparsing === #
def str_to_bool(v):
    """Convert string representations of truth to True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def none_or_str(value):
    if value.lower() in ["none", "null"]:
        return None
    return value


def parse_args_discrete(args=None):
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description="Run preference-based RL experiments")

    # Config selection arguments
    parser.add_argument(
        "-cn",
        "--config-name",
        type=str,
        default=None,
        help="Config name to load (e.g., 'special/debug_starmdp' or 'exps/123')",
    )
    parser.add_argument(
        "-env",
        "--environment",
        type=str,
        default=None,
        choices=["starmdp", "gridworld"],
        help="Environment short (!!) name to load default config for. Only used if no config name specified.",
    )

    # Broad experiment params
    parser.add_argument(
        "--N_experiments",
        "--seeds",
        type=int,
        dest="N_experiments",
        help="Number of experiments (seeds)",
    )
    parser.add_argument("--N_iterations", type=int, help="Online iterations per seed")
    parser.add_argument("--episode_length", type=int, help="Episode length")
    parser.add_argument("--env_move_prob", type=float, help="Environment movement probability")
    parser.add_argument(
        "--phi_name",
        type=str,
        choices=["state_counts", "id_short", "id_long", "final_state"],
        help="Embedding function",
    )
    parser.add_argument("--do_offline_BC", type=str_to_bool, help="Use offline behavioral cloning")
    parser.add_argument("--N_offline_trajs", type=int, help="Number of offline trajectories")

    # Offline learning parameters
    parser.add_argument("--delta_offline", type=float, help="Offline delta parameter")
    parser.add_argument("--N_confset_size", type=int, help="Number of initial policies sampled")
    parser.add_argument(
        "--which_confset_construction_method",
        type=str,
        choices=["noise-matrices", "rejection-sampling"],
        help="Confidence set construction method",
    )
    parser.add_argument(
        "--n_transition_model_epochs_offline",
        type=int,
        help="Number of transition model epochs offline",
    )
    parser.add_argument(
        "--offlineradius_formula",
        type=str,
        choices=[
            "full",
            "ignore_bracket",
            "only_alpha",
            "hardcode_radius_scaled",
            "hardcode_radius",
        ],
        help="Offline radius formula",
    )
    parser.add_argument(
        "--offlineradius_override_value", type=float, help="Offline radius override value"
    )

    # Online learning parameters
    parser.add_argument("--N_rollouts", type=int, help="Number of rollouts per iteration")
    parser.add_argument("--delta_online", type=float, help="Online delta parameter")
    parser.add_argument("--W", type=int, help="W parameter")
    parser.add_argument("--w_MLE_epochs", type=int, help="MLE epochs for w")
    parser.add_argument("--w_initialization", type=str, help="Weight initialization method")
    parser.add_argument("--w_sigmoid_slope", type=float, help="Sigmoid slope for weights")
    parser.add_argument(
        "--xi_formula", type=str, choices=["full", "smaller_start"], help="Xi formula"
    )
    parser.add_argument(
        "--n_transition_model_epochs_online",
        type=int,
        help="Number of transition model epochs online",
    )
    parser.add_argument(
        "--online_confset_bonus_multiplier",
        type=float,
        help="Online confidence set bonus multiplier",
    )
    parser.add_argument("--gamma_t_hardcoded_value", type=float, help="Hardcoded gamma_t value")
    parser.add_argument(
        "--baseline_search_space",
        type=str,
        choices=["all_policies", "random_sample", "augmented_ball"],
        help="Baseline search space. 'all_policies', or 'random_sample' (of size N_confset_size), or 'augmented_ball' (augmenting BRIDGE's ball to N_confset_size's size with random policies)",
    )

    # Verbosity and saving
    parser.add_argument(
        "--verbose",
        nargs="*",  # user can pass multiple options that are concatted into list
        help="Verbosity options",
    )
    parser.add_argument("--run_baseline", type=str_to_bool, help="Run baseline")
    parser.add_argument("--run_bridge", type=str_to_bool, help="Run bridge")
    parser.add_argument("--save_results", type=str_to_bool, help="Save results")
    parser.add_argument("--run_ID", type=str, help="ID")
    parser.add_argument(
        "--loaded_run_behaviour",
        "--load",
        dest="loaded_run_behaviour",
        type=str,
        choices=["continue", "redo", "overwrite"],
        help="Purpose of loaded run",
    )
    parser.add_argument(
        "--which_plot_subopt",
        type=str,
        choices=["suboptimality_percent", "regret", "cumulative_regret"],
        help="Which plot to show",
    )
    parser.add_argument(
        "--plot_slim",
        type=str_to_bool,
        help="Plot in slim mode",
    )
    parser.add_argument(
        "--plot_logy",
        type=str_to_bool,
        help="Plot in logy mode",
    )

    parsed_args = parser.parse_args(args)

    config_path = None
    if parsed_args.config_name:
        config_path = os.path.join("configs", f"{parsed_args.config_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Must specify a config with -cn [config name without the .yaml]"
            )

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")

    # override with command line arguments
    for key, value in vars(parsed_args).items():
        if key in ["config_name", "environment"]:
            continue
        if value is not None:
            params[key] = value

    if "verbose" not in params or params["verbose"] is None:
        params["verbose"] = []
    if "plot_slim" not in params or params["plot_slim"] is None:
        params["plot_slim"] = False
    if "plot_logy" not in params or params["plot_logy"] is None:
        params["plot_logy"] = False
    if "n_embedding_samples" not in params or params["n_embedding_samples"] is None:
        params["n_embedding_samples"] = 100

    return params


def parse_args_mujoco(args=None):
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description="Run preference-based RL experiments")

    # Config selection arguments
    parser.add_argument(
        "-cn",
        "--config-name",
        type=str,
        default=None,
        help="Config name to load (e.g., 'special/debug_starmdp' or 'exps/123')",
    )
    parser.add_argument(
        "-env",
        "--environment",
        type=str,
        default=None,
        choices=["Reacher-v5", "HalfCheetah-v5", "reacher", "halfcheetah"],
        dest="env_id",
        help="Environment name (gym ID) to load default config for. Only used if no config name specified. Options: Reacher-v5, HalfCheetah-v5",
    )

    # Broad experiment params
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the experiment",
    )
    parser.add_argument(
        "--N_experiments",
        "-seeds",
        type=int,
        dest="N_experiments",
        help="Number of experiments (seeds)",
    )
    parser.add_argument(
        "--N_iterations", "-its", type=int, dest="N_iterations", help="Online iterations per seed"
    )
    parser.add_argument("--episode_length", type=int, help="Episode length")
    parser.add_argument(
        "--phi_name",
        "--embedding_name",
        type=str,
        choices=[
            "avg_sa",
            "avg_s",
            "last_s",
            "actionenergy",
            "psm",
            "halfcheetah_xpos",
            "reacher_perf",
            "ant_perf",
            "walker2d_perf",
            "hopper_perf",
            "humanoid_perf",
            "walker2d_extended",
            "hopper_extended",
            "humanoid_extended",
        ],
        dest="embedding_name",
        help="Embedding function",
    )
    parser.add_argument(
        "--N_offline_trajs",
        "--trajs",
        "-trajs",
        type=int,
        dest="N_offline_trajs",
        help="Number of offline trajectories",
    )
    parser.add_argument(
        "--fresh_offline_trajs", type=str_to_bool, help="Fresh offline trajectories"
    )
    parser.add_argument(
        "--initial_pos_noise",
        type=float,
        help="Initial position noise for HalfCheetah. Default is 0.1",
    )

    # Offline learning parameters
    parser.add_argument("--N_confset_size", type=int, help="Number of initial policies sampled")
    parser.add_argument(
        "--confset_base",
        "-conf_base",
        dest="confset_base",
        type=str,
        choices=["bcnoise", "bignoise", "random"],
        help="What the candidate set is made of. 'bcnoise' (BC, +noise), 'bignoise' (BC+10x noise, w/o BC), 'random' (random policies).",
    )
    parser.add_argument(
        "--confset_dilution",
        "-conf_dilution",
        "-dilute",
        "-dilution",
        dest="confset_dilution",
        type=none_or_str,
        default=None,
        help="What gets added to the confset_base to augment the candidate set. 'None', 'random' (add random policies), 'bignoise' (add BC+10x noise policies).",
    )
    parser.add_argument(
        "--N_confset_dilution",
        "-n_dilution",
        dest="N_confset_dilution",
        type=int,
        help="Number of policies to add to the confset_base.",
    )
    parser.add_argument(
        "--confset_noise",
        type=float,
        help="Noise added to BC policy to generate confset. If not provided, filled in w/ environment defaults: Reacher: 0.05, HalfCheetah: ???",
    )
    parser.add_argument("--n_bc_epochs", type=int, help="Number of BC epochs")
    parser.add_argument("--bc_loss", type=str, choices=["log-loss", "mse"], help="BC loss")
    parser.add_argument("--bc_print_evals", type=str_to_bool, help="Print BC evaluations")
    parser.add_argument(
        "--radius",
        "-radius",
        type=float,
        dest="radius",
        help="Radius for filtering offline confset: L2(embed(π_BC) - embed(π_candidate)) < radius. If unspecified, uses hardcoded defaults per embedding.",
    )
    parser.add_argument(
        "--expert_in_candidates",
        "-exp_in_cand",
        dest="expert_in_candidates",
        type=str_to_bool,
        help="Expert in search space",
    )
    parser.add_argument(
        "--expert_in_confset",
        "-exp_in_conf",
        dest="expert_in_confset",
        type=str_to_bool,
        help="Expert in confset",
    )
    parser.add_argument(
        "--expert_in_eval",
        "-exp_in_eval",
        dest="expert_in_eval",
        type=str_to_bool,
        help="Expert in eval",
    )
    parser.add_argument(
        "--which_eval_space",
        "-eval_space",
        dest="which_eval_space",
        type=str,
        choices=["candidates", "pi_zero", "pi_t"],
        help="Which eval space",
    )
    parser.add_argument(
        "--fresh_embeddings",
        type=str_to_bool,
        help="Force recalculation of embeddings",
    )

    # Online learning parameters
    parser.add_argument("--N_rollouts", type=int, help="Number of rollouts per iteration")
    parser.add_argument(
        "--filter_pi_t_yesno",
        "--filter_online",
        dest="filter_pi_t_yesno",
        type=str_to_bool,
        help="Filter Pi_t according to {pi in Pi_0 s.t. for all other pi': Δϕᵀw + γ * sqrt(Δϕᵀ V_inv Δϕ) >= 0}",
    )
    parser.add_argument(
        "--filter_pi_t_gamma",
        "--filter_online_gamma",
        "--filter_gamma",
        dest="filter_pi_t_gamma",
        type=float,
        help="Gamma for filtering Pi_t",
    )
    parser.add_argument(
        "--gamma_debug_mode",
        "--gamma_debug",
        dest="gamma_debug_mode",
        type=str_to_bool,
        default=False,
        help="Debug mode for gamma",
    )
    parser.add_argument("--W", type=int, help="W parameter")
    parser.add_argument(
        "--w_regularization",
        type=none_or_str,
        default=None,
        help="Regularization for w (None or l2)",
    )
    parser.add_argument("--w_epochs", type=int, help="MLE epochs for w")
    parser.add_argument(
        "--w_initialization",
        "-w",
        "-w_init",
        dest="w_initialization",
        type=str,
        choices=["uniform", "zeros", "random"],
        help="Weight initialization method",
    )
    parser.add_argument(
        "--project_w",
        type=str_to_bool,
        help="Project w s.t. ||w|| <= W",
    )
    parser.add_argument(
        "--retrain_w_from_scratch",
        type=str_to_bool,
        help="Retrain w from scratch",
    )
    parser.add_argument("--w_sigmoid_slope", type=float, help="Sigmoid slope for weights")
    parser.add_argument(
        "--which_policy_selection",
        type=str,
        choices=["random", "ucb", "max_uncertainty"],
        help="Policy selection method",
    )
    parser.add_argument(
        "--ucb_beta",
        "--beta",
        "--ucb",
        dest="ucb_beta",
        type=float,
        help="Beta for UCB policy selection. Used when selecting policy pairs with UCB: formula is ucb_score = σ(Δϕᵀ w) + ucb_beta * sqrt(Δϕᵀ V_inv Δϕ)",
    )
    parser.add_argument(
        "--V_init", type=str, choices=["small", "bounds"], help="V initialization method"
    )
    parser.add_argument(
        "--n_embedding_samples",
        "-n_samples",
        type=int,
        dest="n_embedding_samples",
        help="Number of samples for estimating policy embedding",
    )

    # policy model params
    parser.add_argument(
        "--hidden_dim",
        type=int,
        help="Hidden dimension of policy model. SB3 defaults to 64 x2, halfcheetah: 256 x2",
    )

    # Verbosity and saving
    parser.add_argument(
        "--verbose",
        nargs="*",  # user can pass multiple options that are concatted into list
        help="Verbosity options",
    )
    parser.add_argument(
        "--run_baseline", "-baseline", dest="run_baseline", type=str_to_bool, help="Run baseline"
    )
    parser.add_argument(
        "--run_bridge", "-bridge", dest="run_bridge", type=str_to_bool, help="Run bridge"
    )
    parser.add_argument("--save_results", type=str_to_bool, help="Save results")
    parser.add_argument("--run_ID", "-id", dest="run_ID", type=str, help="ID")
    parser.add_argument(
        "--loaded_run_behaviour",
        "--loaded_run_behavior",  # for the americans
        "--load",
        "-load",
        type=str,
        dest="loaded_run_behaviour",
        choices=["continue", "redo", "overwrite"],
        help="Purpose of loaded run",
    )
    parser.add_argument(
        "--which_plot_subopt",
        "-plot",
        "--plot",
        type=str,
        dest="which_plot_subopt",
        nargs="*",
        choices=[
            "suboptimality_percent",
            "regret",
            "regret_indiv",
            "cumulative_regret",
            "raw_reward",
        ],
        help="Which plot to show",
    )
    parser.add_argument(
        "--baseline_or_bridge",
        type=str,
        choices=["baseline", "bridge"],
        help="Which method to run",
    )
    parser.add_argument(
        "--plot_scores",
        type=str_to_bool,
        help="Plot histogram of all candidates' scores at each loop iteration",
    )
    parser.add_argument(
        "--exclude_outliers",
        "--outliers",
        "-outliers",
        "--exclude",
        "-exclude",
        dest="exclude_outliers",
        type=str,
        help="Exclude outliers from the analysis",
        choices=[
            "worst_bcexpertdist",
            "worst_cumregret",
            "95conf_bcexpertdist",
            "95conf_cumregret",
        ],
    )
    parser.add_argument(
        "--use_wandb",
        "-wandb",
        dest="use_wandb",
        type=str_to_bool,
        help="Use wandb",
    )
    parser.add_argument(
        "--plot_slim",
        type=str_to_bool,
        help="Plot in slim mode",
    )
    parser.add_argument(
        "--plot_logy",
        type=str_to_bool,
        help="Plot in logy mode",
    )

    parsed_args = parser.parse_args(args)

    # Load configuration based on priority
    config_path = None
    if parsed_args.config_name:
        config_path = os.path.join("configs", f"{parsed_args.config_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"User-specified config file not found: {config_path}")
        with open(config_path, "r") as f:
            try:
                params = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {config_path}: {e}")
                raise
        print(f"Loaded config from: {config_path}")
    else:
        raise ValueError("Must provide config with -cn [config name without the .yaml]")

    assert params is not None

    # Override with command line arguments
    for key, value in vars(parsed_args).items():
        # Skip the config selection arguments
        if key in ["config_name", "environment"]:
            continue
        # Only override if the value was explicitly provided (not None)
        if value is not None:
            try:
                params[key] = value
                print(f"Overriding {key} with {value}")
            except Exception as e:
                print(f"Error overriding {key} with {value}: {e}")

    # handling of list arguments (in case they come as None)
    if "verbose" not in params or params["verbose"] is None:
        params["verbose"] = []

    if "which_plot_subopt" not in params or params["which_plot_subopt"] is None:
        params["which_plot_subopt"] = []

    if isinstance(params["which_plot_subopt"], str):
        params["which_plot_subopt"] = [params["which_plot_subopt"]]

    return params


# ==================================== #
