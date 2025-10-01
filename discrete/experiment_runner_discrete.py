"""
Contains experiment runner and all helpers/utils exclusively used by the continuous BRIDGE code
"""

import itertools
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import copy
import os
import pickle
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize

from common_utils import SHORT_ENV_NAMES

from .envs.star_mdp import StarMDP_with_random_flinging, StarMDP_with_random_staying
from .envs.gridworld import Gridworld
from .tabular_policy import TabularPolicy


mp.set_start_method("spawn", force=True)  # try 'fork' if you're on Unix, it's faster but may hang


EPS = np.finfo(np.float32).eps

ENV_N_STATES_ACTIONS = {
    "StarMDP": (5, 4),
    "StarMDP_staying": (5, 4),
    "Gridworld": (16, 4),
}


# ==== Experiment runner ==== #
def run_experiment_multiprocessing(params):
    seeds = list(range(params["N_experiments"]))
    results = []
    finished = 0
    maybe_generate_all_policies(SHORT_ENV_NAMES[params["env"]])
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_seed_mp, seed, params) for seed in seeds]
        for future in as_completed(futures):
            results.append(future.result())
            finished += 1
            print(
                f"finished {finished}/{params['N_experiments']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
    results.sort(key=lambda x: seeds.index(x[0]) if isinstance(x, tuple) and len(x) > 1 else 0)
    (
        seeds,
        metrics_per_seed,
        final_objs_per_seed,
        final_values_per_seed,
        bc_reward_per_seed,
        opt_reward_per_seed,
    ) = zip(*results)

    if params["N_experiments"] == 1:
        return (
            metrics_per_seed[0],
            final_objs_per_seed[0],
            final_values_per_seed[0],
            bc_reward_per_seed[0],
            opt_reward_per_seed[0],
        )
    else:
        return (
            list(metrics_per_seed),
            list(final_objs_per_seed),
            list(final_values_per_seed),
            np.mean(bc_reward_per_seed) if bc_reward_per_seed[0] is not None else None,
            np.mean(opt_reward_per_seed),
        )


def _run_single_seed_mp(seed, params):
    # setup env and embeddings
    print(f"... started seed {seed} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _set_random_seeds(seed)
    env_true, which_transition_model, optimal_policy_true, N_search_space_samples = (
        _setup_environment(params)
    )
    N_states = env_true.N_states
    N_actions = env_true.N_actions
    phi, d, B, kappa, lambda_param, eta = _setup_embeddings(params, N_states, N_actions)

    # (optional) offline learning
    # is executed in a BRIDGE run, or if baseline uses augmented ball (i.e. we use this confset to generate the baseline search space by diluting it with random policies)
    if params["do_offline_BC"] or params["baseline_search_space"] == "augmented_ball":
        offline_trajs, _ = generate_offline_trajectories(
            env_true, optimal_policy_true, n_samples=params["N_offline_trajs"]
        )
        confset_offline, bc_policy, env_learned_transitions = offline_learning(
            offline_trajs,
            env_true,
            params["episode_length"],
            params["delta_offline"],
            optimal_policy_true,
            N_search_space_samples,  # only needed for which_confset_construction_method=="rejection-sampling-from-sample"
            params["which_confset_construction_method"],
            which_transition_model,
            params["n_transition_model_epochs_offline"],
            params["offlineradius_formula"],
            params["offlineradius_override_value"],
            verbose=params["verbose"],
        )
        bc_reward = env_true.calc_analytic_discounted_reward_finitehorizon(bc_policy)
    else:  # we may skip BC. E.g. if running baseline and we set its search space to something other than 'augmented_ball=offline_confset+noise'
        bc_reward = None
        confset_offline = None
        bc_policy = None
        offline_trajs = None
        env_learned_transitions = None

    opt_reward = env_true.calc_analytic_discounted_reward_finitehorizon(optimal_policy_true)

    # ==== Online preference-based learning ====
    metrics, final_objs, final_values = online_learning(
        confset_offline,
        offline_trajs,
        env_learned_transitions,
        optimal_policy_true,
        env_true,
        phi,
        B,
        d,
        kappa,
        lambda_param,
        eta,
        which_transition_model,
        params,
    )
    return seed, metrics, final_objs, final_values, bc_reward, opt_reward


# ==== (end experiment runner) ==== #


# === Setup === #
def _setup_environment(params):
    N_search_space_samples = None
    if params["env"] == "StarMDP_with_random_flinging":
        env_true = StarMDP_with_random_flinging(
            discount_factor=0.99,
            episode_length=params["episode_length"],
            move_prob=params["env_move_prob"],
        )
        which_transition_model = "MLE"
    elif params["env"] == "StarMDP_with_random_staying":
        env_true = StarMDP_with_random_staying(
            discount_factor=0.99,
            episode_length=params["episode_length"],
            move_prob=params["env_move_prob"],
        )
        which_transition_model = "MLE"
    elif params["env"] == "Gridworld":
        env_true = Gridworld(
            width=4,
            height=4,
            episode_length=params["episode_length"],
            discount_factor=0.99,
            random_action_prob=1 - params["env_move_prob"],
        )
        which_transition_model = "MLP"
        N_search_space_samples = int(5e5)  # for augmented ball confset
        if params["which_confset_construction_method"] == "rejection-sampling-from-all":
            raise ValueError(
                "rejection-sampling-from-all not supported for Gridworld (policy space too big!). Use rejection-sampling-from-sample, or random_sample instead."
            )
    else:
        raise ValueError(f"Environment {params['env']} not supported")

    # ensure transitions are valid distribution
    env_true = sanity_check_transitions(env_true, fix=True)
    pi_optimal = env_true.get_lp_solution()

    return env_true, which_transition_model, pi_optimal, N_search_space_samples


def _setup_embeddings(params, N_states, N_actions):
    """Set up embedding functions and related parameters."""
    embedding_bounds = {
        "id_long": np.sqrt(2 * params["episode_length"]),
        "id_short": np.sqrt(2) * params["episode_length"],
        "state_counts": params["episode_length"],
        "final_state": 1,  # final_state is one-hot vector of size N_states, so norm=1
    }
    embedding_dims = {
        "id_long": params["episode_length"] * (N_states + N_actions),
        "id_short": N_states + N_actions,
        "state_counts": N_states,
        "final_state": N_states,
    }
    phi = create_embeddings(params["phi_name"], N_states, N_actions)
    d = embedding_dims[params["phi_name"]]
    B = embedding_bounds[params["phi_name"]]
    s = 1 / (1 + np.exp(-params["W"] * B))  # sigmoid(W*B)
    s_deriv = s * (1 - s)  # d/dx(sigmoid(W*B))
    kappa = 1 / s_deriv  # kappa = 1 / (d/dx(sigmoid(W*B)))
    lambda_param = B / kappa
    eta = 2 * params["W"] * B

    return phi, d, B, kappa, lambda_param, eta


def _set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(42 + seed)
    np.random.seed(42 + seed)
    torch.manual_seed(42 + seed)


# ==== (end setup) ==== #


# === Offline learning === #
def offline_learning(
    offline_trajs,
    env_true,
    episode_length,
    delta_offline,
    optimal_policy_true,
    N_search_space_samples,
    which_confset_construction_method,
    which_transition_model,
    n_transition_model_epochs,
    offlineradius_formula,
    offlineradius_override_value,
    verbose=[],
):
    """
    Perform offline learning using behavioral cloning and confidence set generation.

    Args:
        offline_trajs: List of offline trajectories
        env_true: True environment
        episode_length: Length of episodes
        delta_offline: Confidence parameter for offline radius
        overrides: Dict of parameter overrides
        optimal_policy_true: True optimal policy
        N_sampled_initial_policies: Number of policies to sample for confidence set
        which_confset_construction_method: 'rejection-sampling-[from-all, from-sample]'
        which_transition_model: 'MLE', 'linear_classifier', or 'MLP'
        n_transition_model_epochs: Number of epochs to train transition model
        verbose: List of verbosity options

    Returns tuple:
        confset_offline: confidence set of offline policies
        policy_BC_offline: policy obtained via behavioral cloning
        env_T_learned: learned transition model
    """
    N_states = env_true.N_states
    N_actions = env_true.N_actions

    if "loop-summary" in verbose or "full" in verbose:
        print("\n ----- starting OFFLINE part ----- ")

    offline_start_time = time.time()

    # learn transition model, sanity check it's a valid distribution (& fix if needed)
    env_T_learned, _, __ = train_transition_model_wrapper(
        offline_trajs, env_true, which_transition_model, n_transition_model_epochs, verbose
    )
    env_T_learned = sanity_check_transitions(
        env_T_learned, fix=True, verbose=verbose
    )  # sum(rows) must be 1
    if "loop-summary" in verbose or "full" in verbose:
        print(f"Transition model: {which_transition_model}")
    if "full" in verbose:
        print(f"MLE transitions (offline):\n{env_T_learned.transitions}")

    # learn behavioral cloning policy
    policy_BC_offline = train_tabular_BC_policy(
        offline_trajs,
        N_states,
        N_actions,
        init="random",
        n_epochs=10,
        lr=0.01,
        make_deterministic=True,
        verbose=verbose,
    )

    # calculate the radius of the offline confidence set
    offlineradius = calc_offlineradius(
        offline_trajs,
        N_states,
        N_actions,
        episode_length,
        delta_offline,
        formula_version=offlineradius_formula,
        aux_input=offlineradius_override_value,
        verbose=verbose,
    )
    if "full" in verbose:
        print(f"offlineradius: {offlineradius:.3f}")

    # generate confidence set
    if which_confset_construction_method == "rejection-sampling-from-all":
        confset_offline = generate_confidence_set_deterministic_via_rejection_sampling(
            policy_BC_offline,
            env_T_learned,
            episode_length,
            offlineradius,
            optimal_policy_true,
            N_search_space_samples=None,
            verbose=verbose,
        )
    elif which_confset_construction_method == "rejection-sampling-from-sample":
        confset_offline = generate_confidence_set_deterministic_via_rejection_sampling(
            policy_BC_offline,
            env_T_learned,
            episode_length,
            offlineradius,
            optimal_policy_true,
            N_search_space_samples=N_search_space_samples,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unknown confset construction method: {which_confset_construction_method}"
        )

    offline_runtime = time.time() - offline_start_time
    if "loop-summary" in verbose or "full" in verbose:
        print(f"--- offline runtime: {offline_runtime:.3f} seconds ---")
        print(f"Generated offline confset of size {len(confset_offline) if confset_offline else 0}")

    return confset_offline, policy_BC_offline, env_T_learned


# ==== (end offline learning) ==== #


# === Online learning === #
def online_learning(
    confset_offline,
    offline_trajs,
    env_BC_learned,
    expert_policy,
    env_true,
    phi,
    B,
    d,
    kappa,
    lambda_param,
    eta,
    which_transition_model,
    params,
):
    """
    Perform online learning using preference feedback and confidence sets.

    Main logic:
    for t=0, ..., N_iterations-1:
    1) train w_MLE and project it to W-ball
    2) refine online confset
    3) select policy pair (pi1, pi2) that maximizes uncertainty
    4) sample trajectories from (pi1, pi2)
    5) get oracle preference and add to buffer
    6) update V_t, N_ts and retrain transition model on all offline + online trajectories

    The start & end of the loop are slightly different. Start requires initialization and skipping some steps due to 0 online data,
    while end requires extra metrics calculation.
    """
    N_states = env_true.N_states
    N_actions = env_true.N_actions
    verbose = params["verbose"]
    online_start_time = _print_online_start_msg(verbose)
    metrics = setup_metrics()
    annotated_online_buffer, online_traj_list, online_policy_pairs = [], [], []
    if not params["do_offline_BC"]:
        confset_offline, env_BC_learned, offline_trajs = (
            _setup_initial_confidence_set_and_environment(
                expert_policy, confset_offline, env_true, params
            )
        )

    # ==== Loop t=0 ==== #
    t = 0  # any math formulas use "t_" in [1, ..., N_iterations].

    # 0.1) initialize w (can't train yet, no preference data at t=0)
    V_t, V_t_inv, N_ts, w_MLE_t, w_proj_t = _initialize_loop_variables(
        kappa, lambda_param, d, params["W"], offline_trajs, N_states, N_actions
    )

    # 0.2) precompute embeddings and bonuses
    precomputed_phi_bhats, gamma_t, Pi_t = _compute_online_embeds_bonuses_confset(
        env_BC_learned,
        None,  # env_true (unneeded at t=0)
        confset_offline,
        None,  # online_policy_pairs (don't have any yet at t=0)
        None,  # w_proj_t (unneeded at t=0)
        None,  # V_t_inv (unneeded at t=0)
        phi,
        N_ts,
        eta,
        kappa,
        B,
        d,
        lambda_param,
        t,
        params,
    )

    # |Pi_t| = 1 --> BC was enough to solve the problem so we terminate early
    if len(Pi_t) == 1:
        metrics, final_objs, final_values = _handle_loop0_termination(
            metrics,
            confset_offline,
            expert_policy,
            Pi_t,
            w_MLE_t,
            w_proj_t,
            env_BC_learned,
            online_start_time,
            params,
        )
        return metrics, final_objs, final_values
    # |Pi_t| > 1 --> do 1 online loop step
    else:
        # 0.3) select policy pair that maximizes uncertainty
        policy_pair_t, uncertainty_t = get_policy_pair_that_maximizes_uncertainty(
            Pi_t, precomputed_phi_bhats, gamma_t, V_t_inv, verbose
        )
        online_policy_pairs.append(policy_pair_t)
        _check_if_BC_policy_in_pair(confset_offline, policy_pair_t, verbose)

        # 0.4) generate rollouts & save to traj list
        traj_pairs_t = generate_policy_pair_rollouts(
            env_true, policy_pair_t[0], env_true, policy_pair_t[1], params["N_rollouts"]
        )
        for traj_pair in traj_pairs_t:
            online_traj_list.extend(traj_pair)
        all_trajs = offline_trajs + online_traj_list

        # 0.5) get oracle preference & add to buffer
        annotated_online_buffer.extend(
            annotate_buffer(traj_pairs_t, env_true, params["N_rollouts"])
        )

        # 0.6) update V_t, N_ts, transition model
        pi1_phi = precomputed_phi_bhats[hash(policy_pair_t[0].matrix.tobytes())].embed
        pi2_phi = precomputed_phi_bhats[hash(policy_pair_t[1].matrix.tobytes())].embed
        phi_diff = pi1_phi - pi2_phi
        V_t = V_t + np.outer(phi_diff, phi_diff)
        N_ts.append(calc_empirical_counts(online_traj_list, N_states, N_actions))
        env_learned_t, _, _ = train_transition_model_wrapper(
            all_trajs,
            env_true,
            which_transition_model,
            params["n_transition_model_epochs_online"],
            verbose,
        )
        env_learned_t = sanity_check_transitions(env_learned_t, fix=True)

    metrics = _loop0_metrics_and_prints(metrics, Pi_t, uncertainty_t, online_start_time, t, verbose)
    # ==== (end loop 0) ==== #

    # ==== Loop t=1, ..., N_iterations-1 ==== #
    for t in range(1, params["N_iterations"]):
        loop_start_time = _print_loop_start_msg(t, verbose)

        # t.1)train and project w
        w_MLE_t = learn_w_MLE(
            annotated_online_buffer,
            phi,
            d,
            w=None,
            w_initialization=params["w_initialization"],
            sigmoid_slope=params["w_sigmoid_slope"],
            W_norm=params["W"],
            lambda_param=lambda_param,
            n_epochs=params["w_MLE_epochs"],
            lr=0.01,
            verbose=verbose,
        )
        V_t_inv = np.linalg.inv(V_t)
        w_proj_t = project_w(
            w_MLE_t, params["W"], V_t_inv, annotated_online_buffer, phi, lambda_param, verbose
        )

        # t.2) precompute embeddings and bonuses
        precomputed_phi_bhats, gamma_t, Pi_t = _compute_online_embeds_bonuses_confset(
            env_learned_t,
            env_true,
            confset_offline,
            online_policy_pairs,
            w_proj_t,
            V_t_inv,
            phi,
            N_ts,
            eta,
            kappa,
            B,
            d,
            lambda_param,
            t,
            params,
        )

        # we repeat steps 3)-6) from loop 0
        if len(Pi_t) > 1:
            # t.3) select policy pair that maximizes uncertainty
            policy_pair_t, uncertainty_t = get_policy_pair_that_maximizes_uncertainty(
                Pi_t, precomputed_phi_bhats, gamma_t, V_t_inv, verbose=verbose
            )
            online_policy_pairs.append(policy_pair_t)
            _check_if_BC_policy_in_pair(confset_offline, policy_pair_t, verbose)

            # t.4) generate rollouts & save to traj list
            traj_pairs_t = generate_policy_pair_rollouts(
                env_true, policy_pair_t[0], env_true, policy_pair_t[1], params["N_rollouts"]
            )
            for traj_pair in traj_pairs_t:
                online_traj_list.extend(traj_pair)
            all_trajs = offline_trajs + online_traj_list

            # t.5) get oracle preference & add to buffer
            annotated_online_buffer.extend(
                annotate_buffer(traj_pairs_t, env_true, params["N_rollouts"])
            )

            # t.6) update V_t, N_ts, transition model
            pi1_phi = precomputed_phi_bhats[hash(policy_pair_t[0].matrix.tobytes())].embed
            pi2_phi = precomputed_phi_bhats[hash(policy_pair_t[1].matrix.tobytes())].embed
            phi_diff = pi1_phi - pi2_phi
            V_t = V_t + np.outer(phi_diff, phi_diff)
            N_ts.append(calc_empirical_counts(online_traj_list, N_states, N_actions))
            env_learned_t, _, _ = train_transition_model_wrapper(
                all_trajs,
                env_true,
                which_transition_model,
                params["n_transition_model_epochs_online"],
                verbose,
            )
            env_learned_t = sanity_check_transitions(env_learned_t, fix=True)

        metrics = _compute_and_log_online_iteration_metrics(
            metrics,
            confset_offline,
            Pi_t,
            uncertainty_t,
            env_true,
            w_proj_t,
            None,  # w_MLE (unneeded)
            None,  # env_learned (unneeded)
            phi,
            expert_policy,
            loop_start_time,
            t,
            verbose,
            is_final=False,
        )
        if len(Pi_t) == 1:
            print(f"Pi_t contains only 1 policy, terminating at t={t}/{params['N_iterations'] - 1}")
            break
        # ==== (end loop t) ==== #
    # ==== (end loop t=1, ..., N_iterations-1) ==== #

    # ==== final loop iteration T ==== #
    # T.1) train w_MLE -> w_proj one last time
    # (skip all other steps, straight to metrics)

    # T.1) train w_MLE
    w_MLE_final = learn_w_MLE(
        annotated_online_buffer,
        phi,
        d,
        w=None,
        w_initialization=params["w_initialization"],
        sigmoid_slope=params["w_sigmoid_slope"],
        W_norm=params["W"],
        lambda_param=lambda_param,
        n_epochs=params["w_MLE_epochs"],
        lr=0.01,
        verbose=verbose,
    )
    V_T_inv = np.linalg.inv(V_t)
    w_proj_final = project_w(
        w_MLE_final, params["W"], V_T_inv, annotated_online_buffer, phi, lambda_param, verbose
    )

    # final metrics
    metrics, final_objs, final_values = _compute_and_log_online_iteration_metrics(
        metrics,
        confset_offline,
        Pi_t,
        uncertainty_t,
        env_true,
        w_proj_final,
        w_MLE_final,
        env_learned_t,
        phi,
        expert_policy,
        online_start_time,
        t,
        verbose,
        is_final=True,
    )

    return metrics, final_objs, final_values


# ==== (end online learning) ==== #


# ==== Setup: Online learning ==== #
def _print_online_start_msg(verbose):
    if "loop-summary" in verbose or "full" in verbose:
        print("\n ----- starting ONLINE loop ----- ")
    return time.time()


def _setup_initial_confidence_set_and_environment(
    optimal_policy_true, confset_offline, env_true, params
):
    """if we didn't do offline BC (this is a baseline run), we need to setup the initial confidence set and env."""
    if params["baseline_search_space"] == "random_sample":
        confset_offline = [optimal_policy_true]
        confset_offline.extend(
            generate_random_tabular_policies(
                env_true.N_states,
                env_true.N_actions,
                N_policies=params["N_confset_size"],
                make_deterministic=True,
            )
        )
    # avoid kernel crash by checking size of policy space
    elif (
        params["baseline_search_space"] == "all_policies"
        and env_true.N_actions**env_true.N_states < 1e4
    ):
        confset_offline = []
        # first look for them on disk in saved_models/{env_name}/all_policies.pkl
        all_policies_path = (
            f"discrete/saved_policies/{SHORT_ENV_NAMES[params['env']]}_all_policies.pkl"
        )
        if os.path.exists(all_policies_path):
            with open(all_policies_path, "rb") as f:
                all_policies = pickle.load(f)
            if "full" in params["verbose"] or "online-confset" in params["verbose"]:
                print(f"loaded {len(all_policies)} policies from {all_policies_path}")
        else:
            # if not found on disk, generate all policies, then save to disk
            all_policies = generate_all_deterministic_stationary_policies(
                env_true.N_states, env_true.N_actions
            )
            if "full" in params["verbose"] or "online-confset" in params["verbose"]:
                print(
                    f"Generated all {env_true.N_actions**env_true.N_states} possible deterministic policies..."
                )
            os.makedirs(os.path.dirname(all_policies_path), exist_ok=True)
            with open(all_policies_path, "wb") as f:
                pickle.dump(all_policies, f)
        confset_offline.extend(all_policies)
    elif params["baseline_search_space"] == "augmented_ball":
        ## this branch: assumes we start with BRIDGE's ball from offline, then augment it with random policies
        ## to hit size N_search_space_size. Idea is that ball is decent, and random policies dilute it.
        num_to_be_filled = params["N_confset_size"] - len(confset_offline)
        if num_to_be_filled > 0:
            additional_random_policies = generate_random_tabular_policies(
                env_true.N_states,
                env_true.N_actions,
                N_policies=num_to_be_filled,
                make_deterministic=True,
            )
            confset_offline.extend(additional_random_policies)
            if "full" in params["verbose"] or "online-confset" in params["verbose"]:
                print(
                    f"Augmented ball with {num_to_be_filled} random policies to hit size {len(confset_offline)} (target {params['N_confset_size']})"
                )
        else:
            print(
                f"Warning: Search space creation (augmented ball): N_search_space {params['N_confset_size']} is smaller than H2-ball (current size {len(confset_offline)}). Decrease radius, or increase N_search_space."
            )
            # Remove excess policies to match N_confset_size
            confset_offline = confset_offline[: params["N_confset_size"]]
    else:
        raise ValueError("baseline_search_space invalid or MDP too big for all_policies")
    env_BC_learned = copy.deepcopy(env_true)
    env_BC_learned.transitions = (
        np.ones((env_true.N_actions, env_true.N_states, env_true.N_states)) / env_true.N_states
    )
    env_BC_learned = sanity_check_transitions(env_BC_learned, fix=True)
    offline_trajs = []
    return confset_offline, env_BC_learned, offline_trajs


def setup_metrics():
    metrics = {
        # logged at loops {1, ..., N-1, final=N} | metrics related to "best policy prediction", they all use the pref_w trained on dataset of loop t
        "regrets": [],
        "best_iteration_policy": [],
        "scores_best_iteration_policy": [],
        "scores_true_opt": [],
        "avg_rewards_best_iteration_policy": [],
        "avg_rewards_true_opt": [],
        # logged at loops {initial, 1, ..., N-1} | metrics related to in-loop calculations of loop t
        "pi_set_sizes": [],  # 0, 1...N-1
        "uncertainty_t": [],
        "iteration_times": [],
    }
    return metrics


def _initialize_loop_variables(kappa, lambda_param, d, W, offline_trajs, N_states, N_actions):
    V_t = kappa * lambda_param * np.eye(d)
    V_t_inv = np.linalg.inv(V_t)
    # empirical visitation counts: at each t, counts (s,a) visitations in offline+online trajs. is a growing list [N_offline, N_online0, N_online1, ...]
    N_ts = [calc_empirical_counts(offline_trajs, N_states, N_actions)]
    # w: no labeled data yet, so initialize random normal.
    w_MLE_t = np.random.randn(d)
    w_proj_t = w_MLE_t / np.linalg.norm(w_MLE_t) * W
    return V_t, V_t_inv, N_ts, w_MLE_t, w_proj_t


# ==== end setup: Online learning ==== #


# ==== Online learning: Loop 0 ==== #
def _handle_loop0_termination(
    metrics,
    confset_offline,
    optimal_policy_true,
    Pi_t,
    w_MLE_t,
    w_proj_t,
    env_BC_learned,
    online_start_time,
    params,
):
    """if BC was enough to solve the problem and |Pi_t| = 1, we terminate early"""
    if params["do_offline_BC"]:
        # check if BC policy is in Pi_t
        BC_policy_hash = hash(confset_offline[0].matrix.tobytes())
        opt_policy_hash = hash(optimal_policy_true.matrix.tobytes())
        pi_t_hashes = [hash(policy.matrix.tobytes()) for policy in Pi_t]
        print(
            f"len(Pi_t) = 1, terminating at t=0. Pi_t contains... BC_policy: {BC_policy_hash in pi_t_hashes}, opt_policy: {opt_policy_hash in pi_t_hashes}, BC == opt: {BC_policy_hash == opt_policy_hash}"
        )
    else:  # we didn't do offline BC, and yet still have Pi_t = 1? Something went wrong here
        raise ValueError("len(Pi_t) = 1, but we didn't do offline BC. Shouldn't happen!")

    # update metrics -- since we're returning early, pad metrics
    metrics, final_objs, final_values = initial_loop_earlystop_logging(
        metrics, Pi_t, w_MLE_t, w_proj_t, env_BC_learned, confset_offline, online_start_time
    )
    return metrics, final_objs, final_values


def _loop0_metrics_and_prints(metrics, Pi_t, uncertainty_t, online_start_time, t, verbose):
    metrics["pi_set_sizes"].append(len(Pi_t))
    metrics["uncertainty_t"].append(uncertainty_t)
    metrics["iteration_times"].append(time.time() - online_start_time)

    # optionally print loop summary
    if "loop-summary" in verbose or "full" in verbose:
        print(f"-- summary loop {t}:")
        print(f"  size of Pi_t: {len(Pi_t)}, uncertainty: {uncertainty_t:.3f}")
        print(" ----- ending 0-th loop ----- ")
    return metrics


# ==== (end loop 0) ==== #


# ==== Online learning: Loop t=1, ..., N_iterations-1 ==== #
def _print_loop_start_msg(t, verbose):
    if "loop-summary" in verbose or "full" in verbose:
        print(f"\n ----- starting loop {t} ----- ")
    return time.time()


def _compute_online_embeds_bonuses_confset(
    env_learned,  # 0: env_BC_learned, >0: env_learned_t
    env_true,  # 0: unneeded, >0: env_true
    confset_offline,
    online_policy_pairs,  # 0: None, >0: online_policy_pairs
    w_proj_t,  # 0: unneeded, >0: w_proj_t
    V_t_inv,  # 0: unneeded, >0: V_t_inv
    phi,
    N_ts,
    eta,
    kappa,
    B,
    d,
    lambda_param,
    t,
    params,
):
    N_states = env_learned.N_states
    N_actions = env_learned.N_actions
    precomputed_phi_bhats = precompute_phi_Bhats(
        N_states,
        N_actions,
        t,
        phi,
        confset_offline,
        env_learned,  # 0: env_BC_learned, >0: env_learned_t
        eta,
        params["delta_online"],
        params["episode_length"],
        N_ts,
        params["xi_formula"],
        n_embedding_samples=params["n_embedding_samples"],
    )

    # calculate gamma_t (optionally just use hardcoded value)
    if params["gamma_t_hardcoded_value"]:
        gamma_t = params["gamma_t_hardcoded_value"]
        if "full" in params["verbose"] or "online-confset" in params["verbose"]:
            print(f"overriding gamma_{t}: {gamma_t}")
    else:
        gamma_t = calc_gamma_t(
            t,
            kappa,
            lambda_param,
            B,
            params["W"],
            params["N_iterations"],
            d,
            params["delta_online"],
            eta,
            params["episode_length"],
            N_ts,
            params["xi_formula"],
            online_policy_pairs,  # 0: None, >0: online_policy_pairs
            env_learned,
            verbose=params["verbose"],
        )

    # calculate online confset Pi_t
    if t == 0:
        Pi_t = confset_offline
    else:
        Pi_t = calc_online_confset_t(
            confset_offline,
            precomputed_phi_bhats,
            w_proj_t,
            gamma_t,
            V_t_inv,
            params["online_confset_bonus_multiplier"],
            phi,
            env_true,
            params["verbose"],
        )
    return precomputed_phi_bhats, gamma_t, Pi_t


# === Embeddings === #
def create_embeddings(phi_name, N_states, N_actions):
    """
    Create embedding function based on specified type.
    """

    if phi_name == "id_long":
        return _create_id_long_embedding(N_states, N_actions)
    elif phi_name == "id_short":
        return _create_id_short_embedding(N_states, N_actions)
    elif phi_name == "state_counts":
        return _create_state_counts_embedding(N_states, N_actions)
    elif phi_name == "final_state":
        return _create_final_state_embedding(N_states, N_actions)
    else:
        raise ValueError(f"Unknown embedding type: {phi_name}")


def _create_id_long_embedding(N_states, N_actions):
    """Embeds a trajectory into a 1D, one-hot vector, ignoring rewards.
    Trajectory [s0,a0,r0,s1,a1,r1,...] becomes a concatenation of
    one-hot encodings of s0, a0, s1, a1, etc.

    Args:
    traj: A trajectory in the form [s0,a0,r0,s1,a1,r1,...]

    Returns:
        A 1D numpy array with concatenated one-hot vectors, length episode_length * (N_states + N_actions)
        containing 2 * episode_length 1s (two per state-action pair), rest 0s.
    """

    def phi_id_long(traj):
        result = np.array([], dtype=int)
        for t in range(len(traj)):
            if t % 3 == 0:  # states at indices 0,3,6..
                one_hot = np.zeros(N_states)
                one_hot[traj[t]] = 1
                result = np.append(result, one_hot)
            elif t % 3 == 1:  # actions at indices 1,4,7..
                one_hot = np.zeros(N_actions)
                one_hot[traj[t]] = 1
                result = np.append(result, one_hot)
        # skip rewards at indices 2,5,8..
        return result

    phi_id_long.name = "id_long"
    return phi_id_long


def _create_id_short_embedding(N_states, N_actions):
    """
    Embeds a trajectory into a 1D, one-hot vector, ignoring rewards.
    Trajectory [s0,a0,r0,s1,a1,r1,...] (list-like) where s_t in {0,...,N_states-1} and a_t in {0,...,N_actions-1}
    becomes sum_{t=1...H} [one-hot encoding s_t, one-hot encoding a_t].

    Returns:
        A 1D numpy array with concatenated one-hot vectors, length N_states + N_actions.
        2-norm bound B=sqrt(2)*H, since worst-case have ||(H,0,...,0,H,0,...,0)||_2 = sqrt(2*H^2)
    """

    def phi_id_short(traj):
        result_states = np.zeros(N_states)
        result_actions = np.zeros(N_actions)
        for t in range(len(traj)):
            if t % 3 == 0:  # even: state
                result_states[traj[t]] += 1
            elif t % 3 == 1:  # odd: action
                result_actions[traj[t]] += 1
            # skip rewards at indices 2, 5, 8, ..

        return np.concatenate([result_states, result_actions])

    phi_id_short.name = "id_short"
    return phi_id_short


def _create_state_counts_embedding(N_states, N_actions):
    """
    Embeds a trajectory by counting states.

    Args:
        traj: A trajectory in the form [s0,a0,r0,s1,a1,r1,...]

    Returns:
        A 1D numpy array of length N_states where each element represents
        the probability (frequency) of being in that state during the trajectory.
    """

    def phi_state_counts(traj):
        result = np.zeros(N_states)

        for t in range(len(traj)):
            if t % 3 == 0:  # states at indices 0,3,6...
                result[traj[t]] += 1
        return result

    phi_state_counts.name = "state_counts"
    return phi_state_counts


def _create_final_state_embedding(N_states, N_actions):
    """
    Embeds a trajectory by returning the final state as a one-hot vector.
    """

    def phi_final_state(traj):
        result = np.zeros(N_states)
        result[traj[-3]] = 1  # -3 because it's [..., s_H, a_H, r_H]
        return result

    phi_final_state.name = "final_state"
    return phi_final_state


# ==== (end embeddings) ==== #


# ==== Helpers: offline learning ==== #
def calc_offlineradius(
    offline_trajs,
    N_states,
    N_actions,
    episode_length,
    delta,
    formula_version="full",
    aux_input=None,
    verbose=[],
):
    """
    Calculates the offline radius for the confidence set when using behavioral cloning.
    Formula: alpha/sqrt(N) + beta/sqrt(N) * offlineradius_outerbracket. Alpha and beta are MDP-specific,
    and the bracket term represents uncertainty in the transition estimate.

    Args:
        offline_trajs: List of offline trajectories
        N_states: Number of states
        N_actions: Number of actions
        episode_length: Length of episodes
        delta: Confidence parameter for offline radius
        formula_version: Version of the formula to use
        aux_input: Auxiliary input for the formula
        verbose: Verbosity options

    Returns:
        Offline radius

    Formula versions:
    - 'full': full formula -> alpha/sqrt(N) + beta/sqrt(N) * outer_bracket_term
    - 'ignore_bracket': ignore the bracket term (set=1) -> (alpha+beta)/sqrt(N)
    - 'only_alpha': only use alpha term -> alpha/sqrt(N)
    - 'hardcode_radius_scaled': -> (aux_input)/sqrt(N)
    - 'hardcode_radius': -> aux_input
    """
    N_offline_trajs = len(offline_trajs)

    # calculate state-action visitation frequencies
    dhat_t = np.zeros((episode_length, N_states, N_actions))
    ## calculate this: dhat_t(s,a) = 1/N_offline_trajs * sum_{i=1}^N_offline_trajs * (1_{s_t = s and a_t = a})
    for t in range(episode_length):
        for s in range(N_states):
            for a in range(N_actions):
                dhat_t[t, s, a] = (
                    1
                    / N_offline_trajs
                    * sum(
                        [
                            1
                            for traj in offline_trajs
                            if traj[t * 3] == s and traj[t * 3 + 1] == a  # 1_{s_t = s and a_t = a}
                        ]
                    )
                )

    # calculate smallest nonzero visitation probability under optimal policy pi* and true model P*. here we approximate it with offline trajectories.
    gamma_min = np.min(dhat_t[dhat_t > 0])  # if np.any(dhat_t > 0) else 1e-6

    if "full" in verbose or "radius-calc" in verbose:
        print(f"gamma_min (coverage): {gamma_min:.6f}")

    # Calculate radius components
    deltahalf = delta / 2
    alpha = 2 * np.sqrt(N_states * np.log(N_actions / deltahalf))
    beta = 2 * np.sqrt(
        (N_states**2) * N_actions * np.log(N_offline_trajs * episode_length / deltahalf)
    )

    inner_bracket_term = 1 + (2 * alpha / (gamma_min * np.sqrt(N_offline_trajs)))
    outer_bracket_term = 1 + np.sqrt(episode_length * inner_bracket_term)

    alphaterm = alpha / np.sqrt(N_offline_trajs)
    betaterm = beta / np.sqrt(N_offline_trajs)

    if "full" in verbose or "radius-calc" in verbose or "offline-confset" in verbose:
        print(f"alpha/sqrt(N): {alphaterm:.3f}, beta/sqrt(N): {betaterm:.3f}")
        print(
            f"outer_bracket_term: {outer_bracket_term:.3f}, inner_bracket_term: {inner_bracket_term:.3f}"
        )

    formula_calculations = {
        "full": alphaterm + betaterm * outer_bracket_term,
        "ignore_bracket": alphaterm + betaterm,
        "only_alpha": alphaterm,
        "hardcode_radius_scaled": aux_input / np.sqrt(N_offline_trajs),
        "hardcode_radius": aux_input,
    }

    if formula_version in formula_calculations:
        return formula_calculations[formula_version]
    else:
        raise ValueError(f"Unknown formula version: {formula_version}. See docstring for options.")


def generate_confidence_set_deterministic_via_rejection_sampling(
    policy_BC,
    env_emp,
    episode_length,
    offlineradius,
    optimal_policy_true,
    N_search_space_samples=None,
    verbose=[],
):
    """generates an offline confidence set of deterministic policies by rejection sampling.
    first generates ALL possible policies (this is intractable for anything != StarMDP!!!)
    then calculate hellinger distance for each policy (choice of approximation via 'local-avg'
    or exact via 'bhattacharyya'), only include candidate policies with H2-distance <= radius^2."""
    start_time = time.time()

    if hasattr(policy_BC.matrix, "detach"):
        policy_BC = TabularPolicy(policy_BC.matrix.detach().numpy())

    confidence_set = []
    N_states, N_actions = env_emp.N_states, env_emp.N_actions
    ## generate ALL possible policies.
    ## policies are matrices of size N_states x N_actions, each row one-hot for deterministic policy
    ## so to generate ALL of them:
    ## generate all possible action combinations (one action per state)
    ## this gives us N_actions**N_states total combinations.
    ## then check if H2-distance is <= radius^2.

    true_dist_opt_MLE = _calc_squared_hellinger_dist(
        policy_BC,
        env_emp.transitions,
        optimal_policy_true,
        env_emp.transitions,
        env_emp.initial_state_distribution,
        episode_length,
    )
    if true_dist_opt_MLE > offlineradius**2:
        print(
            f"WARNING: (confset gen.) optimal policy not in confset! true H2-dist: {true_dist_opt_MLE} > rad^2: {offlineradius**2}"
        )

    # if N_samples is None, generate all possible policies
    if N_search_space_samples is None:
        total_policies = N_actions**N_states
        if "offline-confset" in verbose:
            print(f"Confset: generating all {total_policies} possible deterministic policies...")
        action_combinations = itertools.product(range(N_actions), repeat=N_states)
        for actions in action_combinations:
            candidate_policy_matrix = np.zeros((N_states, N_actions))
            for state, action in enumerate(actions):
                candidate_policy_matrix[state, action] = 1  # one-hot encoded action
            candidate_policy = TabularPolicy(candidate_policy_matrix)

            squared_hellinger_dist = _calc_squared_hellinger_dist(
                policy_BC,
                env_emp.transitions,
                candidate_policy,
                env_emp.transitions,
                env_emp.initial_state_distribution,
                episode_length,
            )

            if squared_hellinger_dist <= offlineradius**2:
                confidence_set.append(candidate_policy)
    elif offlineradius < 1:
        if "offline-confset" in verbose:
            print(
                f"Confset: generating {N_search_space_samples} random policies, then rejection sampling them."
            )
        # generate N_samples many random policies, then rejection sample them (exactly)
        policies_sample = generate_random_tabular_policies(
            N_states,
            N_actions,
            N_policies=N_search_space_samples,
            make_deterministic=True,
        )
        for policy in policies_sample:
            squared_hellinger_dist = _calc_squared_hellinger_dist(
                policy_BC,
                env_emp.transitions,
                policy,
                env_emp.transitions,
                env_emp.initial_state_distribution,
                episode_length,
            )
            if squared_hellinger_dist <= offlineradius**2:
                confidence_set.append(policy)

        # add the MLE policy, because it's in the confset by definition.
        confidence_set.append(policy_BC)
        # add the optimal policy, if its bhatta dist is <= radius^2.
        if true_dist_opt_MLE <= offlineradius**2:
            confidence_set.append(optimal_policy_true)

    else:
        raise ValueError(
            "Trying to rejection sample from large sample (suggesting big MDP) but radius >= 1. Set radius < 1 to filter large sample to something manageable."
        )

    if "offline-confset" in verbose:
        print(
            f"Confset: generation success, {len(confidence_set)} deterministic policies in {time.time() - start_time:.2f} seconds"
        )

    return confidence_set


def approx_d_pi_BC(offline_trajs, N_states):
    """Calculates the stationary distribution of policy_BC via visitation count of offline trajs.

    Trajectories are of form [s0, a0, r0, s1, a1, r1, ..., sH, aH, rH].
    """
    if not offline_trajs:
        raise ValueError("Offline trajectories are empty")

    d_pi_BC = np.zeros(N_states)
    total_visits = len(offline_trajs) * (len(offline_trajs[0]) // 3)  # N_trajs * episode_length

    for traj in offline_trajs:
        for state in traj[::3]:
            d_pi_BC[state] += 1

    d_pi_BC /= total_visits  # normalize to get distribution
    return d_pi_BC


def generate_offline_trajectories(env, policy, n_samples, verbose=[]):
    """
    Generate offline trajectories by rolling out a policy in an environment.

    Args:
        env: The environment to roll out in
        policy: The policy to roll out
        n_samples: Number of trajectories to generate

    Returns:
        offline_trajs: List of trajectories, each traj [s0,a0,r0,s1,a1,r1,...]
        unique_trajs: Set of unique trajectories (converted to int)
    """
    offline_trajs = []
    for _ in range(n_samples):
        offline_trajs.append(rollout_policy_in_env(env, policy))

    def traj_to_int(traj):
        return [int(x) if not isinstance(x, int) else x for x in traj]

    # get only unique trajectories
    unique_trajs = set(tuple(traj_to_int(traj)) for traj in offline_trajs)

    if "full" in verbose or "offline-trajs" in verbose:
        print(f"Number of unique trajectories: {len(set(tuple(traj) for traj in offline_trajs))}")
        print(f"Unique trajectories: {unique_trajs}")

    return offline_trajs, unique_trajs


def _calc_squared_hellinger_dist(
    policy1,
    t_matrix1,
    policy2,
    t_matrix2,
    d0,  # initial dist, 1-d vec of N_states
    H,  # episode length
    verbose=[],
) -> float:
    """
    Calculates the squared Hellinger distance between two trajectory distributions using a trick with Bhattacharyya coefficient.

    ASSUMES: finite MDP. deterministic, stationary policies.

    The distributions are defined by (policy, transition_model) pairs. This function
    avoids creating the full trajectory distribution vector by using a dynamic
    programming approach:

    Args:
        policy1, policy2: First/second policy objects. Must have a .matrix attribute, which is a Tensor of shape (N_states, N_actions).
        t_matrix1, t_matrix2: The first/second transition model's matrix. Tensor of shape (N_actions, N_states, N_states).
        d0: The initial state distribution vector. Tensor of shape (N_states,).
        H: The episode length (int).

    Algo:
        B_0 = d0
        for t in range(H):
            M_t = sqrt(T1(s'|s,pi1(s)) * T2(s'|s,pi2(s))) * IS_TRUE(pi1(s)==pi2(s))
            B_t+1 = M_t @ B_t
        NOTE: if T, pi deterministic then M_t is constant and we just have
        B_H = (M**H) @ B_0.

    Returns:
        Squared Hellinger distance, a float between 0 and 1
    """
    p1_matrix = policy1.matrix
    p2_matrix = policy2.matrix

    # validate inputs
    if not (
        p1_matrix.shape == p2_matrix.shape
        and t_matrix1.shape == t_matrix2.shape
        and p1_matrix.shape[0] == t_matrix1.shape[1]
        and p1_matrix.shape[1] == t_matrix1.shape[0]
    ):
        raise ValueError("Dimension mismatch between policies and transition models.")
    if not (len(d0.shape) == 1 and d0.shape[0] == p1_matrix.shape[0]):
        raise ValueError("Dimension mismatch for initial state distribution d0.")

    N_states = p1_matrix.shape[0]

    # pre-calc deterministic actions for each policy
    actions1 = np.argmax(p1_matrix, axis=1)
    actions2 = np.argmax(p2_matrix, axis=1)

    # --- dynamic programming for Bhattacharyya coefficient ---
    # Initialize B_0(s) = d0(s), initial state dist vector
    # B_t(s) is the sum of sqrt-probabilities of all trajectories of length t that end in state s
    B = np.copy(d0)

    if "full" in verbose:
        print(f"B_0: {B}")

    ## since have deterministic policies and transitions: the M_t matrix is constant!
    M_t = np.zeros((N_states, N_states))
    # for each starting state 's_from', calculate the transitions.
    # this corresponds to filling the columns of M_t. if pi1(s) != pi2(s),
    # the entire column of M_t stays zero, M[:,s]==0.
    for s_from in range(N_states):
        a1 = actions1[s_from]
        a2 = actions2[s_from]
        if a1 == a2:
            t1_probs = t_matrix1[a1, s_from, :]
            t2_probs = t_matrix2[a1, s_from, :]
            sqrt_joint_probs = np.sqrt(t1_probs * t2_probs)
            M_t[:, s_from] = sqrt_joint_probs

    # iterate for H timesteps. if we didn't have deterministic policies & transitions,
    # this loop would redefine M_t.
    # for _ in range(H):
    #     print(f"Step {_}->{_ + 1}: B_{_ + 1} = M_{_} * B_{_} =\n{M_t} * {B} =\n{M_t @ B}")
    #     B = M_t @ B  # B_t+1 = M_t * B_t

    M_H = np.linalg.matrix_power(M_t, H)
    B = M_H @ B

    if "full" in verbose:
        print(f"matrix M:\n{M_t}")
        print(f"M^H:\n{np.linalg.matrix_power(M_t, H)}")
        print(f"B = M^H @ B_0:\n{B}")

    bhattacharyya_coefficient = np.sum(B)  # sum over all states
    if "full" in verbose:
        print(f"Bhattacharyya coefficient: {bhattacharyya_coefficient}")
        print(f"returning max(0, 1-BhatC) = {max(0.0, 1.0 - bhattacharyya_coefficient)}")
    squared_h_distance = max(0.0, 1.0 - bhattacharyya_coefficient)  # clip to handle FP inaccuracies

    return squared_h_distance


# ==== (end helpers: offline learning) ==== #


# ==== Helpers: online learning ==== #
class PrecomputedPhiBhats:
    def __init__(self, embed, bhat_online_confset, bhat_uncertainty_score):
        self.embed = embed
        self.bhat_online_confset = bhat_online_confset
        self.bhat_uncertainty_score = bhat_uncertainty_score


def precompute_phi_Bhats(
    N_states,
    N_actions,
    t,
    phi_func,
    confset_offline,
    env,
    eta,
    delta_online,
    episode_length,
    N_ts,
    xi_formula,
    n_embedding_samples=100,
):
    """returns precomputed values for all policies in the offline confidence set,
    as a dictionary with policy hash as key.
    delta1: for online confset Pi_t
    delta2: for the uncertainty score (=delta_param)
    Access e.g. policy_values[policy_hash].embed"""
    policy_values = {}
    delta_online_confset = max(delta_online / (2 * (N_actions**N_states)), 1e-10)

    for policy in confset_offline:
        policy_hash = hash(policy.matrix.tobytes())  # assumes policy matrix is np.ndarray

        sampled_trajs = []
        for _ in range(n_embedding_samples):
            sampled_trajs.append(rollout_policy_in_env(env, policy))

        values = PrecomputedPhiBhats(
            # phi(policy)
            _get_policy_embedding(
                phi_func,
                policy,
                env,
                n_trajectories=n_embedding_samples,  # not needed b/c we pre-compute the traj's above
                provided_trajs=sampled_trajs,
            ),
            # Bhat(policy, eta, delta/(2A^S)) for online confset condition
            _calc_Bhat(
                t,
                policy,
                env,
                eta,
                delta_online_confset,
                episode_length,
                N_ts,
                xi_formula,
                sampled_trajs,
                n_trajectories=n_embedding_samples,  # not needed b/c we pre-compute the traj's above
            ),
            # Bhat(policy, eta, delta) for uncertainty score
            _calc_Bhat(
                t,
                policy,
                env,
                eta,
                delta_online,
                episode_length,
                N_ts,
                xi_formula,
                sampled_trajs,
                n_trajectories=n_embedding_samples,  # not needed b/c we pre-compute the traj's above
            ),
        )
        policy_values[policy_hash] = values
    return policy_values


def _get_policy_embedding(phi_func, policy, env, n_trajectories=1000, provided_trajs=None):
    """estimates ϕ(policy) = E_policy[ϕ(traj)]"""
    if provided_trajs is not None:
        return np.mean([phi_func(traj) for traj in provided_trajs], axis=0)
    else:
        embeddings = []
        for _ in range(n_trajectories):
            traj = rollout_policy_in_env(env, policy)
            embeddings.append(phi_func(traj))
        return torch.stack(embeddings, axis=0).mean(0)


def _calc_Bhat(
    t,  # outer loop idx: 0, ..., N_iterations-1
    policy,
    env,
    eta,
    delta,
    episode_length,
    N_ts,
    xi_formula,
    provided_trajs=None,
    n_trajectories=100,
):
    """calculates following formula:
    Bhat = Expectation_{traj [s_0, a_0, s_1, a_1, ...] under policy and transition_model, and initial state determined from env.set}
    of sum_{h=1}^{episode_length-1} xi(t, s_h, a_h, eta, delta, episode_length, N_states, N_actions)
    where:
    - transition_model: learned model at time t

    If trajectories are provided in a list, use those to approx. the expectation.
    Otherwise, generate n_trajectories."""
    N_states, N_actions = env.N_states, env.N_actions
    total_xi_sum = 0
    # process provided trajectories if available
    if provided_trajs is not None:
        for traj in provided_trajs:
            xi_sum = 0
            for h in range(0, len(traj) - 3, 3):
                s_h = traj[h]  # state at time h
                a_h = traj[h + 1]  # action at time h
                xi_sum += _calc_xi(
                    t,
                    s_h,
                    a_h,
                    eta,
                    delta,
                    episode_length,
                    N_states,
                    N_actions,
                    N_ts,
                    xi_formula,
                )
            total_xi_sum += xi_sum
        return total_xi_sum / len(provided_trajs)

    # generate and process trajectories on the fly if none provided
    else:
        for _ in range(n_trajectories):
            traj = rollout_policy_in_env(env, policy)
            xi_sum = 0
            for h in range(0, len(traj) - 3, 3):
                s_h = traj[h]  # state at time h
                a_h = traj[h + 1]  # action at time h
                xi_sum += _calc_xi(
                    t,
                    s_h,
                    a_h,
                    eta,
                    delta,
                    episode_length,
                    N_states,
                    N_actions,
                    N_ts,
                    xi_formula,
                )
            total_xi_sum += xi_sum
        return total_xi_sum / n_trajectories


def _calc_xi(t, s, a, eta, delta, episode_length, N_states, N_actions, N_ts, xi_formula):
    """Calculates xi_t,s,a(eta,delta), the state-action level uncertainty term that flows into the empirical bonus Bhat
    according to either of the following formulas (choose via xi_formula):
    'full': min(2*eta, 4*eta*sqrt(U/(N_ts[0][s,a] + N_ts[t][s,a])))
    'smaller_start': min(log(N_states*N_actions), sqrt(log(N_states*N_actions)/visitations))

    where U = episode_length*log(N_states*N_actions)+log(6*log(N_ts[0][s,a] + N_ts[t][s,a]))-log(delta).

    Note:
    - The 'smaller_start' formula is meant for low N_offline. It has the same scaling in N as 'full', but
    starts off smaller.
    - N_ts is a list, each entry contains a matrix of counts N_t(s,a)
    - N_1 contains the offline counts. N_ts[t] contains the ONLINE counts up to time t.

    """
    safe_delta = max(delta, EPS * 10)  # delta could be VERY small
    # splitting the outer log into two for numerical stability
    if t == 0:
        visitation_sum = N_ts[0][s, a]
    else:
        visitation_sum = N_ts[0][s, a] + N_ts[t][s, a]

    if (
        xi_formula == "smaller_start"
    ):  # replace formula by something with same scaling in N, but starts smaller (in low N).
        if visitation_sum == 0 or visitation_sum == 1:
            return np.log(N_states * N_actions)
        else:
            res = min(
                np.log(N_states * N_actions), np.sqrt(np.log(N_states * N_actions) / visitation_sum)
            )
            return res  # line above used to be: / np.log(visitation_sum)

    elif xi_formula == "full":
        if visitation_sum == 0:  # numerator->-infty, denominator->0, so res->2*eta
            res = 2 * eta
            return res
        else:
            U = (
                episode_length * np.log(N_states * N_actions)
                + np.log(6 * np.log(visitation_sum))
                - np.log(safe_delta)
            )
            res = min(2 * eta, 4 * eta * np.sqrt(U / visitation_sum))
            return res

    else:
        raise ValueError(f"Invalid xi_formula: {xi_formula}. options: 'full', 'smaller_start'")


def calc_gamma_t(
    t,  # outer loop idx: 0, ..., N_iterations-1
    kappa,
    lambda_param,
    B,
    W,
    N_iterations,
    d,
    delta_param,
    eta,
    episode_length,
    N_ts,
    xi_formula,
    policy_pairs,
    env_learned,
    verbose=[],
):
    """computes the gamma_t according to the following formula:
    gamma_t = sqrt(2)*(4*kappa*beta_t(delta) + alpha_{d,N_iterations}(delta)) + 1/t + 2*sqrt(U)
    where
        U = sum_{i=1}^{t-1} Bhat(t, pi1_i, env, eta, delta, episode_length)**2 + Bhat(t, pi2_i, env, eta, delta, episode_length)**2
        pi1_i, pi2_i are the policies picked in loop iteration i, which we obtain as a list [pi1_i, pi2_i] from policy_pairs[i] input (usually online_policy_pairs but in the 1st iteration it's offline_policy_pairs)

    Note: env must use learned transitions at t (not true!)
    Note: has to sample new trajs, not precomputing those b/c amount of policy pairs could be <<< confidence set size so unsure if worth (calc'ing extra Bhats for those in confset but not policy_pairs).
    """
    N_states, N_actions = env_learned.N_states, env_learned.N_actions
    t_formula = t + 1  # for formulas, t=1, ..., N_iterations.
    epsilon = 1 / (t_formula**2 * kappa * lambda_param + 4 * B**2 * t_formula**3)
    if epsilon < 1e-10 and ("full" in verbose or "warnings" in verbose):
        print("gamma_calc: epsilon is too small, hitting safety floor")
    delta_prime = delta_param / ((1 + 4 * W) / max(epsilon, 1e-10)) ** d  # add safety to prevent /0
    alpha_d_T = (
        20 * B * W * np.sqrt(d * np.log((N_iterations * (1 + 2 * N_iterations)) / delta_param))
    )
    beta_t = np.sqrt(lambda_param) * W + np.sqrt(
        np.log(1 / delta_param)
        + 2 * d * np.log(1 + (t_formula * B) / (kappa * lambda_param * delta_param))
    )
    Bhat1s = []
    Bhat2s = []
    if policy_pairs is not None:
        for l_, pair in enumerate(policy_pairs):
            pi1, pi2 = pair[0], pair[1]
            delta_primeprime = max(
                delta_prime / (8 * (t_formula**3) * (N_actions**N_states)), 1e-10
            )
            Bhat1s.append(
                _calc_Bhat(
                    l_,
                    pi1,
                    env_learned,
                    eta,
                    delta_primeprime,
                    episode_length,
                    N_ts,
                    xi_formula,
                    provided_trajs=None,
                    n_trajectories=100,
                )
                ** 2
            )
            Bhat2s.append(
                _calc_Bhat(
                    l_,
                    pi2,
                    env_learned,
                    eta,
                    delta_primeprime,
                    episode_length,
                    N_ts,
                    xi_formula,
                    provided_trajs=None,
                    n_trajectories=100,
                )
                ** 2
            )
        B_term = 2 * np.sqrt(np.sum(Bhat1s) + np.sum(Bhat2s))
    else:  # iteration t=0
        B_term = 0
    alpha_beta_term = np.sqrt(2) * (4 * kappa * beta_t + alpha_d_T)
    gamma_t = alpha_beta_term + 1 / t_formula + B_term
    if "full" in verbose:
        print("calc gamma_t:")
        print(f"  epsilon: {epsilon:.5f}")
        print(f"  alpha_d_T: {alpha_d_T:.3f}")
        print(f"  beta_t: {beta_t:.3f}, 4*kappa*beta_t: {4 * kappa * beta_t:.3f}")
        print(f"  B_term: {B_term:.3f}")
        print(
            f"  gamma_t: {gamma_t:.3f} = alpha&beta term {alpha_beta_term:.3f} + 1/t {1 / t_formula:.3f} + B_term {B_term:.3f}"
        )
    return gamma_t


def calc_empirical_counts(traj_list, N_states, N_actions):
    """
    Get empirical state-action counts from trajectories: N_t(s,a) = number of times we've seen (s,a) up to time t.

    Each trajectory in traj_list is [s0, a0, r0, s1, a1, r1, ...].
    """
    N_t = np.zeros((N_states, N_actions), dtype=np.int32)
    for traj in traj_list:
        for i in range(0, len(traj), 3):
            s, a, _ = traj[i], traj[i + 1], traj[i + 2]
            N_t[s, a] += 1
    return N_t


def calc_online_confset_t(
    confset_offline,
    precomputed_phi_bhats,
    w_proj_t,
    gamma_t,
    V_t_inv,
    online_confset_bonus_multiplier,  # default should be 1
    phi_func=None,
    env=None,
    verbose=[],
):
    """calculates Pi_t, the online confset at time t
    it's a subset of the offline confset, containing all policies that fulfil the condition:
    for all other policies pi' in the offline confset, it holds that
    <(phi(pi)-phi(pi')), w> + gamma_t * ||phi(pi)-phi(pi')||_V^(-1) + bhat_online_confset(pi) + bhat_online_confset(pi') >= 0
    """
    calc_Pi_t_time = time.time()
    Pi_t = []

    min_estimate_win_probability_all_pi = np.inf
    avg_estimate_winprob_size = 0
    avg_B_term_size = 0
    avg_norm_size = 0
    avg_sum_term = 0
    tot_num_calcs = 0

    for i, pi in enumerate(confset_offline):
        pi_hash = hash(pi.matrix.tobytes())
        pi_phi = precomputed_phi_bhats[pi_hash].embed

        pi_bhat_online_confset = precomputed_phi_bhats[pi_hash].bhat_online_confset
        all_conditions_satisfied = True

        for j, pi2 in enumerate(confset_offline):
            pi2_hash = hash(pi2.matrix.tobytes())
            pi2_phi = precomputed_phi_bhats[pi2_hash].embed
            pi2_bhat_online_confset = precomputed_phi_bhats[pi2_hash].bhat_online_confset
            estimate_winprob = np.dot(pi_phi - pi2_phi, w_proj_t)
            norm_term = mahalanobis(pi_phi, pi2_phi, V_t_inv)
            uncertainty_w_estimate = gamma_t * norm_term
            sum_term = (
                estimate_winprob
                + uncertainty_w_estimate
                + online_confset_bonus_multiplier
                * (pi_bhat_online_confset + pi2_bhat_online_confset)
            )

            if estimate_winprob < min_estimate_win_probability_all_pi:
                min_estimate_win_probability_all_pi = estimate_winprob
            B_term_size = pi_bhat_online_confset + pi2_bhat_online_confset
            avg_estimate_winprob_size += estimate_winprob
            avg_norm_size += norm_term
            avg_B_term_size += B_term_size
            avg_sum_term += sum_term
            tot_num_calcs += 1
            if sum_term < 0:
                all_conditions_satisfied = False
                break

        if all_conditions_satisfied:
            Pi_t.append(pi)

    avg_estimate_winprob_size /= tot_num_calcs
    avg_norm_size /= tot_num_calcs
    avg_B_term_size /= tot_num_calcs
    avg_sum_term /= tot_num_calcs
    if "full" in verbose or "online-confset" in verbose:
        print(" -- confset --")
        print(
            f"  formula: <estimate_winprob> ({avg_estimate_winprob_size:.3f}) + gamma_t ({gamma_t:.2f}) * norm-term ({avg_norm_size:.2f}) + bonus_multiplier ({online_confset_bonus_multiplier:.3f}) * B_term ({avg_B_term_size:.3f})"
        )
        print("  policy deleted if above <0")
        print(f"  avg estimate winprob size: {avg_estimate_winprob_size:.3f}")
        print(f"  avg norm size: {avg_norm_size:.3f}")
        print(f"  avg B term size: {avg_B_term_size:.3f}")
        print(f"  avg sum term: {avg_sum_term:.3f}")
        print(f"  min win prob: {min_estimate_win_probability_all_pi:.3f} (all pi)")

    # Check if MLE (or optimal, if override or no BC done) policy is in Pi_t
    policy_BC_or_opt = confset_offline[0]
    bc_or_opt_in_pi_t = policy_BC_or_opt in Pi_t
    if not bc_or_opt_in_pi_t and ("full" in verbose or "warnings" in verbose):
        print("WARNING: BC/optimal policy is not in Pi_t!")

    if "full" in verbose or "online-confset" in verbose:
        print(f"calc Pi_t: success, {len(Pi_t)} policies, in {time.time() - calc_Pi_t_time:.2f}s")
        print(f"  BC policy in Pi_t: {bc_or_opt_in_pi_t}")

    if len(Pi_t) > 0:
        return Pi_t
    else:
        raise ValueError("Pi_t is empty")  # at least pi_MLE should be in Pi_t.


def get_policy_pair_that_maximizes_uncertainty(
    Pi_t, precomputed_phi_bhats, gamma_t, V_t_inv, verbose=[]
):
    """
    Get the policy pair that maximizes uncertainty from the Pi_t set.
    uncertainty defined as, for a pair pi1, pi2 in Pi_t:
    gamma_t * ||phi(pi1)-phi(pi2)||_V^(-1) + 2*bhat_uncertainty_score(pi1) + 2*bhat_uncertainty_score(pi2)
    """
    calc_uncertainty_time = time.time()
    max_uncertainty = -float("inf")

    # generate unique pairs to avoid redundant calculations
    policy_pairs = list(itertools.combinations(Pi_t, 2))

    if len(policy_pairs) == 0:
        raise ValueError("Pi_t is empty")

    for pi1, pi2 in policy_pairs:
        pi1_hash = hash(pi1.matrix.tobytes())
        pi2_hash = hash(pi2.matrix.tobytes())
        pi1_phi = precomputed_phi_bhats[pi1_hash].embed
        pi2_phi = precomputed_phi_bhats[pi2_hash].embed
        pi1_bhat_uncertainty_score = precomputed_phi_bhats[pi1_hash].bhat_uncertainty_score
        pi2_bhat_uncertainty_score = precomputed_phi_bhats[pi2_hash].bhat_uncertainty_score
        mahalanobis_dist = mahalanobis(pi1_phi, pi2_phi, V_t_inv)
        uncertainty = (
            gamma_t * mahalanobis_dist
            + 2 * pi1_bhat_uncertainty_score
            + 2 * pi2_bhat_uncertainty_score
        )

        if uncertainty > max_uncertainty:
            max_uncertainty = uncertainty
            best_pair = (pi1, pi2)

    if "full" in verbose:
        print(
            f"max. uncertainty pair calc'd: uncertainty {max_uncertainty:.3f}, in {time.time() - calc_uncertainty_time:.2f}s"
        )

    return best_pair, uncertainty


def _check_if_BC_policy_in_pair(confset_offline, policy_pair_t, verbose):
    if "full" in verbose or "warnings" in verbose:
        mle_hash = hash(confset_offline[0].matrix.tobytes())
        pair_hashes = [
            hash(policy_pair_t[0].matrix.tobytes()),
            hash(policy_pair_t[1].matrix.tobytes()),
        ]
        print(f"MLE/Opt policy is in policy_pair_t: {mle_hash in pair_hashes}")


def generate_policy_pair_rollouts(
    rollout_env1,
    policy_1,
    rollout_env2,
    policy_2,
    num_rollouts=10,
):
    """For a pair of 2 policies & envs, generate num_rollouts many pairs of trajectories.
    Returns:
        traj_pairs: list of lists [[traj1, traj2]_1, [traj1, traj2]_2, ..., []_Nrollouts]
        where each traj is [s1,a1,r1,s2,...,sN,aN,rN]
    """
    traj_pairs = []
    for _ in range(num_rollouts):
        traj_1 = rollout_policy_in_env(rollout_env1, policy_1)
        traj_2 = rollout_policy_in_env(rollout_env2, policy_2)

        traj_pairs.append([traj_1, traj_2])
    return traj_pairs  # [[traj1, traj2]_1, [traj1, traj2]_2, ..., []_Nrollouts] where each traj [s1,a1,r1,s2,...,sN,aN,rN]


def annotate_buffer(traj_pairs_buffer, oracle_env, N_rollouts):
    """Annotate buffer of offline trajectories using an Oracle represented by oracle_env (usually true env).

    Only the first N_rollouts pairs are annotated.

    IN: buffer (list) of trajectory pairs [[traj1, traj2]_1, [traj1, traj2]_2, ..., [.,.]_bufferlength]
    OUT: augmented list of traj pairs, for every pair add
    y_T: 1 if log-likelihood(traj1) > log-likelihood(traj2), else 0
    y_R: 1 if disc.reward(traj1) > disc.reward(traj2), else 0
    returns: [[traj1, traj2, y_T, y_R]_1, ..., [.,.,.,.]_Nrollouts]

    Note: uses transitions, rewards, discount factor of annotation_env. if bufferlength < N_rollouts: duplicate trajs
    """
    list_trajs = []
    for i in range(min(len(traj_pairs_buffer), N_rollouts)):
        traj_1, traj_2 = traj_pairs_buffer[i]
        P_1 = _compute_log_likelihood_traj(traj_1, oracle_env.transitions)
        R_1 = compute_rewards_traj(traj_1, oracle_env.rewards, oracle_env.discount_factor)
        P_2 = _compute_log_likelihood_traj(traj_2, oracle_env.transitions)
        R_2 = compute_rewards_traj(traj_2, oracle_env.rewards, oracle_env.discount_factor)

        y_t = int(P_1 >= P_2)  # 1 if log-likelihood(traj1) > log-likelihood(traj2), else 0
        if R_1 == R_2:
            y_r = np.random.randint(0, 1)
        else:
            y_r = int(R_1 > R_2)  # 1 if disc.reward(traj1) > disc.reward(traj2), else 0
        list_trajs.append([traj_1, traj_2, y_t, y_r])

    if len(traj_pairs_buffer) < N_rollouts:
        print("Warning: not enough samples in buffer")
        while len(list_trajs) < N_rollouts:
            list_trajs.append(np.random.choice(list_trajs))
    return list_trajs


def _compute_log_likelihood_traj(traj, transition_matrix):
    """Compute log-likelihood of a trajectory under a transition matrix T.
    First gets flattened (1D-vector) counts of (s,a,s') in traj. using unique flat index:
    For 3D array [D_x, D_y, D_z], the unique flat index for (x, y, z) is [x *D_y*D_z + y*D_z + z].
    Here: (x,y,z) = (a,s,s') so index is [a *N_states*N_states + s*N_states + s']

    IN:
    - traj: [s0,a0,r0, ..., sH,aH,rH] all ints (s,a) or floats (r)
    - transition_matrix: [action, state, next_state] np.ndarray

    OUT:
    log(P(traj | T)) = log( PROD_{s,a,s'} T(s,a,s') ^ count(s,a,s') )
                     = SUM_{s,a,s'} log(T(s,a,s')) * count(s,a,s')
    """
    N_actions, N_states, _ = transition_matrix.shape

    triple_counts = np.zeros((N_actions * N_states * N_states))
    for i in range(0, len(traj) - 3, 3):
        state, action, next_state = traj[i], traj[i + 1], traj[i + 3]  # skipping reward at i+2
        index = action * (N_states * N_states) + state * N_states + next_state
        triple_counts[index] += 1

    return np.log(transition_matrix + EPS).flatten().dot(triple_counts)


def compute_rewards_traj_statecounts(traj, reward_vector, discount_factor):
    """Computes discounted rewards of a trajectory.
    Args:
        traj: [s0, a0, r0, ..., sH-1, aH-1, rH-1]
        reward_vector: reward per state, np.ndarray of shape (N_states,)
        discount_factor: float
    """
    N_states = reward_vector.shape
    state_counts = np.zeros((N_states))
    for i in range(0, len(traj), 3):
        state = traj[i]
        state_counts[state] += discount_factor**i
    return reward_vector.dot(state_counts)


def compute_rewards_traj(traj, reward_vector, discount_factor):
    """Computes discounted rewards of a trajectory [s0, a0, r0, ..., sH-1, aH-1, rH-1].
    Reward r_i is given for s_{i+1} = env.step(s_i, a_i). Crucially r_H-1 is given for s_H which is NOT in the traj.
    So when counting rewards of a traj, we must sum all rewards in the traj.

    Args:
        traj: [s0, a0, r0, ..., sH-1, aH-1, rH-1]
        reward_vector: reward per state, np.ndarray of shape (N_states,)
        discount_factor: float
    """
    total_reward = 0.0
    for t, i in enumerate(range(2, len(traj), 3)):
        reward = traj[i]
        total_reward += reward * (discount_factor**t)
    return total_reward


def learn_w_MLE(
    annotated_traj_buffer,
    phi_func,
    dim,
    w=None,
    w_initialization="random",
    sigmoid_slope=1,
    W_norm=1,
    lambda_param=0.01,
    n_epochs=10,
    lr=0.01,
    verbose=[],
):
    """trains a w_MLE parameter on some set of annotated trajectory pairs,
    i.e. [[[s0,a0,r0,s1,...]_traj1, [s0,a0,r0,s1,...]_traj2, y_T, y_R]_pair1, [..]_pair2, ..., [..]_pairbufferlength].
    e.g. warm-starting from offline dataset, or later on using the collected online preferences.
    """
    assert len(phi_func(annotated_traj_buffer[0][0])) == dim
    if w is None:
        if w_initialization == "uniform":
            w = torch.ones(dim)  # initialize uniform vector
            w = w / torch.norm(w, p=2) * W_norm  # normalize to have 2-norm W
        elif w_initialization == "random":
            w = torch.randn(dim)  # initialize random vector
            w = w / torch.norm(w, p=2) * W_norm  # normalize to have 2-norm W
        else:
            raise ValueError(f"w_initialization {w_initialization} not supported")
        w.requires_grad = True

    else:  # if w is provided, ensure it's trainable
        if not w.requires_grad:
            w.requires_grad = True
    optimizer = torch.optim.Adam([w], lr=lr)  # initialize optimizer

    for epoch in range(n_epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for traj_pair in annotated_traj_buffer:
            t1 = traj_pair[0]  # first trajectory
            t2 = traj_pair[1]  # second trajectory
            y_R = traj_pair[3]  # binary pref label (1 if t1>t2 based on rewards)

            phi_t1 = torch.tensor(phi_func(t1), dtype=torch.float32)
            phi_t2 = torch.tensor(phi_func(t2), dtype=torch.float32)
            phi_diff = phi_t1 - phi_t2  # embedded difference phi(t1) - phi(t2)

            logit = torch.dot(phi_diff, w)  # logit = w^T * (phi(t1) - phi(t2))

            # Binary cross-entropy loss with regularization
            if y_R == 1:
                loss = -torch.log(torch.sigmoid(sigmoid_slope * logit))
            else:
                loss = -torch.log(1 - torch.sigmoid(sigmoid_slope * logit))

            # Add regularization term
            reg_term = (lambda_param / 2) * torch.norm(w, p=2) ** 2
            loss += reg_term

            total_loss += loss

        total_loss.backward()
        optimizer.step()

        if ("full" in verbose or "losses" in verbose) and epoch % 2 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss.item():.4f}")
    w = w.detach()
    if "full" in verbose:
        print(f"  w_MLE_t: {w}, 2-norm: {np.linalg.norm(w):.3f}")
    return w


def project_w(w_MLE, W, V_t_inv, annotated_traj_buffer, phi_func, lambda_param, verbose=[]):
    """Projects the learned w_MLE vector to a ball, resulting in w_proj.

    Formula: w_proj_t = argmin_{w in d-dim ball of radius W} ||g_t(w) - g_t(w_MLE)||_{V_t^{-1}}
    where the matrix norm is the Mahalanobis norm with respect to V_t^{-1},
    and where g_t(w) = sum_{i=0}^{t-1} sigmoid(np.dot(phi(t1_i)-phi(t2_i), w))*(phi(t1_i)-phi(t2_i)) + lambda_param * w,
    and where t1_i, t2_i are the i-th pair of trajectories from the dataset we're training on,
    dataset of the form [[[s0,a0,r0,s1,...]_traj1, [s0,a0,r0,s1,...]_traj2, y_T, y_R]_pair1, [..]_pair2, ..., [..]_pairbufferlength].
    """
    if isinstance(w_MLE, torch.Tensor):  # if w is torch tensor, convert to np
        w_MLE = w_MLE.detach().numpy().astype(np.float64)
    else:
        w_MLE = np.array(w_MLE, dtype=np.float64)

    if np.linalg.norm(w_MLE) <= W:
        if "full" in verbose:
            print("project w_MLE: success (was already ||.||<W)")
        return w_MLE

    # Ensure V_t_inv is float64
    V_t_inv = np.array(V_t_inv, dtype=np.float64)

    def g_t(w_):
        """calculates g_t(w) for any w, used in obj. function"""
        w_ = np.array(w_, dtype=np.float64)  # Ensure w_ is float64
        result = np.zeros_like(w_, dtype=np.float64)
        for traj_pair in annotated_traj_buffer:
            t1 = traj_pair[0]
            t2 = traj_pair[1]

            phi_t1 = np.array(phi_func(t1), dtype=np.float64)
            phi_t2 = np.array(phi_func(t2), dtype=np.float64)
            phi_diff = phi_t1 - phi_t2

            logit = np.dot(phi_diff, w_)
            sigmoid_val = _sigmoid(logit)

            result += sigmoid_val * phi_diff

        result += lambda_param * w_
        return result

    g_t_w_MLE = g_t(w_MLE)

    def objective_func(w_):
        w_ = np.array(w_, dtype=np.float64)  # Ensure w_ is float64
        return mahalanobis(g_t(w_), g_t_w_MLE, V_t_inv)

    # Constraint: ||w||_2 <= W
    constraint = {"type": "ineq", "fun": lambda w: float(W - np.linalg.norm(w))}  # >= 0

    # Initial guess: use w_MLE if within the ball, otherwise project it
    initial_w = W * (w_MLE / np.linalg.norm(w_MLE))
    initial_w = np.array(initial_w, dtype=np.float64)

    # Solve the optimization problem
    try:
        result = minimize(objective_func, initial_w, constraints=constraint, method="SLSQP")

        if result.success:
            if "full" in verbose:
                print(f"  w_proj_t: {result.x}, 2-norm: {np.linalg.norm(result.x):.3f} <= W: {W}")
            return result.x
        else:
            # If optimization fails, fall back to simple projection
            w_norm = np.linalg.norm(w_MLE)
            if w_norm <= W:
                if "full" in verbose:
                    print("failed to project w_MLE but ||w_MLE||_2 <= W, returning it")
                return w_MLE
            else:
                if "full" in verbose:
                    print("failed to project w_MLE, returning normalized w_MLE")
                return W * (w_MLE / w_norm)
    except Exception as e:
        if "full" in verbose:
            print(f"Error during optimization: {e}")
        # Fall back to simple projection
        w_norm = np.linalg.norm(w_MLE)
        if w_norm <= W:
            return w_MLE
        else:
            return W * (w_MLE / w_norm)


def _sigmoid(x, slope=1):
    """slope: >1 steeper, [0,1] shallower, <1 flips
    for numerical stability, separate cases"""
    if x >= 0:
        return 1 / (1 + np.exp(-slope * x))
    else:
        exp_x = np.exp(slope * x)
        return exp_x / (1 + exp_x)


def find_most_preferred_policy(w, policy_list, phi_traj_func, env, verbose=[]):
    """finds best policy in a list of policies by maximizing score func

    this list should be the offline confset."""
    best_policy = None
    best_score = -np.inf
    policy_scores = {}

    for policy in policy_list:
        policy_hash = hash(policy.matrix.tobytes())
        score, reward = _score_func_policy(phi_traj_func, policy, w, env, verbose=verbose)
        policy_scores[policy_hash] = {
            "policy": policy,
            "score": score,
            "reward": reward,
        }
        if "full" in verbose:
            print(f"  score: {score:.3f}, avg reward: {reward:.3f}")
        if score > best_score:
            best_score = score
            best_policy = policy
    if "full" in verbose:
        print(f"  best policy score: {best_score:.3f}")
    return best_policy, policy_scores


def calc_avg_reward(policy, env, N_samples=1000):
    """simply calculates expected reward of a policy in an env, over N_samples trajectories"""
    reward_sum = 0
    for _ in range(N_samples):
        traj = rollout_policy_in_env(env, policy)
        traj_reward = compute_rewards_traj(traj, env.rewards, env.discount_factor)
        reward_sum += traj_reward
    return reward_sum / N_samples


def _score_func_policy(phi_traj_func, policy, w, env, N_samples=100, verbose=[]):
    """calculates the "score" function: <phi(policy), w>
    which is defined as the Expectation over trajectories from that policy, of <phi(traj), w>

    also calculates average reward over N_samples trajectories
    """
    # sampled_trajs = []
    score = 0
    reward_sum = 0
    for _ in range(N_samples):
        traj = rollout_policy_in_env(env, policy)
        traj_reward = compute_rewards_traj(traj, env.rewards, env.discount_factor)
        score += np.dot(phi_traj_func(traj), w)
        reward_sum += traj_reward
    if "full" in verbose:
        print(f"phi(traj): {phi_traj_func(traj)}, reward: {traj_reward:.3f}")  # print one
    return score / N_samples, reward_sum / N_samples


def calc_regret(
    w,
    policy_test,
    policy_true_opt,
    phi_traj_func,
    env,
    N_samples=1000,
    verbose=[],
    simmed_or_analytic="simmed",
):
    """compares average performance (in terms of discounted rewards) on the environment (via sampling N_samples trajectories) of
    a policy to the true optimal policy

    Optionally, compute regret analytically. But simmed is more honest: score and reward are off of the same
    simmed trajectories, so in a situation where pi_test is very close to pi_opt, and on the sampled trajs pi_test > pi_opt in rewards, the optimal model would output s(test)>s(opt).
    with analytically computed regret, you'd get positive regret even though the model made the correct decision (on the sample!).
    with simmed regret you get <=0 regret, so matching the correct model decision.
    """
    avg_reward_test, score_test, avg_reward_trueopt, score_trueopt = _calc_avg_rewards_scores(
        w, policy_test, policy_true_opt, phi_traj_func, env, N_samples
    )
    regret = avg_reward_trueopt - avg_reward_test
    comparison_glyph = "<" if regret > 0 else ">"
    if "full" in verbose:
        print(
            f"  avg rewards: TEST {avg_reward_test:.3f} {comparison_glyph} TRUE {avg_reward_trueopt:.3f},\n  regret: {regret:.3f}\n  scores test {score_test:.3f} vs true opt {score_trueopt:.3f}"
        )
    if simmed_or_analytic == "analytic":
        avg_reward_test = env.calc_analytic_discounted_reward_finitehorizon(policy_test)
        avg_reward_trueopt = env.calc_analytic_discounted_reward_finitehorizon(policy_true_opt)
    return (
        regret,
        score_test,
        score_trueopt,
        avg_reward_test,
        avg_reward_trueopt,
    )


def _calc_avg_rewards_scores(w, policy_test, policy_true_opt, phi_traj_func, env, N_samples=1000):
    """compares average performance (in terms of discounted rewards) on the environment (via sampling N_samples trajectories) of
    a policy to the true optimal policy"""
    score_testpolicy, avg_reward_testpolicy = _score_func_policy(
        phi_traj_func, policy_test, w, env, N_samples
    )
    score_trueopt, avg_reward_trueopt = _score_func_policy(
        phi_traj_func, policy_true_opt, w, env, N_samples
    )
    return avg_reward_testpolicy, score_testpolicy, avg_reward_trueopt, score_trueopt


def _compute_and_log_online_iteration_metrics(
    metrics,
    confset_offline,
    Pi_t,
    uncertainty_t,
    env_true,
    w_proj,  # online: w_proj_t,final: w_proj_final
    w_MLE,  # online: None (unneeded), final: w_MLE_final
    env_learned,  # online: None (unneeded), final: env_learned_t
    phi,
    optimal_policy_true,
    start_time,  # online: loop_start_time, final: online_start_time
    t,
    verbose,
    is_final,
):
    best_policy_t, _ = find_most_preferred_policy(
        w_proj, confset_offline, phi, env_true, verbose=[]
    )

    # compute regret of current best vs theoretically optimal policy (difference in average rewards over N_samples trajectories)
    (
        regret,
        score_test,
        score_trueopt,
        avg_reward_test,
        avg_reward_trueopt,
    ) = calc_regret(
        w_proj,
        best_policy_t,
        optimal_policy_true,
        phi,
        env_true,
        N_samples=1000,
    )

    if not is_final:
        metrics = loop_iteration_logging(
            metrics,
            regret,
            uncertainty_t,
            best_policy_t,
            score_test,
            score_trueopt,
            avg_reward_test,
            avg_reward_trueopt,
            Pi_t,
            optimal_policy_true,
            start_time,
            t,
            verbose,
        )
        return metrics
    else:
        metrics, final_objs, final_values = final_iteration_logging(
            metrics,
            regret,
            best_policy_t,
            score_test,
            score_trueopt,
            avg_reward_test,
            avg_reward_trueopt,
            w_MLE,
            w_proj,  # w_proj_final
            env_learned,
            Pi_t,
            confset_offline,
            start_time,
            verbose,
        )
        return metrics, final_objs, final_values


def loop_iteration_logging(
    metrics,
    regret,
    uncertainty_t,
    best_policy_t,
    score_test,
    score_trueopt,
    avg_reward_test,
    avg_reward_trueopt,
    Pi_t,
    optimal_policy_true,
    loop_start_time,
    t,
    verbose=[],
):
    """logs metrics for one iteration"""
    metrics["regrets"].append(regret)
    metrics["best_iteration_policy"].append(best_policy_t)
    metrics["scores_best_iteration_policy"].append(score_test)
    metrics["scores_true_opt"].append(score_trueopt)
    metrics["avg_rewards_best_iteration_policy"].append(avg_reward_test)
    metrics["avg_rewards_true_opt"].append(avg_reward_trueopt)
    metrics["uncertainty_t"].append(uncertainty_t)
    metrics["pi_set_sizes"].append(len(Pi_t))
    metrics["iteration_times"].append(time.time() - loop_start_time)

    if "loop-summary" in verbose or "full" in verbose:
        opt_hash = hash(optimal_policy_true.matrix.tobytes())
        Pi_t_hashes = [hash(policy.matrix.tobytes()) for policy in Pi_t]

        print(f"-- summary loop {t}:")
        print(
            f"  size of Pi_t: {len(Pi_t)}, opt in Pi_t: {opt_hash in Pi_t_hashes}, uncertainty: {uncertainty_t:.3f}"
        )
        print(f"  rewards best policy: {avg_reward_test:.3f} vs true opt {avg_reward_trueopt:.3f}")
        print(f"  scores best policy: {score_test:.3f} vs true opt {score_trueopt:.3f}")
        print(f"  => regret: {regret:.3f}")
        print(f" ----- ending loop {t} (in {time.time() - loop_start_time:.1f} seconds) ----- ")

    if "loop-time" in verbose:
        print(f"-- loop {t} time: {time.time() - loop_start_time:.1f}s, |Pi_t|: {len(Pi_t)}")
    return metrics


def initial_loop_earlystop_logging(
    metrics, Pi_t, w_MLE_t, w_proj_t, env_BC_learned, confset_offline, online_start_time
):
    """produces metrics & return objects for loop 0 when we early-stop because |Pi_t| = 1"""
    metrics["uncertainty_t"].append(0)
    metrics["pi_set_sizes"].append(1)
    metrics["regrets"].append(0)
    metrics["best_iteration_policy"].append(Pi_t[0])
    metrics["scores_best_iteration_policy"].append(0)
    metrics["scores_true_opt"].append(0)
    metrics["avg_rewards_best_iteration_policy"].append(0)
    metrics["avg_rewards_true_opt"].append(0)
    metrics["iteration_times"].append(time.time() - online_start_time)
    final_objs = {
        "w_MLE_final": w_MLE_t,
        "w_proj_final": w_proj_t,
        "final_best_policy": Pi_t[0],
        "env_learned_t": env_BC_learned,
        "Pi_t": Pi_t,
        "confset": confset_offline,
    }
    final_values = {
        "regret": 0,
        "avg_reward_final_best_policy": 0,
        "score_final_best_policy": 0,
        "avg_reward_true_opt": 0,
        "score_true_opt": 0,
    }
    return metrics, final_objs, final_values


def final_iteration_logging(
    metrics,
    regret,
    final_best_policy,
    score_test,
    score_trueopt,
    avg_reward_test,
    avg_reward_trueopt,
    w_MLE_final,
    w_proj_final,
    env_learned_t,
    Pi_t,
    confset_offline,
    online_start_time,
    verbose=[],
):
    metrics["regrets"].append(regret)
    metrics["best_iteration_policy"].append(final_best_policy)
    metrics["scores_best_iteration_policy"].append(score_test)
    metrics["scores_true_opt"].append(score_trueopt)
    metrics["avg_rewards_best_iteration_policy"].append(avg_reward_test)
    metrics["avg_rewards_true_opt"].append(avg_reward_trueopt)

    final_objs = {
        "w_MLE_final": w_MLE_final,
        "w_proj_final": w_proj_final,
        "final_best_policy": final_best_policy,
        "env_learned_t": env_learned_t,
        "Pi_t": Pi_t,
        "confset": confset_offline,
    }

    final_values = {
        "regret": regret,
        "avg_reward_final_best_policy": avg_reward_test,
        "score_final_best_policy": score_test,
        "avg_reward_true_opt": avg_reward_trueopt,
        "score_true_opt": score_trueopt,
    }

    online_runtime = time.time() - online_start_time

    if "full" in verbose:
        print(f"  w_MLE_final: {w_MLE_final}, 2-norm: {np.linalg.norm(w_MLE_final):.3f}")
        print(f"  w_proj_final: {w_proj_final}, 2-norm: {np.linalg.norm(w_proj_final):.3f}")
        print(f"  final best policy: {final_best_policy}, regret: {regret:.3f}")
        print(f"  final best policy score: {score_test:.3f}, true opt score: {score_trueopt:.3f}")
        print(
            f"  final best policy avg reward: {avg_reward_test:.3f}, true opt avg reward: {avg_reward_trueopt:.3f}"
        )
        print(f"  online runtime: {online_runtime:.1f} seconds")

    return metrics, final_objs, final_values


# ==== (end helpers: online learning) ==== #


# ==== Policies & their training (BC) funcs ==== #
def train_tabular_BC_policy(
    offline_trajs,
    N_states,
    N_actions,
    init="random",
    n_epochs=10,
    lr=0.01,
    make_deterministic=True,
    verbose=[],
):
    """
    Train behavioral cloning policy from offline trajectories.

    Args:
        offline_trajs: List of offline trajectories
        N_states: Number of states
        N_actions: Number of actions
        init: Initialization method ("random")
        n_epochs: Number of training epochs
        lr: Learning rate
        make_deterministic: Whether to make policy deterministic

    Returns:
        Trained TabularPolicy object
    """
    if init == "random":
        random_policy_logits = torch.rand((N_states, N_actions), dtype=torch.float32)
        policy_logits = torch.nn.functional.softmax(random_policy_logits, dim=1)
    elif init == "uniform":
        policy_logits = torch.ones((N_states, N_actions), dtype=torch.float32) / N_actions
    else:
        raise ValueError(f"init {init} not supported, use 'random' or 'uniform'")
    policy = TabularPolicy(policy_logits)

    # convert np array to trainable torch tensor
    policy.matrix = torch.nn.Parameter(torch.tensor(policy.matrix, dtype=torch.float32))

    optimizer = torch.optim.Adam([policy.matrix], lr=lr)

    # training loop
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # compute loss for each trajectory
        total_loss = 0
        for traj in offline_trajs:
            states = traj[::3]
            actions = traj[1::3]

            # compute log probabilities for each state-action pair
            for state, action in zip(states, actions):
                state_probs = policy.matrix[state]  # prob of each action given state
                action_prob = state_probs[action]  # prob of action taken
                total_loss -= torch.log(
                    action_prob + 1e-10
                )  # add small epsilon for numerical stability

        total_loss.backward()  # backprop
        optimizer.step()  # update

        # project probabilities to be valid (non-negative and sum to 1)
        with torch.no_grad():
            policy.matrix.data = torch.nn.functional.softmax(policy.matrix, dim=1)

        if ("full" in verbose or "losses" in verbose) and epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss.item():.4f}")

    # after training, optionally make policy deterministic
    if make_deterministic:
        with torch.no_grad():
            max_probs, max_actions = torch.max(policy.matrix, dim=1)
            policy.matrix.zero_()
            for state in range(policy.matrix.shape[0]):
                policy.matrix[state, max_actions[state]] = 1.0
            policy.matrix = policy.matrix.detach().numpy().astype(np.int32)
    else:
        policy.matrix = policy.matrix.detach().numpy().astype(np.float32)

    if "full" in verbose:
        print(f"Trained policy matrix in {n_epochs} epochs:")
        print(policy.matrix)

    return policy


def generate_random_tabular_policies(
    N_states,
    N_actions,
    N_policies=100,
    make_deterministic=True,
    dtype=np.int32,
):
    """
    Generate random tabular policies. Uses vectorized NumPy (fast).

    Returns:
        List of TabularPolicy objects
    """
    policies = []
    if make_deterministic:
        # for each policy, for each state, randomly select one action. shape: (N_policies, N_states)
        actions = np.random.randint(0, N_actions, size=(N_policies, N_states))
        policy_matrices = np.zeros((N_policies, N_states, N_actions), dtype=dtype)
        rows = np.arange(N_states)
        for i in range(N_policies):
            policy_matrices[i, rows, actions[i]] = 1
    else:  # random probs normalized along action axis
        policy_matrices = np.random.rand(N_policies, N_states, N_actions)
        policy_matrices /= policy_matrices.sum(axis=2, keepdims=True)
    for i in range(N_policies):
        policies.append(TabularPolicy(policy_matrices[i]))
    return policies


def generate_all_deterministic_stationary_policies(N_states, N_actions, dtype=np.int32):
    """
    Generate all possible deterministic stationary policies for a given state and action space. itertools.product.
    """
    policies = []
    action_combinations = itertools.product(range(N_actions), repeat=N_states)
    for actions in action_combinations:
        # Create policy matrix
        policy_matrix = np.zeros((N_states, N_actions), dtype=dtype)
        for state, action in enumerate(actions):
            policy_matrix[state, action] = 1  # one-hot encoded action
        policy = TabularPolicy(policy_matrix)
        policies.append(policy)

    return policies


# ==== end: policies ==== #


# ==== helpers: transition models & their training ==== #
class TrajectoriesDataset(Dataset):
    """
    Dataset for trajectory data. Individual datapoint is (s,a,r,s').
    """

    def __init__(self, trajs, N_states=5, N_actions=2, verbose=[]):
        if "full" in verbose:
            print(f"Creating dataset from {len(trajs)} trajectories. Trajs are:\n{trajs}")

        states = torch.tensor([traj[:-3:3] for traj in trajs]).view(-1)
        actions = torch.tensor([traj[1:-2:3] for traj in trajs]).view(-1)
        self.rewards = torch.tensor([traj[2:-1:3] for traj in trajs]).view(-1).float()
        next_states = torch.tensor([traj[3::3] for traj in trajs]).view(-1)

        # One-hot encode:
        self.states = nn.functional.one_hot(states, num_classes=N_states).float()
        self.actions = nn.functional.one_hot(actions, num_classes=N_actions).float()
        self.next_states = nn.functional.one_hot(next_states, num_classes=N_states).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
        )


class TabularTransitionModel(nn.Module):
    """
    Tabular transition model for discrete MDPs.

    Computes the logprob of a trajectory by multiplying triplet counts (s,a,s') with log transition matrix.
    """

    def __init__(self, N_states, N_actions, init_matrix=None):
        super(TabularTransitionModel, self).__init__()
        self.N_states = N_states
        self.N_actions = N_actions
        if init_matrix is None:
            init_matrix = torch.rand(N_actions, N_states, N_states)
        else:
            assert init_matrix.shape == (N_actions, N_states, N_states)
            init_matrix = torch.from_numpy(copy.deepcopy(init_matrix))
            init_matrix += EPS
        self.transition_matrix = nn.Parameter(init_matrix)
        self.normalize()

    def forward(self, triple_count):
        return torch.matmul(triple_count, self.transition_matrix.reshape(-1, 1))

    def normalize(
        self,
    ):
        with torch.no_grad():
            self.transition_matrix -= torch.logsumexp(self.transition_matrix, -1, keepdim=True)


class MLPTransitionModel(nn.Module):
    """
    Transition model that uses a 2-layer MLP or linear layer to predict next state probabilities.
    """

    def __init__(self, N_states, N_actions, use_mlp=False, hidden_dim=32):
        """
        Initialize MLP transition model.

        Args:
            N_states: Number of states
            N_actions: Number of actions
            use_mlp: Whether to use MLP (True) or linear model (False)
            hidden_dim: Hidden dimension for MLP
        """
        super(MLPTransitionModel, self).__init__()
        self.N_states = N_states
        self.N_actions = N_actions

        if use_mlp:  # 2-layer MLP with ReLU activation
            self.model = nn.Sequential(
                nn.Linear(N_states + N_actions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, N_states),
            )
        else:  # linear model
            self.model = nn.Linear(N_states + N_actions, N_states, bias=False)
            torch.nn.init.kaiming_uniform_(self.model.weight)

    def get_device(self):
        """Get device of model parameters."""
        return next(self.parameters()).device

    def forward(self, state, action):
        """
        Predict next state probabilities p(s'|s,a).

        Args: state, action: one-hot encoded vectors of shape [N_states], [N_actions]

        Returns: next state probability distribution, vector p(s'|s,a) of shape [N_states]
        """
        input_tensor = torch.cat((state, action), dim=-1)
        next_state = self.model(input_tensor)
        return next_state

    def extract_transition_matrix(self):
        transition_matrix = torch.zeros(self.N_actions, self.N_states, self.N_states)

        for action_idx in range(self.N_actions):
            for state_idx in range(self.N_states):
                state = torch.zeros(1, self.N_states)
                state[:, state_idx] = 1  # one-hot encoding for state j
                action = torch.zeros(1, self.N_actions)
                action[:, action_idx] = 1  # one-hot encoding for action i

                next_state_prob = nn.Softmax(dim=1)(self.forward(state, action))  # softmax(T(s,a))
                transition_matrix[action_idx, state_idx, :] = next_state_prob.squeeze()

        return transition_matrix.detach().numpy().astype(np.double)


def train_transition_model_wrapper(trajectories, env, model="MLE", n_epochs=5, verbose=[]):
    """
    Wrapper function for training transition models with different methods.

    Args:
        offline_trajs: List of offline trajectories
        env_true: True environment (for initialization)
        model: Model type ("MLE", "linear_classifier", "MLP")

    Returns:
        tuple: (env_updated, transition_model, losses)
    """
    if model == "MLE":
        return train_transition_model_MLE(trajectories, env, verbose)
    elif model == "linear_classifier":
        return train_transition_model_MLP(
            trajectories, env, n_epochs, use_mlp=False, verbose=verbose
        )
    elif model == "MLP":
        return train_transition_model_MLP(
            trajectories, env, n_epochs, use_mlp=True, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown transition model type: {model}")


def train_transition_model_MLE(trajectories, env, verbose):
    """
    Train transition model using Maximum Likelihood Estimation from dataset of trajectories.

    Args:
        trajectories: List of trajectories, each [s0,a0,r0,s1,a1,r1,...]
        env: Environment with N_states, N_actions attributes
        verbose: Verbosity options

    Returns:
        tuple:
         env_MLE: Environment with updated transition matrix
         transitions: Transition matrix
         transition_counts: Counts of transitions
    """
    N_states = env.N_states
    N_actions = env.N_actions

    transition_counts = np.zeros((N_actions, N_states, N_states))

    # count transitions from trajectories
    for traj in trajectories:
        for i in range(0, len(traj) - 3, 3):  # step by 3 to get (s,a,r,s')
            state = traj[i]
            action = traj[i + 1]  # next state is at i+3
            next_state = traj[i + 3]

            if 0 <= state < N_states and 0 <= action < N_actions and 0 <= next_state < N_states:
                transition_counts[action, state, next_state] += 1

    # convert counts to probabilities (MLE)
    transitions = np.zeros((N_actions, N_states, N_states))
    for a in range(N_actions):
        for s in range(N_states):
            state_action_total_count = np.sum(transition_counts[a, s, :])
            if state_action_total_count > 0:
                transitions[a, s, :] = transition_counts[a, s, :] / state_action_total_count
            else:
                # if state-action pair never observed, uniform distribution
                transitions[a, s, :] = 1.0 / N_states

    # create updated environment
    env_MLE = copy.deepcopy(env)
    env_MLE.transitions = transitions
    if "full" in verbose:
        print(f"MLE transitions:\n{transitions}")

    return env_MLE, transitions, transition_counts


def train_transition_model_MLP(trajectories, env, n_epochs, use_mlp=False, verbose=[]):
    """
    Train transition model using neural networks.

    Args:
        offline_trajs: List of offline trajectories
        env_true: True environment
        use_mlp: Whether to use MLP (True) or linear model (False)
        n_epochs: Number of training epochs
        verbose: List of verbosity options

    Returns:
        tuple:
          env_updated: Environment with updated transition matrix
          transition_model: Trained transition model
          losses: Losses per epoch
    """
    if "full" in verbose:
        print(f"Training {'MLP' if use_mlp else 'linear'} transition model")

    dataset = TrajectoriesDataset(trajectories, env.N_states, env.N_actions, verbose)
    dataloader = DataLoader(dataset, batch_size=min(4, len(trajectories)), shuffle=True)

    transition_model = MLPTransitionModel(env.N_states, env.N_actions, use_mlp=use_mlp)
    optimizer = torch.optim.Adam(transition_model.parameters(), lr=0.01)

    transition_model, losses = train_model(
        transition_model, dataloader, optimizer, loss_fn_T_trajs, n_epochs, verbose
    )

    transitions = transition_model.extract_transition_matrix()

    env_trained = copy.deepcopy(env)
    env_trained.transitions = transitions

    if "full" in verbose:
        print(f"Transition model: {transition_model}")

    return env_trained, transition_model, losses


def train_model(model, dataloader, optimizer, loss_fn, n_epochs=10, verbose=[]):
    """
    Generic model training function.

    Args:
        model: Model to train
        dataloader: Data loader
        optimizer: Optimizer
        loss_fn: Loss function
        n_epochs: Number of epochs
        verbose: Verbosity options

    Returns:
        tuple: (trained_model, loss_per_epoch)
    """
    loss_per_epoch = []

    for epoch in range(n_epochs):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(batch, model)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = np.mean(losses)
        loss_per_epoch.append(epoch_loss)

        if "full" in verbose:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    return model, loss_per_epoch


def loss_fn_T_trajs(batch, transition_model):
    """Transition model: cross-entropy loss, to predict single successor state:
    L = CE(T(.|s,a), s')
    right now: s' (next_state) is one-hot vector, treated as 'class probabilities' target.
    TODO: documentation says nn.CrossEntropyLoss computation is faster if target is 'class index'.
    so change next_state to be just the correct idx.
    """
    state, action, _, next_state = batch  # all one-hot encoded.
    device = transition_model.get_device()
    state = state.to(device)
    action = action.to(device)
    next_state = next_state.to(device)
    pred_next_state_logit = transition_model(state, action)  # size (batch_size, N_states)
    return nn.CrossEntropyLoss()(pred_next_state_logit, next_state)


def sanity_check_transitions(env, fix=False, verbose=[]):
    """
    Check that transition probabilities sum to 1 and fix if needed.

    Args:
        env: Environment with transitions to check
        fix: Whether to fix invalid transitions

    Returns:
        Environment with valid transitions

    """
    if env.transitions is None:
        return env

    # using stricter tolerances that are closer to what np.random.choice expects
    row_sums = np.sum(env.transitions, axis=2)
    all_close = np.allclose(row_sums, 1, rtol=1e-10, atol=1e-10)

    if not all_close:
        if "full" in verbose:
            print("\nRows that don't sum to 1 detected.")
        bad_indices = np.where(~np.isclose(row_sums, 1, rtol=1e-10, atol=1e-10))
        for i in range(len(bad_indices[0])):
            a, s = bad_indices[0][i], bad_indices[1][i]
            if "full" in verbose:
                print(f"Action {a}, State {s}: sum = {row_sums[a, s]:.10f}")
                print(f"Row values: {env.transitions[a, s]}")

        if fix:
            for a in range(env.N_actions):
                for s in range(env.N_states):
                    row_sum = np.sum(env.transitions[a, s])
                    if row_sum > 0:
                        env.transitions[a, s] /= row_sum
                    else:
                        env.transitions[a, s] = 1.0 / env.N_states
            if "full" in verbose:  # verify fix worked
                new_row_sums = np.sum(env.transitions, axis=2)
                print(
                    f"Fixing transition model: all rows now sum to 1: {np.allclose(new_row_sums, 1, rtol=1e-10, atol=1e-10)}\nFixed transitions:\n{env.transitions}"
                )
        else:
            raise ValueError("Transition model rows must sum to 1")

    return env


# ==== end: transition models & their training ==== #


# ==== Helpers: Misc ==== #
def pad_metrics(metrics, params):
    """
    if loop terminated early in one seed, then some of the metrics don't have the right length.
    fix this by padding them at the end: all with 0, except avg_rewards_best_iteration_policy, avg_rewards_true_opt, and pi_set_sizes with 1
    """
    no_padding_needed = True
    for seed in range(len(metrics)):
        for metric in metrics[seed]:
            if len(metrics[seed][metric]) < params["N_iterations"]:
                padding_length = params["N_iterations"] - len(metrics[seed][metric])
                if (
                    metric == "avg_rewards_best_iteration_policy"
                    or metric == "avg_rewards_true_opt"
                ):
                    metrics[seed][metric].extend([1] * padding_length)
                elif metric == "pi_set_sizes":
                    metrics[seed][metric].extend([1] * padding_length)
                else:
                    metrics[seed][metric].extend([0] * padding_length)
                if "plotting" in params["verbose"] or "full" in params["verbose"]:
                    print(f"metric {metric} padded {padding_length} for seed {seed}")
                no_padding_needed = False
    if no_padding_needed and ("plotting" in params["verbose"] or "full" in params["verbose"]):
        print("no padding needed")
    return metrics


def maybe_generate_all_policies(env_name):
    N_states, N_actions = ENV_N_STATES_ACTIONS[env_name]
    if N_actions**N_states >= 1e4:
        print(f"Skip gen'ing all policies, MDP too big, {N_actions**N_states} pol's (>1e4)")
        return None
    all_policies_path = f"discrete/saved_policies/{env_name}_all_policies.pkl"
    if os.path.exists(all_policies_path):
        print(f"All policies already gen'd at {all_policies_path}")
    else:
        t = time.time()
        all_policies = generate_all_deterministic_stationary_policies(N_states, N_actions)
        with open(all_policies_path, "wb") as f:
            pickle.dump(all_policies, f)
        print(f"Gen'd all policies & saved to {all_policies_path} in {time.time() - t:.1f} seconds")
    return None


def rollout_policy_in_env(env, policy):
    """
    Rollout a policy in an environment, for H=episode_length many steps.

    Returns trajectory [s0, a0, r0, s0'(=s1), a1, r1, s1'(=s2), ..., a_{H-1},r_{H-1}].

    Note: the order is
    - s_i given
    - a_i is policy action
    - r_i, s_{i+1} generated by env.step(s_i, a_i).

    Crucially: reward r_i is given for s_{i+1} = env.step(s_i, a_i). So to calc total traj reward,
    must do so by summing all rewards in the traj.
    You CANNOT count states [s0...s_{H-1}] and apply the reward vector!
    If you do that, you'd count a reward for s0 (but r1 is actually given for s1),
    and you'd miss r_{H-1} b/c that's given for s_H, which we delete.
    """
    obs = env.reset()
    done = False
    traj = [obs[0]]
    while not done:
        a = policy.get_action(obs)
        obs, reward, done, _ = env.step(a)
        traj.extend([a, reward, obs[0]])  # obs[0] b/c TabularMDP
    traj.pop()  # removes very last obs
    return traj  # [s1,a1,r1,s1'(=s2),a2,r2,s2'(=s3), ..., aN,rN]
