"""
Contains experiment runner and all helpers/utils exclusively used by the continuous BRIDGE code
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import hashlib
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from functools import partial
from typing import List, Optional
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import ObservationWrapper
from gymnasium import spaces

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize
from scipy.stats import spearmanr

import wandb
from wandb.integration.sb3 import WandbCallback

from .embeddings import setup_embeddings, Trajectory


# ==== experiment runner ==== #
def run_experiment_mp(params):
    mp.set_start_method("spawn", force=True)
    seeds = list(range(params.seed, params.seed + params.N_experiments))
    results = []
    finished = 0
    which_mp = "processpoolexecutor"
    if which_mp == "processpoolexecutor":
        with ProcessPoolExecutor() as executor:
            print("starting subprocesses (ProcessPoolExecutor)...")
            futures = [executor.submit(run_single_seed, seed, params) for seed in seeds]
            for future in as_completed(futures):
                results.append(future.result())
                finished += 1
                print(
                    f"finished {finished}/{params.N_experiments} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
        results.sort(key=lambda x: seeds.index(x[0]) if isinstance(x, tuple) and len(x) > 1 else 0)
        seeds, metrics_per_seed = zip(*results)
    elif which_mp == "pool":
        with mp.Pool() as pool:
            print("starting subprocesses (Pool)...")
            worker_func = partial(run_single_seed, params=params)
            results = pool.map(worker_func, seeds)
        seeds, metrics_per_seed = zip(*results)

    import gc
    import matplotlib.pyplot as plt

    gc.collect()
    plt.close("all")

    if params.N_experiments == 1:
        metrics_per_seed = metrics_per_seed[0]  # {metrics0}
        avg_expert_reward = np.mean(metrics_per_seed["avg_rewards_true_opt"])
        avg_bc_reward = np.mean(metrics_per_seed["avg_rewards_bc"])
    else:
        metrics_per_seed = list(metrics_per_seed)  # [{metrics0}, {metrics1}, ...]
        avg_expert_reward = np.mean(
            [np.mean(metrics["avg_rewards_true_opt"]) for metrics in metrics_per_seed]
        )
        avg_bc_reward = np.mean(
            [np.mean(metrics["avg_rewards_bc"]) for metrics in metrics_per_seed]
        )
    return metrics_per_seed, avg_expert_reward, avg_bc_reward


def run_single_seed(seed, params):
    """End-to-end experiment: load/train expert, get offline data, train BC, build candidates,
    run online preference learning, and return metrics.
    Environment-agnostic assuming gym env wrapped as SB3 VecEnv with num_envs=1.
    """
    log_file_path = os.environ.get("EXPERIMENT_LOG_FILE")
    original_stdout = None
    env = None
    if log_file_path:
        original_stdout = sys.stdout  # save original stdout
        sys.stdout = Tee(log_file_path, mode="a")  # redirect stdout to log file

    try:
        # ==== Setup ==== #
        print(f"...seed {seed} started")
        params.seed = seed
        _set_random_seeds(seed)  # sets random, np, torch

        tag, expert_policy, env, phi_func = setup_experiment_environment(seed, params)

        offline_trajs = load_or_generate_offline_trajs(env, expert_policy, params)
        _set_random_seeds(seed)  # re-set seeds b/c potentially just stepped creating offline trajs
        env.seed(seed=seed)

        # ==== Offline: BC ==== #
        bc_policy = train_bc_policy_with_defaults(offline_trajs, env, expert_policy, params)

        if params.confset_noise is None:
            params.confset_noise = 0.01

        # ==== Offline: generate confset ====
        offline_confset, candidates, all_policies = generate_policy_sets(
            bc_policy, expert_policy, params
        )

        metrics = setup_metrics()

        # ==== Bridging: precompute embeddings for policy list ==== #
        vals = get_precomputed_embeddings_names(
            bc_policy, expert_policy, offline_confset, all_policies, phi_func, tag, env, params
        )
        precomputed_embeddings, policy_names, bc_embed_precalc, recompute_embeds = vals

        # re-set seeds b/c potentially just stepped precomputing embeddings
        _set_random_seeds(seed)
        env.seed(seed=seed)

        # ==== Online: define policy search space ====
        Pi_0 = get_online_search_space(
            candidates,
            bc_policy,
            expert_policy,
            phi_func,
            env,
            precomputed_embeddings,
            tag,
            params,
        )

        # ==== make a plot of dists-to-bc and rewards, save it ====
        make_cand_bc_scatterplots(
            all_policies,
            precomputed_embeddings,
            bc_policy,
            expert_policy,
            phi_func,
            recompute_embeds,
            params,
        )

        # ==== Online: main preference loop ====
        online_trajs, preference_data = [], []
        pref_model, V, V_inv = initialize_preference_learning(phi_func, params)

        for t in range(params.N_iterations):
            loop_start_time = time.time()

            # ==== (if enabled) Refine online confidence set Pi_t ====
            if params.filter_pi_t_yesno:
                Pi_t = refine_confset(
                    t,
                    Pi_0,
                    pref_model,
                    env,
                    phi_func,
                    params.episode_length,
                    V_inv,
                    params.filter_pi_t_gamma,
                    gamma_debug_mode=params.gamma_debug_mode,
                    precomputed_embeddings=precomputed_embeddings,
                    n_samples=params.n_embedding_samples,
                )
            else:
                Pi_t = Pi_0
            Pi_t = maybe_add_expert_to_Pi_t(Pi_t, expert_policy, tag, params)

            if len(Pi_t) > 1:
                # ==== select policy pair ====
                π1, π2, uncertainty = select_policy_pair(
                    Pi_t,
                    pref_model,
                    env,
                    V_inv,
                    params,
                    phi_func,
                    n_samples=params.n_embedding_samples,
                    precomputed_embeddings=precomputed_embeddings,
                )

                # ==== Oracle preferences ====
                # get oracle preference (higher true reward), update V, add sampled trajs and preference labels to buffer
                for _ in range(params.N_rollouts):
                    online_trajs, preference_data, V = get_preferences(
                        env, π1, π2, phi_func, params, online_trajs, preference_data, V
                    )
                V_inv = torch.linalg.inv(V)
                V.detach_()
                V_inv.detach_()

                # ==== Train pref model on buffer ====
                pref_model, prev_w = train_pref_model_maybe_project(
                    pref_model, preference_data, phi_func, t, V_inv, params
                )
            else:  # in case |Pi_t| == 1 we're early stopping. ensure metrics don't break
                prev_w = 0
                uncertainty = 0
                π1, π2 = expert_policy, expert_policy

            # ==== METRICS ====
            metrics = collect_and_print_metrics(
                metrics,
                precomputed_embeddings,
                candidates,
                all_policies,
                policy_names,
                bc_policy,
                expert_policy,
                Pi_0,
                Pi_t,
                V,
                V_inv,
                π1,
                π2,
                uncertainty,
                pref_model,
                prev_w,
                env,
                phi_func,
                loop_start_time,
                t,
                tag,
                params,
            )
            if len(Pi_t) == 1:
                print("%%%    only one policy in Pi_t, terminating algo")
                break

    # some cleanup for logging
    finally:
        if env is not None:
            env.close()
        if original_stdout is not None:
            tee_obj = sys.stdout
            sys.stdout = original_stdout
            tee_obj.close()
        plt.close("all")
    return seed, metrics


# ==== end experiment runner ==== #


# ==== Setup ==== #
def setup_experiment_environment(seed, params):
    expert_policy, env, _ = load_or_train_expert_and_env(params)
    env.seed(seed=seed)  # SB3 VecEnv. In gym, would be env.reset(seed=..)
    if params.env_id in ["Walker2d-v5", "Hopper-v5"]:  # env-specific
        v_index = 8 if params.env_id == "Walker2d-v5" else 5
        v_ref = estimate_speed_scale_from_policy(env, expert_policy, v_index, params.episode_length)
        phi_func = setup_embeddings(params, env, v_ref=v_ref)
    else:
        phi_func = setup_embeddings(params, env)
    print("expert & env loaded, seeds set")
    if params.baseline_or_bridge == "baseline":
        tag = f"(L{seed})"
    elif params.baseline_or_bridge == "bridge":
        tag = f"(B{seed})"
    return tag, expert_policy, env, phi_func


def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# == Setup: Experiment Params == #
@dataclass
class ExperimentParams:
    env_id: str
    seed: int
    N_experiments: int
    N_iterations: int
    episode_length: int
    embedding_name: str
    N_offline_trajs: int
    fresh_offline_trajs: bool
    initial_pos_noise: float  # only HalfCheetah. default is 0.1
    N_confset_size: int
    confset_base: Optional[str]  # "bcnoise", "bignoise", "random"
    confset_dilution: Optional[str]  # "random", "bignoise", None
    N_confset_dilution: int
    confset_noise: float  # noise added to BC policy to generate confset. if not provided, filled in w/ environment defaults
    n_bc_epochs: int
    bc_loss: str  # "mse" or "log-loss"
    bc_print_evals: bool
    radius: float  # for filtering offline confset: L2(embed(π_BC) - embed(π_candidate)) < radius
    expert_in_candidates: bool  # if True, offline confset is [BC, noise, expert]
    expert_in_confset: bool  # if True, expert is added to search space if not present
    expert_in_eval: bool  # if True, expert is added to chosen eval space
    which_eval_space: str  # 'offline_confset', 'pi_zero', 'pi_t' -- only for BRIDGE. these are all the same for baseline
    fresh_embeddings: bool
    # online learning
    N_rollouts: int
    filter_pi_t_yesno: bool
    filter_pi_t_gamma: float
    gamma_debug_mode: bool
    # online pref model
    W: int
    w_regularization: Optional[str]  # "None" or "l2"
    w_epochs: int
    w_initialization: str
    w_sigmoid_slope: int
    project_w: bool
    retrain_w_from_scratch: bool
    # online policy selection
    which_policy_selection: str
    ucb_beta: float
    # online misc
    V_init: str
    n_embedding_samples: int
    # model params
    hidden_dim: int
    # experiment params
    verbose: List[str]
    # these are for experiment setup (needed only to avoid errors)
    run_baseline: bool
    run_bridge: bool
    save_results: bool
    run_ID: str
    loaded_run_behaviour: str
    which_plot_subopt: str
    baseline_or_bridge: str
    plot_scores: str
    exclude_outliers: str
    run_dir: None
    norm_obs: bool  # default False. set to True if env_id is halfcheetah or reacher.
    use_wandb: bool
    plot_slim: bool
    plot_logy: bool


def preprocess_params_dict(params_dict):
    """Takes in params dict from after argparsing, fills in any defaults
    that are missing or None, and returns ExperimentParams obj"""

    defaults = {
        "exclude_outliers": False,
        "plot_scores": False,
        "expert_in_candidates": True,
        "expert_in_confset": False,
        "expert_in_eval": False,
        "which_eval_space": "pi_zero",
        "fresh_embeddings": False,
        "run_dir": None,
        "norm_obs": False,
        "use_wandb": False,
        "confset_base": "bcnoise",
        "confset_dilution": None,
        "N_confset_dilution": 0,
        "plot_slim": True,
        "plot_logy": True,
        "initial_pos_noise": None,
        "gamma_debug_mode": False,
        "bc_print_evals": False,
        "w_regularization": None,
        "w_sigmoid_slope": 1,
        "project_w": False,
        "retrain_w_from_scratch": False,
        "V_init": "small",
        "W": 1,
        "bc_loss": "log-loss",
        "baseline_or_bridge": None,
        "fresh_offline_trajs": False,
        "hidden_dim": None,
    }

    for key, default_value in defaults.items():
        if params_dict.get(key) is None:
            params_dict[key] = default_value

    return ExperimentParams(**params_dict), params_dict


# == Setup: load Env, load/train Expert == #
def load_or_train_expert_and_env(params: ExperimentParams):
    """Returns expert policy (e.g. SB3 PPO) and eval env (VecEnv)"""
    # dict of: model_config (dict), normalize_obs, normalize_reward, n_train_steps, model (class)
    training_cfg = _get_training_config(params)
    if params.initial_pos_noise is not None:  # e.g. for halfcheetah, initial pos&velocity
        noiselevel = str(params.initial_pos_noise).replace(".", "_")
        model_base, model_ext = os.path.splitext(training_cfg["model_path"])
        env_base, env_ext = os.path.splitext(training_cfg["env_path"])
        training_cfg["model_path"] = f"{model_base}_noise{noiselevel}{model_ext}"
        training_cfg["env_path"] = f"{env_base}_noise{noiselevel}{env_ext}"

    # maybe need to train expert:
    if not os.path.exists(training_cfg["model_path"]) or not os.path.exists(
        training_cfg["env_path"]
    ):
        print(f"no expert found, training fresh & saving to {training_cfg['model_path']}")
        os.makedirs(training_cfg["save_path"], exist_ok=True)
        train_env = _make_train_env(params, training_cfg)
        expert_model = _train_expert(train_env, training_cfg, params)

    # load model & eval env with stats
    expert_model = training_cfg["model"].load(training_cfg["model_path"])
    print(f"expert loaded from {training_cfg['model_path']}")
    env = _load_eval_env(params, training_cfg)
    print(f"env loaded from {training_cfg['env_path']}")
    n_expert_samples = 10000
    try:
        avg_reward_expert = _load_expert_reward(
            env, expert_model, training_cfg, params, n_samples=n_expert_samples
        )
    except Exception as e:
        print(f"ERROR: couldn't load expert reward: {e}")
        avg_reward_expert = 0
    print(f"expert avg reward: {avg_reward_expert:.2f} ({n_expert_samples} samples)")

    return expert_model, env, avg_reward_expert


def _train_expert(train_env, training_cfg, params=None):
    """Train expert policy on train env. Saves model and env stats (e.g. SB3 VecNormalize stats)."""
    stamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    total_train_steps = training_cfg["n_train_steps"]

    use_wandb = params.use_wandb if params is not None else False
    if use_wandb:
        experiment_name = f"{training_cfg['model']}_{int(time.time())}"
        params_asdict = asdict(params)
        wandb_config = {
            "exp_params": params_asdict,
            "training_params": training_cfg,
        }
        run = wandb.init(
            name=experiment_name,
            project="BRIDGE_experts",
            config=wandb_config,
            sync_tensorboard=True,  # we don't use TB, but SB3 requires this
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=training_cfg["save_path"] + f"/{stamp}/",
            model_save_freq=total_train_steps // 5,
        )
        tb_logpath = training_cfg["save_path"] + "/tb_logs/"
        model = training_cfg["model"](
            env=train_env, tensorboard_log=tb_logpath, **training_cfg["model_config"]
        )
        model.learn(total_train_steps, callback=wandb_callback)
        run.finish()
    else:
        model = training_cfg["model"](env=train_env, **training_cfg["model_config"])
        checkpoint_callback = CheckpointCallback(
            save_freq=total_train_steps // 5,
            save_path=training_cfg["save_path"] + f"/{stamp}/",
            name_prefix="iter",
        )
        model.learn(total_train_steps, callback=checkpoint_callback, progress_bar=True)

    model.save(training_cfg["model_path"])
    train_env.save(training_cfg["env_path"])

    return model


def _load_expert_reward(env, expert_model, training_cfg, params, n_samples: int = 10000):
    try:
        # env-specific (if env has initial position noise)
        if params.env_id == "HalfCheetah-v5" and params.initial_pos_noise is not None:
            noiselevel = str(params.initial_pos_noise).replace(".", "_")
            exprewardpath = os.path.join(
                training_cfg["save_path"],
                f"expertreward_n{n_samples}_H{params.episode_length}_noise{noiselevel}.json",
            )
        else:
            exprewardpath = os.path.join(
                training_cfg["save_path"],
                f"expertreward_n{n_samples}_H{params.episode_length}.json",
            )
        with open(exprewardpath, "r") as f:
            avg_reward_expert = json.load(f)
            avg_reward_expert = np.float32(avg_reward_expert)
            print(f"loaded expert reward from {exprewardpath}")
    except FileNotFoundError:  # no expert reward file found: calculate it
        print(f"no expert reward file found, calculating it on {n_samples} samples")
        avg_reward_expert = calc_policy_avg_reward(
            env, expert_model, params.episode_length, n_samples=n_samples
        )
        with open(exprewardpath, "w") as f:
            json.dump(str(avg_reward_expert), f)
        print(f"saved expert reward to {exprewardpath}")
    return avg_reward_expert


def _get_training_config(params):
    """Returns config dict incl. model config (e.g. PPO) and
    training params for given environment. uses defaults from RLZoo"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # env-specific
    if params.env_id == "HalfCheetah-v5":
        cfg = {
            "model_config": {
                "policy": "MlpPolicy",
                "learning_rate": 2.0633e-05,
                "n_steps": 512,
                "batch_size": 64,
                "n_epochs": 20,
                "gamma": 0.98,
                "gae_lambda": 0.92,
                "clip_range": 0.1,
                "normalize_advantage": True,
                "ent_coef": 0.000401762,
                "vf_coef": 0.58096,
                "max_grad_norm": 0.8,
                "policy_kwargs": {
                    "log_std_init": -2,
                    "ortho_init": False,
                    "activation_fn": nn.ReLU,
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                },
                "verbose": 0,
                "device": device,  # had set to cpu
            },
            "n_train_steps": int(2e6),  # int(1e6),
            "model": PPO,
            "save_path": "continuous/saved_experts/halfcheetah/ppo",
            "model_file": "ppo_optimal.zip",
            "env_file": "vec_normalize.pkl",
        }
    if params.env_id == "Reacher-v5":
        cfg = {
            "model_config": {
                "policy": "MlpPolicy",
                "learning_rate": 0.000104019,
                "n_steps": 512,
                "batch_size": 32,
                "n_epochs": 5,
                "gamma": 0.9,
                "gae_lambda": 1.0,
                "clip_range": 0.3,
                "ent_coef": 7.52585e-08,
                "vf_coef": 0.950368,
                "max_grad_norm": 0.9,
                "verbose": 0,
                "device": device,  # had set to cpu
            },
            "n_train_steps": int(1e6),
            "model": PPO,
            "save_path": "continuous/saved_experts/reacher/ppo",
            "model_file": "ppo_optimal.zip",
            "env_file": "vec_normalize.pkl",
        }
    if params.env_id == "Ant-v5":
        cfg = {
            "model_config": {
                "policy": "MlpPolicy",
                "learning_rate": 1e-3,
                "learning_starts": 10000,
                "action_noise": NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)),
                "train_freq": 1,
                "gradient_steps": 1,
                "batch_size": 256,
                "policy_kwargs": {
                    "net_arch": [400, 300],
                },
                "device": device,  # had set to mps (but didn't test if cpu faster?)
            },
            "n_train_steps": int(1e6),
            "model": TD3,
            "save_path": "continuous/saved_experts/ant/TD3",
            "model_file": "td3_optimal.zip",
            "env_file": "",  # no normalized obs
        }
    if params.env_id == "Walker2d-v5":
        cfg = {
            "save_path": "continuous/saved_experts/walker",
            "model_file": "TD3_rlzoo.zip",
            "env_file": "",  # no normalized obs
        }
    if params.env_id == "Hopper-v5":
        cfg = {
            "save_path": "continuous/saved_experts/hopper",
            "model_file": "TD3_rlzoo.zip",
            "env_file": "",  # no normalized obs
        }
    if params.env_id == "Humanoid-v5":
        cfg = {
            "model_config": {
                "policy": "MlpPolicy",
                "learning_starts": 10000,
            },
            "n_train_steps": int(2e6),
            "save_path": "continuous/saved_experts/humanoid/sac",
            "model_file": "sac_optimal.zip",
            "env_file": "",  # no normalized obs
        }
    cfg["model_path"] = os.path.join(cfg["save_path"], cfg["model_file"])
    cfg["env_path"] = os.path.join(cfg["save_path"], cfg["env_file"])
    return cfg


def _load_eval_env(params, training_cfg):
    """Setup mujoco envs, for eval any reward normalization is turned off.
    Note:
    - halfcheetah, reacher: RLZoo experts use normalized obs"""
    # env-specific
    if params.env_id not in [
        None,
        "HalfCheetah-v5",
        "Reacher-v5",
        "Walker2d-v5",
        "Hopper-v5",
        "Ant-v5",
        "Humanoid-v5",
    ]:
        raise ValueError(f"Environment {params.env_id} not supported in _load_eval_env")

    elif params.env_id == "HalfCheetah-v5":
        reset_noise_scale = (
            params.initial_pos_noise if params.initial_pos_noise is not None else 0.1
        )

        def make_eval_env():
            base_env = gym.make(params.env_id, reset_noise_scale=reset_noise_scale)
            timelimited_env = TimeLimit(base_env, max_episode_steps=params.episode_length)
            return timelimited_env

    elif params.env_id in ["Reacher-v5", "Walker2d-v5", "Hopper-v5", "Ant-v5", "Humanoid-v5"]:

        def make_eval_env():
            base_env = gym.make(params.env_id)
            timelimited_env = TimeLimit(base_env, max_episode_steps=params.episode_length)
            return timelimited_env

    env = make_vec_env(make_eval_env, n_envs=1)
    if params.env_id in ["HalfCheetah-v5", "Reacher-v5"]:
        env = VecNormalize.load(training_cfg["env_path"], env)
        env.training = False  # don't update statistics
        env.norm_reward = False
    elif params.env_id in ["Walker2d-v5", "Hopper-v5", "Ant-v5", "Humanoid-v5"]:
        env.norm_obs = False
    return env


class MakeObsFloat32Wrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0],), dtype=np.float32
        )

    def observation(self, obs):
        return obs  # obs.astype(np.float32)


def _make_train_env(params, training_cfg):
    """creates VecEnv for training expert in.
    crucially:
    - no time limit
    - VecNormalize (obs and reward norm) with statistics turned on
    """
    train_env = make_vec_env(params.env_id, n_envs=1, wrapper_class=MakeObsFloat32Wrapper)

    if params.env_id == "HalfCheetah-v5" or params.env_id == "Reacher-v5":
        train_env = VecNormalize(
            train_env, gamma=training_cfg["model_config"]["gamma"]
        )  # because normalize: True, and gamma=0.98
    return train_env


# == Setup: precomputing embeddings == #
def create_embedding_filename(policy_set, seed, base, dilution):
    """Create a unique hash for the policy set and embedding parameters to use as cache key."""
    policy_hashes = sorted([hash_policy(policy) for policy in policy_set])

    # Create a string representation of all the key parameters
    cache_key_components = [
        str(policy_hashes),
    ]

    combined_string = "|".join(cache_key_components)
    hash_obj = hashlib.sha256(combined_string.encode())
    short_hash = hash_obj.hexdigest()[:16]
    filename = (
        f"embeddings_seed{seed}_b{base.capitalize()}_d{dilution.capitalize()}_{short_hash}.pkl"
    )
    return filename


def precompute_policy_embeddings(policy_set, phi_func, env, params):
    """Precompute (embedding, expected reward) tuples for all policies in a policy set. Use if rollouts are costly."""
    precomputed_embeddings = {}
    for policy in policy_set:
        policy_hash = hash_policy(policy)  # should work for both GaussianPolicy and SB3 agent
        embedding, exp_reward = get_policy_embedding(
            phi_func, policy, env, params.episode_length, n_samples=params.n_embedding_samples
        )
        precomputed_embeddings[policy_hash] = (embedding, exp_reward)
    return precomputed_embeddings


def get_policy_embedding(
    phi_func, policy, env, episode_length, n_samples=1000, provided_trajs=None
):
    """estimates ϕ(policy) = E_policy[ϕ(traj)]"""
    if provided_trajs is not None:
        mean_embedding = np.mean([phi_func(traj) for traj in provided_trajs], axis=0)
        mean_reward = np.mean([traj.total_return() for traj in provided_trajs])
        return mean_embedding, mean_reward
    else:
        embeddings = []
        rewards = []
        for _ in range(n_samples):
            traj = rollout_sb3(env, policy, episode_length, deterministic=True)
            embeddings.append(phi_func(traj))
            rewards.append(traj.total_return())
        return torch.stack(embeddings, axis=0).mean(0), np.mean(rewards)


# ==== end Setup ==== #


# ==== Trajectory generation ==== #
def rollout_sb3(
    env,
    policy,
    episode_length,
    deterministic=True,
) -> Trajectory:
    """Annoying: need to pass extract_extras, right now only for HalfCheetah (get x-pos)"""
    states, actions, rewards, infos, true_obs = [], [], [], [], []
    # try:
    #     normalized_obs = env.norm_obs
    # except AttributeError:
    #     pass
    obs = env.reset()
    for _ in range(episode_length):
        act, _ = policy.predict(obs, deterministic)  # works for both ours and SB3
        next_obs, reward, done, info = env.step(act)  # Assume SB3 VecEnv!
        if env.norm_obs:
            true_ob = env.unnormalize_obs(obs).squeeze(0)
            true_obs.append(true_ob)
        states.append(obs.squeeze(0))
        actions.append(act.squeeze(0))
        rewards.append(reward.squeeze(0))
        infos.append(info[0])
        obs = next_obs
        if done:
            break
    if not env.norm_obs:
        true_obs = None
    return Trajectory(states, actions, rewards, infos, true_obs)


def generate_offline_trajectories(env, expert_policy, params):
    """returns offline dataset as list of Trajectory objects"""
    offline_trajs = []
    for _ in range(params.N_offline_trajs):
        traj = rollout_sb3(
            env=env,
            policy=expert_policy,
            episode_length=params.episode_length,
            deterministic=True,
        )
        offline_trajs.append(traj)
    return offline_trajs


def load_or_generate_offline_trajs(env, expert_policy, params, reset_seed=False):
    maybe_expert_str = getattr(params, "which_expert", "")
    maybe_expert_str = maybe_expert_str + "_" if maybe_expert_str else ""
    if reset_seed:
        traj_save_path = f"continuous/offline_trajs/{params.env_id}/eplength{params.episode_length}-{maybe_expert_str}{params.N_offline_trajs}trajs/seed{params.seed}.pkl"
    else:
        traj_save_path = f"continuous/offline_trajs/{params.env_id}/eplength{params.episode_length}-{maybe_expert_str}{params.N_offline_trajs}trajs_fixedseed/seed{params.seed}.pkl"
        # traj_save_path = f"continuous/offline_trajs/{params.env_id}/eplength{params.episode_length}-{maybe_expert_str}{params.N_offline_trajs}trajs.pkl"
    if os.path.exists(traj_save_path) and not params.fresh_offline_trajs:
        with open(traj_save_path, "rb") as f:
            offline_trajs = pickle.load(f)
        print(f"loaded offline trajs from {traj_save_path}")
    else:
        if reset_seed:
            _set_random_seeds(params.seed)
            env.seed(seed=params.seed)
        offline_trajs = generate_offline_trajectories(env, expert_policy, params)
        os.makedirs(os.path.dirname(traj_save_path), exist_ok=True)
        with open(traj_save_path, "wb") as f:
            pickle.dump(offline_trajs, f)
        print(f"generated offline trajs and saved to {traj_save_path}")
    return offline_trajs


# ==== end trajectory generation ==== #


# ==== Policies ==== #
class GaussianPolicy(nn.Module):
    """Gaussian policy, with MLP.

    Note: to get an action, call the external function policy_action.
    Reason to not have .act() or .predict() is that we want the BRIDGE code
    to work with SB3 policies, with ours, and with obs from SB3 VecEnvs or gym envs
    """

    # init
    def __init__(self, state_dim, action_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2)
        self.device = device
        # self._computing_hash = False  # add recursion guard

    # forward: used in training
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    # predict: used in eval
    def predict(self, obs, deterministic=False):
        """predicts action given obs (only used in eval)"""
        single_obs = False
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            single_obs = True

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mean, std = self(obs)[:2]  # forward pass
            if deterministic:
                action = mean
            else:
                action = torch.normal(mean, std)
            action = action.detach().cpu().numpy()
            if single_obs:
                action = action.squeeze(0)
        return action, None  # None for hidden state

    def get_param_hash(self, truncate_to=16):
        # if self._computing_hash:
        # return "hash-recursion-detected"
        # self._computing_hash = True
        # try:
        param_bytes = []
        # for param in self.parameters():
        # param_bytes.append(param.data.cpu().numpy().tobytes())
        for param in self.state_dict().values():
            param_bytes.append(param.cpu().numpy().tobytes())
        combined = b"".join(param_bytes)
        hash_obj = hashlib.sha256(combined)
        if truncate_to:
            return hash_obj.hexdigest()[:truncate_to]
        return hash_obj.hexdigest()
        # finally:
        # self._computing_hash = False

    def __hash__(self):
        return int(self.get_param_hash(), 16)  # truncates 16 chars (by default), of base 16

    def __eq__(self, other):
        if not isinstance(other, GaussianPolicy):
            return False
        return torch.allclose(
            torch.cat([p.flatten() for p in self.parameters()]),
            torch.cat([p.flatten() for p in other.parameters()]),
        )
        # self.get_param_hash() == other.get_param_hash()


def hash_policy(policy, truncate_to=16):
    """returns hash (str) for both GaussianPolicy and SB3 policies (e.g. PPO), by default truncates to 16 chars"""
    if hasattr(policy, "get_param_hash"):  # isinstance(policy, GaussianPolicy)
        hex_str = policy.get_param_hash(truncate_to)
        return int(hex_str, 16)
    elif hasattr(policy, "policy"):  # isinstance(policy, PPO)
        param_bytes = []
        for param in policy.policy.parameters():
            param_bytes.append(param.data.cpu().numpy().tobytes())
        combined = b"".join(param_bytes)
        hash_obj = hashlib.sha256(combined)
        hex_str = hash_obj.hexdigest()[:truncate_to]
        return int(hex_str, 16)
    else:
        raise ValueError(f"Policy type {type(policy)} not supported in hash_policy")


# ==== end Policies ==== #


# ==== Offline learning: Behavioral Cloning ==== #
def train_bc_policy_with_defaults(offline_trajs, env, expert_policy, params):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Get hidden dim from expert (hide this logic)
    if params.env_id in ["Reacher-v5", "HalfCheetah-v5"]:
        hidden_dim = expert_policy.policy.mlp_extractor.latent_dim_pi  # PPO
    else:
        hidden_dim = 256  # TD3, SAC

    bc_policy = GaussianPolicy(state_dim, action_dim, hidden_dim)

    print("training BC policy...")
    start_time = time.time()

    bc_policy, _, _ = train_behavioral_cloning(
        offline_trajs,
        bc_policy,
        n_epochs=params.n_bc_epochs,
        batch_size=32,
        lr=1e-3,
        which_bc=params.bc_loss,
        print_eval=params.bc_print_evals,
        print_every=10,
        start_at_epoch=0,
        eval_env=env,
        params=params,
    )

    print(f"...BC policy trained in {time.time() - start_time:.2f}s")
    return bc_policy


def train_behavioral_cloning(
    offline_trajs: List[Trajectory],
    policy: GaussianPolicy,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    which_bc: str = "log-loss",
    optimizer_state=None,
    print_eval: bool = False,
    print_every: int = 10,
    start_at_epoch: int = 0,
    eval_env: Optional[VecEnv] = None,
    params: ExperimentParams = None,
):
    """Train a given GaussianPolicy using log-loss BC. Returns trained GaussianPolicy and optimizer state.

    To continue training an already trained policy, pass in optimizer_state.

    Options for which_bc:
    - log-loss: min. loss=-mean_{batch (s,a) ~ D}(log(p(a|s))), policy stochastic
    - mse: min. loss=mean_batch((a - mean(a|s))^2), policy deterministic
    """
    device = torch.device("cpu")
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    states = torch.tensor([s for traj in offline_trajs for s in traj.states], dtype=torch.float32)
    actions = torch.tensor([a for traj in offline_trajs for a in traj.actions], dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(params.seed)
    for epoch in range(n_epochs):
        # indices = torch.randperm(len(states))
        indices = torch.randperm(len(states), generator=generator)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(states), batch_size):
            idx = indices[i : i + batch_size]
            s_batch = states[idx]
            a_batch = actions[idx]
            mean, std = policy(s_batch)

            if which_bc == "mse":
                loss = nn.functional.mse_loss(mean, a_batch)
            elif which_bc == "log-loss":  # negative log-likelihood loss
                var = std.pow(2) + 1e-6
                nll = 0.5 * (((a_batch - mean) / std).pow(2) + torch.log(2 * np.pi * var))
                nll = nll.sum(dim=1)
                loss = nll.mean()
            else:
                raise ValueError(f"Invalid BC loss: {which_bc}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        actual_epoch = start_at_epoch + epoch
        print_epoch = print_every and ((actual_epoch + 1) % print_every == 0)
        if print_epoch:
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {actual_epoch + 1}/{start_at_epoch + n_epochs}, Loss: {avg_loss:.4f}")

        if print_eval and print_epoch:
            foo_trajs = [
                rollout_sb3(
                    eval_env,
                    policy,
                    episode_length=params.episode_length,
                    deterministic=True,
                )
                for _ in range(100)
            ]
            foo_rewards = [traj.compute_cumulative_reward() for traj in foo_trajs]
            avg_reward = np.mean(foo_rewards)
            std_reward = np.std(foo_rewards)

            if params.env_id == "HalfCheetah-v5":
                avg_x_pos = np.mean([traj.get_final_x_position() for traj in foo_trajs])
                print(
                    f"  (train) avg distance achieved by BC: {avg_x_pos:.2f}, reward: {avg_reward:.2f}"
                )
            elif params.env_id == "Reacher-v5":
                # Get distance to target from last state (indices 8 and 9 are agent-target x and y)
                avg_dist_to_target = np.mean(
                    [
                        np.sqrt(traj.true_obs[-1][8] ** 2 + traj.true_obs[-1][9] ** 2)
                        for traj in foo_trajs
                    ]
                )
                print(
                    f"  (train) avg distance to target by BC: {avg_dist_to_target:.2f}, reward: {avg_reward:.2f}"
                )
            elif params.env_id == "Walker2d-v5":
                survivaltimes = [len(traj.states) for traj in foo_trajs]
                dists = [traj.get_final_x_position() for traj in foo_trajs]
                print(
                    f"  (train) BC reward {avg_reward:.2f} (± {std_reward:.2f}), survival: {np.mean(survivaltimes):.2f} (± {np.std(survivaltimes):.2f}), dist {np.mean(dists):.2f} (± {np.std(dists):.2f})"
                )

    return policy, optimizer.state_dict(), actual_epoch + 1


# ==== end Offline learning: Behavioral Cloning ==== #


# ==== Offline learning: generate confset ==== #
def generate_policy_sets(bc_policy, expert_policy, params):
    print(
        f"generating set of {params.N_confset_size} candidates w/ noise {params.confset_noise}..."
    )
    offline_confset = generate_confset(
        bc_policy,
        num_candidates=params.N_confset_size,
        noise_std=params.confset_noise,
        confset_base=params.confset_base,
        confset_dilution=params.confset_dilution,
        N_confset_dilution=params.N_confset_dilution,
    )  # = confset_base + confset_dilution (may not contain BC if base is not bcnoise!)
    candidates = offline_confset.copy()  # =offline_confset + expert, defines Pi_0, Pi_t.
    if params.expert_in_candidates:
        candidates.append(expert_policy)

    all_policies = offline_confset.copy()  # =BC + offline_confset + expert, used for metrics later
    if params.confset_base in ["bignoise", "random"]:
        all_policies = [bc_policy] + all_policies
    all_policies.append(expert_policy)  # [BC, confset_base, confset_dilution, expert]
    return offline_confset, candidates, all_policies


def generate_confset(
    bc_policy: GaussianPolicy,
    num_candidates: int,
    noise_std: float,
    confset_base: str,
    confset_dilution: Optional[str],
    N_confset_dilution: int,
) -> List[GaussianPolicy]:
    """Generate a set of policy candidates around the BC policy.
    Base: "bcnoise": BC, BC+noise; "bignoise": BC+10x noise; "random": random policies
    Dilution: "random": random policies; "bignoise": BC+10x noise; None: none.

    """
    state_dim = bc_policy.net[0].in_features
    action_dim = bc_policy.mean.out_features
    hidden_dim = bc_policy.net[0].out_features
    # confset base policies
    candidates = []
    if confset_base == "bcnoise":
        candidates = [bc_policy]
        noise_mult = 1
    elif confset_base == "bignoise":
        noise_mult = 5
    if confset_base in ["random", "bcnoise", "bignoise"]:
        for _ in range(num_candidates):
            candidate = GaussianPolicy(state_dim, action_dim, hidden_dim)
            if confset_base in ["bcnoise", "bignoise"]:
                candidate.load_state_dict(bc_policy.state_dict())
                for param in candidate.parameters():
                    param.data += noise_mult * noise_std * torch.randn_like(param)
            candidates.append(candidate)
    else:
        raise ValueError(
            f"Invalid confset base: {confset_base}. Choices 'bcnoise', 'bignoise', 'random'."
        )

    # dilution policies
    dilution = []
    if confset_dilution == "random":
        dilution = [
            GaussianPolicy(state_dim, action_dim, hidden_dim) for _ in range(N_confset_dilution)
        ]
    elif confset_dilution == "bignoise":
        for _ in range(N_confset_dilution):
            dilution_cand = GaussianPolicy(state_dim, action_dim, hidden_dim)
            dilution_cand.load_state_dict(bc_policy.state_dict())
            for param in dilution_cand.parameters():
                param.data += 5 * noise_std * torch.randn_like(param)
            dilution.append(dilution_cand)
    elif confset_dilution is not None:
        print(type(confset_dilution))
        raise ValueError(
            f"Invalid confset dilution: {confset_dilution}. Choices 'random', 'bignoise', None."
        )
    candidates.extend(dilution)
    return candidates


def l2_dist_vs_bc(
    policy, bc_embedding, phi_func, env, episode_length, n_samples=100, precomputed_embeddings=None
):
    """Calculates L2 distance between policy embedding and BC embedding (or theoretically any other already computed embedding)"""
    if precomputed_embeddings is not None:
        pi_embedding, _ = precomputed_embeddings[hash_policy(policy)]
    else:
        pi_embedding, _ = get_policy_embedding(phi_func, policy, env, episode_length, n_samples)
    return torch.norm(pi_embedding - bc_embedding, p=2).item()


# ==== end Offline learning: generate confset ==== #


# ==== Bridging: precompute embeddings for policy list ==== #
def get_precomputed_embeddings_names(
    bc_policy, expert_policy, offline_confset, all_policies, phi_func, tag, env, params
):
    # for printing, give each candidate policy a name
    policy_names = {}
    policy_names[hash_policy(bc_policy)] = "bc"
    policy_names[hash_policy(expert_policy)] = "exp"
    for i, policy in enumerate(all_policies[1:-1]):  # skip BC & expert
        policy_names[hash_policy(policy)] = f"n{i}"

    # precompute embeddings (or load from disk if already exist)
    precomp_embed_filename = create_embedding_filename(
        all_policies, params.seed, str(params.confset_base), str(params.confset_dilution)
    )
    precomp_embed_filedir = f"continuous/cached_embeddings/{params.env_id}/{phi_func.name}/n{params.n_embedding_samples}"
    precomp_embed_filepath = os.path.join(precomp_embed_filedir, precomp_embed_filename)
    recompute_embeds = params.fresh_embeddings or not os.path.exists(precomp_embed_filepath)
    if not recompute_embeds:
        with open(precomp_embed_filepath, "rb") as f:
            precomputed_embeddings = pickle.load(f)
        print(f"...loaded precomputed embeddings {precomp_embed_filename} from cache {tag}")
    else:
        print(
            f"Computing policy embeddings {precomp_embed_filename} on {params.n_embedding_samples} samples... {tag}"
        )
        precomputed_embeddings = precompute_policy_embeddings(
            offline_confset, phi_func, env, params
        )
        print(f"precomputing BC and expert embeddings on 1000 samples {tag}")

        # for BC and expert, use high-fidelity estimates. BC for filtering BRIDGE, expert for eval.
        precomputed_embeddings[hash_policy(bc_policy)] = get_policy_embedding(
            phi_func, bc_policy, env, params.episode_length, n_samples=1000
        )
        precomputed_embeddings[hash_policy(expert_policy)] = get_policy_embedding(
            phi_func, expert_policy, env, params.episode_length, n_samples=1000
        )

        # save to disk
        os.makedirs(precomp_embed_filedir, exist_ok=True)
        with open(precomp_embed_filepath, "wb") as f:
            pickle.dump(precomputed_embeddings, f)

    bc_embed_precalc, _ = precomputed_embeddings[hash_policy(bc_policy)]
    return precomputed_embeddings, policy_names, bc_embed_precalc, recompute_embeds


# ==== end Bridging: precompute embeddings for policy list ==== #


# ==== Bridging: get initial online search space ==== #
def get_online_search_space(
    candidates,
    bc_policy,
    expert_policy,
    phi_func,
    env,
    precomputed_embeddings,
    tag,
    params,
):
    if params.baseline_or_bridge == "baseline":
        print("running baseline -- not filtering confset")
        Pi_0 = candidates  # .copy()

    # BRIDGE: online searches in filtered candidates set:
    # only pick policies close to pi_BC (in terms of L2 distance in embedding space)
    elif params.baseline_or_bridge == "bridge":
        filteringtime = time.time()
        print(
            f"filtering confset once before loop... epsilon = {phi_func.filter_epsilon:.2f} w/ {phi_func.name}"
        )
        Pi_0 = []
        for candidate in candidates:
            dist = l2_dist_vs_bc(
                candidate,
                precomputed_embeddings[hash_policy(bc_policy)][0],  # BC embedding
                phi_func,
                env,
                params.episode_length,
                n_samples=params.n_embedding_samples,
                precomputed_embeddings=precomputed_embeddings,
            )
            if dist < phi_func.filter_epsilon:
                Pi_0.append(candidate)
        if len(Pi_0) == 0:
            raise ValueError("No candidates left after filtering (impossible as BC has L2=0)")
        print(f"|Pi_0| = {len(Pi_0)}/{len(candidates)}")
        print(f"...filtered offline confset in {time.time() - filteringtime:.2f}s")

    # test if expert is in initial training set Pi_0 (should be) via checking hashes
    Pi_0_hashes = [hash_policy(π) for π in Pi_0]
    if hash_policy(expert_policy) not in Pi_0_hashes:
        print("=" * 80)
        if params.expert_in_confset:
            Pi_0.append(expert_policy)
            print(f"WARNING {tag}: EXPERT HAD TO BE ADDED TO Pi_0!")
        else:
            print(f"WARNING {tag}: EXPERT POLICY NOT IN SET Pi_0!")
        print("=" * 80)
    return Pi_0


def make_cand_bc_scatterplots(
    all_policies,
    precomputed_embeddings,
    bc_policy,
    expert_policy,
    phi_func,
    recompute_embeds,
    params,
):
    """make a plot of dists-to-bc and rewards, save it"""
    os.makedirs(f"{params.run_dir}/BC-cand-scatterplots", exist_ok=True)
    fig_scatter_path = f"{params.run_dir}/BC-cand-scatterplots/seed{params.seed}.png"
    if not os.path.exists(fig_scatter_path):
        dists = []
        rewards = []
        bc_embed = precomputed_embeddings[hash_policy(bc_policy)][0]
        for π in all_policies:
            π_embed = precomputed_embeddings[hash_policy(π)][0]
            dist = torch.norm(π_embed - bc_embed, p=2).item()
            dists.append(dist)
            rewards.append(precomputed_embeddings[hash_policy(π)][1])

        assert rewards[0] == precomputed_embeddings[hash_policy(bc_policy)][1]
        assert rewards[-1] == precomputed_embeddings[hash_policy(expert_policy)][1]
        _, fig_scatter = plot_dists_rewards(
            dists,
            rewards,
            phi_func.filter_epsilon,
            title=f"env: {params.env_id}, embed: {phi_func.name} (fresh: {recompute_embeds}), traj: {params.N_offline_trajs} (fresh: {params.fresh_offline_trajs}),\nbc_epochs: {params.n_bc_epochs}, confset_noise: {params.confset_noise}, n_samples: {params.n_embedding_samples}",
        )
        fig_scatter.savefig(fig_scatter_path)


# ==== end Bridging: get initial online search space ==== #


# ==== Online learning: Preference Learning ==== #
def initialize_preference_learning(phi_func, params):
    pref_model = PreferenceModel(phi_func.dim, params)
    if params.V_init == "small":
        V = torch.eye(phi_func.dim) * 1e-2
    elif params.V_init == "bounds":
        V = torch.eye(phi_func.dim) * phi_func.bound  # bound is ca in [1, 10].
    V_inv = torch.linalg.inv(V)
    return pref_model, V, V_inv


class PreferenceModel(nn.Module):
    def __init__(self, embedding_dim, params):
        super().__init__()
        if params.w_initialization == "zeros":
            self.w = torch.zeros(embedding_dim)
        elif params.w_initialization == "uniform":
            self.w = torch.ones(embedding_dim)
            self.w = self.w / torch.norm(self.w, p=2) * params.W
        elif params.w_initialization == "random":
            self.w = torch.randn(embedding_dim)
        self.w = nn.Parameter(self.w)

    def score(self, ϕ):  # <ϕ(traj), w>
        return torch.matmul(ϕ, self.w)

    def forward(self, diff: torch.Tensor) -> torch.Tensor:  # sigmoid(<ϕ(traj), w>) = probability
        return torch.sigmoid(torch.matmul(diff, self.w))

    def normalize(self, bound):
        with torch.no_grad():
            self.w.data = self.w / torch.norm(self.w, p=2) * bound

    def project(self, bound, V_inv, pref_buffer, reg_param=0.1):
        """project learned w_MLE vector to ball ||.||<W, resulting in w_proj (like in tabular case).

        Formula: w_proj = argmin_{w in d-dim ball of radius W} ||g_t(w) - g_t(w_MLE)||_{V_t^{-1}}
        where the matrix norm is the Mahalanobis norm with respect to V_t^{-1},
        and where g_t(w) = sum_{i=0}^{t-1} σ(Δϕ_iᵀ w)*Δϕ_i + lambda_param * w

        Modifies self.w in place.
        """
        if torch.norm(self.w) <= bound:
            print("||w|| < W, skipping projection")
            return

        w_MLE = self.w.detach().numpy().astype(np.float64)

        def g(w_):
            w_ = np.array(w_, dtype=np.float64)  # ensure w_ is float64
            result = np.zeros_like(w_, dtype=np.float64)
            for diff, _ in pref_buffer:
                diff_np = diff.detach().numpy().astype(np.float64)
                sigmoid_val = 1 / (1 + np.exp(-np.dot(diff_np, w_MLE)))
                result += sigmoid_val * diff_np
            result += reg_param * w_
            return result

        def objective_func(w_):
            """find w_ = argmin ||g(w_) - g(self.w)||_{V_inv}"""
            w_ = np.array(w_, dtype=np.float64)  # ensure w_ is float64
            return mahalanobis(g(w_), g(w_MLE), V_inv)

        # Constraint: ||w||_2 <= W (bound)
        constraint = {"type": "ineq", "fun": lambda w: float(bound - np.linalg.norm(w))}  # >= 0

        # initial guess: use w_MLE if within the ball, else project to edge
        initial_w = bound * (w_MLE / np.linalg.norm(w_MLE))
        initial_w = np.array(initial_w, dtype=np.float64)

        # solve the optimization problem
        try:
            result = minimize(objective_func, initial_w, constraints=constraint, method="SLSQP")
            if result.success:
                self.w.data = torch.tensor(result.x, dtype=torch.float32)
                self.w.requires_grad = True
            else:
                self.w.data = bound * (w_MLE / np.linalg.norm(w_MLE))
                self.w.requires_grad = True
        except Exception as e:
            print(f"Error during optimization: {e}\nFalling back to simple normalization.")
            self.normalize(bound)


def train_pref_model_maybe_project(pref_model, preference_data, phi_func, t, V_inv, params):
    prev_w = pref_model.w.detach().clone().numpy()
    if params.retrain_w_from_scratch:
        pref_model = None
        print(f"  {t}: training pref model (from scratch")
    else:
        print(f"  {t}: training pref model continually")
    trainpreftime = time.time()
    pref_model = train_preference_model(
        pref_model,
        preference_data,
        phi_func.dim,
        params,
        epochs=params.w_epochs,
        lr=1e-2,
    )
    print(f"  {t}: ...trained pref model in {time.time() - trainpreftime:.2f}s")

    ## (optionally: project pref model vector to W-ball)
    ## regularization param is lambda = B/kappa roughly 0.1
    if params.project_w:
        projecttime = time.time()
        pref_model.project(params.W, V_inv, preference_data, reg_param=0.01)
        print(f"  {t}: ...projected pref model in {time.time() - projecttime:.2f}s")
    return pref_model, prev_w


def train_preference_model(model, data, embedding_dim, params, epochs=1, lr=1e-2):
    """Idea: get MLE of w on preference data
    - train on all data, full-batch
    - reinitialize w each call if model is None
    - L2 via weight_decay
    - can use sigmoid slope to control differentiation of diffs close to 0
    """
    if model is None:
        model = PreferenceModel(embedding_dim, params)

    if params.w_regularization == "l2":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.stack([diff for diff, _ in data]).float()
    y = torch.tensor([label for _, label in data], dtype=torch.float32)

    epoch_print_interval = epochs // 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = params.w_sigmoid_slope * torch.matmul(X, model.w)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % epoch_print_interval == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return model


# ==== end Online learning: Preference Model ==== #


# ==== Online learning: refine confset ==== #
def refine_confset(
    t,
    confset,
    pref_model,
    env,
    phi_func,
    episode_length,
    V_inv,
    gamma,
    gamma_debug_mode=False,
    precomputed_embeddings=None,
    n_samples=100,
):
    """refines online confidence set Pi_0 into Pi_t. Checks any member π for likelihood to be the best,
    i.e. for each pairwise comparison, its score difference plus an uncertainty bonus is > 0 (and thus win probability > 50%):
      for all π': <ϕ(π) - ϕ(π'), w> + γ * sqrt[(ϕ(π) - ϕ(π'))^T V_inv (ϕ(π) - ϕ(π'))] > 0
      where
      - ϕ(π) is the embedding of π
      - w is the preference model vector
      - V_inv is the inverse of the covariance matrix of the preference model
      - γ is a hyperparameter that controls the amount of uncertainty bonus
    if the criterion is satisfied, π is added to Pi_t.
    """
    if precomputed_embeddings is None:
        precomputed_embeddings = {}
        for π in confset:
            precomputed_embeddings[hash_policy(π)] = get_policy_embedding(
                phi_func, π, env, episode_length, n_samples=n_samples
            )
    Pi_t = []
    tot_num_calcs = 0
    avg_abs_scorediff = 0
    avg_uncertainty = 0
    avg_sum_term = 0
    gammas_needed = []
    for π1 in confset:
        all_conditions_satisfied = True
        pi1_gammas = []
        for π2 in confset:
            if π1 == π2:
                continue
            ϕ1 = precomputed_embeddings[hash_policy(π1)][0]
            ϕ2 = precomputed_embeddings[hash_policy(π2)][0]
            diff = ϕ1 - ϕ2
            score = pref_model.score(diff).detach()
            uncertainty = torch.sqrt(diff @ V_inv @ diff).detach()
            sum_term = score + gamma * uncertainty
            tot_num_calcs += 1
            avg_abs_scorediff += abs(score)
            avg_uncertainty += uncertainty
            avg_sum_term += sum_term
            gamma_needed = -score / uncertainty
            pi1_gammas.append(gamma_needed)
            if sum_term < 0:
                all_conditions_satisfied = False
                if not gamma_debug_mode:
                    break  # idea: if one condition is not satisfied, we can stop checking the rest (but if we want to output the γ thresholds, we need to check all)
        gammas_needed.append(max(pi1_gammas))
        if all_conditions_satisfied:
            Pi_t.append(π1)
    avg_abs_scorediff /= tot_num_calcs
    avg_uncertainty /= tot_num_calcs
    avg_sum_term /= tot_num_calcs
    if gamma_debug_mode:
        print(f"  {t}: confset refinement: |Pi_t| = {len(Pi_t)}")
        print(
            f"  {t}:   formula: <scorediff> ({avg_abs_scorediff:.3f}) + gamma ({gamma:.2f}) * <uncertainty> ({avg_uncertainty:.2f})"
        )
        print(f"  {t}:   avg sum term: {avg_sum_term:.3f}")
        if gammas_needed:
            gammas_needed_sorted = sorted(gammas_needed)
            percentiles = [25, 50, 75, 90, 100]
            printstr = f"  {t}:   γ thresholds needed to retain"
            for p in percentiles:
                idx = min(int(p / 100 * len(gammas_needed_sorted)), len(gammas_needed_sorted) - 1)
                printstr += f" {p}%: γ={gammas_needed_sorted[idx]:.3f},"
            print(printstr)
    return Pi_t


def maybe_add_expert_to_Pi_t(Pi_t, expert_policy, tag, params):
    Pi_t_hashes = [hash_policy(π) for π in Pi_t]
    if (hash_policy(expert_policy) not in Pi_t_hashes) and params.expert_in_confset:
        Pi_t.append(expert_policy)
        print(f"WARNING {tag}: EXPERT HAD TO BE ADDED TO Pi_t!")
    return Pi_t


# ==== end Online learning: refine confset ==== #


# ==== Online learning: pair selection ==== #
def select_policy_pair(
    Pi_t, pref_model, env, V_inv, params, phi_func, n_samples=100, precomputed_embeddings=None
):
    ep_length = params.episode_length
    if precomputed_embeddings is None:
        precomputed_embeddings = {}
        for π in Pi_t:
            precomputed_embeddings[hash_policy(π)] = get_policy_embedding(
                phi_func, π, env, ep_length, n_samples=n_samples
            )

    if params.which_policy_selection == "ucb":
        # highest σ(Δϕᵀ w) + β * sqrt(Δϕᵀ V_inv Δϕ)
        π1, π2, best_score = select_ucb_pair(
            Pi_t,
            pref_model,
            V_inv,
            env,
            phi_func,
            episode_length=ep_length,
            n_samples=n_samples,
            beta=params.ucb_beta,
            precomputed_embeddings=precomputed_embeddings,
        )
        print(f"  -> selection via ucb, score: {best_score:.2f}")
    elif params.which_policy_selection == "random":
        print("  -> selection via pi1: get_best_policy, pi2: random")
        π1 = get_best_policy(
            Pi_t,
            pref_model,
            env,
            phi_func,
            ep_length,
            n_samples=n_samples,
            precomputed_embeddings=precomputed_embeddings,
        )
        remaining_policies = [π for π in Pi_t if π != π1]
        π2 = np.random.choice(remaining_policies, size=1, replace=False)[0]
    elif params.which_policy_selection == "max_uncertainty":
        # highest sqrt(Δϕᵀ V_inv Δϕ)
        π1, π2, max_uncertainty = select_max_uncertainty_pair(
            Pi_t,
            V_inv,
            env,
            phi_func,
            ep_length,
            n_samples=n_samples,
            precomputed_embeddings=precomputed_embeddings,
        )
        print(f"  -> selection via max uncertainty, uncertainty: {max_uncertainty:.2f}")
    else:
        raise ValueError(f"Invalid policy selection method: {params.which_policy_selection}")

    # calc uncertainty for metrics
    Δϕ = precomputed_embeddings[hash_policy(π1)][0] - precomputed_embeddings[hash_policy(π2)][0]
    uncertainty = torch.sqrt(Δϕ @ V_inv @ Δϕ).detach().cpu().numpy()
    return π1, π2, uncertainty


def select_max_uncertainty_pair(
    policies, V_inv, env, phi_func, episode_length, n_samples=5, precomputed_embeddings=None
):
    """selects the pair of policies with the highest uncertainty in their score difference"""
    max_uncertainty = -np.inf
    best_pair = None
    if precomputed_embeddings is not None:
        embeddings = [precomputed_embeddings[hash_policy(π)][0] for π in policies]
    else:
        embeddings = [
            get_policy_embedding(phi_func, π, env, episode_length, n_samples=n_samples)[0]
            for π in policies
        ]
    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            Δϕ = embeddings[i] - embeddings[j]
            uncertainty = torch.sqrt(Δϕ @ V_inv @ Δϕ)  # same as scipy mahalanobis
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_pair = (i, j)
    return policies[best_pair[0]], policies[best_pair[1]], max_uncertainty


def select_ucb_pair(
    policies,
    model,
    V_inv,
    env,
    phi_func,
    episode_length,
    n_samples=5,
    beta=1,
    precomputed_embeddings=None,
):
    best_score = -np.inf
    best_pair = None
    if precomputed_embeddings is not None:
        embeddings = [precomputed_embeddings[hash_policy(π)][0] for π in policies]
    else:
        embeddings = [
            get_policy_embedding(phi_func, π, env, episode_length, n_samples=n_samples)[0]
            for π in policies
        ]

    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            Δϕ = embeddings[i] - embeddings[j]
            score = ucb_score(model, Δϕ, V_inv, beta)
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    return policies[best_pair[0]], policies[best_pair[1]], best_score


def ucb_score(model, delta_phi, V_inv, beta):
    """UCB score: mean + beta * sqrt(delta_phi^T V_inv delta_phi)."""
    mean = model(delta_phi).item()  # σ(Δϕᵀ w) = P(π1 > π2)
    uncertainty = torch.sqrt(
        delta_phi @ V_inv @ delta_phi
    ).item()  # √(Δϕᵀ V⁻¹ Δϕ) uncertainty in the prediction of P(π1 > π2)
    return mean + beta * uncertainty


# ==== end Online learning: pair selection ==== #


# ==== Online learning: oracle step ==== #
def get_preferences(env, π1, π2, phi_func, params, online_trajs, preference_data, V):
    """Gets oracle preference on single sample of trajectory pair"""

    def oracle_step(π, phi_func, env, params):
        traj = rollout_sb3(env, π, params.episode_length, deterministic=True)
        total_reward = np.sum(traj.rewards)
        ϕ = phi_func(traj)
        return traj, total_reward, ϕ

    traj1, r1, ϕ1 = oracle_step(π1, phi_func, env, params)
    traj2, r2, ϕ2 = oracle_step(π2, phi_func, env, params)
    diff = ϕ1 - ϕ2
    if r1 > r2:
        preference_data.append((diff, 1))
    elif r1 == r2:
        preference_data.append((diff, np.random.randint(0, 1)))
    else:
        preference_data.append((diff, 0))
    # online_trajs.append(traj1)
    # online_trajs.append(traj2)
    V += torch.outer(diff, diff)
    return online_trajs, preference_data, V


# ==== end Online learning: oracle step ==== #


# ==== Online learning: get best policy ==== #
def get_best_policy(
    policy_list, model, env, phi_func, episode_length, n_samples=100, precomputed_embeddings=None
):
    """Return the policy with highest predicted preference score."""
    best_score = -np.inf
    best_policy = None
    for π in policy_list:
        if precomputed_embeddings is not None:
            ϕ, _ = precomputed_embeddings[hash_policy(π)]
        else:
            ϕ, _ = get_policy_embedding(phi_func, π, env, episode_length, n_samples=n_samples)
        score = model.score(ϕ).detach().item()
        if score > best_score:
            best_score = score
            best_policy = π
    return best_policy


def calc_policy_avg_reward(env, policy, episode_length, n_samples=100):
    """calc expected cumulative reward of a policy in an env, over N_samples trajectories"""
    returns = []
    for _ in range(n_samples):
        traj = rollout_sb3(env, policy, episode_length, deterministic=True)
        traj_reward = (
            traj.total_return()
        )  # np.sum(traj.rewards)  # alternatively: discounted rewards!
        returns.append(traj_reward)
    mean_return = np.mean(returns)
    return mean_return


# ==== end Online learning: get best policy ==== #


# ==== Metrics & Logging ==== #
def setup_metrics():
    metrics = {
        "loop_time": [],
        "regrets": [],
        "score_selected": [],
        "score_expert": [],
        "avg_rewards_best_iteration_policy": [],
        "avg_rewards_true_opt": [],
        "avg_rewards_bc": [],
        "pi_set_sizes": [],
        "norm_delta_w": [],
        "norm_w": [],
        "uncertainty": [],
        "var_exp_return_k_topscorers": [],
        "logdet_V": [],
        "trace_V_inv": [],
        "dist_bc_expert": [],
        "spearman_corr_all": [],
        "spearman_corr_eval": [],
        "expert_score_rank": [],
        "expert_reward_rank": [],
        "highest_rew_score_rank": [],
    }
    return metrics


def collect_and_print_metrics(
    metrics,
    precomputed_embeddings,
    candidates,
    all_policies,
    policy_names,
    bc_policy,
    expert_policy,
    Pi_0,
    Pi_t,
    V,
    V_inv,
    π1,
    π2,
    uncertainty,
    pref_model,
    prev_w,
    env,
    phi_func,
    loop_start_time,
    t,
    tag,
    params,
):
    if params.which_eval_space == "pi_zero":
        eval_policy_list = Pi_0
    elif params.which_eval_space == "pi_t":
        eval_policy_list = Pi_t
    elif params.which_eval_space == "candidates":
        eval_policy_list = candidates
    best_policy = get_best_policy(
        eval_policy_list,
        pref_model,
        env,
        phi_func,
        params.episode_length,
        precomputed_embeddings=precomputed_embeddings,
    )
    best_embedding_t, avg_reward_best_t = precomputed_embeddings[hash_policy(best_policy)]
    expert_embedding_t, avg_reward_expert_t = precomputed_embeddings[hash_policy(expert_policy)]
    bc_embedding_t, avg_reward_bc_t = precomputed_embeddings[hash_policy(bc_policy)]
    all_scores = {}
    all_rewards = {}
    for π in all_policies:
        all_scores[hash_policy(π)] = (
            pref_model.score(precomputed_embeddings[hash_policy(π)][0]).detach().cpu().numpy()
        )
        all_rewards[hash_policy(π)] = precomputed_embeddings[hash_policy(π)][1]
    eval_hashes = [hash_policy(π) for π in eval_policy_list]
    eval_scores = [all_scores[h] for h in eval_hashes]
    eval_rewards = [all_rewards[h] for h in eval_hashes]
    spearman_corr_eval = spearmanr(eval_scores, eval_rewards)
    spearman_corr_all = spearmanr(list(all_scores.values()), list(all_rewards.values()))
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    score_selected = all_scores[hash_policy(best_policy)]
    score_expert = all_scores[hash_policy(expert_policy)]
    score_bc = all_scores[hash_policy(bc_policy)]
    expert_score_idx = sorted_scores.index((hash_policy(expert_policy), score_expert))
    bc_score_idx = sorted_scores.index((hash_policy(bc_policy), score_bc))
    selected_score_idx = sorted_scores.index((hash_policy(best_policy), score_selected))
    sorted_rewards = sorted(all_rewards.items(), key=lambda x: x[1], reverse=True)
    # get score rank of highest reward policy -- should be expert, but maybe not (if BC too good/if bad luck with embedding estimate)
    highest_rew_hash = sorted_rewards[0][0]
    highest_rew_score = all_scores[highest_rew_hash]
    highest_rew_score_idx = sorted_scores.index((highest_rew_hash, highest_rew_score))
    # the "true" indices the model should have found (ranking by reward)
    selected_reward_idx = sorted_rewards.index((hash_policy(best_policy), avg_reward_best_t))
    expert_reward_idx = sorted_rewards.index((hash_policy(expert_policy), avg_reward_expert_t))
    bc_reward_idx = sorted_rewards.index((hash_policy(bc_policy), avg_reward_bc_t))
    regret = max(0, avg_reward_expert_t - avg_reward_best_t)  # !!!!
    sign, logdet_V = torch.linalg.slogdet(V)
    trace_V_inv = torch.trace(V_inv)

    metrics["loop_time"].append(time.time() - loop_start_time)
    metrics["regrets"].append(regret)
    metrics["score_selected"].append(score_selected)
    metrics["score_expert"].append(score_expert)
    metrics["avg_rewards_best_iteration_policy"].append(avg_reward_best_t)
    metrics["avg_rewards_true_opt"].append(avg_reward_expert_t)
    metrics["avg_rewards_bc"].append(avg_reward_bc_t)
    metrics["pi_set_sizes"].append(len(Pi_t))
    metrics["norm_delta_w"].append(np.linalg.norm(prev_w - pref_model.w.detach().numpy()))
    metrics["norm_w"].append(np.linalg.norm(pref_model.w.detach().numpy()))
    metrics["uncertainty"].append(uncertainty)
    metrics["var_exp_return_k_topscorers"].append(
        metrics_var_exp_return_k_topscorers(Pi_t, pref_model, precomputed_embeddings, k=10)
    )
    metrics["logdet_V"].append(logdet_V.item())
    metrics["trace_V_inv"].append(trace_V_inv.item())
    metrics["dist_bc_expert"].append(
        torch.norm(bc_embedding_t - expert_embedding_t, p=2).item()
    )  # same as used in filtering Pi_0
    metrics["spearman_corr_all"].append(spearman_corr_all.statistic)
    metrics["spearman_corr_eval"].append(spearman_corr_eval.statistic)
    metrics["expert_score_rank"].append(expert_score_idx)
    metrics["expert_reward_rank"].append(expert_reward_idx)
    metrics["highest_rew_score_rank"].append(highest_rew_score_idx)

    # ==== LOOP PRINTS ====
    print(
        f"%%% {t}: time: {time.time() - loop_start_time:.2f}s, |Pi_t|: {len(Pi_t)}, (π1, π2): ({policy_names[hash_policy(π1)]}, {policy_names[hash_policy(π2)]}), corr: {spearman_corr_all.statistic:.2f} | tag: {tag}"
    )
    print(
        f"%%%    pred: {policy_names[hash_policy(best_policy)]}, scores: S_pred: {score_selected:.2f} ({selected_score_idx} | {selected_reward_idx}), S_expert: {score_expert:.2f} ({expert_score_idx} | {expert_reward_idx}), S_BC: {score_bc:.2f} ({bc_score_idx} | {bc_reward_idx})"
    )
    print(
        f"%%%    regret: {regret:.2f}, R_pred: {avg_reward_best_t:.2f}, R_expert: {avg_reward_expert_t:.2f}, R_BC: {avg_reward_bc_t:.2f}"
    )
    w_vec_t = pref_model.w.detach().numpy()
    print(
        f"%%%    embed pred ({best_embedding_t[0]:.2f}, {best_embedding_t[1]:.2f}), expert ({expert_embedding_t[0]:.2f}, {expert_embedding_t[1]:.2f}), w=({w_vec_t[0]:.2f}, {w_vec_t[1]:.2f})"
    )

    Pi_t_hashes = [hash_policy(π) for π in Pi_t]
    if hash_policy(expert_policy) not in Pi_t_hashes and params.expert_in_candidates:
        print("%%%    WARNING: EXPERT POLICY NOT IN Pi_t!")
    if len(Pi_t) <= 5:
        printstr = "%%%    "
        for i, π in enumerate(Pi_t):
            phi_π, rew_π = precomputed_embeddings[hash_policy(π)]
            printstr += f"s(π{i}): {pref_model.score(phi_π).detach().cpu().numpy():.2f}, R(π{i}): {rew_π:.2f}, "
        print(printstr)
    return metrics


def metrics_var_exp_return_k_topscorers(policy_list, pref_model, embeddings, k=10):
    """calculates Var(expected return of k top scoring policies in the list, judged by pref model)

    Assumes embeddings are precomputed, ie embeddings[hash_policy(π)] = (embedding, exp_reward)"""
    policy_scores = []
    for policy in policy_list:
        policy_hash = hash_policy(policy)
        embedding, exp_reward = embeddings[policy_hash]
        score = pref_model.score(embedding).detach().item()
        policy_scores.append((policy, score, exp_reward))

    top_k_policies = sorted(policy_scores, key=lambda x: x[1], reverse=True)[:k]
    top_k_rewards = [exp_reward for _, _, exp_reward in top_k_policies]

    return np.var(top_k_rewards)


def pad_metrics_mujoco(metrics, params):
    """
    if loop terminated early in one seed, then some of the metrics don't have the right length.
    fix this by padding them at the end with the last known value of each metric
    """
    no_padding_needed = True
    for seed in range(len(metrics)):
        for metric in metrics[seed]:
            if len(metrics[seed][metric]) < params["N_iterations"]:
                padding_length = params["N_iterations"] - len(metrics[seed][metric])
                # Pad with the last known value
                last_value = metrics[seed][metric][-1] if metrics[seed][metric] else 0
                metrics[seed][metric].extend([last_value] * padding_length)
                if "plotting" in params["verbose"] or "full" in params["verbose"]:
                    print(f"metric {metric} padded {padding_length} for seed {seed}")
                no_padding_needed = False
    if no_padding_needed and ("plotting" in params["verbose"] or "full" in params["verbose"]):
        print("no padding needed")
    return metrics


# ==== end Metrics ==== #


# ==== Logging ==== #
class Tee:
    """Tee class for logging to both file and console (needed for multiprocessing logs)."""

    def __init__(self, file_path, mode="w"):
        self.terminal = sys.stdout
        self.log_file = open(file_path, mode)

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate writing to file
        except Exception:
            pass  # errors shouldnt crash experiment

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ==== end Logging ==== #


# ==== Helpers: misc ==== #
def estimate_speed_scale_from_policy(env, policy, v_index, episode_length, n=200, q=0.95):
    """To normalize embeddings with a velocity component, this func estimates a good top speed
    using a reference policy (expert or BC)"""
    vals = []
    for _ in range(n):
        traj = rollout_sb3(env, policy, episode_length, deterministic=True)
        s = torch.tensor(traj.states, dtype=torch.float32)
        vals.append(s[:, v_index].abs().mean().item())
    return float(np.quantile(vals, q))


def plot_dists_rewards(dists_x, rewards, epsilon, title=None):
    fig_hist = plt.figure(figsize=(8, 6))
    plt.hist(dists_x)
    plt.axvline(epsilon, color="red", linestyle="--", label=f"epsilon = {epsilon}")
    plt.show()
    plt.close()

    num_accepted = 0
    for dist in dists_x:
        if dist < epsilon:
            num_accepted += 1

    basetitle = "Distance to BC Policy vs Average Reward"

    if title:
        basetitle += f"\n{title}"
    fig_scatter = plt.figure(figsize=(8, 6))
    plt.scatter(dists_x, rewards, alpha=0.6)
    plt.scatter(dists_x[0], rewards[0], color="green", s=100, alpha=0.8, label="BC Policy")
    plt.scatter(dists_x[-1], rewards[-1], color="red", s=100, alpha=0.8, label="Expert Policy")
    plt.xlabel("Distance to BC Policy")
    plt.ylabel("Average Reward")
    plt.title(basetitle)
    plt.axvline(epsilon, color="red", linestyle="--", label=f"epsilon = {epsilon}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()

    print(
        f"acc.: {num_accepted}/{len(dists_x)}, skipped: {len(dists_x) - num_accepted}/{len(dists_x)}"
    )
    return fig_hist, fig_scatter


# ==== end Helpers: misc ==== #
