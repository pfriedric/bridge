import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass(slots=True)
class Trajectory:
    """Represents a trajectory with states, actions, rewards, and info dictionaries.

    For some envs the infos are important. E.g., for HalfCheetah, they contain the x-coordinate (that measures progress&reward)"""

    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]  # assume scalar reward
    infos: List[Dict[str, Any]]
    true_obs: List[np.ndarray] = None  # in case obs are normalized (reacher, HC)
    # extras: Optional[np.ndarray] = None  # shape [T, K] or None

    def __post_init__(self):
        if self.true_obs is None:
            self.true_obs = self.states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """Allow indexing like traj[0] -> (state, action, reward, info)"""
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.infos[idx],
            self.true_obs[idx],
        )

    def to_sa_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns [(s0, a0), (s1, a1), ...]"""
        return list(zip(self.true_obs, self.actions))

    def total_return(self) -> float:
        return float(np.sum(self.rewards))

    def to_tensors(self):
        """Returns tuple (states, actions), both tensors"""
        return (
            torch.tensor(self.true_obs, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.float32),
        )

    # HalfCheetah/Walker-specific:
    def get_x_positions(self) -> np.ndarray:
        """Extract x_positions from info dictionaries."""
        return np.array([info["x_position"] for info in self.infos])

    def get_final_x_position(self) -> float:
        """Get the final x_position."""
        return self.infos[-1]["x_position"]

    def compute_discounted_cumulative_reward(self, discount_factor):
        return np.dot(self.rewards, np.power(discount_factor, np.arange(len(self.rewards))))

    def compute_cumulative_reward(self):
        return np.sum(self.rewards)


def extract_x_pos_from_info(info: Dict[str, Any]) -> Optional[np.ndarray]:
    if "x_position" in info:
        return np.array([info["x_position"]], dtype=np.float32)
    return None


def setup_embeddings(params, env, **kwargs):
    phi_funcs = {
        "avg_sa": _create_avg_sa_embedding(),  # d: (dim_actions+dim_states)
        "avg_s": _create_avg_s_embedding(),
        "last_s": _create_last_s_embedding(),
        "actionenergy": _create_actionenergy_embedding(),
        "psm": _create_psm_embedding(),
        "reacher_perf": _create_reacher_perf_embedding(),
        "halfcheetah_xpos": _create_halfcheetah_xpos_embedding(),
        "halfcheetah_perf": _create_halfcheetah_perf_embedding(),
        "halfcheetah_extended": _create_halfcheetah_extended_embedding(),
        "walker2d_perf": _create_walker2d_perf_embedding(
            params.episode_length,
            v_ref=kwargs.get("v_ref", 5),
            action_dim=env.action_space.shape[0],
        ),
        "hopper_perf": _create_hopper_perf_embedding(
            params.episode_length,
            v_ref=kwargs.get("v_ref", 5),
            action_dim=env.action_space.shape[0],
        ),
        "ant_perf": _create_ant_perf_embedding(params.episode_length),
        "walker2d_extended": _create_walker2d_extended_embedding(
            params.episode_length,
            v_ref=kwargs.get("v_ref", 5),
            action_dim=env.action_space.shape[0],
        ),
        "hopper_extended": _create_hopper_extended_embedding(
            params.episode_length,
            v_ref=kwargs.get("v_ref", 5),
            action_dim=env.action_space.shape[0],
        ),
        "humanoid_perf": _create_humanoid_perf_embedding(params.episode_length),
    }
    dim_actions = env.action_space.shape[0]
    dim_states = env.observation_space.shape[0]
    embedding_dims = {
        "avg_sa": dim_actions + dim_states,
        "avg_s": dim_states,
        "last_s": dim_states,
        "actionenergy": 2,
        "psm": 7,
        "reacher_perf": 2,
        "halfcheetah_xpos": 1,
        "halfcheetah_perf": 2,
        "halfcheetah_extended": 4,
        "walker2d_perf": 3,
        "hopper_perf": 3,
        "ant_perf": 4,
        "humanoid_perf": 4,
        "walker2d_extended": 6,
        "hopper_extended": 6,
    }
    d = embedding_dims[params.embedding_name]
    # maximum L2 norm of trajectory embedding (eyeballing this)
    embedding_bounds = {
        "avg_sa": np.sqrt(d),  # < [ds, da]
        "avg_s": np.sqrt(d),  # < [ds]
        "last_s": np.sqrt(d),  # < [ds]
        "actionenergy": np.sqrt(5),  # < [1, 2]
        "psm": np.sqrt(6 * dim_states + 5 * dim_actions + 2),  # < [2ds, 2ds, 2ds, da, 4da, 1, 1]
        "reacher_perf": np.sqrt(2),  # < [1, 1]
        "halfcheetah_xpos": 20,  # < [ds]
        "halfcheetah_perf": np.sqrt(d),  # < [ds]
        "halfcheetah_extended": np.sqrt(d),  # < [ds]
        "walker2d_perf": np.sqrt(d),  # < [ds]
        "hopper_perf": np.sqrt(d),  # < [ds]
        "ant_perf": np.sqrt(d),  # < [ds]
        "humanoid_perf": np.sqrt(d),  # < [ds]
        "walker2d_extended": np.sqrt(d),  # < [ds]
        "hopper_extended": np.sqrt(d),  # < [ds]
    }
    embedding_filter_epsilons = {
        "Reacher-v5": {
            "avg_sa": 2,  # 90%
            "avg_s": 2,  # ???
            "last_s": 3,  # 90%
            "actionenergy": 0.05,  # 80%. 0.03-70%, 0.02-50%
            "psm": 2,  # ???
            "reacher_perf": 0.6,  # 50%. this one's really good
        },
        "HalfCheetah-v5": {  # used for L2(embed(π_BC) - embed(π_candidate)) < eps
            "avg_sa": 0.5,  #
            "avg_s": 2,  # ???
            "last_s": 3,  # ???
            "actionenergy": 0.05,  # ???
            "psm": 2,  # ???
            "halfcheetah_xpos": 2,
            "halfcheetah_perf": 0.3,
            "halfcheetah_extended": 3,
        },
        "Walker2d-v5": {
            "avg_sa": 2,  # exp-BC: ~0.5
            "avg_s": None,  # ???
            "last_s": None,  # ???
            "actionenergy": None,  # ???
            "psm": None,  # ???
            "walker2d_perf": 0.5,
            "walker2d_extended": None,
        },
        "Hopper-v5": {
            "avg_sa": None,
            "last_s": None,
            "hopper_perf": None,
        },
        "Ant-v5": {
            "avg_sa": None,
            "last_s": None,
            "ant_perf": None,
        },
        "Humanoid-v5": {
            "avg_sa": None,
            "last_s": None,
            "humanoid_perf": None,
        },
    }
    embedding_filter_epsilon = embedding_filter_epsilons[params.env_id][params.embedding_name]

    B = embedding_bounds[params.embedding_name]
    phi = phi_funcs[params.embedding_name]

    embedding_env_map = {
        "reacher_perf": "Reacher-v5",
        "halfcheetah_xpos": "HalfCheetah-v5",
        "halfcheetah_perf": "HalfCheetah-v5",
        "walker2d_extended": "Walker2d-v5",
        "walker2d_perf": "Walker2d-v5",
        "hopper_extended": "Hopper-v5",
        "hopper_perf": "Hopper-v5",
        "ant_perf": "Ant-v5",
        "humanoid_perf": "Humanoid-v5",
    }
    if params.embedding_name in embedding_env_map:
        required_env = embedding_env_map[params.embedding_name]
        if params.env_id != required_env:
            raise ValueError(
                f"'{params.embedding_name}' embedding only works in '{required_env}' env"
            )
    phi.dim = d
    phi.bound = B
    if params.radius is not None:
        phi.filter_epsilon = params.radius
    else:
        phi.filter_epsilon = embedding_filter_epsilon
    phi.name = params.embedding_name

    return phi


def _create_avg_sa_embedding():
    def embed_trajectory_avg_sa(traj: Trajectory) -> torch.Tensor:
        states = torch.tensor(traj.true_obs, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        assert states.shape[0] == actions.shape[0]
        try:
            return torch.cat([states.mean(0), actions.mean(0)])
        except RuntimeError:
            print(f"states: {states.shape}, actions: {actions.shape}")
            raise RuntimeError("states and actions have different shapes")

    embed_trajectory_avg_sa.name = "avg_sa"
    return embed_trajectory_avg_sa


def _create_avg_s_embedding():
    def embed_trajectory_avg_s(traj: Trajectory) -> torch.Tensor:
        states = torch.tensor(traj.true_obs, dtype=torch.float32)
        return states.mean(0)

    embed_trajectory_avg_s.name = "avg_s"
    return embed_trajectory_avg_s


def _create_last_s_embedding():
    def embed_trajectory_last_s(traj: Trajectory) -> torch.Tensor:
        return torch.tensor(traj.true_obs[-1], dtype=torch.float32)

    embed_trajectory_last_s.name = "last_s"
    return embed_trajectory_last_s


def _create_actionenergy_embedding():
    def embed_trajectory_actionenergy(traj: Trajectory) -> torch.Tensor:
        """action energy & smoothness"""
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        a2_mean = (actions.pow(2).sum(dim=1)).mean(0, keepdim=True)
        da = actions[1:] - actions[:-1]  # delta actions
        smooth_a = (da.pow(2)).sum(dim=1).mean(0, keepdim=True)
        return torch.cat([a2_mean, smooth_a])

    embed_trajectory_actionenergy.name = "actionenergy"
    return embed_trajectory_actionenergy


def _create_reacher_perf_embedding():
    def embed_trajectory_reacher_perf(traj: Trajectory) -> torch.Tensor:
        """emulates the reward signal, of averages of [dist to target, action energy]"""
        states = torch.tensor(traj.true_obs, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        dist_to_target = torch.norm(states[:, 8:10], p=2, dim=1).mean().view(1)
        action_energy = (actions.pow(2).sum(dim=1)).mean(0, keepdim=True)
        return torch.cat([dist_to_target, action_energy])

    embed_trajectory_reacher_perf.name = "reacher_perf"
    return embed_trajectory_reacher_perf


def _create_psm_embedding():
    def embed_trajectory_psm(traj):
        """experimental: Progress-Smoothness-Effort
        captures net progress, motion stability, control effort, control smoothness, actuator saturation, temporal coherence"""

        states = torch.tensor(traj.true_obs, dtype=torch.float32)  # [T, d_s]
        actions = torch.tensor(traj.actions, dtype=torch.float32)  # [T, d_a]
        T = states.shape[0]

        # displacement (progress)
        disp = torch.norm(states[-1] - states[0], p=2).view(1)

        # finite difference
        if T > 1:
            ds = states[1:] - states[:-1]
            da = actions[1:] - actions[:-1]
            speed = torch.norm(ds, dim=1)  # ||Δs||
            speed_mean = speed.mean().view(1)
            speed_std = speed.std().view(1)
            jerk = torch.norm(da, dim=1).mean().view(1)  # ||Δa||
            # action autocorrelation (temporal coherence)
            ac = F.cosine_similarity(actions[:-1], actions[1:], dim=1)
        else:
            speed_mean = torch.zeros(1)
            speed_std = torch.zeros(1)
            jerk = torch.zeros(1)
            ac = torch.zeros(1)

        # effort
        effort = torch.norm(actions, dim=1).mean().view(1)  # ||a||

        # saturation
        alpha = 0.9
        sat = (actions.abs() > alpha).float().mean().view(1)

        return torch.cat([disp, speed_mean, speed_std, effort, jerk, sat, ac], dim=0)

    embed_trajectory_psm.name = "psm"
    return embed_trajectory_psm


def _create_halfcheetah_xpos_embedding():
    def embed_trajectory_halfcheetah_xpos(traj: Trajectory) -> torch.Tensor:
        """Emulates reward signal of x-position (needs to read out x-position that's usually not in state)"""
        return torch.tensor([traj.get_final_x_position()], dtype=torch.float32)

    embed_trajectory_halfcheetah_xpos.name = "halfcheetah_xpos"
    return embed_trajectory_halfcheetah_xpos


def _create_halfcheetah_perf_embedding():
    def embed_trajectory_halfcheetah_perf(traj):
        """Emulates reward w/o knowing x-position. [mean velocity, mean action energy] = mean(true reward)"""
        states = torch.tensor(traj.true_obs, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)

        # from docs: 8: rootx velocity (m/s), 9: rootz velocity, 1: rooty (pitch)
        v_x = states[:, 8].mean().view(1)  # fwd speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1)
        return torch.cat([v_x, action_energy])

    embed_trajectory_halfcheetah_perf.name = "halfcheetah_perf"
    return embed_trajectory_halfcheetah_perf


def _create_halfcheetah_extended_embedding():
    def embed_trajectory_halfcheetah_extended(traj):
        """Emulates reward w/o knowing x-position. includes bounce & pitch to disambiguate gaits"""
        states = torch.tensor(traj.true_obs, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        v_x = states[:, 8].mean().view(1)  # fwd speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1)
        bounce = states[:, 9].abs().mean().view(1)  # vertical speed magnitude
        pitch = states[:, 1].abs().mean().view(1)  # torso pitch magnitude
        return torch.cat([v_x, action_energy, bounce, pitch])

    embed_trajectory_halfcheetah_extended.name = "halfcheetah_extended"
    return embed_trajectory_halfcheetah_extended


def _create_walker2d_perf_embedding(episode_length, v_ref=5, action_dim=6):
    def embed_trajectory_walker2d_perf(traj):
        """Walker rewards:
        - "be alive/healthy": +1 if rootz=obs[0] is in [0.8, 1] and abs(angle)=abs(obs[1]) in [-1,1]
        - "forward": +1 * dx/dt where dx=rootx (hidden, we use velocity in obs[8] to emulate), dt=0.008
        - "ctrl_cost": -1e-3 * ||a||^2
        we emulate this via
        - survival: total length of traj / episode_length (normalize s.t. all entries roughly equal magnitude for L2-dist)
        - forward: average velocity in obs[8]
        - ctrl_cost: average action energy / action_dim=6
        """
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)

        # typical layout: 0: rootz (height), 1: rooty (pitch), 8: rootx velocity
        survival = torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        v_x = states[:, 8].mean().view(1) / v_ref  # forward speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1) / action_dim
        ## this does the same thing as survival, needed in case we don't terminate traj's as soon as agent is unhealthy
        # healthy_height = ((states[:, 0] >= 0.8) & (states[:, 0] <= 2.0)).float()
        # healthy_angle = (states[:, 1].abs() <= 1.0).float()
        # healthy = (healthy_height * healthy_angle).sum().view(1)  # total amount of healthy timesteps

        return torch.cat([survival, v_x, action_energy])

    embed_trajectory_walker2d_perf.name = "walker2d_perf"
    return embed_trajectory_walker2d_perf


def _create_walker2d_extended_embedding(episode_length, v_ref=5, action_dim=6):
    def embed_trajectory_walker2d_extended(traj):
        """Same as walker2d_perf, but with additional signals that allow disambiguation of gaits"""
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)

        survival = torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        v_x = states[:, 8].mean().view(1) / v_ref  # forward speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1) / action_dim
        height = states[:, 0].mean().view(1)  # mean torso height. "healthy" if [0.8, 2.0]
        upright = (-states[:, 1].abs()).mean().view(1)  # torso angle. "healthy" if [-1,1]
        smooth = (
            (actions[1:] - actions[:-1]).pow(2).sum(dim=1).mean().view(1)
            if len(actions) > 1
            else torch.zeros(1)
        )  # smoothness
        return torch.cat([survival, v_x, action_energy, height, upright, smooth])

    embed_trajectory_walker2d_extended.name = "walker2d_extended"
    return embed_trajectory_walker2d_extended


def _create_hopper_perf_embedding(episode_length, v_ref=5, action_dim=3):
    def embed_trajectory_hopper_perf(traj):
        """hopper rewards:
        - "be alive/healthy": +1 if agent hasn't died (by default episode terminates if agent dies). agent is alive if:
          - all elements in obs[1:] are in [-100,100]
          - height of hopper (obs[0]) inside [0.7, infty) (i.e. it hasn't fallen)
          - angle of torso (obs[1]) inside [-0.2,0.2]
        - "forward": +1 * dx/dt where dx=rootx (hidden, we use velocity in obs[8] to emulate), dt=0.008
        - "ctrl_cost": -1e-3 * ||a||^2
        we emulate this via
        - survival: count length of traj / episode_length (normalize s.t. all entries roughly equal magnitude for L2-dist)
        - forward: average velocity in obs[5]
        - ctrl_cost: average action energy / action_dim=3 (normalize)
        """
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)

        # Typical layout: 0: rootz (height), 1: rooty (angle), 5: rootx velocity, 6: rootz velocity
        survival = torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        v_x = states[:, 5].mean().view(1) / v_ref  # forward speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1) / action_dim
        return torch.cat([survival, v_x, action_energy])

    embed_trajectory_hopper_perf.name = "hopper_perf"
    return embed_trajectory_hopper_perf


def _create_hopper_extended_embedding(episode_length, v_ref=5, action_dim=3):
    def embed_trajectory_hopper_extended(traj):
        """same as hopper_perf, but with additional signals that allow disambiguation of gaits"""
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)

        survival = torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        v_x = states[:, 5].mean().view(1) / v_ref  # forward speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1) / action_dim
        height = states[:, 0].mean().view(1)  # torso height
        upright = (-states[:, 1].abs()).mean().view(1)  # uprightness
        if states.shape[1] > 2:  # vertical velocity magnitude
            bounce = states[:, 6].abs().mean().view(1)
        else:
            bounce = torch.zeros(1)

        return torch.cat([survival, v_x, action_energy, height, upright, action_energy, bounce])

    embed_trajectory_hopper_extended.name = "hopper_extended"
    return embed_trajectory_hopper_extended


def _create_ant_perf_embedding(episode_length):
    def embed_trajectory_ant_perf(traj):
        """emulates ant rewards. per timestep:
        + healthy_reward (survival): +1
        + forward_reward (velocity): +v
        - ctrl_cost (action energy): -0.5 * ||a||^2
        - contact_cost (contact): -5e-4 * ||F||^2 (on by default)
        """
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        survival = (
            torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        )  # mean survival
        v_x = states[:, 13].mean().view(1)  # mean speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1)  # mean action energy
        contact_forces = states[:, 27:].pow(2).sum(dim=1).mean().view(1)  # mean contact forces
        scaled_contact_forces = 5e-4 * contact_forces
        return torch.cat([survival, v_x, action_energy, scaled_contact_forces])

    embed_trajectory_ant_perf.name = "ant_perf"
    return embed_trajectory_ant_perf


def _create_humanoid_perf_embedding(episode_length):
    def embed_trajectory_humanoid_perf(traj):
        """emulates humanoid rewards. per timestep:
        + healthy_reward (survival): +1
        + forward_reward (velocity): +v
        - ctrl_cost (action energy): -0.5 * ||a||^2
        - contact_cost (contact): -5e-4 * ||c||^2 (this one is off by default)
        """
        states = torch.tensor(traj.states, dtype=torch.float32)
        actions = torch.tensor(traj.actions, dtype=torch.float32)
        survival = (
            torch.tensor([len(traj.states)], dtype=torch.float32) / episode_length
        )  # mean survival (fraction of 1)
        v_x = states[:, 22].mean().view(1)  # mean speed
        action_energy = (actions.pow(2).sum(dim=1)).mean().view(1)  # mean action energy
        contact_forces = states[:, 270:].pow(2).sum(dim=1).mean().view(1)  # mean contact forces
        scaled_contact_forces = 5e-7 * contact_forces
        return torch.cat([survival, v_x, action_energy, scaled_contact_forces])

    embed_trajectory_humanoid_perf.name = "humanoid_perf"
    return embed_trajectory_humanoid_perf
