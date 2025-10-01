# Some code adapted from https://github.com/david-lindner/idrl
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba_array
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base_tabular_mdp import TabularPolicy, TabularMDP

KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN = 275, 276, 273, 274

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4

ACTION_STEP = {LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1), STAY: (0, 0)}

COLOR_CYCLE = sns.color_palette("deep")


class Gridworld(TabularMDP):
    """
    Gridworld environment that includes different tile types with different rewards.

    The environment uses the `TabularMDP` representation and therefore gives
    access to handling value functions and using the LP solver. Also, this means
    that the state space is represented one-dimensionally which has advantages
    for using linear algebra and handeling the full transition matrix. The
    `_get_agent_pos_from_state` and `_get_state_from_agent_pos` provide transformations
    from the one-dimensional representation to an x and y position within the
    gridworld. However, the agent's current position is always stored in the
    `agent_xpos` and `agent_ypos` attributes. Accessing these should usually
    be sufficient.

    Attributes
    -------------
    height (int): height of the gridworld
    width (int): width of the gridworld
    n_objects (int): number of distinct objects / tile types
    n_per_object (int): number of occurences of each object in the environment
                        (currently the same for each type)
    env_seed (int): seed for generation of the environment
    """

    def __init__(
        self,
        width: int,
        height: int,
        discount_factor: float,
        episode_length: int = 10,
        random_action_prob: float = 0.1,
        add_optimal_policy_to_candidates: bool = False,
        observation_noise: float = 0,
        observation_type: str = "state",
    ):
        assert height > 0
        assert width > 0
        assert 0 <= random_action_prob <= 1

        self.width = width
        self.height = height
        self.N_states = height * width
        self.render_scale = 50
        self.render_width = self.width * self.render_scale
        self.render_height = self.height * self.render_scale

        self.random_action_prob = random_action_prob
        self.add_optimal_policy_to_candidates = add_optimal_policy_to_candidates
        self.gaussian_peaks_as_rewards = True

        # self.Ndim_repr = self.N_states
        self.Ndim_repr = 7

        # start at (0,0)
        self.agent_xpos, self.agent_ypos = (0, 0)

        self.tiles = self._create_tiles()
        (
            rewards,
            self.reward_grid,
            self.object_states,
        ) = self._create_rewards_from_tiles_and_object_rewards()

        self.walls = self._get_walls()

        initial_state = 0
        terminal_states: List[int] = []
        transitions = self._get_transition_matrix()
        use_sparse_transitions = self.random_action_prob == 0

        super().__init__(
            height * width,
            5,
            rewards,
            transitions,
            discount_factor,
            terminal_states,
            episode_length,
            initial_state,
            use_sparse_transitions=use_sparse_transitions,
            observation_noise=observation_noise,
            observation_type=observation_type,
        )

        if observation_type == "raw":
            self.observation_space = gym.spaces.Box(0, max(width, height), shape=(2,))

        self._create_graph()

    def get_candidate_policies(self) -> List[np.ndarray]:
        """
        Generate a set of candidate policies.

        This set is meant to be used to test algorithms that query rewards to find
        the best policy in a restricted policy space.

        The policies are generated as follows:
            - for each object in the gridworld (i.e. each non-empty field):
                - for each other state find the shortest path to the object
                - set the policy to perform the first action of the shortest path

        As a result, we get one deterministic policy per target object that selects
        the shortest path to the target from every state.
        """
        candidate_policies = []
        for target in self.object_states:
            policy = np.zeros((self.N_states, self.N_actions), dtype=np.float32)
            for start in range(self.N_states):
                if start == target:
                    action = STAY
                else:
                    try:
                        path = nx.shortest_path(self.graph, start, target)
                    except nx.NetworkXNoPath:
                        path = None
                    if path:
                        first_edge = self.graph.edges[path[0], path[1]]
                        if len(first_edge["actions"]) > 1 and STAY in first_edge["actions"]:
                            action = STAY
                        elif len(first_edge["actions"]) >= 1:
                            action = first_edge["actions"][0]
                        else:
                            raise Exception("Unexpected error: no action stored in graph edge")
                    else:
                        action = STAY
                policy[start, action] = 1
            candidate_policies.append(TabularPolicy(policy))

        if self.add_optimal_policy_to_candidates:
            optimal_policy = self.get_lp_solution()
            already_exists = False
            for policy in candidate_policies:
                if np.all(optimal_policy == policy):
                    already_exists = True
                    break
            if not already_exists:
                candidate_policies.append(optimal_policy)

        return candidate_policies

    def _create_graph(self) -> None:
        """
        Create a networkx graph that represents the gridworld and can be used
        to find shortest paths, etc.

        Each tile is represented as a node. The node has the 1d-state-representation
        as a label and it has properties `xpos` and `ypos` that store the 2d
        position of the tile.

        Each possible transition is represented by an edge. The edges have a
        property `actions` that stores a list of actions that can cause this
        transition.
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.N_states))
        for s1 in range(self.N_states):
            xpos, ypos = self._get_agent_pos_from_state(s1)
            self.graph.nodes[s1]["xpos"] = xpos
            self.graph.nodes[s1]["ypos"] = ypos
            for a in range(5):
                s2 = self._get_next_state_for_action(s1, a)
                if self.graph.has_edge(s1, s2):
                    self.graph.edges[s1, s2]["actions"].append(a)
                else:
                    self.graph.add_edge(s1, s2)
                    self.graph.edges[s1, s2]["actions"] = [a]

    def _create_tiles(self) -> np.ndarray:
        """convention:
        ax 0 is x is horizontal,
        ax 1 is y is vertical
        so when printing, imagine it as transpose"""
        tiles = np.zeros((self.height, self.width), dtype=int)
        tiles[2, 0] = 0
        tiles[2, 1] = 0
        tiles[3, 0] = 0
        tiles[3, 1] = 0

        tiles[1, 2] = 1  # walled in

        tiles[3, 3] = 2  # goal
        return tiles

    def _create_rewards_from_tiles_and_object_rewards(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """convention:
        ax 0 is x is horizontal,
        ax 1 is y is vertical
        so when printing, imagine it as transpose"""

        reward_grid = np.zeros((self.height, self.width), dtype=np.float32)
        reward_grid[2, 0] = -1
        reward_grid[2, 1] = -1
        reward_grid[3, 0] = -1
        reward_grid[3, 1] = -1

        reward_grid[1, 2] = 20  # walled in (unreachable)

        reward_grid[3, 3] = 10  # goal

        object_states = [
            self._get_state_from_agent_pos(x, y)
            for x, y in [(2, 0), (2, 1), (3, 0), (3, 1), (1, 2), (3, 3)]
        ]

        rewards = np.zeros(self.N_states, dtype=np.float32)
        for s in range(self.N_states):
            x, y = self._get_agent_pos_from_state(s)
            rewards[s] = reward_grid[y, x]  # y,x because x is horizontal, y is vertical!

        return rewards, reward_grid, object_states

    def _get_walls(self) -> np.ndarray:
        """
        Return a random array of walls (here: hardcoded to wall off [1,2] ONLY)

        Between each two tiles a wall is being generated with a fixed probability.

        The `seed` can be used for reproducibility. It does not affect the
        numpy random state (which is saved initially and then restored after
        generating the environment).
        """

        walls = np.zeros((self.N_states, self.N_states), dtype=bool)

        x1, y1 = (2, 1)  # USED TO BE (1, 2) -- wrong?
        s1 = self._get_state_from_agent_pos(x1, y1)
        for x2, y2 in [(x1 - 1, y1), (x1 + 1, y1), (x1, y1 - 1), (x1, y1 + 1)]:
            s2 = self._get_state_from_agent_pos(x2, y2)
            walls[s1, s2] = True
            walls[s2, s1] = True
        return walls

    def _get_transition_matrix(self) -> np.ndarray:
        """
        Get the full transition matrix according to standard gridworld dynamics
        and a custom probability to take a random move instead of the action chosen.

        In particular:
            - actions UP, DOWN, LEFT, RIGHT always move the agent deterministically
              in the corresponding direction if it is not at the boundary of the
              world
            - if the agent would leave the grid with an action, the action has
              no effect, i.e. the agent stays in the same tile
            - the STAY action makes the agent stay in the same tile with probability 1
        """
        n_states = self.N_states
        N_actions = 5
        transitions = np.zeros((N_actions, n_states, n_states), dtype=np.double)
        for a in range(N_actions):
            for s1 in range(n_states):
                s2 = self._get_next_state_for_action(s1, a)
                transitions[a, s1, s2] = 1

        stochastic_transitions = np.zeros((N_actions, n_states, n_states), dtype=np.double)
        for action_taken in range(N_actions):
            stochastic_transitions[action_taken] = (1 - self.random_action_prob) * transitions[
                action_taken
            ]
            for action_random in range(N_actions):
                stochastic_transitions[action_taken] += (
                    self.random_action_prob / N_actions * transitions[action_random]
                )

        return stochastic_transitions

    def _get_next_state_for_action(self, s1, a):
        x1, y1 = self._get_agent_pos_from_state(s1)
        diff = ACTION_STEP[a]
        x2, y2 = x1 + diff[0], y1 + diff[1]
        s2 = self._get_state_from_agent_pos(x2, y2)
        if self._transition_possible((x1, y1), (x2, y2)):
            s2 = self._get_state_from_agent_pos(x2, y2)
        else:
            s2 = s1
        return s2

    def _within_grid(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def _transition_possible(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
        """
        Returns True iff the transition from `point1` to `point2` is possible.
        A transition is not possible iff it would go outside the grid or there
        is a wall between these two cells.
        """
        x1, y1 = point1
        x2, y2 = point2
        assert self._within_grid(x1, y1)
        if not self._within_grid(x2, y2):
            return False
        else:
            s1 = self._get_state_from_agent_pos(x1, y1)
            s2 = self._get_state_from_agent_pos(x2, y2)
            return not self.walls[s1, s2]

    def _get_state_from_agent_pos(self, x: Optional[int] = None, y: Optional[int] = None) -> int:
        """
        Get the integer that represent the current agent position in the one-
        dimensional state representation of the TabularMDP base.
        """
        if x is None:
            assert self.agent_xpos is not None
            x = self.agent_xpos
        if y is None:
            assert self.agent_ypos is not None
            y = self.agent_ypos
        return x + y * self.width

    def _get_agent_pos_from_state(self, state: int) -> Tuple[int, int]:
        """
        Get the agent position from a state index.
        """
        x = int(state % self.width)
        y = int(state // self.width)
        return x, y

    def reset(self) -> int:
        """
        Reset the environment and store the agents position.
        """
        obs = super().reset()
        self.agent_xpos, self.agent_ypos = self._get_agent_pos_from_state(self.current_state)
        return obs

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Perform one step in the environment and store the agents position.
        """
        obs, reward, done, info = super().step(action)
        self.agent_xpos, self.agent_ypos = self._get_agent_pos_from_state(self.current_state)
        info["x"] = self.agent_xpos
        info["y"] = self.agent_ypos
        return obs, reward, done, info

    def render(self, mode: str = "human", close: bool = False) -> Optional[Union[str, np.ndarray]]:
        """
        Implements the render modes "ansi", "rgb_array" and "human".

        ansi: provides an ansi representation of the tiles array
        rgb_array: returns a visualization of the tiles array using fixed rgb values
        human: visualizes the rgb_array using `plt.imshow`
        """
        assert self.tiles is not None
        if mode == "ansi":
            rows = []
            for i, reward_row in enumerate(self.tiles):
                row = []
                for j, reward in enumerate(self.tiles[i]):
                    if i == self.agent_ypos and j == self.agent_xpos:
                        row.append("A")
                    else:
                        row.append("{:.2f}".format(reward))
                rows.append("  ".join(row))
            return "\n".join(rows)
        elif mode == "human" or mode == "rgb_array":
            rgb = np.zeros((self.render_height, self.render_width, 3), dtype=int)
            for x in range(self.width):
                for y in range(self.height):
                    if x == self.agent_xpos and y == self.agent_ypos:
                        color = "#0000ff"
                    elif self.tiles[y, x] == 0:
                        color = "#dddddd"
                    else:
                        color = COLOR_CYCLE[(self.tiles[y, x] - 1) % len(COLOR_CYCLE)]

                    rgb[
                        y * self.render_scale : (y + 1) * self.render_scale,
                        x * self.render_scale : (x + 1) * self.render_scale,
                        :,
                    ] = to_rgba_array(color)[0, :3] * int(255)

            # plot walls
            for s1 in range(self.N_states):
                for s2 in range(s1 + 1, self.N_states):
                    if self.walls[s1, s2]:
                        x1, y1 = self._get_agent_pos_from_state(s1)
                        x2, y2 = self._get_agent_pos_from_state(s2)
                        wall_width = 6
                        if x1 == x2:
                            x_from = x1 * self.render_scale
                            x_to = (x1 + 1) * self.render_scale
                        elif x1 == x2 + 1:
                            x_from = x1 * self.render_scale - wall_width // 2
                            x_to = x1 * self.render_scale + wall_width // 2
                        elif x1 == x2 - 1:
                            x_from = x2 * self.render_scale - wall_width // 2
                            x_to = x2 * self.render_scale + wall_width // 2
                        else:
                            raise Exception("unexpected error: wrong walls")
                        if y1 == y2:
                            y_from = y1 * self.render_scale
                            y_to = (y1 + 1) * self.render_scale
                        elif y1 == y2 + 1:
                            y_from = y1 * self.render_scale - wall_width // 2
                            y_to = y1 * self.render_scale + wall_width // 2
                        elif y1 == y2 - 1:
                            y_from = y2 * self.render_scale - wall_width // 2
                            y_to = y2 * self.render_scale + wall_width // 2
                        else:
                            raise Exception("unexpected error: wrong walls")

                        rgb[y_from:y_to, x_from:x_to, :] = [0, 0, 0]

            if mode == "human":
                plt.axis("off")
                plt.imshow(rgb, origin="upper", extent=(0, self.width, self.height, 0))
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
                return None
            else:
                return rgb
        else:
            raise NotImplementedError(
                "Mode '{}' unsupported. ".format(mode)
                + "Mode should be in ('human', 'ansi', 'rgb_array')"
            )

    def get_keys_to_action(self):
        """
        Provides the controls for using the environment with gym.util.play
        """
        return {
            (): STAY,
            (KEY_UP,): UP,
            (KEY_DOWN,): DOWN,
            (KEY_UP, KEY_DOWN): STAY,
            (KEY_LEFT,): LEFT,
            (KEY_RIGHT,): RIGHT,
            (KEY_LEFT, KEY_RIGHT): STAY,
        }  # control with arrow keys

    def visualize_reward_estimate(self, mu: np.ndarray, sigma: np.ndarray, ax) -> None:
        states = np.arange(self.N_states)
        ax.plot(states, self.rewards, color="black", label="true")
        ax.plot(states, mu, color="orange", label="prediction")
        ax.fill_between(states, mu - sigma, mu + sigma, alpha=0.3, color="orange")

    def visualize_value(
        self,
        value: np.ndarray,
        ax: matplotlib.axes.Axes,
        highlight_states: Optional[List[int]] = None,
    ) -> None:
        """
        Plot a value function to a pyplot axes.

        Shows the gridworld using the `render` function and overlays it with
        a imshow plot of the value function.

        Args:
        -----------
        value (np.ndarray): 1d-array containing the value of each state
        ax (matplotlib.axes.Axes): axes to plot to
        highlight_states (list): if given this specific state will be hightlighted
        """
        assert value.shape == (self.N_states,)
        self.reset()
        rgb_gridworld = self.render(mode="rgb_array")

        value_scaled = np.zeros((self.render_height, self.render_width))
        for x in range(self.width):
            for y in range(self.height):
                value_scaled[
                    y * self.render_scale : (y + 1) * self.render_scale,
                    x * self.render_scale : (x + 1) * self.render_scale,
                ] = value[self._get_state_from_agent_pos(x, y)]

        ax.imshow(rgb_gridworld, origin="upper", extent=(0, self.width, self.height, 0))
        value_im = ax.imshow(
            value_scaled,
            cmap="hot",
            origin="upper",
            alpha=0.3,
            extent=(0, self.width, self.height, 0),
        )
        if highlight_states is not None:
            for highlight_state in highlight_states:
                x, y = self._get_agent_pos_from_state(highlight_state)
                ax.plot(x + 0.5, y + 0.5, marker="o", color="black")

        fig = plt.gcf()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        fig.colorbar(value_im, cax=cax)

    def visualize_policy(self, policy: TabularPolicy, ax: matplotlib.axes.Axes) -> None:
        """
        Plot a policy to a pyplot axes.

        Shows the gridworld using the `render` function and overlays it with
        a quiver plot of the policie's action.

        Args:
        -----------
        policy (np.ndarray): 2d-array containing the actions of the policy for each state
        ax (matplotlib.axes.Axes): axes to plot to
        """
        policy_matrix = np.argmax(policy.matrix, axis=1)
        self.reset()
        # rgb_gridworld = self.render(mode="rgb_array")
        X = np.zeros(self.N_states)
        Y = np.zeros(self.N_states)
        U = np.zeros(self.N_states)
        V = np.zeros(self.N_states)
        for s in range(self.N_states):
            xpos, ypos = self._get_agent_pos_from_state(s)
            X[s] = xpos + 0.5
            Y[s] = ypos + 0.5
            action_step = ACTION_STEP[int(policy_matrix[s])]
            U[s] = action_step[0]
            V[s] = action_step[1]
        # im = ax.imshow(rgb_gridworld, extent=(0, self.width, self.height, 0))
        ax.quiver(X, Y, U, V, angles="xy")

    def visualize_reward_grid(self, ax: matplotlib.axes.Axes) -> None:
        """
        Plot a reward function to a pyplot axes.

        Shows the gridworld using the `render` function and overlays it with
        a imshow plot of `self.reward_grid`.

        Args:
        -----------
        ax (matplotlib.axes.Axes): axes to plot to
        """
        self.reset()
        # rgb_gridworld = self.render(mode="rgb_array")
        # im_grid = ax.imshow(rgb_gridworld, extent=(0, self.width, self.height, 0))
        im_reward = ax.imshow(
            self.reward_grid,
            extent=(0, self.width, self.height, 0),
            alpha=0.3,
            cmap="hot",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        fig = plt.gcf()
        fig.colorbar(im_reward, cax=cax)

    def get_state_repr(self, state: int) -> np.ndarray:
        """
        This can be implemented to define vector representations of the states.

        These can e.g. be used to define GP kernels.
        """
        x, y = self._get_agent_pos_from_state(state)
        if self.gaussian_peaks_as_rewards:
            return np.array((x, y))
        else:
            assert self.tiles is not None
            repr = np.arange(7) == self.tiles[y, x]
            repr = np.array(repr, dtype=int)
            return repr

    def get_number_of_reachable_tiles(
        self, xpos: Optional[int] = None, ypos: Optional[int] = None
    ) -> int:
        if xpos is None:
            assert ypos is None
            source = self.initial_state
        else:
            assert self._within_grid(xpos, ypos)
            source = self._get_state_from_agent_pos(xpos, ypos)
        reachable_nodes = nx.dfs_preorder_nodes(self.graph, source=source)
        n_reachable = sum(1 for _ in reachable_nodes)
        return n_reachable

    def _get_raw_observation(self, state: int) -> np.ndarray:
        x, y = self._get_agent_pos_from_state(state)
        return np.array([x, y])

    def _get_feature_observation(self, state: int) -> np.ndarray:
        distances = []
        x1, y1 = self._get_agent_pos_from_state(state)
        for target in self.object_states:
            x2, y2 = self._get_agent_pos_from_state(target)
            dx = x1 - x2
            dy = y1 - y2
            distances.append(dx)
            distances.append(dy)
        return np.array(distances)
