"""
StarMDP environment implementations.

This module contains the StarMDP variants used in experiments.
"""

import numpy as np
from .base_tabular_mdp import TabularMDP


class StarMDP_with_random_staying(TabularMDP):
    """MDP with central state & 4 branching states.
    States: left (start, +0); middle (-, +0); down (bad, -1); right (good, +6); up (goal, +10)
    Actions: 0=right, 1=left, 2=up, 3=down
    Transitions: move with P=move_prob, stay at state with P=1-move_prob

    Expert policy has only 6 possible trajectories, as it'll try to go (s=left,a=right)->(s=middle,a=up)

    CARE:
    - inadmissible actions aren't penalized, just result in not moving
    - discount < 1 or else LP solving fails"""

    def __init__(self, discount_factor: float = 1, episode_length: int = 3, move_prob=0.9):
        N_states = 5
        N_actions = 4

        transitions = np.zeros((N_actions, N_states, N_states))
        stay_prob = 1 - move_prob

        # state 0 (left)
        transitions[0, 0, 1] = move_prob
        transitions[0, 0, 0] = stay_prob
        transitions[1, 0, 0] = 1
        transitions[2, 0, 0] = 1
        transitions[3, 0, 0] = 1

        # state 1 (middle)
        transitions[0, 1, 2] = move_prob
        transitions[0, 1, 1] = stay_prob
        transitions[1, 1, 0] = move_prob
        transitions[1, 1, 1] = stay_prob
        transitions[2, 1, 4] = move_prob
        transitions[2, 1, 1] = stay_prob
        transitions[3, 1, 3] = move_prob
        transitions[3, 1, 1] = stay_prob

        # state 2 (right)
        transitions[0, 2, 2] = 1
        transitions[1, 2, 1] = move_prob
        transitions[1, 2, 2] = stay_prob
        transitions[2, 2, 2] = 1
        transitions[3, 2, 2] = 1

        # state 3 (down)
        transitions[0, 3, 3] = 1
        transitions[1, 3, 3] = 1
        transitions[2, 3, 1] = move_prob
        transitions[2, 3, 3] = stay_prob
        transitions[3, 3, 3] = 1

        # state 4 (up)
        transitions[0, 4, 4] = 1
        transitions[1, 4, 4] = 1
        transitions[2, 4, 4] = 1
        transitions[3, 4, 4] = move_prob
        transitions[3, 4, 1] = stay_prob

        rewards = np.array([0, 0, 6, -1, 10])

        super().__init__(
            N_states,
            N_actions,
            rewards,
            transitions,
            discount_factor,
            [],
            episode_length,
            initial_state=0,
            observation_type="state",
            observation_noise=0,
        )


class StarMDP_with_random_flinging(TabularMDP):
    """MDP with central state & 4 branching states, and probabilistic transitions.
    States: left (start, +0); middle (-, +0); down (bad, -1); right (good, +6); up (goal, +10)
    Actions: 0=right, 1=left, 2=up, 3=down
    Transitions:
      - move with P = move_prob,
      - get flung to random state (that's NEITHER current state NOR the state you want to move to) with P = 1-move_prob

    Expert policy has only 6 possible trajectories, as it'll try to go (s=left,a=right)->(s=middle,a=up)->(s=up,a=DONT_MOVE)

    CARE:
    - inadmissible actions aren't penalized, just result in not moving (P=move_prob) or getting flung (P=1-move_prob)
    - discount < 1 or else LP solving fails"""

    def __init__(
        self, discount_factor: float = 1, episode_length: int = 3, move_prob=0.8, initial_state=0
    ):
        N_states = 5
        N_actions = 4

        transitions = np.zeros((N_actions, N_states, N_states))
        # -2 because can't stay at current state; can't get flung to state you WANT to move to.
        fling_prob = (1 - move_prob) / (N_states - 2)

        ### read these as `transitions[action, current_state, next_state] = probability`
        ## State 0 (left)
        # go right (try to go to middle)
        transitions[0, 0, 1] = move_prob  # right -> middle
        transitions[0, 0, 2] = fling_prob  # random to right
        transitions[0, 0, 3] = fling_prob  # random to down
        transitions[0, 0, 4] = fling_prob  # random to up
        # go left (try to stay)
        transitions[1, 0, 0] = move_prob  # left -> stay at left
        transitions[1, 0, 2] = fling_prob  # random to right
        transitions[1, 0, 3] = fling_prob  # random to down
        transitions[1, 0, 4] = fling_prob  # random to up
        # go up (try to stay)
        transitions[2, 0, 0] = move_prob  # up -> stay at left
        transitions[2, 0, 1] = fling_prob  # random to middle
        transitions[2, 0, 3] = fling_prob  # random to down
        transitions[2, 0, 4] = fling_prob  # random to up
        # go down (try to stay)
        transitions[3, 0, 0] = move_prob  # down -> stay at left
        transitions[3, 0, 1] = fling_prob  # random to middle
        transitions[3, 0, 2] = fling_prob  # random to right
        transitions[3, 0, 4] = fling_prob  # random to up

        ## State 1 (middle)
        # go right (try to go to right)
        transitions[0, 1, 2] = move_prob  # right -> right
        transitions[0, 1, 0] = fling_prob  # random to left
        transitions[0, 1, 3] = fling_prob  # random to down
        transitions[0, 1, 4] = fling_prob  # random to up
        # go left (try to go to left)
        transitions[1, 1, 0] = move_prob  # left -> left
        transitions[1, 1, 2] = fling_prob  # random to right
        transitions[1, 1, 3] = fling_prob  # random to down
        transitions[1, 1, 4] = fling_prob  # random to up
        # go up (try to go to up)
        transitions[2, 1, 4] = move_prob  # up -> up
        transitions[2, 1, 0] = fling_prob  # random to left
        transitions[2, 1, 2] = fling_prob  # random to right
        transitions[2, 1, 3] = fling_prob  # random to down
        # go down (try to go to down)
        transitions[3, 1, 3] = move_prob  # down -> down
        transitions[3, 1, 0] = fling_prob  # random to left
        transitions[3, 1, 2] = fling_prob  # random to right
        transitions[3, 1, 4] = fling_prob  # random to up

        ## State 2 (right)
        # go right (try to stay)
        transitions[0, 2, 2] = move_prob  # right -> stay at right
        transitions[0, 2, 0] = fling_prob  # random to left
        transitions[0, 2, 1] = fling_prob  # random to middle
        transitions[0, 2, 3] = fling_prob  # random to down
        # go left (try to go to middle)
        transitions[1, 2, 1] = move_prob  # left -> middle
        transitions[1, 2, 0] = fling_prob  # random to left
        transitions[1, 2, 3] = fling_prob  # random to down
        transitions[1, 2, 4] = fling_prob  # random to up
        # go up (try to stay)
        transitions[2, 2, 2] = move_prob  # up -> stay at right
        transitions[2, 2, 0] = fling_prob  # random to left
        transitions[2, 2, 1] = fling_prob  # random to middle
        transitions[2, 2, 3] = fling_prob  # random to down
        # go down (try to stay)
        transitions[3, 2, 2] = move_prob  # down -> stay at right
        transitions[3, 2, 0] = fling_prob  # random to left
        transitions[3, 2, 1] = fling_prob  # random to middle
        transitions[3, 2, 4] = fling_prob  # random to up

        ## State 3 (down)
        # go right (try to stay)
        transitions[0, 3, 3] = move_prob  # right -> stay at down
        transitions[0, 3, 0] = fling_prob  # random to left
        transitions[0, 3, 1] = fling_prob  # random to middle
        transitions[0, 3, 4] = fling_prob  # random to up
        # go left (try to stay)
        transitions[1, 3, 3] = move_prob  # left -> stay at down
        transitions[1, 3, 0] = fling_prob  # random to left
        transitions[1, 3, 2] = fling_prob  # random to right
        transitions[1, 3, 4] = fling_prob  # random to up
        # go up (try to go to middle)
        transitions[2, 3, 1] = move_prob  # up -> middle
        transitions[2, 3, 0] = fling_prob  # random to left
        transitions[2, 3, 2] = fling_prob  # random to right
        transitions[2, 3, 4] = fling_prob  # random to up
        # go down (try to stay)
        transitions[3, 3, 3] = move_prob  # down -> stay at down
        transitions[3, 3, 0] = fling_prob  # random to left
        transitions[3, 3, 1] = fling_prob  # random to middle
        transitions[3, 3, 2] = fling_prob  # random to right

        ## State 4 (up)
        # go right (try to stay)
        transitions[0, 4, 4] = move_prob  # right -> stay at up
        transitions[0, 4, 0] = fling_prob  # random to left
        transitions[0, 4, 1] = fling_prob  # random to middle
        transitions[0, 4, 3] = fling_prob  # random to down
        # go left (try to stay)
        transitions[1, 4, 4] = move_prob  # left -> stay at up
        transitions[1, 4, 0] = fling_prob  # random to left
        transitions[1, 4, 2] = fling_prob  # random to right
        transitions[1, 4, 3] = fling_prob  # random to down
        # go up (try to stay)
        transitions[2, 4, 4] = move_prob  # up -> stay at up
        transitions[2, 4, 0] = fling_prob  # random to left
        transitions[2, 4, 1] = fling_prob  # random to middle
        transitions[2, 4, 2] = fling_prob  # random to right
        # go down (try to go to middle)
        transitions[3, 4, 1] = move_prob  # down -> middle
        transitions[3, 4, 0] = fling_prob  # random to left
        transitions[3, 4, 2] = fling_prob  # random to right
        transitions[3, 4, 3] = fling_prob  # random to down

        rewards = np.array([0, 0, 6, -1, 10])

        super().__init__(
            N_states,
            N_actions,
            rewards,
            transitions,
            discount_factor,
            [],
            episode_length,
            initial_state=initial_state,
            observation_type="state",
            observation_noise=0,
        )
