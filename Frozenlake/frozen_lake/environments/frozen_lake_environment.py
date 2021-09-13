import contextlib
from itertools import product
from typing import List, Optional

import numpy as np

from environments.base_environment import Environment


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


# actions are represented in integers
ACTIONS = {
    0: (-1, 0),  # UP
    1: (0, -1),  # LEFT
    2: (1, 0),  # DOWN
    3: (0, 1),  # RIGHT
}

# tiles symbols
GOAL_SYMBOL = '$'
HOLE_SYMBOL = '#'
FROZEN_SYMBOL = '.'
START_SYMBOL = '&'


class FrozenLake(Environment):
    def __init__(self, lake: List, slip: float, max_steps: int, seed: Optional[int] = None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        self.seed = seed
        self.seed_random_state()

        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        self.max_steps = max_steps

        n_actions = len(ACTIONS.values())
        # all the lake's tiles + absorbing state
        n_states = self.lake.size + 1

        # initialize policy with arbitrary values (zeros), except for the start state, set policy 1
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        # Initializing environment
        super().__init__(n_states, n_actions, max_steps, pi, self.seed)

        # define the absorbing state: index and coordinates
        self.absorbing_state_idx = n_states - 1

        # indices to coordinates states representation
        self.indices_to_coords = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        # add additional coordinates to represent the absorbing state, an out of grid coordination (-1,-1)
        self.indices_to_coords.append((-1, -1))
        # coordinates to  indices states representation
        self.coords_to_indices = {coord: index for (index, coord) in enumerate(self.indices_to_coords)}

        # initiate transition probabilities:
        # for each possible state `s` to each possible state `s'` through each possible action `a`
        self.transition_probabilities = np.zeros((n_states, n_states, self.n_actions))

        # set the probabilities stochastically
        self.set_transitioning_probabilities()

    def set_transitioning_probabilities(self):
        """
        Calculate the 3D matrix of probabilities of transitioning to state `s'` from state `s` through action `a`
        The 3D matrix is contains the combination of the following order:
        [next_state, current_state, action]
        The states' indices are from 0 to the lake_size + 1, the last state being the absorbing state
        The action indices are 0, 1, 2, 3 corresponding to UP, LEFT, DOWN and RIGHT respectively
        :return:
        """
        # Example, small frozen lake: 4x4 -> 16+1 states (plus absorbing state), so the 3D array size is [17,17,4]
        # each cell in the array represents the probability of transitioning from state s to state s' under action a

        # get all final states, the GOAL and HOLE states which should transition the agent to the absorbing state
        final_states_indices = self.get_states_indices_by_symbol(GOAL_SYMBOL, HOLE_SYMBOL)
        # also add the absorbing state index, because it is a finale state, and you can't get out of it
        final_states_indices.append(self.absorbing_state_idx)
        # the probability of any action when the final_states_indices is being the current state is 1.0

        for _state_idx, _state_coords in enumerate(self.indices_to_coords):
            # 1. for ANY action, whenever the current state `s` is final [GOAL, HOLE, ABSORBING]
            # the next state `s'` is the ABSORBING state, the probability of taking that action is 1.0
            if _state_idx in final_states_indices:
                for _action in ACTIONS.keys():
                    # next state, current state, action
                    self.transition_probabilities[self.absorbing_state_idx, _state_idx, _action] = 1.0

            # 2. for all other states [FROZEN, START], we calculate the probability of taking an action `a`
            # and the next possible state s'
            else:
                for _action in ACTIONS.keys():
                    # first find the next state s` through this action , if it is invalid, stay in the same state

                    # calculate the new coordinates according to the taken action
                    current_state_coord = self.indices_to_coords[_state_idx]
                    next_state_coords = tuple(np.add(current_state_coord, ACTIONS[_action]))
                    # if next_state_coords are invalid (outside the lake grid), next_desired_state_idx will be the
                    # current state (_state_idx)
                    next_desired_state_idx = self.coords_to_indices.get(next_state_coords, _state_idx)

                    # the agent can take this action with a probability of slipping into other directions
                    # so it can't take this action 100%, if slipping chance is 10%, it can take it 90% of the times
                    self.transition_probabilities[next_desired_state_idx, _state_idx, _action] = 1 - self.slip

                    # the rest of probability (10% chance of slipping) should be split over the four possible directions
                    # "the agent slips (moves one tile in a random direction, `which may be the desired direction`)"

                    # So we need to find all corresponding next states for all actions
                    # and set a portion of the slip probability to each action
                    all_possible_next_states = []
                    # iterate over all action, and find all possible next positions
                    for _action_idx, _action_value in ACTIONS.items():
                        _next_state_coords = tuple(np.add(current_state_coord, _action_value))
                        # if the state is out of the grid, set it to current state
                        _next_state_idx = self.coords_to_indices.get(_next_state_coords, _state_idx)
                        all_possible_next_states.append(_next_state_idx)

                    # for each possible action in this current state, set a portion of the slip to be
                    # the probability of transitioning to their corresponding next state through the taken action
                    for _possible_next_state in all_possible_next_states:
                        _p = self.slip / len(all_possible_next_states) if len(all_possible_next_states) > 0 else 0
                        # here we are accumulating the probabilities for each combination
                        # because when slipping, we might move using the desired direction
                        # so we dont want to override the probability value set above `next_desired_state_idx`
                        # hence, we add on it
                        self.transition_probabilities[_possible_next_state, _state_idx, _action] += _p

    def get_states_indices_by_symbol(self, *symbols: str) -> List[int]:
        """
        Takes a one or multiple symbols and return an array of indices of the symbol state
        :param symbols: only allowed '$', '.', '#', '&'
        :return: np.array, of all indices that matches the passed symbol state
        """
        _allowed_symbols = ('$', '.', '#', '&')
        # check if the symbols are valid, raise an error if not
        if not set(symbols).issubset(_allowed_symbols):
            raise ValueError(f'Symbol should be one of theses characters: {_allowed_symbols}, '
                             f'the passed symbols are {symbols}.')
        indices = []
        for _symbol in symbols:
            indices.extend(np.where(self.lake_flat == _symbol)[0].astype(int))
        return indices

    def step(self, action: int):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state_idx) or done
        return state, reward, done

    def p(self, next_state: int, state: int, action: int) -> float:
        """
        Calculate the single probability of transitioning from `state` to `next_state` through `action`
        :param next_state:
        :param state:
        :param action:
        :return:
        """
        return self.transition_probabilities[next_state, state, action]

    def r(self, next_state: int, state: int, action: int) -> int:
        """
        Calculate the single reward of transitioning from `state` to `next_state` through `action`
        :param next_state:
        :param state:
        :param action:
        :return:
        """
        reward = 0
        # if the current state is goal, the reward is 1, for any other state, reward is 0
        if state == self.get_states_indices_by_symbol(GOAL_SYMBOL)[0]:
            reward = 1
        return reward

    def seed_random_state(self, seed: Optional[int] = None) -> None:
        """
        Set the random state to use it whenever you want to deal with random variables
        :param seed: Optional, an int to seed the random state, if not given, will use the one that initiates this env
        :return:
        """
        # for debugging purposes, you need to seed random state every time when you want to generate random variables
        # this way you will get the exact same random numbers, making debugging easier
        # ex:
        # rng = np.random.RandomState(0)
        # rng.rand(4)
        # # Out[1]: array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        # rng = np.random.RandomState(0)
        # rng.rand(4)
        # Out[2]: array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        # same output everytime
        if not seed:
            seed = self.seed
        self.random_state = np.random.RandomState(seed)
        return None

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state_idx:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # actions = ['^', '_', '<', '>']
            actions = ['^', '<', '_', '>']  # Corrected action array

            print('Lake: ')
            print(self.lake)

            print('Policy: ')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
