from typing import Tuple

import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations) -> np.array:
    """
    Evaluate a given policy within a threshold theta
    :param env: an initialized environment object (frozen lake)
    :param policy: an array of size n_states of policy π
    :param gamma: a value between 0 and 1 that represents the `discount factor`, as it gets closer to 1,
                  the agent will value future rewards more
    :param theta: a tolerance threshold, once the threshold reached, the evaluation will stop
    :param max_iterations: maximum number of iterations to avoid letting the program run indefinitely
    :return: an array of n_states of that represents the evaluated policy
    """
    # the objective is to converge to the true value function for a given policy π.

    # initialization of value function to 0 for each state
    value = np.zeros(env.n_states, dtype=np.float)
    delta = abs(theta) + 1  # Force the loop entry
    i = 0

    # repeat until change in value is below the threshold or the max iterations reached
    while delta > theta and i < max_iterations:
        delta = 0
        # iterate through each state
        for state in range(env.n_states):
            old_value = value[state]
            total_expected_return = 0
            # try all possible actions that can be taken from this state
            for next_state in range(env.n_states):
                # probability of transitioning to this next state
                next_state_probability = env.p(next_state, state, action=policy[state])
                # calculate the discounted reward
                discounted_reward = env.r(next_state, state, action=policy[state]) + (gamma * value[next_state])
                # calculate the expected return
                total_expected_return += next_state_probability * discounted_reward
            # record the new expected state
            value[state] = total_expected_return
            # calculate the difference to see if the threshold is reached
            delta = max(delta, np.abs(old_value - value[state]))
        i += 1

    return value


def policy_improvement(env, policy, value, gamma) -> Tuple[np.array, bool]:
    """
    Make an improvement on a given policy
    :param env: an initialized environment object (frozen lake)
    :param policy: an array of size n_states of policy π
    :param value: an array of size n_states of evaluated policy π
    :param gamma: a value between 0 and 1 that represents the `discount factor`, as it gets closer to 1,
                  the agent will value future rewards more
    :return: an array of the expected value
    """
    # initialize with zeros
    improved_policy = np.zeros(env.n_states, dtype=int)
    # boolean used to keep track if the policy is improved or not
    policy_stable = True

    # iterate over each state
    for state in range(env.n_states):
        old_action = policy[state]
        new_actions = []
        new_action_values = []

        # for taken action, calculate the expected value of the next state given this action
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                new_actions.append(action)
                # probability of transitioning to this next state
                next_state_probability = env.p(next_state, state, action=action)
                # calculate the discounted reward
                discounted_reward = env.r(next_state, state, action=action) + (gamma * value[next_state])
                new_action_values.append(next_state_probability * discounted_reward)

        # choose the action that leads to maximum reward
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        improved_policy[state] = best_action

        # if there is change, the policy is still not stable, we need to keep improving, set `policy_stable` False
        if old_action != best_action:  # and state != env.absorbing_state_idx:
            policy_stable = False

    return improved_policy, policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None) -> Tuple[np.array, np.array]:
    """
    The policy iteration algorithm
    An iterative process of policy evaluation and policy improvement until the algorithm converged
    and policy is not changing anymore
    :param env: an initialized environment object (frozen lake)
    :param gamma: a value between 0 and 1 that represents the `discount factor`, as it gets closer to 1,
                  the agent will value future rewards more
    :param theta: a tolerance threshold, once the threshold reached, the evaluation will stop
    :param max_iterations: maximum number of iterations to avoid letting the program run indefinitely
    :param policy: optional, an array of size n_states of policy π
    :return: a tuple of the optimal policy array and value function for each state according to policy iteration
    """
    # if policy is not passed, initialized it
    policy = np.zeros(env.n_states, dtype=int) if policy is None else np.array(policy, dtype=int)

    # keep track of number of iterations
    iteration = 0

    while True:
        # evaluate the policy first
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        # make improvement, `policy_stable` will be True when there is no further change on policy
        policy, policy_stable = policy_improvement(env, policy, value, gamma)
        iteration += 1
        if policy_stable:
            break

    print(f"Number of iterations required in `policy iteration`: {iteration}")

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None) -> Tuple[np.array, np.array]:
    """
    The value iteration algorithm
    :param env: an initialized environment object (frozen lake)
    :param gamma: a value between 0 and 1 that represents the `discount factor`, as it gets closer to 1,
                  the agent will value future rewards more
    :param theta: a tolerance threshold, once the threshold reached, the evaluation will stop
    :param max_iterations: maximum number of iterations to avoid letting the program run indefinitely
    :param value:
    :return: a tuple of the optimal policy array and value function for each state according to value iteration
    """
    # initialize with zeros if value is not passed
    value = np.zeros(env.n_states) if value is None else np.array(value, dtype=np.float)
    # initialize policy with zeros
    policy = np.zeros(env.n_states, dtype=int)

    delta = abs(theta) + 1  # Force the loop entry
    iteration = 0
    # repeat until change in value is below the threshold or the max iterations reached
    while delta > theta and iteration < max_iterations:
        delta = 0
        for state in range(env.n_states):
            old_value = value[state]
            new_value = []
            for action in range(env.n_actions):
                total_expected_return = 0
                for next_state in range(env.n_states):
                    # probability of transitioning to this next state
                    next_state_probability = env.p(next_state, state, action=action)
                    # calculate the discounted reward
                    discounted_reward = env.r(next_state, state, action=action) + (gamma * value[next_state])
                    # accumulate for each next state
                    total_expected_return += next_state_probability * discounted_reward

                # add this action expected value for all next states
                new_value.append(total_expected_return)

            # choose the best value
            value[state] = max(new_value)
            # calculate the difference to see if the threshold is reached
            delta = max(delta, np.abs(old_value - value[state]))
        iteration += 1

    # now find the best action and add it to the policy
    for state in range(env.n_states):
        new_actions = []
        new_action_values = []
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                # probability of transitioning to this next state
                next_state_probability = env.p(next_state, state, action=action)
                # calculate the discounted reward
                discounted_reward = env.r(next_state, state, action=action) + (gamma * value[next_state])
                # add the action and its value
                new_actions.append(action)
                new_action_values.append(next_state_probability * discounted_reward)
        # choose the best action that gives the maximum value
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        policy[state] = best_action

    print(f"Number of iterations required in `value iteration`: {iteration}")

    return policy, value
