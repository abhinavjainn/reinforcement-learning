import numpy as np

from rl_algorithms.tabular_model_based import policy_evaluation

# used to compare model free algorithms' policy value with the policy_evaluation value
DIFFERENCE_TOLERANCE = 0.005   #0.5  #0.08  #0.05  #0.005


def e_greedy(epsilon: float, q: np.array, action: int):
    """
    Simulate e-greedy algorithm
    :param epsilon: the current exploration factor
    :param q: an array of `action value function` for each action at the current state
    :param action: the action index
    :return: an int represents the action chosen by the e-greedy algorithm
    """
    # initiate a random state
    rng = np.random.default_rng()

    # if this value is smaller than epsilon e-greedy acts greedily
    if rng.uniform(0, 1) < (1 - epsilon):
        # e-greedy is exploitative
        noise = rng.normal(loc=0.0, scale=1e-10, size=4)
        q = np.add(q, noise)
        return q.argmax()
    else:
        # e-greedy is explorative
        return rng.choice(action)


def sarsa(env, max_episodes: int, eta: float, gamma: float, epsilon: float, optimal_value: np.array, seed=None):
    """
    Implement `sarsa control` algorithm to find the optimal policy
    :param env: initialized frozen lake environment
    :param max_episodes: max number of episodes
    :param eta: an initial learning rate
    :param gamma: discount factor
    :param epsilon: an initial exploration factor
    :param optimal_value: an array of the value of the optimal policy, according to policy iteration
                          used to compare if sarsa reached the optimal policy or not
    :param seed: (optional) seed that controls the pseudorandom number generator
    :return: a tuple of the optimal policy array and value function for each state according to sarsa algorithm
    """
    # make learning rate `eta` and exploring factor `epsilon` decaying
    # "decrease linearly as the number of episodes increases"
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialize estimated `action value function` for each state and action
    q = np.zeros((env.n_states, env.n_actions))

    # iterate over the number of episodes
    for i in range(max_episodes):
        # start from a fresh environment each time
        state = env.reset()
        # select action `a` according to e-greedy policy based on Q
        action = e_greedy(epsilon[i], q[state], env.n_actions)

        # iterate until state `s` is terminal
        done = False
        while not done:
            # get the observed next_state `s'` and observed reward `r` for action `a` at state `s`
            next_state, reward, done = env.step(action)
            # Select action `a'` for state `s'` according to an e-greedy policy based on Q.
            next_action = e_greedy(epsilon[i], q[next_state], env.n_actions)

            # fill the action value for this action in this state according to the equation in sarsa algorithm
            q[state, action] += eta[i] * (reward + (gamma * q[next_state, next_action]) - q[state, action])

            # update the current state and action with the new ones
            state = next_state
            action = next_action

        # update policy and value by taking the maximum value
        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        # evaluate this policy using `policy evaluation`
        value_from_policy_eval = policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)

        # compare the obtained values from policy evaluation and the values from sarsa
        # if the difference is very small (difference is within 0.01 for each value)
        # then this is the optimal policy, not need to continue the episodes, return this policy and its value
        differences = abs(value_from_policy_eval - optimal_value)
        if np.sum(differences) < DIFFERENCE_TOLERANCE * (env.n_states - 1):  # exclude absorbing state -> always 0
            print('Number of Sarsa Control episodes to reach optimal policy: ', str(i + 1))
            return policy, value, True

    # if we reached this part of code, that means we did not find the optimal policy and all the episodes are consumed
    print(f'All the {max_episodes} episodes are consumed, Sarsa Control did not reach an optimal policy.')
    return policy, value, False


def q_learning(env, max_episodes: int, eta: float, gamma: float, epsilon: float, optimal_value: np.array, seed=None):
    """
    Implement `q-learning control` algorithm to find the optimal policy
    :param env: initialized frozen lake environment
    :param max_episodes: max number of episodes
    :param eta: an initial learning rate
    :param gamma: discount factor
    :param epsilon: an initial exploration factor
    :param optimal_value: an array of the value of the optimal policy, according to policy iteration
                              used to compare if sarsa reached the optimal policy or not
    :param seed: (optional) seed that controls the pseudorandom number generator
    :return: a tuple of the optimal policy array and value function for each state according to q-learning algorithm
    """
    # make learning rate `eta` and exploring factor `epsilon` decaying
    # "decrease linearly as the number of episodes increases"
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialize estimated `action value function` for each state and action
    q = np.zeros((env.n_states, env.n_actions))

    # iterate over the number of episodes
    for i in range(max_episodes):
        # start from a fresh environment each time
        state = env.reset()

        # iterate until state `s` is terminal
        done = False
        while not done:
            # select action `a` according to e-greedy policy based on Q
            action = e_greedy(epsilon[i], q[state], env.n_actions)
            # get the observed next_state `s'` and observed reward `r` for action `a` at state `s`
            next_state, reward, done = env.step(action)

            # fill the action value for this action in this state according to the equation in q-learning algorithm
            q[state, action] += eta[i] * (reward + (gamma * max(q[next_state])) - q[state, action])

            # update the current state
            state = next_state

        # update policy and value by taking the maximum value
        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        # evaluate this policy using `policy evaluation`
        value_from_policy_eval = policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100)

        # compare the obtained values from policy evaluation and the values from sarsa
        # if the difference is very small (difference is within 0.01 for each value)
        # then this is the optimal policy, not need to continue the episodes, return this policy and its value
        differences = abs(value_from_policy_eval - optimal_value)
        if np.sum(differences) < DIFFERENCE_TOLERANCE * (env.n_states - 1):  # exclude absorbing state -> always 0
            print('Number of Q-learning Control episodes to reach optimal policy: ', str(i + 1))
            return policy, value,True

    # if we reached this part of code, that means we did not find the optimal policy and all the episodes are consumed
    print(f'All the {max_episodes} episodes are consumed, Q-learning Control did not reach an optimal policy.')
    return policy, value, False
