import numpy as np

from rl_algorithms.tabular_model_free import e_greedy


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    # make learning rate `eta` and exploring factor `epsilon` decaying
    # "decrease linearly as the number of episodes increases"
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = np.inner(features, theta)  # For all actions a:  Q(a) ← ∑ θi φ(s, a)i
        # Select an action according to an e-greedy policy
        action = e_greedy(epsilon[i], q, env.n_actions)
        terminal_state = False

        while not terminal_state:
            # r ← observed reward for action a at state s, s′ ← observed next state for action a at state s
            next_state, reward, terminal_state = env.step(action)
            td = reward - q[action]  # Temporal difference δ ← r − Q(a')
            q = np.inner(next_state, theta)  # For all actions: Q(a′) ← ∑ θi φ(s′, a′)i
            # Select an action according to an e-greedy policy
            next_action = e_greedy(epsilon[i], q, env.n_actions)
            td += gamma * max(q)  # δ ← δ + γ max_a′Q(a′)
            theta += eta[i] * td * features[action]  # θ ← θ + αδφ(s, a)
            features = next_state  # s ← s′
            action = next_action  # a ← a′
    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    # make learning rate `eta` and exploring factor `epsilon` decaying
    # "decrease linearly as the number of episodes increases"
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = np.inner(features, theta)  # For all actions: Q(a) ← ∑ θi φ(s, a)i
        terminal_state = False

        while not terminal_state:
            # Select an action according to an e-greedy policy
            action = e_greedy(epsilon[i], q, env.n_actions)
            # r ← observed reward for action a at state s, s′ ← observed next state for action a at state s
            next_state, reward, terminal_state = env.step(action)
            td = reward - q[action]  # Temporal difference δ ← r − Q(a)
            q = np.inner(next_state, theta)  # For all actions: Q(a′) ← ∑ θi φ(s′, a′)i
            td += gamma * max(q)  # δ ← δ + γ max_a′Q(a′)
            theta += eta[i] * td * features[action]  # θ ← θ + αδφ(s, a)
            features = next_state  # s ← s′
    return theta
