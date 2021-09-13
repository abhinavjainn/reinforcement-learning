import argparse
from datetime import datetime

from environments.frozen_lake_environment import FrozenLake
from rl_algorithms.non_tabular_model_free import LinearWrapper, linear_sarsa, linear_q_learning
from rl_algorithms.tabular_model_based import policy_iteration, value_iteration
from rl_algorithms.tabular_model_free import sarsa, q_learning


def run_forzen_lake_rl(lake, seed=0):
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # play(env)

    # print('# Model-based algorithms\n')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('## Policy iteration')
    start_time = datetime.now()

    optimal_policy, optimal_value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(optimal_policy, optimal_value)

    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}\n')

    # print('## Value iteration')
    # start_time = datetime.now()
    #
    # policy, value = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # end_time = datetime.now()
    # print(f'Duration: {end_time - start_time}\n')

    print('# Model-free algorithms')
    # max_episodes = 2000
    # eta = 0.5
    # epsilon = 0.5


    # print('')

    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, optimal_value, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, optimal_value, seed=seed)
    # env.render(policy, value)

    # print('')
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta,
    #                                gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)

# ----------- BEGIN TEST Q5-------------- #
    from datetime import date

    print('')
    _max_episodes = [10000]
    _eta = [0.5]
    _epsilon = [0.5]
    optimal = False
    print("Start time", date.today().strftime("%d.%m.%Y"),",",datetime.now().strftime("%H:%M:%S"))
    for eta in _eta:
        for epsilon in _epsilon:
            for max_episodes in _max_episodes:
                print('')
                print('## Sarsa')
                # print('## Q-learning')
                print("eta:",eta,"epsilon:",epsilon,"max_episodes:",max_episodes)
                policy, value, optimal = sarsa(env, max_episodes, eta, gamma, epsilon, optimal_value, seed=seed)
                # policy, value,optimal = q_learning(env, max_episodes, eta, gamma, epsilon, optimal_value, seed=seed)
                if optimal == True:
                    print("optimal policy found, eta:",eta,"epsilon:",epsilon,"Episodes:",max_episodes)
                    env.render(policy, value)
                    break
            if optimal == True:
                break
        if optimal == True:
            break
    print("End time", date.today().strftime("%d.%m.%Y"),",",datetime.now().strftime("%H:%M:%S"))
    if optimal == False:
        env.render(policy, value)

        # print('')
        #
        # print('## Q-learning')
        # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, optimal_value, seed=seed)
        # env.render(policy, value)
#------------END TEST Q5---------#

def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid Action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


if __name__ == '__main__':
    # Small lake
    small_lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
    # Big lake
    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    # arg parser to choose which lake to run
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lake",
        help="Choose which lake to run: `small` for small lake OR `big` for big lake, default: small",
        choices=['small', 'big'],
        default='small')
    args = parser.parse_args()
    lake = big_lake if args.lake == 'big' else small_lake

    # run reinforcement learning methods on the chosen lake
    run_forzen_lake_rl(lake=lake)
