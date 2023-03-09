import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import model_free

def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)), env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise Exception(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    print('test_x_to_s: passed')


def main():
    # Create environment
    #
    #   By default, both the state space (theta, thetadot) and the action space
    #   (tau) are discretized with 31 grid points in each dimension, for a total
    #   of 31 x 31 states and 31 actions.
    #
    #   You can change the number of grid points as follows (for example):
    #
    #       env = discrete_pendulum.Pendulum(n_theta=11, n_thetadot=51, n_tau=21)
    #
    #   Note that there will only be a grid point at "0" along a given dimension
    #   if the number of grid points in that dimension is odd.
    #
    #   How does performance vary with the number of grid points? What about
    #   computation time?
    env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)

    # Apply unit test to check state representation
    test_x_to_s(env)

    # Initialize simulation
    s = env.reset()

    # Have to change these parameters based on model
    gamma = 0.95
    alpha = 0.3
    epsilon = 0.7
    episodes = 5000

    pendulum = True
    
    Q_SARSA, ep_reward_SARSA, log_SARSA = model_free.SARSA(env, episodes, gamma, alpha, epsilon, pendulum)
    policy_Q_SARSA = np.argmax(Q_SARSA, axis = 1)
    # print(policy_Q_SARSA.reshape(15, 21))

    Q_LEARNING, ep_reward_LEARNING, log_Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon, pendulum)
    policy_Q_LEARNING = np.argmax(Q_LEARNING, axis = 1)
    # print(policy_Q_LEARNING.reshape(15, 21))

    V_SARSA = model_free.TD_0(env, Q_SARSA, episodes, gamma, alpha)
    # print(V_SARSA.reshape(15, 21))
    
    V_Q_LEARNING = model_free.TD_0(env, Q_LEARNING, episodes, gamma, alpha)
    # print(V_Q_LEARNING.reshape(15, 21))

    # for epsilon variation
    epsilon_vals = np.linspace(0,1, num=5, endpoint=True)
    Q_SARSA_Returns_ep = []
    Q_LEARNING_Returns_ep = []
    alpha1 = 0.2
    for ep in epsilon_vals:
        _, ep_return = model_free.SARSA(env, episodes, gamma, alpha1, ep)
        Q_SARSA_Returns_ep.append(ep_return)

        _, ep_return = model_free.Q_learning(env, episodes, gamma, alpha1, ep)
        Q_LEARNING_Returns_ep.append(ep_return)

    # for alpha variation
    alpha_vals = np.linspace(0,1, num=5, endpoint=True)
    Q_SARSA_Returns_alp = []
    Q_LEARNING_Returns_alp = []
    epsilon1 = 0.5
    for alp in alpha_vals:
        _, ep_return = model_free.SARSA(env, episodes, gamma, alp, epsilon1)
        Q_SARSA_Returns_alp.append(ep_return)

        _, ep_return = model_free.Q_learning(env, episodes, gamma, alp, epsilon1)
        Q_LEARNING_Returns_alp.append(ep_return)

    path = 'figures/pendulum/SARSA_epsilon_and_alpha_variation'
    model_free.plot_vary_parameter(episodes, Q_SARSA_Returns_ep, Q_SARSA_Returns_alp, epsilon_vals, alpha_vals, path)

    path = 'figures/pendulum/Q_LEARNING_epsilon_and_alpha_variation'
    model_free.plot_vary_parameter(episodes, Q_LEARNING_Returns_ep, Q_LEARNING_Returns_alp, epsilon_vals, alpha_vals, path)

    path = 'figures/pendulum/SARSA_policy_and_state_value_function.png'
    model_free.policy_and_state_value_function_plotting(policy_Q_SARSA.reshape(15,21), V_SARSA.reshape(15, 21), path)

    path = 'figures/pendulum/Q_LEARNING_policy_and_state_value_function.png'
    model_free.policy_and_state_value_function_plotting(policy_Q_LEARNING.reshape(15,21), V_Q_LEARNING.reshape(15, 21), path)

    path = 'figures/pendulum/SARSA_trace_and_return.png'
    model_free.plotting_sar_trace_model_free(episodes, ep_reward_SARSA, log_SARSA, path, pendulum)

    path = 'figures/pendulum/Q_LEARNING_trace_and_return.png'
    model_free.plotting_sar_trace_model_free(episodes, ep_reward_LEARNING, log_Q_LEARNING, path, pendulum)

if __name__ == '__main__':
    main()
