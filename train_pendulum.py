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

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Have to change these parameters based on model
    gamma = 0.95
    alpha = 0.2
    epsilon = 0.75
    episodes = 1000

    pendulum = True

    path = 'figures/pendulum/SARSA_trace_and_reward.png'
    Q_SARSA = model_free.SARSA(env, episodes, gamma, alpha, epsilon, path, pendulum)
    policy_Q_SARSA = np.argmax(Q_SARSA, axis = 1)
    # print(policy_Q_SARSA.reshape(15, 21))

    path = 'figures/pendulum/Q_LEARNING_trace_and_reward.png'
    Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon, path, pendulum)
    policy_Q_LEARNING = np.argmax(Q_LEARNING, axis = 1)
    # print(policy_Q_LEARNING.reshape(15, 21))

    V_SARSA = model_free.TD_0(env, Q_SARSA, episodes, gamma, alpha)
    # print(V_SARSA.reshape(15, 21))
    
    V_Q_LEARNING = model_free.TD_0(env, Q_LEARNING, episodes, gamma, alpha)
    # print(V_Q_LEARNING.reshape(15, 21))

    path = 'figures/pendulum/SARSA_policy_and_state_value_function.png'
    model_free.policy_and_state_value_function_plotting(policy_Q_SARSA.reshape(15,21), V_SARSA.reshape(15, 21), path)

    path = 'figures/pendulum/Q_LEARNING_policy_and_state_value_function.png'
    model_free.policy_and_state_value_function_plotting(policy_Q_LEARNING.reshape(15,21), V_Q_LEARNING.reshape(15, 21), path)

if __name__ == '__main__':
    main()
