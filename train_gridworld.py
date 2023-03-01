import random
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn')
import gridworld
import model_based
import model_free


def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # they converge to the same policy and value function !
    gamma = 0.95
    policy_pi, V_pi, mvf_pi = model_based.policy_improvement(env, gamma)
    # print(V_pi.reshape(5, 5))
    # print(policy_pi)
    
    policy_vi, V_vi, mvf_vi = model_based.value_iteration(env, gamma)

    # change these parameters based on model
    gamma = 0.95
    alpha = 0.5
    epsilon = 0.5
    episodes = 1000

    path = 'figures/gridworld/SARSA_trace_and_reward.png'
    Q_SARSA = model_free.SARSA(env, 500, gamma, alpha, epsilon, path)
    policy_Q_SARSA = np.argmax(Q_SARSA, axis = 1)

    V_SARSA = model_free.TD_0(env, Q_SARSA, episodes, gamma, alpha)
    # print(V_SARSA.reshape(5, 5))

    path = 'figures/gridworld/Q_LEARNING_trace_and_reward.png'
    Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon, path)
    policy_Q_LEARNING = np.argmax(Q_LEARNING, axis = 1)
    
    V_Q_LEARNING = model_free.TD_0(env, Q_LEARNING, episodes, gamma, alpha)
    # print(V_Q_LEARNING.reshape(5, 5))

    path = 'figures/gridworld/policy_iteration_sar_trace_and_mean_V.png'
    model_based.plotting_sar_trace_model_based(env, policy_pi, mvf_pi, path)

    path = 'figures/gridworld/value_iteration_sar_trace_and_mean_V.png'
    model_based.plotting_sar_trace_model_based(env, policy_vi, mvf_vi, path)

    path = 'figures/gridworld/policy_iteration_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(np.reshape(np.argmax(policy_pi, axis=1), (5,5)), V_pi.reshape(5, 5), path)

    path = 'figures/gridworld/value_iteration_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(np.reshape(np.argmax(policy_vi, axis=1), (5,5)), V_vi.reshape(5, 5), path)

    path = 'figures/gridworld/SARSA_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(policy_Q_SARSA.reshape(5,5), V_SARSA.reshape(5, 5), path)

    path = 'figures/gridworld/Q_LEARNING_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(policy_Q_LEARNING.reshape(5,5), V_Q_LEARNING.reshape(5, 5), path)

if __name__ == '__main__':
    main()
