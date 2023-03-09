import random
import numpy as np
import matplotlib.pyplot as plt
from discrete_pendulum import Pendulum
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
    # print(np.reshape(np.argmax(policy_pi, axis=1), (5,5)))
    
    policy_vi, V_vi, mvf_vi = model_based.value_iteration(env, gamma)

    # change these parameters based on model
    gamma = 0.95
    alpha = 0.5
    epsilon = 0.4
    episodes = 5000

    pendulum = False

    Q_SARSA, ep_reward_SARSA, log_SARSA = model_free.SARSA(env, episodes, gamma, alpha, epsilon, pendulum)
    policy_Q_SARSA = np.argmax(Q_SARSA, axis = 1)

    V_SARSA = model_free.TD_0(env, Q_SARSA, episodes, gamma, alpha)

    Q_LEARNING, ep_reward_LEARNING, log_Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon, pendulum)
    policy_Q_LEARNING = np.argmax(Q_LEARNING, axis = 1)
    
    V_Q_LEARNING = model_free.TD_0(env, Q_LEARNING, episodes, gamma, alpha)

    # for epsilon variation
    epsilon_vals = np.linspace(0,1, num=5, endpoint=True)
    Q_SARSA_Returns_ep = []
    Q_LEARNING_Returns_ep = []
    alpha1 = 0.4
    for ep in epsilon_vals:
        _, ep_return, _ = model_free.SARSA(env, episodes, gamma, alpha1, ep, pendulum)
        Q_SARSA_Returns_ep.append(ep_return)

        _, ep_return, _ = model_free.Q_learning(env, episodes, gamma, alpha1, ep, pendulum)
        Q_LEARNING_Returns_ep.append(ep_return)

    # for alpha variation
    alpha_vals = np.linspace(0,1, num=5, endpoint=True)
    Q_SARSA_Returns_alp = []
    Q_LEARNING_Returns_alp = []
    epsilon1 = 0.5
    for alp in alpha_vals:
        _, ep_return, _ = model_free.SARSA(env, episodes, gamma, alp, epsilon1, pendulum)
        Q_SARSA_Returns_alp.append(ep_return)

        _, ep_return, _ = model_free.Q_learning(env, episodes, gamma, alp, epsilon1, pendulum)
        Q_LEARNING_Returns_alp.append(ep_return)

    path = 'figures/gridworld/SARSA_epsilon_and_alpha_variation'
    model_free.plot_vary_parameter(episodes, Q_SARSA_Returns_ep, Q_SARSA_Returns_alp, epsilon_vals, alpha_vals, path)

    path = 'figures/gridworld/Q_LEARNING_epsilon_and_alpha_variation'
    model_free.plot_vary_parameter(episodes, Q_LEARNING_Returns_ep, Q_LEARNING_Returns_alp, epsilon_vals, alpha_vals, path)

    path = 'figures/gridworld/policy_iteration_sar_trace_and_mean_V.png'
    model_based.plotting_sar_trace_model_based(env, policy_pi, mvf_pi, path)

    path = 'figures/gridworld/value_iteration_sar_trace_and_mean_V.png'
    model_based.plotting_sar_trace_model_based(env, policy_vi, mvf_vi, path)

    path = 'figures/gridworld/policy_iteration_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(np.reshape(np.argmax(policy_pi, axis=1), (5,5)), V_pi.reshape(5, 5), path)

    path = 'figures/gridworld/value_iteration_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(np.reshape(np.argmax(policy_vi, axis=1), (5,5)), V_vi.reshape(5, 5), path)

    path = 'figures/gridworld/SARSA_trace_and_return.png'
    model_free.plotting_sar_trace_model_free(episodes, ep_reward_SARSA, log_SARSA, path, pendulum)

    path = 'figures/gridworld/Q_LEARNING_trace_and_return.png'
    model_free.plotting_sar_trace_model_free(episodes, ep_reward_LEARNING, log_Q_LEARNING, path, pendulum)

    path = 'figures/gridworld/SARSA_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(policy_Q_SARSA.reshape(5,5), V_SARSA.reshape(5, 5), path)

    path = 'figures/gridworld/Q_LEARNING_policy_and_state_value_function.png'
    model_based.policy_and_state_value_function_plotting(policy_Q_LEARNING.reshape(5,5), V_Q_LEARNING.reshape(5, 5), path)

if __name__ == '__main__':
    main()
