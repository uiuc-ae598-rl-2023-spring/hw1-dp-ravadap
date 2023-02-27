import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import gridworld
import model_based
import model_free

def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # they converge to the same policy and value function !
    gamma = 0.95
    policy_pi, V_pi, mvf_pi = model_based.policy_improvement(env, gamma)
    print(V_pi)
    
    policy_vi, V_vi, mvf_vi = model_based.value_iteration(env, gamma)

    # change these parameters based on model
    gamma = 0.95
    alpha = 0.5
    epsilon = 0.5
    episodes = 500

    Q_SARSA = model_free.SARSA(env, episodes, gamma, alpha, epsilon)
    # print(Q_SARSA)

    V_SARSA = model_free.TD_0(env, Q_SARSA, episodes, gamma, alpha)
    print(V_SARSA)

    Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon)
    
    V_Q_LEARNING = model_free.TD_0(env, Q_LEARNING, episodes, gamma, alpha)
    print(V_Q_LEARNING)

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(4)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    plt.figure(0)
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')

    # plot for value iteration
    value_iters = np.arange(1, np.size(mvf_vi)+1)
    policy_iters = np.arange(1, np.size(mvf_pi)+1)
    plt.figure(1)
    plt.plot(mvf_vi, value_iters)
    plt.plot(mvf_pi, policy_iters)
    plt.xlabel('Mean Value of Value Function')
    plt.ylabel('Iterations')
    plt.savefig('figures/gridworld/value_iter_plot.png')

    # plot for policy iteration

if __name__ == '__main__':
    main()
