import random
import numpy as np
import matplotlib.pyplot as plt
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
    policy_pi, V_pi = model_based.policy_improvement(env, gamma)
    
    policy_vi, V_vi = model_based.value_iteration(env, gamma)

    # change these parameters based on model
    gamma = 0.95
    alpha = 0.05
    epsilon = 0.1
    episodes = 100

    Q_SARSA = model_free.SARSA(env, episodes, gamma, alpha, epsilon)
    print(Q_SARSA)

    Q_LEARNING = model_free.Q_learning(env, episodes, gamma, alpha, epsilon)
    print(Q_LEARNING)

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
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/gridworld/test_gridworld.png')


if __name__ == '__main__':
    main()
