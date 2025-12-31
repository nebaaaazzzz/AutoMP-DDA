import numpy as np
import matplotlib.pyplot as plt

def simulate_bandit(epsilon, steps=1000, n_arms=10):
    # Hidden true reward means for each arm
    true_values = np.random.normal(0, 1, n_arms)
    
    # Estimates and counts
    q_estimates = np.zeros(n_arms)
    action_counts = np.zeros(n_arms)
    
    rewards = np.zeros(steps)
    
    for t in range(steps):
        # Epsilon-Greedy selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(q_estimates)
            
        # Get actual reward (true mean + noise)
        reward = true_values[action] + np.random.normal(0, 1)
        
        # --- SAMPLE AVERAGE UPDATE ---
        action_counts[action] += 1
        # Step size is 1/k
        step_size = 1.0 / action_counts[action]
        # NewEstimate = OldEstimate + StepSize * [Target - OldEstimate]
        q_estimates[action] += step_size * (reward - q_estimates[action])
        
        rewards[t] = reward
        
    return rewards

# Simulation settings
n_runs = 2000
n_steps = 1000

# Run both strategies
greedy_results = np.mean([simulate_bandit(0.0) for _ in range(n_runs)], axis=0)
egreedy_results = np.mean([simulate_bandit(0.1) for _ in range(n_runs)], axis=0)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(greedy_results, label="Greedy ($\epsilon=0$)")
plt.plot(egreedy_results, label="$\epsilon$-greedy ($\epsilon=0.1$)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Sample-Average Action-Value Estimates")
plt.legend()
plt.show()