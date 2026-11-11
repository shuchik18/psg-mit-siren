import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

num_points = 100
turn_length = 10

x_true = np.zeros(num_points)
alt_true = np.zeros(num_points)
alt_true[0] = 0.

q = 0.1

for i in range(1, num_points):
  # x_true[i] = x_true[i - 1] + np.random.normal(0.5, np.sqrt(q))
  x_true[i] = np.sin(i / 10) * 5 + np.random.normal(0, np.sqrt(q))
  alt_true[i] = i/10

noise_std = 0.1
x_noisy = x_true.copy()
alt_noisy = alt_true.copy()

large_noise_indices = np.where(alt_true < 5, True, False)
large_noise_indices = np.argwhere(large_noise_indices).flatten()

large_noise = 1.

noise = np.full(num_points, noise_std)

for idx in large_noise_indices:
  noise[idx] += large_noise

for idx in range(num_points):
  x_noisy[idx] += np.random.normal(0, noise[idx])
  alt_noisy[idx] += np.random.normal(0, noise[idx])

plt.figure(figsize=(14, 7))
plt.plot(alt_true, label='Altitude', color='blue', linestyle='-', marker='.')
plt.plot(alt_noisy, label='Measured Altitude', color='red', marker='x')
plt.legend()
plt.xlabel('t')
plt.ylabel('Altitude')

plt.figure(figsize=(14, 7))
plt.plot(x_true, label='True Position (Ground Truth)', color='blue', linestyle='-', marker='.')
plt.plot(x_noisy, label='Noisy Measurements with Glint Errors', color='red', marker='x')
plt.legend()
plt.xlabel('t')
plt.ylabel('X')
plt.grid(True)
plt.show()

data = pd.DataFrame({'true_x': x_true, 'true_alt': alt_true, 'true_v': noise, 'x_obs': x_noisy, 'alt_obs': alt_noisy,})

data.to_csv('data.csv', index=False)

with open('../../config.json') as f:
  config = json.load(f)

  config['true_vars']['smc'][0][1] = data['true_x'].values.tolist()
  config['true_vars']['smc'][1][1] = data['true_alt'].values.tolist()

with open('../../config.json', 'w') as f:
  json.dump(config, f, indent=2)