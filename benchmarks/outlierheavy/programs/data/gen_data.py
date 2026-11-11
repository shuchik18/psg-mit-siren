import numpy as np

n = 100
# x = np.random.normal(0, 50)
x = 22
# clutter_prob = np.random.beta(100, 1000)
clutter_prob = 0.09113206700990846

print(str(clutter_prob) + '\n')

with open('processed_data.csv', 'w') as f:
  f.write('true_x, obs\n')

for i in range(n):
  x = np.random.normal(x, 1)
  is_clutter = np.random.binomial(1, clutter_prob)
  if is_clutter:
    print('clutter', i)
    y = np.random.standard_t(1.1) * 1 + x
  else:
    y = np.random.normal(x, 1)
  with open('processed_data.csv', 'a') as f:
    f.write(str(x) + ', ' + str(y) + '\n')
