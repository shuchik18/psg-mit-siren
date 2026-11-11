import numpy as np

n = 500
x = np.random.normal(0, 50)
clutter_prob = np.random.beta(100, 1000)

with open('data.csv', 'w') as f:
  f.write(str(clutter_prob) + '\n')

for i in range(n):
  x = np.random.normal(x, 1)
  is_clutter = np.random.binomial(1, clutter_prob)
  if is_clutter:
    y = np.random.normal(0, 100)
  else:
    y = np.random.normal(x, 1)
  with open('data.csv', 'a') as f:
    f.write(str(x) + ', ' + str(y) + '\n')
