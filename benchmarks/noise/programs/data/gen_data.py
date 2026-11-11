import numpy as np

n = 100
h = 2
f = 1.001
# g = 1

q = 2
r = 1


with open('processed_data.csv', 'w') as out:
  out.write('true_x, obs\n')

prev_x = 0
for i in range(n):
  x = np.random.normal(f * prev_x, np.sqrt(q))
  z = np.random.normal(h * x, np.sqrt(r))

  prev_x = x

  with open('processed_data.csv', 'a') as out:
    out.write(str(x) + ', ' + str(z) + '\n')
