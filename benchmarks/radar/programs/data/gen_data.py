import numpy as np

n = 100
h = 2
f = 5

q = 2
r = 5

print(f'q: {q}\nr: {r}')

with open('data.csv', 'w') as out:
  out.write('true_x, obs\n')
with open('true_x.csv', 'w') as out:
  out.write('true_x\n')

# j = np.random.randint(0, n)
j = 200

x0 = 0

prev_x = x0
for i in range(n):
  x = np.random.normal(prev_x+f, np.sqrt(q))

  env = i == j
  if env:
    other = np.random.uniform(900, 1000)
    z = np.random.normal(x, np.sqrt(r + other))
  else:
    z = np.random.normal(x, np.sqrt(r))

  prev_x = x

  with open('data.csv', 'a') as out:
    out.write(str(x) + ', ' + str(z) + '\n')
  with open('true_x.csv', 'a') as out:
    out.write(str(x) + ',\n')
