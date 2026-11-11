import numpy as np

n = 100

sx0 = 0.1
sy0 = 0.1

x0 = 0
y0 = 0

with open('processed_data.csv', 'w') as out:
  out.write('sx,sy,x,y,sx_obs,sy_obs,a_obs\n')

sx = sx0
sy = sy0
x = x0
y = y0

def alt(x, y):
  if y > 50:
    return 10 - 0.01 * x * x + 0.0001 * x * x * x
  else:
    return x + 0.1

for i in range(n):
  x = np.random.normal(x + sx, np.sqrt(1))
  y = np.random.normal(y + sy, np.sqrt(1))
  sx = np.random.normal(sx, np.sqrt(0.1))
  sy = np.random.normal(sy, np.sqrt(0.1))

  a = alt(x, y)

  sx_obs = np.random.normal(sx, np.sqrt(1))
  sy_obs = np.random.normal(sy, np.sqrt(1))
  a_obs = np.random.normal(a, np.sqrt(1))

  prev_x = x

  with open('processed_data.csv', 'a') as out:
    out.write(f'{sx},{sy},{x},{y},{sx_obs},{sy_obs},{a_obs}\n')
