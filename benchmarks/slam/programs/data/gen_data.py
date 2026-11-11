import numpy as np

map = [False, False, True, False, False, True, True, True, False, True, True]
wheel_noise = 0.1
sensor_noise = 0.1
max_pos = 10
# n = max_pos * 2 + 1
n = 100

with open('processed_data.csv', 'w') as out:
  out.write('true_x, obs, cmd\n')

prev_x = 0
direction = 1
for i in range(n):
  if prev_x == 0:
    direction = 1
    cmd = 1
  elif prev_x == max_pos:
    direction = -1
    cmd = -1
  else:
    cmd = direction

  wheel_slip = np.random.binomial(1, wheel_noise) == 1
  sensor_error = np.random.binomial(1, sensor_noise) == 1

  x = max(0, min(max_pos, prev_x if wheel_slip else prev_x + cmd))

  b = map[int(x)]

  obs = not b if sensor_error else b
  
  prev_x = x
  
  with open('processed_data.csv', 'a') as out:
    out.write(str(x) + ', ' + str(1 if obs else 0) + ', ' + str(cmd) + '\n')
