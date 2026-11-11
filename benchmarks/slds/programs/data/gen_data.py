import numpy as np

n = 100
s_trans = np.array([[0.9, 0.1], [0.25, 0.75]])
trans_noise = [0.02248262, 1.2429]
emit_noise = [2., 0.1]

s = 0
ss = []

x = 0
xs = []

ys = []

# step through time and save s to ss
for i in range(n):
  s = np.random.choice([0, 1], size=1, p=s_trans[s])[0]
  ss.append(s)

  x_noise = trans_noise[s]

  x = np.random.normal(loc=x, scale=np.sqrt(x_noise), size=1)[0]
  xs.append(x)

  y_noise = emit_noise[s]

  y = np.random.normal(loc=x, scale=np.sqrt(y_noise), size=1)[0]
  ys.append(y)

with open('processed_data.csv', 'w') as f:
  f.write('s,x,y\n')
  for i in range(len(xs)):
    f.write(str(ss[i]) + ',' + str(xs[i]) + ',' + str(ys[i]) + '\n')

print(ss)
print(xs)
print(ys)