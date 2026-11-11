import math

def logcomb(n, k):
  return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def logbeta(a, b):
  return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)