from dataclasses import dataclass
from enum import Enum
from typing import Tuple, TypeVar, Generic, List
import numpy as np
import math

from siren.probability import logbeta, logcomb

class Annotation(Enum):
  symbolic = 1
  sample = 2

### Symbolic expressions ###
### All hybrid inference engines should be able to handle these ###
  
T = TypeVar("T")
Number = int | float
  
### External grammar

# Program expressions can use the SymExpr type (symbolic expressions for program evaluation) 
# or AbsSymExpr type (abstract symbolic expressions for analysis)
S = TypeVar("S", bound='SymExpr | AbsSymExpr')

# Before operators gets evaluated into symbolic expressions,
# they are represented as Operator objects.
Operator = Enum("Operator", [
  "add", "sub", "mul", "div", 
  "eq", "lt", 
  "cons",
  "lst",
  "pair",
  "gaussian", "beta", "bernoulli", 
  "binomial", "beta_binomial", "negative_binomial", "exponential", "gamma", 
  "poisson", "delta", "categorical", "uniform_int", "student_t", 
])

@dataclass(frozen=True)
class Expr(Generic[S]):
  pass

@dataclass(frozen=True)
class Op(Expr[S]):
  pass
  
@dataclass(frozen=True)
class Identifier(Expr):
  module: str | None
  name: str | None

  def __str__(self):
    m = "" if self.module is None else f"{self.module}."
    n = "()" if self.name is None else f"{self.name}"
    return f"{m}{n}"
  
  def __repr__(self) -> str:
    return str(self)
  
@dataclass(frozen=True)
class GenericOp(Op[S]):
  op: Operator
  args: List['Expr[S]']

  def __str__(self):
    return f"{self.op.name}({self.args})"
  
@dataclass(frozen=True)
class Fold(Expr[S]):
  func: Identifier
  init: 'Expr[S]'
  acc: 'Expr[S]'

  def __str__(self):
    return f"fold({self.func}, {self.init}, {self.acc})"
  
@dataclass(frozen=True)
class Apply(Expr[S]):
  func: Identifier
  args: List['Expr[S]']

  def __str__(self):
    return f"{self.func}({self.args})"

@dataclass(frozen=True)
class IfElse(Expr[S]):
  cond: 'Expr[S]'
  then: 'Expr[S]'
  else_: 'Expr[S]'

  def __str__(self):
    return f"if {self.cond} then {self.then} else {self.else_}"
  
@dataclass(frozen=True)
class Let(Expr[S]):
  var: List[Identifier]
  value: 'Expr[S]'
  body: 'Expr[S]'

  def __str__(self):
    if len(self.var) == 0:
      return f"let () = {self.value} in {self.body}"
    return f"let {', '.join(map(str, self.var))} = {self.value} in {self.body}"
  
@dataclass(frozen=True)
class LetRV(Expr[S]):
  var: Identifier
  annotation: Annotation | None
  distribution: 'Op[S]'
  body: 'Expr[S]'

  def __str__(self):
    if self.annotation is None:
      return f"let {self.var} <- {self.distribution} in {self.body}"
    return f"let {self.annotation} {self.var} <- {self.distribution} in {self.body}"
  
@dataclass(frozen=True)
class Observe(Expr[S]):
  condition: 'Op[S]'
  observation: 'Expr[S]'

  def __str__(self):
    return f"observe({self.condition}, {self.observation})"
  
@dataclass(frozen=True)
class Resample(Expr):

  def __str__(self):
    return f"resample()"

@dataclass(frozen=True)
class Function(Expr[S]):
  name: Identifier
  args: List[Identifier]
  body: Expr[S]

  def __str__(self):
    return f"val {self.name} = fun ({', '.join(map(str, self.args))}) = {self.body}"

@dataclass(frozen=True)
class Program:
  functions: List[Function]
  main: Expr

  def __str__(self):
    funcs = '\n'.join(map(str, self.functions))
    return f"{funcs}\n{self.main}"

@dataclass(frozen=True)
class SymExpr(Generic[T], Expr['SymExpr']):
  def __str__(self):
    return "SymExpr"
  
  def rvs(self) -> List['RandomVar']:
    raise NotImplementedError()
  
  def depends_on(self, rv: 'RandomVar', transitive: bool = False) -> bool:
    if transitive:
      return any((rv in rvs.rvs() for rvs in self.rvs()))
    else:
      return rv in self.rvs()
    
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    raise NotImplementedError()
  
@dataclass(frozen=True)
class Const(SymExpr[T]):
  v: T

  def __str__(self):
    if self.v is None:
      return "()"
    elif self.v is True:
      return "true"
    elif self.v is False:
      return "false"
    elif isinstance(self.v, list):
      return f"[{', '.join(map(str, self.v))}]"
    elif isinstance(self.v, tuple):
      return f"({', '.join(map(str, self.v))})"
    return f"{self.v}"
  
  def rvs(self) -> List['RandomVar']:
    return []
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return self
  
  def __round__(self, n=None):
    if isinstance(self.v, Number):
      return round(self.v, n)
    else:
      raise TypeError(f"Cannot round {self.v}")

@dataclass(frozen=True)
class RandomVar(SymExpr[T]):
  rv: str

  def __str__(self):
    return f"RV({self.rv})"
  
  def rvs(self) -> List['RandomVar']:
    return [self]
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    if self == rv:
      return value
    return self
    
@dataclass(frozen=True)
class Add(SymExpr[Number], Op[SymExpr]):
  left: 'SymExpr[Number]'
  right: 'SymExpr[Number]'
    
  def __str__(self):
    return f"({self.left} + {self.right})"
  
  def rvs(self) -> List['RandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  @staticmethod
  def make(left: 'SymExpr[Number]', right: 'SymExpr[Number]') -> 'SymExpr[Number]':
    match left, right:
      case Const(v1), Const(v2):
        return Const(v1 + v2)
      case Const(v1), Add(Const(v2), e3):
        return Add.make(Const(v1 + v2), e3) 
      case Const(v1), Add(e3, Const(v2)):
        return Add.make(Const(v1 + v2), e3) 
      case _:
        return Add(left, right)
      
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Add(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class Mul(SymExpr[Number], Op[SymExpr]):
  left: 'SymExpr[Number]'
  right: 'SymExpr[Number]'

  def __str__(self):
    return f"({self.left} * {self.right})"
  
  def rvs(self) -> List['RandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Mul(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))
  
  @staticmethod
  def make(left: 'SymExpr[Number]', right: 'SymExpr[Number]') -> 'SymExpr[Number]':
    match left, right:
      case Const(v1), Const(v2):
        return Const(v1 * v2)
      case Const(v1), Mul(Const(v2), e3):
        return Mul.make(Const(v1 * v2), e3)
      case Const(v1), Mul(e3, Const(v2)):
        return Mul.make(Const(v1 * v2), e3)
      case Const(v1), Add(Const(v2), e3):
        return Add.make(Const(v1 * v2), Mul.make(Const(v1), e3))
      case _:
        return Mul(left, right)

@dataclass(frozen=True)
class Div(SymExpr[Number], Op[SymExpr]):
  left: 'SymExpr[Number]'
  right: 'SymExpr[Number]'

  def __str__(self):
    return f"({self.left} / {self.right})"
  
  def rvs(self) -> List['RandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Div(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))
  
  @staticmethod
  def make(left: 'SymExpr[Number]', right: 'SymExpr[Number]') -> 'SymExpr[Number]':
    match left, right:
      case Const(v1), Const(v2):
        return Const(v1 / v2)
      case _:
        return Div(left, right)

@dataclass(frozen=True)
class Ite(SymExpr[T], Op[SymExpr]):
  cond: 'SymExpr[bool]'
  true: 'SymExpr[T]'
  false: 'SymExpr[T]'

  def __str__(self):
    return f"ite({self.cond}, {self.true}, {self.false})"
  
  def rvs(self) -> List['RandomVar']:
    return self.cond.rvs() + self.true.rvs() + self.false.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Ite(self.cond.subst_rv(rv, value), self.true.subst_rv(rv, value), self.false.subst_rv(rv, value))
  
  @staticmethod
  def make(cond: 'SymExpr[bool]', true: 'SymExpr[T]', false: 'SymExpr[T]') -> 'SymExpr[T]':
    match cond:
      case Const(v):
        return true if v else false
      case _:
        return Ite(cond, true, false)

@dataclass(frozen=True)
class Eq(SymExpr[bool], Op[SymExpr]):
  left: 'SymExpr'
  right: 'SymExpr'

  def __str__(self):
    return f"({self.left} = {self.right})"
  
  def rvs(self) -> List['RandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Eq(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))
  
  @staticmethod
  def make(left: 'SymExpr', right: 'SymExpr') -> 'SymExpr[bool]':
    match left, right:
      case Const(v1), Const(v2):
        return Const(v1 == v2)
      case _:
        return Eq(left, right)
  
@dataclass(frozen=True)
class Lt(SymExpr[bool], Op[SymExpr]):
  left: 'SymExpr[Number]'
  right: 'SymExpr[Number]'

  def __str__(self):
    return f"({self.left} < {self.right})"
  
  def rvs(self) -> List['RandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Lt(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))
  
  @staticmethod
  def make(left: 'SymExpr[Number]', right: 'SymExpr[Number]') -> 'SymExpr[bool]':
    match left, right:
      case Const(v1), Const(v2):
        return Const(v1 < v2)
      case _:
        return Lt(left, right)
  
@dataclass(frozen=True)
class Pair(SymExpr[T]):
  fst: 'SymExpr[T]'
  snd: 'SymExpr[T]'

  def __str__(self):
    return f"({self.fst}, {self.snd})"
  
  def rvs(self) -> List['RandomVar']:
    return self.fst.rvs() + self.snd.rvs()
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Pair(self.fst.subst_rv(rv, value), self.snd.subst_rv(rv, value))
  
@dataclass(frozen=True)
class Lst(SymExpr[T]):
  exprs: List['SymExpr[T]']

  def __str__(self):
    return f"[{', '.join(map(str, self.exprs))}]"
  
  def rvs(self) -> List['RandomVar']:
    return sum((e.rvs() for e in self.exprs), [])
  
  def subst_rv(self, rv: 'RandomVar', value: 'SymExpr') -> 'SymExpr':
    return Lst([e.subst_rv(rv, value) for e in self.exprs])

# Symbolic distributions are a type of symbolic expression and are built-in operators
@dataclass(frozen=True)
class SymDistr(SymExpr[T], Op[SymExpr]):
  
  def draw(self, rng: np.random.Generator) -> T:
    raise NotImplementedError()

  def score(self, v: T) -> float:
    raise NotImplementedError()

  def mean(self) -> float:
    raise NotImplementedError()
  
  def variance(self) -> float:
    raise NotImplementedError()
  
@dataclass(frozen=True)
class Normal(SymDistr[float]):
  mu: 'SymExpr[float]'
  var: 'SymExpr[float]'

  def __str__(self):
    return f"Normal({self.mu}, {self.var})"
  
  # Variance must be > 0
  def marginal_parameters(self) -> Tuple[float, float]:
    assert isinstance(self.mu, Const)
    assert isinstance(self.var, Const)
    assert self.var.v > 0

    return self.mu.v, self.var.v

  def draw(self, rng: np.random.Generator) -> float:
    mu, var = self.marginal_parameters()
    return rng.normal(mu, np.sqrt(var))

  def score(self, v: float) -> float:
    mu, var = self.marginal_parameters()
    return -0.5 * math.log(2 * math.pi * var) - ((v - mu) ** 2) / (2 * var)
  
  def mean(self) -> float:
    mu, _ = self.marginal_parameters()
    return mu
  
  def variance(self) -> float:
    _, var = self.marginal_parameters()
    return var
  
  def rvs(self) -> List['RandomVar']:
    return self.mu.rvs() + self.var.rvs()
  
@dataclass(frozen=True)
class Bernoulli(SymDistr[bool]):
  p: 'SymExpr[float]'

  def __str__(self):
    return f"Bernoulli({self.p})"
  
  # Probability must be in [0, 1]
  def marginal_parameters(self) -> float:
    assert isinstance(self.p, Const)
    assert 0 <= self.p.v <= 1

    return self.p.v
  
  def draw(self, rng: np.random.Generator) -> bool:
    p = self.marginal_parameters()
    return bool(rng.binomial(1, p))

  def score(self, v: int) -> float:
    p = self.marginal_parameters()
    return math.log(p) if v else math.log(1 - p)

  def mean(self) -> float:
    p = self.marginal_parameters()
    return p
  
  def variance(self) -> float:
    p = self.marginal_parameters()
    return p * (1 - p)
  
  def rvs(self) -> List['RandomVar']:
    return self.p.rvs()
  
@dataclass(frozen=True)
class Beta(SymDistr[float]):
  a: 'SymExpr[float]'
  b: 'SymExpr[float]'

  def __str__(self):
    return f"Beta({self.a}, {self.b})"
  
  # Parameters must be > 0
  def marginal_parameters(self) -> Tuple[float, float]:
    assert isinstance(self.a, Const)
    assert isinstance(self.b, Const)
    assert self.a.v > 0
    assert self.b.v > 0

    return self.a.v, self.b.v
  
  def draw(self, rng: np.random.Generator) -> float:
    a, b = self.marginal_parameters()
    return rng.beta(a, b)
  
  def score(self, v: float) -> float:
    a, b = self.marginal_parameters()
    if v >= 0 and v <= 1:
      return - logbeta(a, b) + (a - 1) * math.log(v) + (b - 1) * math.log(1 - v)
    return -np.inf
  
  def mean(self) -> float:
    a, b = self.marginal_parameters()
    return a / (a + b)
  
  def variance(self) -> float:
    a, b = self.marginal_parameters()
    return (a * b) / ((a + b) ** 2 * (a + b + 1))
  
  def rvs(self) -> List['RandomVar']:
    return self.a.rvs() + self.b.rvs()
    
@dataclass(frozen=True)
class Binomial(SymDistr[int]):
  n: 'SymExpr[int]'
  p: 'SymExpr[float]'

  def __str__(self):
    return f"Binomial({self.n}, {self.p})"
  
  # n must be >= 0, p must be in [0, 1]
  def marginal_parameters(self) -> Tuple[int, float]:
    assert isinstance(self.n, Const)
    assert isinstance(self.p, Const)
    assert self.n.v >= 0
    assert 0 <= self.p.v <= 1

    return self.n.v, self.p.v
  
  def draw(self, rng: np.random.Generator) -> int:
    n, p = self.marginal_parameters()
    return rng.binomial(n, p)
  
  def score(self, v: int) -> float:
    n, p = self.marginal_parameters()
    if v >= 0 and v <= n:
      return logcomb(n, v) + v * math.log(p) + (n - v) * math.log(1 - p)
    return -np.inf
  
  def mean(self) -> float:
    n, p = self.marginal_parameters()
    return n * p
  
  def variance(self) -> float:
    n, p = self.marginal_parameters()
    return n * p * (1 - p)
  
  def rvs(self) -> List['RandomVar']:
    return self.n.rvs() + self.p.rvs()
  
@dataclass(frozen=True)
class BetaBinomial(SymDistr[int]):
  n : 'SymExpr[int]'
  a : 'SymExpr[float]'
  b : 'SymExpr[float]'

  def __str__(self):
    return f"BetaBinomial({self.n}, {self.a}, {self.b})"
  
  # n must be >= 0, a and b must be > 0
  def marginal_parameters(self) -> Tuple[int, float, float]:
    assert isinstance(self.n, Const)
    assert isinstance(self.a, Const)
    assert isinstance(self.b, Const)
    assert self.n.v >= 0
    assert self.a.v > 0
    assert self.b.v > 0

    return self.n.v, self.a.v, self.b.v
  
  def draw(self, rng: np.random.Generator) -> int:
    n, a, b = self.marginal_parameters()
    return rng.binomial(n, rng.beta(a, b))
  
  def score(self, v: int) -> float:
    n, a, b = self.marginal_parameters()
    if v >= 0 and v <= n:
      return logcomb(n, v) + logbeta(v + a, n - v + b) - logbeta(a, b)
    return -np.inf
  
  def mean(self) -> float:
    n, a, b = self.marginal_parameters()
    return n * a / (a + b)
  
  def variance(self) -> float:
    n, a, b = self.marginal_parameters()
    return (n * a * b * (a + b + n)) / ((a + b) ** 2 * (a + b + 1))
  
  def rvs(self) -> List['RandomVar']:
    return self.n.rvs() + self.a.rvs() + self.b.rvs()
  
@dataclass(frozen=True)
class NegativeBinomial(SymDistr[int]):
  n: 'SymExpr[int]'
  p: 'SymExpr[float]'

  def __str__(self):
    return f"NegativeBinomial({self.n}, {self.p})"
  
  # Probability must be in [0, 1]
  def marginal_parameters(self) -> Tuple[int, float]:
    assert isinstance(self.n, Const)
    assert isinstance(self.p, Const)
    assert 0 <= self.p.v <= 1

    return self.n.v, self.p.v
  
  def draw(self, rng: np.random.Generator) -> int:
    n, p = self.marginal_parameters()
    return rng.negative_binomial(n, p)
  
  def score(self, v: int) -> float:
    n, p = self.marginal_parameters()
    if v >= 0:
      return logcomb(v + n - 1, v) + n * math.log(p) + v * math.log(1 - p)
    return -np.inf
  
  def mean(self) -> float:
    n, p = self.marginal_parameters()
    return n * (1 - p) / p
  
  def variance(self) -> float:
    n, p = self.marginal_parameters()
    return n * (1 - p) / (p ** 2)
  
  def rvs(self) -> List['RandomVar']:
    return self.n.rvs() + self.p.rvs()
  
@dataclass(frozen=True)
class Gamma(SymDistr[float]):
  a: 'SymExpr[float]'
  b: 'SymExpr[float]'

  def __str__(self):
    return f"Gamma({self.a}, {self.b})"
  
  # Parameters must be > 0
  def marginal_parameters(self) -> Tuple[float, float]:
    assert isinstance(self.a, Const)
    assert isinstance(self.b, Const)
    assert self.a.v > 0
    assert self.b.v > 0

    return self.a.v, self.b.v
  
  def draw(self, rng: np.random.Generator) -> float:
    a, b = self.marginal_parameters()
    return rng.gamma(a, 1 / b)
  
  def score(self, v: float) -> float:
    a, b = self.marginal_parameters()
    if v >= 0:
      return (a - 1) * math.log(v) - b * v - math.lgamma(a) + a * math.log(b)
    return -np.inf

  def mean(self) -> float:
    a, b = self.marginal_parameters()
    return a / b
  
  def variance(self) -> float:
    a, b = self.marginal_parameters()
    return a / (b ** 2)
  
  def rvs(self) -> List['RandomVar']:
    return self.a.rvs() + self.b.rvs()
  
@dataclass(frozen=True)
class Poisson(SymDistr[int]):
  l: 'SymExpr[float]'

  def __str__(self):
    return f"Poisson({self.l})"
  
  # Rate must be >= 0
  def marginal_parameters(self) -> float:
    assert isinstance(self.l, Const)
    assert self.l.v >= 0

    return self.l.v
  
  def draw(self, rng: np.random.Generator) -> int:
    l = self.marginal_parameters()
    return rng.poisson(l)
  
  def score(self, v: int) -> float:
    l = self.marginal_parameters()
    if v >= 0:
      return v * math.log(l) - l - math.lgamma(v + 1)
    return -np.inf
  
  def mean(self) -> float:
    l = self.marginal_parameters()
    return l
  
  def variance(self) -> float:
    l = self.marginal_parameters()
    return l
  
  def rvs(self) -> List['RandomVar']:
    return self.l.rvs()

@dataclass(frozen=True)
class StudentT(SymDistr[float]):
  mu: 'SymExpr[float]'
  tau2: 'SymExpr[float]'
  nu: 'SymExpr[float]'

  def __str__(self):
    return f"StudentT({self.mu}, {self.tau2}, {self.nu})"
  
  # Variance and degrees of freedom must be > 0
  def marginal_parameters(self) -> Tuple[float, float, float]:
    assert isinstance(self.mu, Const)
    assert isinstance(self.tau2, Const)
    assert isinstance(self.nu, Const)
    assert self.tau2.v > 0
    assert self.nu.v > 1

    return self.mu.v, self.tau2.v, self.nu.v
  
  def draw(self, rng: np.random.Generator) -> float:
    mu, tau2, nu = self.marginal_parameters()
    return rng.standard_t(nu) * np.sqrt(tau2) + mu
  
  def score(self, v: float) -> float:
    mu, tau2, nu = self.marginal_parameters()
    return math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu) - 0.5 * math.log(math.pi) - 0.5 * math.log(tau2) - 0.5 * (nu + 1) * math.log(1 + (v - mu) ** 2 / (nu * tau2))
  
  def mean(self) -> float:
    mu, _, _ = self.marginal_parameters()
    return mu
  
  def variance(self) -> float:
    _, tau2, nu = self.marginal_parameters()
    if nu > 2:
      return tau2 * nu / (nu - 2)
    return np.inf
  
  def rvs(self) -> List['RandomVar']:
    return self.mu.rvs() + self.tau2.rvs() + self.nu.rvs()
  
@dataclass(frozen=True)
class Categorical(SymDistr[int]):
  lower: 'SymExpr[int]'
  upper: 'SymExpr[int]'
  probs: 'SymExpr[List[float]]'

  def __str__(self):
    return f"Categorical({self.lower}, {self.upper}, {self.probs})"
  
  # Lower must be >= 0, upper must be >= lower, probabilities must sum to 1
  # Upper is inclusive
  def marginal_parameters(self) -> Tuple[int, int, List[float]]:
    assert isinstance(self.lower, Const)
    assert isinstance(self.upper, Const)
    assert isinstance(self.probs, Const)
    assert self.lower.v >= 0
    assert self.lower.v <= self.upper.v
    assert len(self.probs.v) == self.upper.v - self.lower.v + 1
    assert all(0 <= p <= 1 for p in self.probs.v)
    assert np.isclose(sum(self.probs.v), 1)

    return self.lower.v, self.upper.v, self.probs.v
  
  def draw(self, rng: np.random.Generator) -> int:
    lower, upper, probs = self.marginal_parameters()
    return rng.choice(range(lower, upper + 1), p=probs)
  
  def score(self, v: int) -> float:
    lower, _, probs = self.marginal_parameters()
    return math.log(probs[v - lower])
  
  def mean(self) -> float:
    lower, _, probs = self.marginal_parameters()
    return sum((i + lower) * p for i, p in enumerate(probs))
  
  def variance(self) -> float:
    lower, _, probs = self.marginal_parameters()
    return sum((i + lower) ** 2 * p for i, p in enumerate(probs)) - self.mean() ** 2
  
  def rvs(self) -> List['RandomVar']:
    return self.lower.rvs() + self.upper.rvs() + self.probs.rvs()
  
@dataclass(frozen=True)
class Delta(SymDistr[T]):
  v: 'SymExpr[T]'
  sampled: bool = False

  def __str__(self):
    return f"Delta({self.v}, {self.sampled})"
  
  def marginal_parameters(self) -> T:
    assert isinstance(self.v, Const)

    return self.v.v
  
  def draw(self, rng: np.random.Generator):
    inner_v = self.marginal_parameters()
    return inner_v
  
  def score(self, v: T) -> float:
    inner_v = self.marginal_parameters()
    return 0 if v == inner_v else -np.inf
  
  def mean(self) -> float:
    v = self.marginal_parameters()

    if isinstance(v, bool) or isinstance(v, Number):
      return float(v)
    else:
      raise ValueError(f"Delta distribution {str(self)} does not have a mean")
  
  def variance(self) -> float:
    return 0
  
  def rvs(self) -> List['RandomVar']:
    return self.v.rvs()

### Abstract Symbolic expressions ###
### All hybrid inference engines should be able to handle these ###
  
@dataclass(frozen=True)
class AbsSymExpr(Generic[T], Expr['AbsSymExpr']):
  def __str__(self):
    return "AbsSymExpr"
  
  def rvs(self) -> List['AbsRandomVar']:
    raise NotImplementedError()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsSymExpr':
    raise NotImplementedError()
  
  def depends_on(self, rv: 'AbsRandomVar', transitive: bool = False) -> bool:
    if transitive:
      return any((rv in rvs.rvs() for rvs in self.rvs()))
    else:
      return rv in self.rvs()
    
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    raise NotImplementedError()
  
@dataclass(frozen=True)
class TopE(AbsSymExpr[T]):
  def __str__(self):
    return "TopE"
  
  # TopE is the top element of the lattice, so it "references" all random variables
  # But this should be treated as a special case
  def rvs(self) -> List['AbsRandomVar']:
    return []
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'TopE':
    return self
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return self
  
  def depends_on(self, rv: 'AbsRandomVar', transitive: bool = False) -> bool:
    return True
  
# UnkE represents an unknown expression that depends on a set of random variables
# It's a refinement of TopE
@dataclass(frozen=True)
class UnkE(AbsSymExpr[T]):
  parents: List['AbsRandomVar']

  def __str__(self):
    return f"UnkE({self.rvs()})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.parents
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'UnkE':
    parents = []
    for p in self.parents:
      if p not in parents:
        parents.append(p)
    if old in parents:
      parents.remove(old)
      parents.append(new)
    return UnkE(parents)
    
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    new_parents = value.rvs()
    parents = []
    for p in self.parents:
      if p not in parents:
        parents.append(p)
    if rv in parents:
      parents.remove(rv)
    for p in new_parents:
      if p not in parents:
        parents.append(p)
    return UnkE(parents)

# UnkC represents an unknown constant
@dataclass(frozen=True)
class UnkC:
  def __str__(self):
    return "UnkC"
  
  def __add__(self, other):
    return UnkC()
  
  def __radd__(self, other):
    return UnkC()
  
  def __sub__(self, other):
    return UnkC()
  
  def __rsub__(self, other):
    return UnkC()
  
  def __mul__(self, other):
    return UnkC()
  
  def __rmul__(self, other):
    return UnkC()
  
  def __truediv__(self, other):
    return UnkC()
  
  def __rtruediv__(self, other):
    return UnkC()
  
  def __neg__(self):
    return UnkC()
    
@dataclass(frozen=True)
class AbsConst(AbsSymExpr[T]):
  v: T | UnkC

  def __str__(self):
    if self.v is None:
      return "()"
    elif self.v is True:
      return "true"
    elif self.v is False:
      return "false"
    elif isinstance(self.v, list):
      return f"[{', '.join(map(str, self.v))}]"
    elif isinstance(self.v, tuple):
      return f"({', '.join(map(str, self.v))})"
    return f"{self.v}"
  
  def rvs(self) -> List['AbsRandomVar']:
    return []
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsConst':
    return self
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return self
  
@dataclass(frozen=True)
class AbsRandomVar(AbsSymExpr[T]):
  rv: str

  def __str__(self):
    return f"RV({self.rv})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return [self]
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsRandomVar':
    if self == old:
      return new
    return self
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    if self == rv:
      return value
    return self
    
@dataclass(frozen=True)
class AbsAdd(AbsSymExpr[Number], Op[AbsSymExpr]):
  left: 'AbsSymExpr[Number]'
  right: 'AbsSymExpr[Number]'
    
  def __str__(self):
    return f"({self.left} + {self.right})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsAdd':
    return AbsAdd(self.left.rename(old, new), self.right.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsAdd(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsMul(AbsSymExpr[Number], Op[AbsSymExpr]):
  left: 'AbsSymExpr[Number]'
  right: 'AbsSymExpr[Number]'

  def __str__(self):
    return f"({self.left} * {self.right})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsMul':
    return AbsMul(self.left.rename(old, new), self.right.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsMul(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsDiv(AbsSymExpr[Number], Op[AbsSymExpr]):
  left: 'AbsSymExpr[Number]'
  right: 'AbsSymExpr[Number]'

  def __str__(self):
    return f"({self.left} / {self.right})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsDiv':
    return AbsDiv(self.left.rename(old, new), self.right.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsDiv(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsIte(AbsSymExpr[T], Op[AbsSymExpr]):
  cond: 'AbsSymExpr[bool]'
  true: 'AbsSymExpr[T]'
  false: 'AbsSymExpr[T]'

  def __str__(self):
    return f"if {self.cond} then {self.true} else {self.false}"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.cond.rvs() + self.true.rvs() + self.false.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsIte':
    return AbsIte(self.cond.rename(old, new), self.true.rename(old, new), self.false.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsIte(self.cond.subst_rv(rv, value), self.true.subst_rv(rv, value), self.false.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsEq(AbsSymExpr[bool], Op[AbsSymExpr]):
  left: 'AbsSymExpr'
  right: 'AbsSymExpr'

  def __str__(self):
    return f"({self.left} == {self.right})"
    
  def rvs(self) -> List['AbsRandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsSymExpr':
    return AbsEq(self.left.rename(old, new), self.right.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsEq(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsLt(AbsSymExpr[bool], Op[AbsSymExpr]):
  left: 'AbsSymExpr[Number]'
  right: 'AbsSymExpr[Number]'

  def __str__(self):
    return f"({self.left} < {self.right})"
    
  def rvs(self) -> List['AbsRandomVar']:
    return self.left.rvs() + self.right.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsLt':
    return AbsLt(self.left.rename(old, new), self.right.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsLt(self.left.subst_rv(rv, value), self.right.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsPair(AbsSymExpr[T]):
  fst: 'AbsSymExpr[T]'
  snd: 'AbsSymExpr[T]'

  def __str__(self):
    return f"({self.fst}, {self.snd})"
    
  def rvs(self) -> List['AbsRandomVar']:
    return self.fst.rvs() + self.snd.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsPair':
    return AbsPair(self.fst.rename(old, new), self.snd.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsPair(self.fst.subst_rv(rv, value), self.snd.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsLst(AbsSymExpr[T]):
  exprs: List['AbsSymExpr[T]']

  def __str__(self):
    return f"[{'; '.join(map(str, self.exprs))}]"
  
  def rvs(self) -> List['AbsRandomVar']:
    return sum([e.rvs() for e in self.exprs], [])
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsLst':
    return AbsLst([e.rename(old, new) for e in self.exprs])
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymExpr':
    return AbsLst([e.subst_rv(rv, value) for e in self.exprs])

@dataclass(frozen=True)
class AbsSymDistr(AbsSymExpr[T], Op[AbsSymExpr]):
  def __str__(self):
    return "AbsSymDistr"
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsSymDistr':
    raise NotImplementedError()
  
  def subst_rv(self, rv: AbsRandomVar, value: AbsSymExpr) -> 'AbsSymDistr':
    raise NotImplementedError()
  
# Analogous to TopE
@dataclass(frozen=True)
class TopD(AbsSymDistr[T]):
  def __str__(self):
    return "TopD"
  
  def rvs(self) -> List['AbsRandomVar']:
    return []
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'TopD':
    return self
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsSymDistr':
    return self

# Analogous to UnkE
@dataclass(frozen=True)
class UnkD(AbsSymDistr[T]):
  parents: List[AbsRandomVar]

  def __str__(self):
    return f"UnkD({self.parents})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.parents
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'UnkD':
    parents = []
    for p in self.parents:
      if p not in parents:
        parents.append(p)
    if old in parents:
      parents.remove(old)
      parents.append(new)
    return UnkD(parents)
  
  def subst_rv(self, rv: AbsRandomVar, value: AbsSymExpr) -> AbsSymDistr:
    new_parents = value.rvs()
    parents = []
    for p in self.parents:
      if p not in parents:
        parents.append(p)
    if rv in parents:
      parents.remove(rv)
    parents.extend(new_parents)
    return UnkD(parents)

@dataclass(frozen=True)
class AbsNormal(AbsSymDistr[float]):
  mu: 'AbsSymExpr[float]'
  var: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Normal({self.mu}, {self.var})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.mu.rvs() + self.var.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsNormal':
    return AbsNormal(self.mu.rename(old, new), self.var.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsNormal':
    return AbsNormal(self.mu.subst_rv(rv, value), self.var.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsBernoulli(AbsSymDistr[bool]):
  p: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Bernoulli({self.p})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.p.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsBernoulli':
    return AbsBernoulli(self.p.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsBernoulli':
    return AbsBernoulli(self.p.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsBeta(AbsSymDistr[float]):
  a: 'AbsSymExpr[float]'
  b: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Beta({self.a}, {self.b})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.a.rvs() + self.b.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsBeta':
    return AbsBeta(self.a.rename(old, new), self.b.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsBeta':
    return AbsBeta(self.a.subst_rv(rv, value), self.b.subst_rv(rv, value))
    
@dataclass(frozen=True)
class AbsBinomial(AbsSymDistr[int]):
  n: 'AbsSymExpr[int]'
  p: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Binomial({self.n}, {self.p})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.n.rvs() + self.p.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsBinomial':
    return AbsBinomial(self.n.rename(old, new), self.p.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsBinomial':
    return AbsBinomial(self.n.subst_rv(rv, value), self.p.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsBetaBinomial(AbsSymDistr[int]):
  n: 'AbsSymExpr[int]'
  a: 'AbsSymExpr[float]'
  b: 'AbsSymExpr[float]'

  def __str__(self):
    return f"BetaBinomial({self.n}, {self.a}, {self.b})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.n.rvs() + self.a.rvs() + self.b.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsBetaBinomial':
    return AbsBetaBinomial(self.n.rename(old, new), self.a.rename(old, new), self.b.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsBetaBinomial':
    return AbsBetaBinomial(self.n.subst_rv(rv, value), self.a.subst_rv(rv, value), self.b.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsNegativeBinomial(AbsSymDistr[int]):
  k: 'AbsSymExpr[int]'
  p: 'AbsSymExpr[float]'

  def __str__(self):
    return f"NegativeBinomial({self.k}, {self.p})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.k.rvs() + self.p.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsNegativeBinomial':
    return AbsNegativeBinomial(self.k.rename(old, new), self.p.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsNegativeBinomial':
    return AbsNegativeBinomial(self.k.subst_rv(rv, value), self.p.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsGamma(AbsSymDistr[float]):
  a: 'AbsSymExpr[float]'
  b: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Gamma({self.a}, {self.b})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.a.rvs() + self.b.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsGamma':
    return AbsGamma(self.a.rename(old, new), self.b.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsGamma':
    return AbsGamma(self.a.subst_rv(rv, value), self.b.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsPoisson(AbsSymDistr[int]):
  l: 'AbsSymExpr[float]'

  def __str__(self):
    return f"Poisson({self.l})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.l.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsPoisson':
    return AbsPoisson(self.l.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsPoisson':
    return AbsPoisson(self.l.subst_rv(rv, value))

@dataclass(frozen=True)
class AbsStudentT(AbsSymDistr[float]):
  mu: 'AbsSymExpr[float]'
  tau2: 'AbsSymExpr[float]'
  nu: 'AbsSymExpr[float]'

  def __str__(self):
    return f"StudentT({self.mu}, {self.tau2}, {self.nu})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.mu.rvs() + self.tau2.rvs() + self.nu.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsStudentT':
    return AbsStudentT(self.mu.rename(old, new), self.tau2.rename(old, new), self.nu.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsStudentT':
    return AbsStudentT(self.mu.subst_rv(rv, value), self.tau2.subst_rv(rv, value), self.nu.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsCategorical(AbsSymDistr[int]):
  lower: 'AbsSymExpr[int]'
  upper: 'AbsSymExpr[int]'
  probs: 'AbsSymExpr[List[float]]'

  def __str__(self):
    return f"Categorical({self.lower}, {self.upper}, {self.probs})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.lower.rvs() + self.upper.rvs() + self.probs.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsCategorical':
    return AbsCategorical(self.lower.rename(old, new), self.upper.rename(old, new), self.probs.rename(old, new))
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsCategorical':
    return AbsCategorical(self.lower.subst_rv(rv, value), self.upper.subst_rv(rv, value), self.probs.subst_rv(rv, value))
  
@dataclass(frozen=True)
class AbsDelta(AbsSymDistr[T]):
  v: 'AbsSymExpr[T]'
  sampled: bool = False

  def __str__(self):
    return f"Delta({self.v}, {self.sampled})"
  
  def rvs(self) -> List['AbsRandomVar']:
    return self.v.rvs()
  
  def rename(self, old: 'AbsRandomVar', new: 'AbsRandomVar') -> 'AbsDelta':
    return AbsDelta(self.v.rename(old, new), self.sampled)
  
  def subst_rv(self, rv: 'AbsRandomVar', value: 'AbsSymExpr') -> 'AbsDelta':
    return AbsDelta(self.v.subst_rv(rv, value), self.sampled)