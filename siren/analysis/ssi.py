from typing import Tuple
import numpy as np
from copy import copy
from collections import deque

from siren.analysis.interface import *
from siren.inference_plan import DistrEnc
import siren.analysis.conjugate as conj

class NonConjugate(Exception):
  def __init__(self, message, rv_nc: AbsRandomVar):
    super().__init__(message)

    self.rv_nc = rv_nc

class AbsSSIState(AbsSymState):
  def __str__(self):
    s = '\n\t'.join(map(str, self.state.items()))
    return f"AbsSSIState(\n\t{s}\n)" if s else "AbsSSIState()"
  
  # Returns reachable random variables from the given set of random variables
  # Used for determining which random variables to consider during fixpoint
  def entry_referenced_rvs(self, rvs: Set[AbsRandomVar]) -> Set[AbsRandomVar]:
    # Don't need to remember the Deltas because always eval before anything

    referenced_rvs = set()

    for rv in rvs:
      parents = self.distr(rv).rvs()
      for rv_par in parents:
        if rv_par in referenced_rvs:
          continue
        else:
          d = self.distr(rv_par)
          if isinstance(d, AbsDelta):
            for rv2 in self.vars():
              self.set_distr(rv2, self.distr(rv2).subst_rv(rv_par, d.v))
            
            for x, e in self.ctx.context.items():
              self.ctx.context[x] = e.subst_rv(rv_par, d.v)
          else:
            referenced_rvs.add(rv_par)

    return referenced_rvs

  ### Symbolic Interface ###
  def assume(self, rv: AbsRandomVar, name: Optional[Identifier], annotation: Optional[Annotation], distribution: AbsSymDistr[T]) -> AbsRandomVar[T]:
    pv = {name} if name is not None else set()
    if annotation is not None:
      if name is None:
        raise ValueError('Cannot annotate anonymous variable')
    self.set_annotation(rv, annotation)

    self.set_pv(rv, pv)
    self.set_distr(rv, distribution)
    return rv

  def observe(self, rv: AbsRandomVar[T], value: AbsConst[T]) -> None:
    def _observe() -> None:
      try:
        self.hoist_and_eval(rv)
        return
      except NonConjugate as e:
        self.value(e.rv_nc)
        _observe()
      
    _observe()
    self.intervene(rv, AbsDelta(value, sampled=False))
    return

  def inner_value(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
    def _value() -> None:
      try:
        self.hoist_and_eval(rv)
        return 
      except NonConjugate as e:
        self.value(e.rv_nc)
        return _value()

    _value()
    self.intervene(rv, AbsDelta(AbsConst(UnkC()), sampled=True))
    
    return AbsConst(UnkC())
  
  def marginalize(self, rv: AbsRandomVar) -> None:
    try:
      self.hoist_and_eval(rv)
    except NonConjugate as e:
      self.value(e.rv_nc)
      self.marginalize(rv)
  
  ########################################################################

  def parents(self, rv: AbsRandomVar) -> List[AbsRandomVar]:
    return self.distr(rv).rvs()

  def intervene(self, rv: AbsRandomVar[T], v: AbsDelta[T]) -> None:
    self.set_distr(rv, v)

  def rv_depends_on_transitive(self, expr: AbsRandomVar, rv: AbsRandomVar) -> bool:
    match self.distr(expr):
      case AbsNormal(mu, var):
        return self.depends_on(mu, rv, True) or self.depends_on(var, rv, True)
      case AbsBernoulli(p):
        return self.depends_on(p, rv, True)
      case AbsBeta(a, b):
        return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
      case AbsBinomial(n, p):
        return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
      case AbsBetaBinomial(n, a, b):
        return self.depends_on(n, rv, True) or self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
      case AbsNegativeBinomial(n, p):
        return self.depends_on(n, rv, True) or self.depends_on(p, rv, True)
      case AbsGamma(a, b):
        return self.depends_on(a, rv, True) or self.depends_on(b, rv, True)
      case AbsPoisson(l):
        return self.depends_on(l, rv, True)
      case AbsStudentT(mu, tau2, nu):
        return self.depends_on(mu, rv, True) or self.depends_on(tau2, rv, True) or self.depends_on(nu, rv, True)
      case AbsCategorical(lower, upper, probs):
        return self.depends_on(lower, rv, True) or self.depends_on(upper, rv, True) or self.depends_on(probs, rv, True)
      case AbsDelta(v, _):
        return self.depends_on(v, rv, True)
      case UnkD(parents):
        return (rv in parents) or any(self.depends_on(p, rv, True) for p in parents)
      case TopD():
        return True
      case _:
        raise ValueError(self.distr(expr))

  def depends_on(self, expr: AbsSymExpr, rv: AbsRandomVar, transitive: bool) -> bool:
    match expr:
      case AbsConst(_):
        return False
      case AbsRandomVar(_):
        if expr == rv:
          return True
        else:
          if not transitive:
            return False
          else:
            return self.rv_depends_on_transitive(expr, rv)
      case AbsAdd(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case AbsMul(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case AbsDiv(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case AbsIte(cond, true, false):
        return self.depends_on(cond, rv, transitive) or self.depends_on(true, rv, transitive) or self.depends_on(false, rv, transitive)
      case AbsEq(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case AbsLt(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case AbsLst(es):
        return any(self.depends_on(e, rv, transitive) for e in es)
      case AbsPair(e1, e2):
        return self.depends_on(e1, rv, transitive) or self.depends_on(e2, rv, transitive)
      case UnkE(parents):
        return (rv in parents) or any(self.depends_on(p, rv, transitive) for p in parents)
      case TopE():
        return True
      case _:
        raise ValueError(expr)

  def hoist(self, rv: AbsRandomVar) -> None:
    def _topo_sort(rvs: List[AbsRandomVar]) -> List[AbsRandomVar]:
      sorted_nodes = []

      def _visit(rv: AbsRandomVar) -> None:
        try:
          parents = self.parents(rv)
        except RecursionError:
          raise RecursionError(rv)

        for parent in parents:
          _visit(parent)

        if rv not in sorted_nodes:
          sorted_nodes.append(rv)

      for rv in rvs:
        _visit(rv)

      nodes = []
      for node in sorted_nodes:
        if node in rvs:
          nodes.append(node)
      
      return nodes

    def _can_swap(rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
      def _has_other_deps_on_par(expr: AbsSymExpr) -> bool:
        match expr:
          case AbsConst(_):
            return False
          case AbsRandomVar(_):
            if expr == rv_par:
              return False
            else:
              return self.rv_depends_on_transitive(expr, rv_par)
          case AbsAdd(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case AbsMul(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case AbsDiv(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case AbsIte(cond, true, false):
            return _has_other_deps_on_par(cond) or _has_other_deps_on_par(true) or _has_other_deps_on_par(false)
          case AbsEq(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case AbsLt(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case AbsLst(es):
            return any(_has_other_deps_on_par(e) for e in es)
          case AbsPair(e1, e2):
            return _has_other_deps_on_par(e1) or _has_other_deps_on_par(e2)
          case UnkE(parents):
            return any(self.rv_depends_on_transitive(p, rv_par) for p in parents)
          case TopE():
            return True
          case _:
            raise ValueError(expr)

      match self.distr(rv_child):
        case AbsNormal(mu, var):
          return (self.depends_on(mu, rv_par, False) or self.depends_on(var, rv_par, False)) \
            and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(var))
        case AbsBernoulli(p):
          return self.depends_on(p, rv_par, False) and (not _has_other_deps_on_par(p))
        case AbsBeta(a, b):
          return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
            and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
        case AbsBinomial(n, p):
          return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
            and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
        case AbsBetaBinomial(n, a, b):
          return (self.depends_on(n, rv_par, False) or self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
            and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
        case AbsNegativeBinomial(n, p):
          return (self.depends_on(n, rv_par, False) or self.depends_on(p, rv_par, False)) \
            and (not _has_other_deps_on_par(n)) and (not _has_other_deps_on_par(p))
        case AbsGamma(a, b):
          return (self.depends_on(a, rv_par, False) or self.depends_on(b, rv_par, False)) \
            and (not _has_other_deps_on_par(a)) and (not _has_other_deps_on_par(b))
        case AbsPoisson(l):
          return self.depends_on(l, rv_par, False) and (not _has_other_deps_on_par(l))
        case AbsStudentT(mu, tau2, nu):
          return (self.depends_on(mu, rv_par, False) or self.depends_on(tau2, rv_par, False) or self.depends_on(nu, rv_par, False)) \
            and (not _has_other_deps_on_par(mu)) and (not _has_other_deps_on_par(tau2)) and (not _has_other_deps_on_par(nu))
        case AbsCategorical(lower, upper, probs):
          return (self.depends_on(lower, rv_par, False) or self.depends_on(upper, rv_par, False) or self.depends_on(probs, rv_par, False)) \
            and (not _has_other_deps_on_par(lower)) and (not _has_other_deps_on_par(upper)) and (not _has_other_deps_on_par(probs))
        case AbsDelta(v, _):
          return self.depends_on(v, rv_par, False) and (not _has_other_deps_on_par(v))
        case UnkD(parents):
          return not (rv_par in parents)
        case TopD():
          return False
        case _:
          raise ValueError(self.distr(rv_child))

    def _swap(rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
      def _update(marginal_posterior: Optional[Tuple[AbsSymDistr, AbsSymDistr]]) -> bool:
        if marginal_posterior is None:
          return False
        
        marginal, posterior = marginal_posterior
        self.set_distr(rv_par, posterior)
        self.set_distr(rv_child, marginal)
        return True

      match self.distr(rv_par), self.distr(rv_child):
        case AbsNormal(_), AbsNormal(_):
          if _update(conj.gaussian_conjugate(self, rv_par, rv_child)):
            return True
          else:
            return _update(conj.normal_inverse_gamma_normal_conjugate(self, rv_par, rv_child))
        case AbsBernoulli(_), AbsBernoulli(_):
          return _update(conj.bernoulli_conjugate(self, rv_par, rv_child))
        case AbsBeta(_), AbsBernoulli(_):
          return _update(conj.beta_bernoulli_conjugate(self, rv_par, rv_child))
        case AbsBeta(_), AbsBinomial(_):
          return _update(conj.beta_binomial_conjugate(self, rv_par, rv_child))
        case AbsGamma(_), AbsPoisson(_):
          return _update(conj.gamma_poisson_conjugate(self, rv_par, rv_child))
        case AbsGamma(_), AbsNormal(_):
          return _update(conj.gamma_normal_conjugate(self, rv_par, rv_child))
        case UnkD(_), _:
          self.set_dynamic(rv_par)
          return False
        case TopD(), _:
          self.set_dynamic(rv_par)
          return False
        case _:
          return False
    
    # Detects if the distribution contains a TopD or TopE
    def _contains_top(rv: AbsRandomVar) -> bool:
      def _contains_top_expr(expr: AbsSymExpr) -> bool:
        match expr:
          case AbsConst(_):
            return False
          case AbsRandomVar(_):
            return False
          case AbsAdd(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case AbsMul(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case AbsDiv(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case AbsIte(cond, true, false):
            return _contains_top_expr(cond) or _contains_top_expr(true) or _contains_top_expr(false)
          case AbsEq(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case AbsLt(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case AbsLst(es):
            return any(_contains_top_expr(e) for e in es)
          case AbsPair(e1, e2):
            return _contains_top_expr(e1) or _contains_top_expr(e2)
          case UnkE(parents):
            return any(_contains_top(rv_par) for rv_par in parents)
          case TopE():
            return True
          case _:
            raise ValueError(expr)
      match self.distr(rv):
        case TopD():
          return True
        case AbsNormal(mu, var):
          return _contains_top_expr(mu) or _contains_top_expr(var)
        case AbsBernoulli(p):
          return _contains_top_expr(p)
        case AbsBeta(a, b):
          return _contains_top_expr(a) or _contains_top_expr(b)
        case AbsBinomial(n, p):
          return _contains_top_expr(n) or _contains_top_expr(p)
        case AbsBetaBinomial(n, a, b):
          return _contains_top_expr(n) or _contains_top_expr(a) or _contains_top_expr(b)
        case AbsNegativeBinomial(n, p):
          return _contains_top_expr(n) or _contains_top_expr(p)
        case AbsGamma(a, b):
          return _contains_top_expr(a) or _contains_top_expr(b)
        case AbsPoisson(l):
          return _contains_top_expr(l)
        case AbsStudentT(mu, tau2, nu):
          return _contains_top_expr(mu) or _contains_top_expr(tau2) or _contains_top_expr(nu)
        case AbsCategorical(lower, upper, probs):
          return _contains_top_expr(lower) or _contains_top_expr(upper) or _contains_top_expr(probs)
        case AbsDelta(v, _):
          return _contains_top_expr(v)
        case UnkD(parents):
          return any(_contains_top(rv_par) for rv_par in parents)
        case _:
          raise ValueError(self.distr(rv))
        
    def _contains_cycle(rv: AbsRandomVar) -> bool:
      visited = set()
      stack = []

      def dfs(rv_cur):
        if rv_cur not in visited:
          visited.add(rv_cur)
          stack.append(rv_cur)

          parents = []
          for par in self.parents(rv_cur):
            if not par in parents:
              parents.append(par)

          for par in parents:
            if par not in visited and dfs(par):
              return True
            elif par in stack:
              return True
          stack.remove(rv_cur)
        return False
      
      return dfs(rv)
        
    def _hoist_inner(rv_cur: AbsRandomVar, ghost_roots: Set[AbsRandomVar]) -> None:
      # Joins can cause cycles so we need to detect them
      # Before we do topological sort
      if _contains_cycle(rv_cur):
        # print(rv_cur)
        # print(self.state)
        self.set_dynamic(rv_cur)
        self.set_distr(rv_cur, TopD())
        return

      # Hoist parents
      parents = _topo_sort(self.parents(rv_cur))
      ghost_roots1 = copy(ghost_roots)
      for rv_par in parents:
        if rv_par not in ghost_roots1:
          _hoist_inner(rv_par, ghost_roots1)
        ghost_roots1.add(rv_par)

      # Hoist current node
      for rv_par in parents[::-1]:
        if rv_par not in ghost_roots:
          # Circular dependencies caused by joins means we don't know
          # what to do, so just set to Top
          # which will cause analysis to conservatively raise exception
          # if variable is annotated
          if not _can_swap(rv_par, rv_cur):
            self.set_dynamic(rv_cur)
            self.set_distr(rv_cur, TopD())
            self.set_distr(rv_par, TopD())
            # raise ValueError(f'Cannot swap {rv_par} and {rv_cur}')
            continue
          
          if not _swap(rv_par, rv_cur):
            raise NonConjugate(f'Nonconjugate {rv_par} and {rv_cur}', rv_par)

    _hoist_inner(rv, set())

  def hoist_and_eval(self, rv: AbsRandomVar) -> None:
    self.set_distr(rv, self.eval_distr(self.distr(rv)))
    self.hoist(rv)
    self.set_distr(rv, self.eval_distr(self.distr(rv)))

  # Sets pvs of rv to be dynamic, and recursively sets pvs of parents to be
  # dynamic
  def set_dynamic(self, rv: AbsRandomVar) -> None:
    super().set_dynamic(rv)
    
    for rv_par in self.parents(rv):
      self.set_dynamic(rv_par)

