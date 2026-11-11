from typing import Any, Set
from copy import copy
from collections import deque

import siren.analysis.conjugate as conj
from siren.analysis.interface import *

@dataclass(frozen=True)
class AbsBPNode():
  pass

@dataclass(frozen=True)
class AbsBPRealized(AbsBPNode):
  def __str__(self) -> str:
    return 'AbsBPRealized'

@dataclass(frozen=True)
class AbsBPMarginalized(AbsBPNode):
  def __str__(self):
    return f'AbsBPMarginalized()'

@dataclass(frozen=True)
class AbsBPInitialized(AbsBPNode):
  parent: AbsRandomVar

  def __str__(self):
    return f'AbsBPInitialized({self.parent})'
  
@dataclass(frozen=True)
class AbsBPUnk(AbsBPNode):
  def __str__(self):
    return f'AbsBPUnk()'

class AbsBPState(AbsSymState):
  ###
  # State entry:
  #   rv: (pv, distribution, node)
  ###
  def __str__(self):
    s = '\n\t'.join(map(str, self.state.items()))
    return f"AbsBPState(\n\t{s}\n)" if s else "AbsBPState()"
  
  ## Accessors
  def node(self, rv: AbsRandomVar) -> AbsBPNode:
    return self.get_entry(rv, 'node')

  ## Mutators
  def set_node(self, rv: AbsRandomVar, node: AbsBPNode) -> None:
    self.set_entry(rv, node=node)

  # Get reachable random variables from rvs
  def entry_referenced_rvs(self, rvs: Set[AbsRandomVar]) -> Set[AbsRandomVar]:
    ref_rvs = super().entry_referenced_rvs(rvs)

    for rv in rvs:
      match self.node(rv):
        case AbsBPInitialized(rv_par):
          ref_rvs.add(rv_par)

    return ref_rvs
  
  # Also check parent to rename  
  def entry_rename_rv(self, rv: AbsRandomVar, old: AbsRandomVar, new: AbsRandomVar) -> None:
    super().entry_rename_rv(rv, old, new)
    match self.node(rv):
      case AbsBPInitialized(rv_par):
        self.set_node(rv, AbsBPInitialized(new if rv_par == old else rv_par))

  # Join nodes by checking node type. If they don't match or the edges don't match
  # then make the node top
  def entry_join(self, rv: AbsRandomVar, other: 'AbsBPState') -> None:
    def _make_unk(
      parents1: Set[AbsRandomVar],
      parents2: Set[AbsRandomVar],
    ) -> None:
      for rv_par in parents1:
        self.set_dynamic(rv_par)
      for rv_par in parents2:
        other.set_dynamic(rv_par)
      self.set_distr(rv, TopD())
      self.set_node(rv, AbsBPUnk())

    match self.node(rv), other.node(rv):
      case AbsBPRealized(), AbsBPRealized():
        super().entry_join(rv, other)
      case AbsBPMarginalized(), AbsBPMarginalized():
        super().entry_join(rv, other)
      case AbsBPInitialized(self_par), AbsBPInitialized(other_par):
        cdistr = self.join_distr(self.distr(rv), other.distr(rv), other)

        if isinstance(cdistr, TopD):
          _make_unk(set(), set())
          return

        parents = []
        for rv_par in cdistr.rvs():
          if rv_par not in parents:
            parents.append(rv_par)

        if len(parents) == 0:
          # Parent is actually realized
          if self_par == other_par:
            self.set_distr(rv, cdistr)
            self.set_pv(rv, self.pv(rv) | other.pv(rv))
            self.set_node(rv, AbsBPInitialized(self_par))
          else:
            raise ValueError(f'{rv} is {self.node(rv)}')
        elif len(parents) == 1:
          self.set_distr(rv, cdistr)
          self.set_pv(rv, self.pv(rv) | other.pv(rv))
          self.set_node(rv, AbsBPInitialized(parents[0]))
        else:
          _make_unk(self.distr(rv).rvs(), other.distr(rv).rvs())
      case _, _:
        super().entry_join(rv, other)

        parents1 = set()
        match self.node(rv):
          case AbsBPRealized():
            parents1 = set()
          case AbsBPMarginalized():
            parents1 = set()
          case AbsBPInitialized(rv_par):
            parents1 = {rv_par}
          case AbsBPUnk():
            parents1 = set()
          case _:
            raise ValueError(f'{rv} is {self.node(rv)}')
        
        parents2 = set()
        match other.node(rv):
          case AbsBPRealized():
            parents2 = set()
          case AbsBPMarginalized():
            parents2 = set()
          case AbsBPInitialized(rv_par):
            parents2 = {rv_par}
          case AbsBPUnk():
            parents2 = set()
          case _:
            raise ValueError(f'{rv} is {self.node(rv)}')

        _make_unk(parents1, parents2)

  ### Symbolic Interface ###
  def assume(self, rv:AbsRandomVar, name: Optional[Identifier], annotation: Optional[Annotation], distribution: AbsSymDistr[T]) -> AbsRandomVar[T]:
    def _check_conjugacy(prior : AbsSymDistr, likelihood : AbsSymDistr, rv_par : AbsRandomVar, rv_child : AbsRandomVar) -> bool:
      match prior, likelihood:
        case AbsNormal(_), AbsNormal(_):
          return conj.gaussian_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case _:
          return False

    if annotation is not None:
      if name is None:
        raise ValueError('Cannot annotate anonymous variable')
    distribution = self.eval_distr(distribution)

    if len(distribution.rvs()) == 0:
      node = AbsBPMarginalized()
    else:
      parents = []
      for rv_par in distribution.rvs():
        if rv_par not in parents:
          parents.append(rv_par)

      # If any parent is BPUnk,
      # then rv and all ancestors are BPUnk
      # Because we don't know which parent is the canonical parent
      if any([isinstance(self.node(rv_par), AbsBPUnk) for rv_par in parents]):
        pv = {name} if name is not None else set()
        self.set_pv(rv, pv)
        # UnkD because we don't know which is the canonical parent
        self.set_distr(rv, UnkD(parents))
        self.set_annotation(rv, annotation)

        for rv_par in parents:
          if not isinstance(self.node(rv_par), AbsBPRealized):
            self.set_dynamic(rv_par)

        self.set_node(rv, AbsBPUnk())
        return rv

      # keep if conjugate, else sample it
      canonical_parent = None
      has_parent = False
      for rv_par in parents:
        if rv_par not in distribution.rvs():
          continue
        if not has_parent:
          match self.node(rv_par):
            case AbsBPRealized():
              distribution = self.eval_distr(distribution)
              continue
            case AbsBPUnk():
              raise ValueError(f'{rv_par} is {self.node(rv_par)}')
            case _: # BPInitialized or BPMarginalized
              if _check_conjugacy(self.distr(rv_par), distribution, rv_par, rv):
                canonical_parent = rv_par
                has_parent = True
                continue
        self.value(rv_par)
        distribution = self.eval_distr(distribution)

        if any([isinstance(self.node(rv_par), AbsBPUnk) for rv_par in parents]):
          pv = {name} if name is not None else set()
          self.set_pv(rv, pv)
          # UnkD because we don't know which is the canonical parent
          self.set_distr(rv, UnkD(parents))

          for rv_par in parents:
            if not isinstance(self.node(rv_par), AbsBPRealized):
              self.set_dynamic(rv_par)

          self.set_node(rv, AbsBPUnk())
          self.set_annotation(rv, annotation)
          return rv

      # all parents were sampled
      if len(distribution.rvs()) == 0:
        node = AbsBPMarginalized()
      else:
        assert canonical_parent is not None
        node = AbsBPInitialized(canonical_parent)

    pv = {name} if name is not None else set()
    self.set_pv(rv, pv)
    self.set_distr(rv, distribution)
    self.set_node(rv, node)
    self.set_annotation(rv, annotation)

    return rv

  def observe(self, rv: AbsRandomVar[T], value: AbsConst[T]) -> None:
    match self.node(rv):
      case AbsBPRealized():
        raise ValueError(f'Cannot observe {rv} twice')
      case AbsBPMarginalized():
        self.intervene(rv, AbsDelta(value, sampled=False))
      case AbsBPInitialized(rv_par):
        assert rv_par is not None
        self.marginalize(rv_par)

        if self.condition_cd(rv_par, rv):
          self.intervene(rv, AbsDelta(value, sampled=False))
          self.set_distr(rv_par, self.eval_distr(self.distr(rv_par)))
          self.set_node(rv_par, AbsBPMarginalized())
        else:
          self.value(rv_par)
          self.observe(rv, value)
      case AbsBPUnk():
        self.intervene(rv, AbsDelta(value, sampled=False))
          
  def inner_value(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
    self.marginalize(rv)
    match self.node(rv):
      case AbsBPInitialized(_):
        raise ValueError(f'{rv} is {self.node(rv)}')
      case _: # AbsBPRealized or AbsBPMarginalized or AbsBPUnk
        self.intervene(rv, AbsDelta(AbsConst(UnkC()), sampled=True))
        return AbsConst(UnkC())
  
  # make rv a root
  def marginalize(self, rv: AbsRandomVar) -> None:
    match self.node(rv):
      case AbsBPInitialized(rv_par):
        self.marginalize(rv_par)

        if self.condition_cd(rv_par, rv):
          return
        else:
          self.value(rv_par)
          self.marginalize(rv)

  ########################################################################

  def intervene(self, rv: AbsRandomVar[T], v: AbsDelta[T]) -> None:
    self.set_node(rv, AbsBPRealized())
    self.set_distr(rv, v)

  def condition_cd(self, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
    def _update(marginal_posterior: Optional[Tuple[AbsSymDistr, AbsSymDistr]]) -> bool:
      if marginal_posterior is None:
        return False
      
      marginal, posterior = marginal_posterior
      self.set_distr(rv_par, posterior)
      self.set_distr(rv_child, marginal)

      self.set_node(rv_child, AbsBPMarginalized())

      match self.node(rv_par):
        case AbsBPRealized():
          pass
        case AbsBPUnk():
          raise ValueError(f'{rv_par} is {self.node(rv_par)}')
        case _: # AbsBPInitialized or AbsBPMarginalized
          self.set_node(rv_par, AbsBPInitialized(rv_child))

      return True

    match self.distr(rv_par), self.distr(rv_child):
      case AbsDelta(v), cdistr:
        return _update((AbsDelta(v), self.eval_distr(cdistr)))
      case AbsNormal(_), AbsNormal(_):
        return _update(conj.gaussian_conjugate(self, rv_par, rv_child))
      case UnkD(_), _:
        self.set_dynamic(rv_par)
        return False
      case TopD(), _:
        self.set_dynamic(rv_par)
        return False
      case _:
        return False
      
  # Makes rv BPUnk and deal with side effects 
  def set_dynamic(self, rv: AbsRandomVar) -> None:
    super_set_dynamic = super().set_dynamic

    nodes = set()

    def _set_unk_node(rv: AbsRandomVar) -> None:
      if rv in nodes:
        return

      match self.node(rv):
        case AbsBPRealized():
          pass
        case AbsBPMarginalized():
          pass
        case AbsBPInitialized(rv_par):
          _set_unk_node(rv_par)
        case AbsBPUnk():
          pass
        case _:
          raise ValueError(f'{rv} is {self.node(rv)}')

      nodes.add(rv)

    _set_unk_node(rv)

    for rv in nodes:
      super_set_dynamic(rv)
      self.set_node(rv, AbsBPUnk())
      self.set_distr(rv, TopD())
