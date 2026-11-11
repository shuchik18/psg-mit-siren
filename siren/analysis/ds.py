from typing import Any, Set
from copy import copy
from siren.grammar import AbsRandomVar

import siren.analysis.conjugate as conj
from siren.analysis.interface import *

@dataclass(frozen=True)
class AbsDSNode():
  pass

# Invariant: All children are also AbsDSUnk
# Top node 
@dataclass(frozen=True)
class AbsDSUnk(AbsDSNode):
  def __str__(self) -> str:
    return 'AbsDSUnk'

@dataclass(frozen=True)
class AbsDSRealized(AbsDSNode):
  def __str__(self) -> str:
    return 'AbsDSRealized'

@dataclass(frozen=True)
class AbsDSMarginalized(AbsDSNode):
  edge: Optional[Tuple[AbsRandomVar, AbsSymDistr]]

  def __str__(self):
    return f'AbsDSMarginalized({self.edge})'

@dataclass(frozen=True)
class AbsDSInitialized(AbsDSNode):
  edge: Tuple[AbsRandomVar, AbsSymDistr]

  def __str__(self):
    return f'AbsDSInitialized({self.edge})'


class AbsDSState(AbsSymState):
  ###
  # State entry:
  #   rv: (pv, distribution, type, children, parent, cdistr)
  ###

  def __str__(self):
    s = '\n\t'.join(map(str, self.state.items()))
    return f"AbsDSState(\n\t{s}\n)" if s else "AbsDSState()"
  
  ## Accessors
  def children(self, rv: AbsRandomVar) -> List[AbsRandomVar]:
    return self.get_entry(rv, 'children')
  
  def node(self, rv: AbsRandomVar) -> AbsDSNode:
    return self.get_entry(rv, 'node')

  ## Mutators
  def set_children(self, rv: AbsRandomVar, children: List[AbsRandomVar]) -> None:
    self.set_entry(rv, children=children)

  def set_node(self, rv: AbsRandomVar, node: AbsDSNode) -> None:
    self.set_entry(rv, node=node)

  def dedup_children(self, rv: AbsRandomVar) -> None:
    children = self.children(rv)
    seen = set()
    new_children = []
    for rv_child in children:
      if rv_child not in seen:
        seen.add(rv_child)
        new_children.append(rv_child)

    self.set_children(rv, new_children)

  # Returns reachable random variables from rvs
  def entry_referenced_rvs(self, rvs: Set[AbsRandomVar]) -> Set[AbsRandomVar]:
    ref_rvs = super().entry_referenced_rvs(rvs)

    for rv in rvs:
      # Add children
      for rv_child in self.children(rv):
        ref_rvs.add(rv_child)
      
      # Add parents
      match self.node(rv):
        case AbsDSRealized():
          pass
        case AbsDSMarginalized(None):
          pass
        case AbsDSMarginalized((rv_par, cdistr)):
          ref_rvs.add(rv_par)
          ref_rvs = ref_rvs.union(cdistr.rvs())
        case AbsDSInitialized((rv_par, cdistr)):
          ref_rvs.add(rv_par)
          ref_rvs = ref_rvs.union(cdistr.rvs())
        case AbsDSUnk():
          pass
        case _:
          raise ValueError(f'{rv} is {self.node(rv)}')

    return ref_rvs
  
  def entry_rename_rv(self, rv: AbsRandomVar, old: AbsRandomVar, new: AbsRandomVar) -> None:
    super().entry_rename_rv(rv, old, new)
    # Rename children
    self.set_children(rv, [rv_child if rv_child != old else new for rv_child in self.children(rv)])
    self.dedup_children(rv)

    # Rename parents
    match self.node(rv):
      case AbsDSRealized():
        pass
      case AbsDSMarginalized(None):
        pass
      case AbsDSMarginalized((rv_par, cdistr)):
        cdistr = cdistr.rename(old, new)
        rv_par = rv_par if rv_par != old else new
        self.set_node(rv, AbsDSMarginalized((rv_par, cdistr)))
      case AbsDSInitialized((rv_par, cdistr)):
        cdistr = cdistr.rename(old, new)
        rv_par = rv_par if rv_par != old else new
        self.set_node(rv, AbsDSInitialized((rv_par, cdistr)))
      case AbsDSUnk():
        pass
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def entry_join(self, rv: AbsRandomVar, other: 'AbsDSState') -> None:
    def _make_unk(
      parents1: Set[AbsRandomVar],
      parents2: Set[AbsRandomVar],
    ) -> None:
      for rv_par in parents1:
        if rv_par not in self.state:
          other.set_dynamic(rv_par)
        else:
          self.set_dynamic(rv_par)
      for rv_par in parents2:
        other.set_dynamic(rv_par)
      self.children(rv).extend(other.children(rv))
      self.set_distr(rv, TopD())
      self.set_node(rv, AbsDSUnk())

    # print('join', rv, self.node(rv), other.node(rv))
    super().entry_join(rv, other)
    if len(self.children(rv)) > self.max_depth:
      parents1 = set()
      match self.node(rv):
        case AbsDSRealized():
          parents1 = set()
        case AbsDSMarginalized(None):
          parents1 = set()
        case AbsDSMarginalized((rv_par, cdistr)):
          parents1 = {rv_par}
        case AbsDSInitialized((rv_par, cdistr)):
          parents1 = {rv_par}
        case AbsDSUnk():
          parents1 = set()
        case _:
          raise ValueError(f'{rv} is {self.node(rv)}')
      
      parents2 = set()
      match other.node(rv):
        case AbsDSRealized():
          parents2 = set()
        case AbsDSMarginalized(None):
          parents2 = set()
        case AbsDSMarginalized((rv_par, cdistr)):
          parents2 = {rv_par}
        case AbsDSInitialized((rv_par, cdistr)):
          parents2 = {rv_par}
        case AbsDSUnk():
          parents2 = set()
        case _:
          raise ValueError(f'{rv} is {other.node(rv)}')

      _make_unk(parents1, parents2)

    # We try joining two node types, matching on edges if possible
    # If node type mismatch or the edges are different, we set to top node
    match self.node(rv), other.node(rv):
      case AbsDSRealized(), AbsDSRealized():
        self.set_node(rv, AbsDSRealized())
      case AbsDSMarginalized(None), AbsDSMarginalized(edge):
        self.set_node(rv, AbsDSMarginalized(edge))
      case AbsDSMarginalized(edge), AbsDSMarginalized(None):
        self.set_node(rv, AbsDSMarginalized(edge))
      case AbsDSMarginalized((self_par, self_cdistr)), AbsDSMarginalized((other_par, other_cdistr)):
        cdistr = self.join_distr(self_cdistr, other_cdistr, other)

        if isinstance(cdistr, TopD):
          _make_unk(set(), set())
          return
        
        parents = []
        for rv_par in cdistr.rvs():
          if rv_par not in parents:
            parents.append(rv_par)
        
        if len(parents) == 0:
          self.set_node(rv, AbsDSMarginalized(None))
          self.set_distr(rv, cdistr)
          self.children(rv).extend(other.children(rv))
          return
        if len(parents) == 1:
          self.set_node(rv, AbsDSMarginalized((parents[0], cdistr)))
          self.set_distr(rv, cdistr)
          self.children(rv).extend(other.children(rv))
          return
        
        # Set to top because only allowed one parent
        _make_unk(self_cdistr.rvs(), other_cdistr.rvs())
        
      case AbsDSInitialized((self_par, self_cdistr)), AbsDSInitialized((other_par, other_cdistr)):
        cdistr = self.join_distr(self_cdistr, other_cdistr, other)
        
        if isinstance(cdistr, TopD):
          _make_unk(set(), set())
          return
        
        parents = []
        for rv_par in cdistr.rvs():
          if rv_par not in parents:
            parents.append(rv_par)
        
        if len(parents) == 0:
          raise ValueError(f'{rv} is {self.node(rv)}')
        if len(parents) == 1:
          self.set_node(rv, AbsDSInitialized((parents[0],  cdistr)))
          self.set_distr(rv, cdistr)
          self.children(rv).extend(other.children(rv))
          return
        
        # Set to top because only allowed one parent
        _make_unk(self_cdistr.rvs(), other_cdistr.rvs())
      case _, _:
        parents1 = set()
        match self.node(rv):
          case AbsDSRealized():
            parents1 = set()
          case AbsDSMarginalized(None):
            parents1 = set()
          case AbsDSMarginalized((rv_par, cdistr)):
            parents1 = {rv_par}
          case AbsDSInitialized((rv_par, cdistr)):
            parents1 = {rv_par}
          case AbsDSUnk():
            parents1 = set()
          case _:
            raise ValueError(f'{rv} is {self.node(rv)}')
        
        parents2 = set()
        match other.node(rv):
          case AbsDSRealized():
            parents2 = set()
          case AbsDSMarginalized(None):
            parents2 = set()
          case AbsDSMarginalized((rv_par, cdistr)):
            parents2 = {rv_par}
          case AbsDSInitialized((rv_par, cdistr)):
            parents2 = {rv_par}
          case AbsDSUnk():
            parents2 = set()
          case _:
            raise ValueError(f'{rv} is {other.node(rv)}')

        _make_unk(parents1, parents2)

    self.dedup_children(rv)

  ### Symbolic Interface ###
  def assume(self, rv: AbsRandomVar, name: Optional[Identifier], annotation: Optional[Annotation], distribution: AbsSymDistr[T]) -> AbsRandomVar[T]:
    def _check_conjugacy(prior : AbsSymDistr, likelihood : AbsSymDistr, rv_par : AbsRandomVar, rv_child : AbsRandomVar) -> bool:
      match prior, likelihood:
        case AbsNormal(_), AbsNormal(_):
          return conj.gaussian_conjugate_check(self, prior, likelihood, rv_par, rv_child) or \
            conj.normal_inverse_gamma_normal_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case AbsBernoulli(_), AbsBernoulli(_):
          return conj.bernoulli_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case AbsBeta(_), AbsBernoulli(_):
          return conj.beta_bernoulli_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case AbsBeta(_), AbsBinomial(_):
          return conj.beta_binomial_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case AbsGamma(_), AbsPoisson(_):
          return conj.gamma_poisson_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case AbsGamma(_), AbsNormal(_):
          return conj.gamma_normal_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case UnkD(_), _:
          self.set_dynamic(rv_par)
          return False
        case TopD(), _:
          raise ValueError(f'{rv_par} is {self.node(rv_par)}')
        case _:
          return False
    if annotation is not None:
      if name is None:
        raise ValueError('Cannot annotate anonymous variable')
    distribution = self.eval_distr(distribution)
    
    children = []
    if len(distribution.rvs()) == 0:
      node = AbsDSMarginalized(None)
    else:
      parents = []
      for rv_par in distribution.rvs():
        if rv_par not in parents:
          parents.append(rv_par)

      # If any parent is DSUnk,
      # then rv and all ancestors are DSUnk
      # Because we don't know which parent is the canonical parent
      if any([isinstance(self.node(rv_par), AbsDSUnk) for rv_par in parents]):
        pv = {name} if name is not None else set()
        self.set_pv(rv, pv)
        self.set_children(rv, children)
        # TopD because we don't know which is the canonical parent
        self.set_distr(rv, TopD())
        self.set_annotation(rv, annotation)

        for rv_par in parents:
          if not isinstance(self.node(rv_par), AbsDSRealized):
            self.set_dynamic(rv_par)

        self.set_node(rv, AbsDSUnk())
        return rv

      # keep if conjugate, else sample it
      canonical_parent = None
      has_parent = False
      for rv_par in parents:
        if not has_parent:
          match self.node(rv_par):
            case AbsDSRealized():
              distribution = self.eval_distr(distribution)
              continue
            case AbsDSMarginalized(_):
              parent_dist = self.distr(rv_par)
              if _check_conjugacy(parent_dist, distribution, rv_par, rv):
                if rv not in self.children(rv_par):
                  self.children(rv_par).append(rv)

                canonical_parent = rv_par
                has_parent = True

                continue
            case AbsDSInitialized((_, parent_dist)):
              if _check_conjugacy(parent_dist, distribution, rv_par, rv):
                if rv not in self.children(rv_par):
                  self.children(rv_par).append(rv)

                canonical_parent = rv_par
                has_parent = True
                continue
            case _:
              raise ValueError(f'{rv_par} is {self.node(rv_par)}')

        self.value(rv_par)
        distribution = self.eval_distr(distribution)

        # If any parent is DSUnk,
        # then rv and all ancestors are DSUnk
        # Because we don't know which parent is the canonical parent
        if any([isinstance(self.node(rv_par), AbsDSUnk) for rv_par in parents]):
          pv = {name} if name is not None else set()
          self.set_pv(rv, pv)
          self.set_children(rv, children)
          # TopD because we don't know which is the canonical parent
          self.set_distr(rv, TopD())
          self.set_annotation(rv, annotation)

          for rv_par in parents:
            if not isinstance(self.node(rv_par), AbsDSRealized):
              self.set_dynamic(rv_par)

          self.set_node(rv, AbsDSUnk())
          return rv

      # all parents were sampled
      if len(distribution.rvs()) == 0:
        node = AbsDSMarginalized(None)
      else:
        assert canonical_parent is not None
        node = AbsDSInitialized((canonical_parent, distribution))

    pv = {name} if name is not None else set()
    self.set_annotation(rv, annotation)
    self.set_pv(rv, pv)
    self.set_distr(rv, distribution)
    self.set_children(rv, children)
    self.set_node(rv, node)
          
    return rv

  def observe(self, rv: AbsRandomVar[T], value: AbsConst[T]) -> None:
    # Turn rv into a terminal node
    self.graft(rv)
    # observe
    match self.node(rv):
      case AbsDSMarginalized(_):
        self.realize(rv, AbsDelta(value, sampled=False))
      case AbsDSUnk():
        # Can still realize
        self.realize(rv, AbsDelta(value, sampled=False))
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def inner_value(self, rv: AbsRandomVar[T]) -> AbsConst[T]:
    # Turn rv into terminal node
    self.graft(rv)
    return self.do_sample(rv)
  
  # Make rv marginal
  def marginalize(self, rv: AbsRandomVar) -> None:
    match self.node(rv):
      case AbsDSRealized():
        assert len(self.children(rv)) == 0
      case AbsDSMarginalized(_):
        if len(self.children(rv)) > 0:
          self.graft(rv)
      case AbsDSInitialized((rv_par, _)):
        match self.node(rv_par):
          case AbsDSInitialized(_):
            self.marginalize(rv_par)
            assert not isinstance(self.node(rv_par), AbsDSInitialized)
          case AbsDSUnk():
            self.marginalize(rv_par)
            assert not isinstance(self.node(rv_par), AbsDSInitialized)
        self.do_marginalize(rv)
            
        if len(self.children(rv)) > 0:
          self.graft(rv)
      case AbsDSUnk():
        # do both marginalize and graft because we don't know which node type it is
        self.do_marginalize(rv)

        if len(self.children(rv)) > 0:
          self.graft(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')
    
  ########################################################################

  def eval_entry(self, rv: AbsRandomVar) -> None:
    self.set_distr(rv, self.eval_distr(self.distr(rv)))

  # moves rv from I to M and updates its distribution by marginalizing over its parent
  def do_marginalize(self, rv: AbsRandomVar) -> None:
    match self.node(rv):
      case AbsDSInitialized((rv_par, cdistr)):
        match self.node(rv_par):
          case AbsDSInitialized(_):
            raise ValueError(f'Cannot marginalize {rv} because {rv_par} is not marginalized')
          case AbsDSRealized():
            d = self.distr(rv_par)
            # simplify delta
            if not isinstance(d, AbsDelta):
              raise ValueError(d)
            # convert to marginal
            self.eval_entry(rv)
            self.set_node(rv, AbsDSMarginalized(None))
          case AbsDSMarginalized(_):
            if self.make_marginal(rv_par, rv):
              self.eval_entry(rv)
              self.set_node(rv, AbsDSMarginalized((rv_par, cdistr)))
            else:
              self.value(rv_par)
              self.eval_entry(rv)
              self.set_node(rv, AbsDSMarginalized(None))
          case AbsDSUnk():
            # Don't know if we kept rv_par or valued it, so set it to unk
            self.set_dynamic(rv_par)
      case AbsDSUnk():
        pass
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def do_sample(self, rv: AbsRandomVar) -> AbsConst:
    # sample
    match self.node(rv):
      case AbsDSMarginalized(_):
        self.realize(rv, AbsDelta(AbsConst(UnkC()), sampled=True))
        return AbsConst(UnkC())
      case AbsDSUnk():
        self.realize(rv, AbsDelta(AbsConst(UnkC()), sampled=True))
        return AbsConst(UnkC())
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  # Invariant 2: A node always has at most one child that is marginalized
  # But because of uncertainty, there can be multiple marginalized children
  def marginal_child(self, rv: AbsRandomVar) -> Set[AbsRandomVar]:
    mc = set()
    for rv_child in self.children(rv):
      if rv_child not in self:
        self.set_distr(rv_child, TopD())
        self.set_node(rv_child, AbsDSUnk())
        self.set_children(rv_child, [])
      match self.node(rv_child):
        case AbsDSMarginalized(_):
          mc.add(rv_child)
        case _:
          continue
      
    return mc
  
  def make_marginal(self, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
    def _update(marginal: Optional[AbsSymDistr]) -> bool:
      if marginal is None:
        return False

      self.set_distr(rv_child, self.eval_distr(marginal))
      return True

    prior = self.distr(rv_par)
    match self.node(rv_child):
      case AbsDSInitialized((_, likelihood)):
        # Only looking at the likelihood based on the rv_par edge
        match prior, likelihood:
          case AbsNormal(_), AbsNormal(_):
            if _update(conj.gaussian_marginal(self, prior, likelihood, rv_par, rv_child)):
              return True
            else:
              return _update(conj.normal_inverse_gamma_normal_marginal(self, prior, likelihood, rv_par, rv_child))
          case AbsBernoulli(_), AbsBernoulli(_):
            return _update(conj.bernoulli_marginal(self, prior, likelihood, rv_par, rv_child))
          case AbsBeta(_), AbsBernoulli(_):
            return _update(conj.beta_bernoulli_marginal(self, prior, likelihood, rv_par, rv_child))
          case AbsBeta(_), AbsBinomial(_):
            return _update(conj.beta_binomial_marginal(self, prior, likelihood, rv_par, rv_child))
          case AbsGamma(_), AbsPoisson(_):
            return _update(conj.gamma_poisson_marginal(self, prior, likelihood, rv_par, rv_child))
          case AbsGamma(_), AbsNormal(_):
            return _update(conj.gamma_normal_marginal(self, prior, likelihood, rv_par, rv_child))
          case UnkD(_), _:
            self.set_dynamic(rv_par)
            return False
          case TopD(), _:
            self.set_dynamic(rv_par)
            return False
          case _:
            return False
      case _:
        raise ValueError(f'{rv_child} is {self.node(rv_child)}')

  def make_conditional(self, rv_par: AbsRandomVar, rv_child: AbsRandomVar, x: AbsSymExpr) -> bool:
    def _update(posterior: Optional[AbsSymDistr]) -> bool:
      if posterior is None:
        return False
      
      posterior = self.eval_distr(posterior)
      self.set_distr(rv_par, posterior)
      # Update original distr
      return True

    prior = self.distr(rv_par)
    match self.node(rv_child):
      case AbsDSMarginalized((_, likelihood)):
        match prior, likelihood:
          case AbsNormal(_), AbsNormal(_):
            if _update(conj.gaussian_posterior(self, prior, likelihood, rv_par, rv_child, x)):
              return True
            else:
              return _update(conj.normal_inverse_gamma_normal_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case AbsBernoulli(_), AbsBernoulli(_):
            return _update(conj.bernoulli_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case AbsBeta(_), AbsBernoulli(_):
            return _update(conj.beta_bernoulli_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case AbsBeta(_), AbsBinomial(_):
            return _update(conj.beta_binomial_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case AbsGamma(_), AbsPoisson(_):
            return _update(conj.gamma_poisson_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case AbsGamma(_), AbsNormal(_):
            return _update(conj.gamma_normal_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case UnkD(_), _:
            self.set_dynamic(rv_par)
            return False
          case TopD(), _:
            self.set_dynamic(rv_par)
            return False
          case _:
            return False
      case _:
        raise ValueError(f'{rv_child} is {self.node(rv_child)}')

  def realize(self, rv: AbsRandomVar, x: AbsDelta) -> None:
    match self.node(rv):
      case AbsDSMarginalized(None):
        pass
      case AbsDSMarginalized((rv_par, cdistr)):
        match self.node(rv_par):
          case AbsDSMarginalized(par_edge):
            if self.make_conditional(rv_par, rv, x.v):
              self.set_node(rv_par, AbsDSMarginalized(par_edge))
              self.children(rv_par).remove(rv)
            else:
              raise ValueError(f'Cannot realize {rv} because {rv_par} is not conjugate')
          case AbsDSUnk():
            assert isinstance(self.distr(rv_par), UnkD)
            self.children(rv_par).remove(rv)
          case _:
            raise ValueError(f'{rv_par} is {self.node(rv_par)}')
      case AbsDSUnk():
        pass
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

    self.set_distr(rv, x)
    self.set_node(rv, AbsDSRealized())

    # new roots from children
    dedup = []
    for rv_child in self.children(rv):
      if rv_child not in dedup:
        dedup.append(rv_child)
    for rv_child in dedup:
      self.do_marginalize(rv_child)

    self.set_children(rv, [])
    
  def graft(self, rv: AbsRandomVar) -> None:
    match self.node(rv):
      case AbsDSRealized():
        raise ValueError(f'Cannot graft {rv} because it is already realized')
      case AbsDSMarginalized(_):
        rv_children = self.marginal_child(rv)
        if len(rv_children) > 1:
          for rv_child in rv_children:
            self.set_dynamic(rv_child)
          # raise ValueError(f'Cannot graft {rv} because it has multiple marginalized children')
        else:
          for rv_child in rv_children:
            self.prune(rv_child)
      case AbsDSInitialized((rv_par, _)):
        self.graft(rv_par)
        self.do_marginalize(rv)
      case AbsDSUnk():
        self.set_dynamic(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def prune(self, rv: AbsRandomVar) -> None:
    match self.node(rv):
      case AbsDSMarginalized(_):
        rv_children = self.marginal_child(rv)
        if len(rv_children) > 1:
          # raise ValueError(f'Cannot prune {rv} because it has multiple marginalized children')
          for rv_child in rv_children:
            self.set_dynamic(rv_child)
        else:
          for rv_child in rv_children:
            self.prune(rv_child)
        
        self.value(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')
      
  # Makes rv DSUnk and deal with side effects 
  def set_dynamic(self, rv: AbsRandomVar) -> None:
    super_set_dynamic = super().set_dynamic

    nodes = set()

    def _set_unk_node(rv: AbsRandomVar) -> None:
      if rv in nodes:
        return
      
      # not in state yet but will
      if rv not in self.state:
        nodes.add(rv)

      match self.node(rv):
        case AbsDSRealized():
          pass
        case AbsDSMarginalized(None):
          pass
        case AbsDSMarginalized((rv_par, _)):
          _set_unk_node(rv_par)
        case AbsDSInitialized((rv_par, _)):
          _set_unk_node(rv_par)
        case AbsDSUnk():
          pass
        case _:
          raise ValueError(f'{rv} is {self.node(rv)}')

      nodes.add(rv)

      for child in self.children(rv):
        _set_unk_node(child)


    _set_unk_node(rv)

    for rv in nodes:
      self.set_node(rv, AbsDSUnk())
      self.set_distr(rv, TopD())
      super_set_dynamic(rv)
