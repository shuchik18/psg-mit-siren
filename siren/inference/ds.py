from typing import Any, Set, Union
from copy import copy
from siren.grammar import RandomVar

import siren.inference.conjugate as conj
from siren.inference.interface import *

# Implementation of delayed sampling
# Adapted from https://github.com/IBM/probzelus/

### Directed Sampling node types
@dataclass(frozen=True)
class DSNode():
  pass

# DSRealized: the node has been observed/sampled
@dataclass(frozen=True)
class DSRealized(DSNode):
  def __str__(self) -> str:
    return 'DSRealized'

# DSMarginalized: the node has been marginalized
@dataclass(frozen=True)
class DSMarginalized(DSNode):
  edge: Optional[Tuple[RandomVar, SymDistr]]

  def __str__(self):
    return f'DSMarginalized({self.edge})'

# DSInitialized: the node is initialized
@dataclass(frozen=True)
class DSInitialized(DSNode):
  edge: Tuple[RandomVar, SymDistr]

  def __str__(self):
    return f'DSInitialized({self.edge})'

class DSState(SymState):
  ###
  # State entry:
  #   rv: (pv, distribution, children, node)
  ###

  def __str__(self):
    s = '\n\t'.join(map(str, self.state.items()))
    return f"DSState(\n\t{s}\n)" if s else "DSState()"
  
  ## Accessors
  def children(self, rv: RandomVar) -> List[RandomVar]:
    return self.get_entry(rv, 'children')
  
  def node(self, rv: RandomVar) -> DSNode:
    return self.get_entry(rv, 'node')

  ## Mutators
  def set_children(self, rv: RandomVar, children: List[RandomVar]) -> None:
    self.set_entry(rv, children=children)

  def set_node(self, rv: RandomVar, node: DSNode) -> None:
    self.set_entry(rv, node=node)

  def assume(self, rv: RandomVar, name: Optional[Identifier], annotation: Optional[Annotation], distribution: SymDistr[T]) -> RandomVar[T]:
    def _check_conjugacy(prior : SymDistr, likelihood : SymDistr, rv_par : RandomVar, rv_child : RandomVar) -> bool:
      match prior, likelihood:
        case Normal(_), Normal(_):
          return conj.gaussian_conjugate_check(self, prior, likelihood, rv_par, rv_child) or \
            conj.normal_inverse_gamma_normal_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case Bernoulli(_), Bernoulli(_):
          return conj.bernoulli_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case Beta(_), Bernoulli(_):
          return conj.beta_bernoulli_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case Beta(_), Binomial(_):
          return conj.beta_binomial_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case Gamma(_), Poisson(_):
          return conj.gamma_poisson_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case Gamma(_), Normal(_):
          return conj.gamma_normal_conjugate_check(self, prior, likelihood, rv_par, rv_child)
        case _:
          return False

    # create a new random variable and assign annotation
    if annotation is not None:
      if name is None:
        raise ValueError('Cannot annotate anonymous variable')
    
    # simplify distribution
    distribution = self.eval_distr(distribution)
    children = []

    # if the distribution refers to no random variables, the RV is marginalized
    if len(distribution.rvs()) == 0:
      node = DSMarginalized(None)
    else:
      # Otherwise, can only keep one parent and it has to be conjugate
      # Greedily choose the first conjugate parent

      # Remove duplicate parents, in place
      parents = []
      for rv_par in distribution.rvs():
        if rv_par not in parents:
          parents.append(rv_par)

      # keep if conjugate, else sample it
      canonical_parent = None
      has_parent = False
      for rv_par in parents:
        # if already sampled, skip
        if rv_par not in distribution.rvs():
          continue
        # still need a conjugate parent
        if not has_parent:
          match self.node(rv_par):
            # If the parent is already realized, just simplify the distribution
            case DSRealized():
              distribution = self.eval_distr(distribution)
              continue
            case DSMarginalized(_):
              # If the parent is marginalized, check if it is conjugate
              parent_dist = self.distr(rv_par)
              if _check_conjugacy(parent_dist, distribution, rv_par, rv):
                if rv not in self.children(rv_par):
                  self.children(rv_par).append(rv)

                canonical_parent = rv_par
                has_parent = True

                continue
            case DSInitialized((_, parent_dist)):
              # If the parent is initialized, check if it is conjugate
              # using the conditional distribution expression of the node
              if _check_conjugacy(parent_dist, distribution, rv_par, rv):
                if rv not in self.children(rv_par):
                  self.children(rv_par).append(rv)

                canonical_parent = rv_par
                has_parent = True
                continue
            case _:
              raise ValueError(f'{rv_par} is {self.node(rv_par)}')

        # Parent was not conjugate, sample it
        self.value(rv_par)
        distribution = self.eval_distr(distribution)

      # all parents were sampled, RV is marginalized
      if len(distribution.rvs()) == 0:
        node = DSMarginalized(None)
      else:
        # If there is a conjugate parent, initialize the RV with the parent
        assert canonical_parent is not None
        node = DSInitialized((canonical_parent, distribution))

    self.set_annotation(rv, annotation)
    self.set_pv(rv, name)
    self.set_distr(rv, distribution)
    self.set_children(rv, children)
    self.set_node(rv, node)

    return rv

  def observe(self, rv: RandomVar[T], value: Const[T]) -> float:
    # Turn rv into a terminal node
    self.graft(rv)
    # observe
    match self.node(rv):
      case DSMarginalized(_):
        s = self.score(rv, value.v)
        self.realize(rv, Delta(value, sampled=False))
        return s
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def value_impl(self, rv: RandomVar[T]) -> Const[T]:
    # Turn rv into terminal node
    self.graft(rv)
    return self.do_sample(rv)
  
  # Make rv marginal and terminal
  def marginalize(self, rv: RandomVar) -> None:
    match self.node(rv):
      case DSRealized():
        assert len(self.children(rv)) == 0
      case DSMarginalized(_):
        # Turn rv into terminal node if it has children
        if len(self.children(rv)) > 0:
          self.graft(rv)
      case DSInitialized((rv_par, _)):
        match self.node(rv_par):
          case DSInitialized(_):
            # recursively marginalize parent
            self.marginalize(rv_par)
            assert not isinstance(self.node(rv_par), DSInitialized)
        # Turn RV into DSMarginalized
        self.do_marginalize(rv)
            
        # Turn rv into terminal node if it has children
        if len(self.children(rv)) > 0:
          self.graft(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')
    
  ########################################################################
              
  def eval_entry(self, rv: RandomVar) -> None:
    self.set_distr(rv, self.eval_distr(self.distr(rv)))

  # moves rv from I to M and updates its distribution by marginalizing over its parent
  def do_marginalize(self, rv: RandomVar) -> None:
    match self.node(rv):
      case DSInitialized((rv_par, cdistr)):
        match self.node(rv_par):
          case DSInitialized(_):
            raise ValueError(f'Cannot marginalize {rv} because {rv_par} is not marginalized')
          case DSRealized():
            # Parent is realized, simplify the distribution
            d = self.distr(rv_par)
            # simplify delta
            if not isinstance(d, Delta):
              raise ValueError(d)
            # convert to marginal
            self.eval_entry(rv)
            self.set_node(rv, DSMarginalized(None))
          case DSMarginalized(_):
            # Try to make rv marginal, preserving the edge and conditional distribution
            if self.make_marginal(rv_par, rv):
              self.eval_entry(rv)
              self.set_node(rv, DSMarginalized((rv_par, cdistr)))
            else:
              # Parent is not conjugate, sample it
              # This will automatically make RV marginal with no edge
              self.value(rv_par)
              self.eval_entry(rv)
              self.set_node(rv, DSMarginalized(None))
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  def do_sample(self, rv: RandomVar) -> Const:
    # sample only works on marginalized nodes
    match self.node(rv):
      case DSMarginalized(_):
        v = self.draw(rv)
        self.realize(rv, Delta(Const(v), sampled=True))
        return Const(v)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')
    
  def score(self, rv: RandomVar[T], v: T) -> float:
    return self.distr(rv).score(v)

  def draw(self, rv: RandomVar) -> Any:
    return self.distr(rv).draw(self.rng)

  # Invariant 2: A node always has at most one child that is marginalized
  def marginal_child(self, rv: RandomVar) -> Optional[RandomVar]:
    for rv_child in self.children(rv):
      match self.node(rv_child):
        case DSMarginalized(_):
          return rv_child
        case _:
          continue
      
    return None
  
  def make_marginal(self, rv_par: RandomVar, rv_child: RandomVar) -> bool:
    # Update saves the marginal distribution to rv_child
    def _update(marginal: Optional[SymDistr]) -> bool:
      if marginal is None:
        return False

      self.set_distr(rv_child, self.eval_distr(marginal))
      return True

    # Try to make rv_child marginal, using the original conditional distribution
    # to check for conjugacy
    prior = self.distr(rv_par)
    match self.node(rv_child):
      case DSInitialized((_, likelihood)):
        match prior, likelihood:
          case Normal(_), Normal(_):
            if _update(conj.gaussian_marginal(self, prior, likelihood, rv_par, rv_child)):
              return True
            else:
              return _update(conj.normal_inverse_gamma_normal_marginal(self, prior, likelihood, rv_par, rv_child))
          case Bernoulli(_), Bernoulli(_):
            return _update(conj.bernoulli_marginal(self, prior, likelihood, rv_par, rv_child))
          case Beta(_), Bernoulli(_):
            return _update(conj.beta_bernoulli_marginal(self, prior, likelihood, rv_par, rv_child))
          case Beta(_), Binomial(_):
            return _update(conj.beta_binomial_marginal(self, prior, likelihood, rv_par, rv_child))
          case Gamma(_), Poisson(_):
            return _update(conj.gamma_poisson_marginal(self, prior, likelihood, rv_par, rv_child))
          case Gamma(_), Normal(_):
            return _update(conj.gamma_normal_marginal(self, prior, likelihood, rv_par, rv_child))
          case _:
            return False
      case _:
        raise ValueError(f'{rv_child} is {self.node(rv_child)}')
      
  # Tries to update parent node with the updated child (that should have been Marginalized)
  def make_conditional(self, rv_par: RandomVar, rv_child: RandomVar, x: SymExpr) -> bool:
    # Update saves the posterior distribution to rv_par
    def _update(posterior: Optional[SymDistr]) -> bool:
      if posterior is None:
        return False
      
      posterior = self.eval_distr(posterior)
      self.set_distr(rv_par, posterior)
      return True

    # Using the original conditional distribution
    # to check for conjugacy
    prior = self.distr(rv_par)
    match self.node(rv_child):
      case DSMarginalized((_, likelihood)):
        match prior, likelihood:
          case Normal(_), Normal(_):
            if _update(conj.gaussian_posterior(self, prior, likelihood, rv_par, rv_child, x)):
              return True
            else:
              return _update(conj.normal_inverse_gamma_normal_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case Bernoulli(_), Bernoulli(_):
            return _update(conj.bernoulli_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case Beta(_), Bernoulli(_):
            return _update(conj.beta_bernoulli_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case Beta(_), Binomial(_):
            return _update(conj.beta_binomial_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case Gamma(_), Poisson(_):
            return _update(conj.gamma_poisson_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case Gamma(_), Normal(_):
            return _update(conj.gamma_normal_posterior(self, prior, likelihood, rv_par, rv_child, x))
          case _:
            return False
      case _:
        raise ValueError(f'{rv_child} is {self.node(rv_child)}')

  # Move RV from M to R and update its distribution
  def realize(self, rv: RandomVar, x: Delta) -> None:
    match self.node(rv):
      case DSMarginalized(None):
        pass
      case DSMarginalized((rv_par, _)):
        # If it has a parent, update the parent with the updated child values
        # Assumes that parent is conjugate, otherwise make_marginal would have failed earlier
        match self.node(rv_par):
          case DSMarginalized(edge):
            if self.make_conditional(rv_par, rv, x.v):
              self.set_node(rv_par, DSMarginalized(edge))
              self.children(rv_par).remove(rv)
            else:
              raise ValueError(f'Cannot realize {rv} because {rv_par} is not conjugate')
          case _:
            raise ValueError(f'{rv_par} is {self.node(rv_par)}')
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

    self.set_distr(rv, x)
    self.set_node(rv, DSRealized())

    # new roots from children being orphaned
    dedup = []
    for rv_child in self.children(rv):
      if rv_child not in dedup:
        dedup.append(rv_child)
    for rv_child in dedup:
      self.do_marginalize(rv_child)

    self.set_children(rv, [])
    
  # Makes RV a terminal node
  def graft(self, rv: RandomVar) -> None:
    match self.node(rv):
      case DSRealized():
        raise ValueError(f'Cannot graft {rv} because it is already realized')
      case DSMarginalized(_):
        # Terminal nodes do not have marginal children
        rv_child = self.marginal_child(rv)
        if rv_child is not None:
          self.prune(rv_child)
      case DSInitialized((rv_par, _)):
        # Recurse on parent, then try to marginalize rv
        self.graft(rv_par)
        self.do_marginalize(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')

  # Samples the RV to remove the edge to its parent
  def prune(self, rv: RandomVar) -> None:
    match self.node(rv):
      case DSMarginalized(_):
        # Recurse on children before sampling
        rv_child = self.marginal_child(rv)
        if rv_child is not None:
          self.prune(rv_child)
        
        self.value(rv)
      case _:
        raise ValueError(f'{rv} is {self.node(rv)}')