from typing import Optional, Tuple

from siren.grammar import *
from siren.inference.interface import SymState

# Helper functions for manipulating conjugate distributions

# If expr scales rv, returns the scaling factor
# Not complete
def is_scaled(state: SymState, expr: SymExpr, e: SymExpr) -> Optional[SymExpr]:
  if expr == e:
    return Const(1)
  
  match expr:
    case Const(_):
      return None
    case RandomVar(_):
      return None
    case Add(e1, e2):
      s1 = is_scaled(state, e1, e)
      s2 = is_scaled(state, e2, e)
      if s1 is None or s2 is None:
        return None
      else:
        return state.ex_add(s1, s2)
    case Mul(e1, e2):
      s1 = is_scaled(state, e1, e)
      s2 = is_scaled(state, e2, e)
      if s1 is None and s2 is None:
        return None
      elif s1 is None and s2 is not None:
        return state.ex_mul(s2, e1)
      elif s1 is not None and s2 is None:
        return state.ex_mul(s1, e2)
      else:
        assert s1 is not None and s2 is not None
        return state.ex_mul(s1, s2)
    case Div(e1, e2):
      s1 = is_scaled(state, e1, e)
      s2 = is_scaled(state, e2, e)
      if s1 is not None and s2 is not None:
        return None # e cancels out
      elif s1 is None and s2 is not None:
        return state.ex_div(e1, s2)
      elif s1 is not None and s2 is None:
        return state.ex_div(s1, e2)
      else:
        return None
    case Ite(_):
      return None
    case Eq(_):
      return None
    case Lt(_):
      return None
    case Lst(_):
      return None
    case Pair(_):
      return None
    case _:
      raise ValueError(expr)

# Returns (a, b) such that expr = a * rv + b
def is_affine(state: SymState, expr: SymExpr, rv: RandomVar) -> Optional[Tuple[SymExpr, SymExpr]]:
  match expr:
    case Const(_):
      return (Const(0), expr)
    case RandomVar(_):
      if expr == rv:
        return (Const(1), Const(0))
      else:
        return (Const(0), expr)
    case Add(e1, e2):
      coefs1 = is_affine(state, e1, rv)
      coefs2 = is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        return (state.ex_add(a1, a2), state.ex_add(b1, b2))
    case Mul(e1, e2):
      coefs1 = is_affine(state, e1, rv)
      coefs2 = is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.eval(a1), state.eval(a2):
          case Const(0), Const(0):
            return (Const(0), state.ex_mul(b1, b2))
          case a1, Const(0):
            return (state.ex_mul(a1, b2), state.ex_mul(b1, b2))
          case Const(0), a2:
            return (state.ex_mul(b1, a2), state.ex_mul(b1, b2))
          case _:
            return None
    case Div(e1, e2):
      coefs1 = is_affine(state, e1, rv)
      coefs2 = is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.eval(a2):
          case Const(0):
            return (state.ex_div(a1, b2), state.ex_div(b1, b2))
          case _:
            return None
    case Ite(_):
      return None
    case Eq(_):
      return None
    case Lt(_):
      return None
    case Lst(_):
      return  None
    case Pair(_):
      return None
    case _:
      raise ValueError(expr)
    
### Affine Gaussian Conjugate ###

def gaussian_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr, 
                             rv_par: RandomVar, rv_child: RandomVar) -> bool:
  match prior, likelihood:
    case Normal(mu0, var0), Normal(mu, var):
      coefs = is_affine(state, mu, rv_par)
      if coefs is None:
        return False

      return not mu0.depends_on(rv_child, True) \
            and not var0.depends_on(rv_child, True) \
            and not var.depends_on(rv_par, True)
    case _:
      return False
    
def gaussian_marginal(state: SymState, prior: Normal, likelihood: Normal, 
                      rv_par: RandomVar, rv_child: RandomVar) -> Optional[Normal]:
  
  if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var0 = prior.mu, prior.var
  mu, var = likelihood.mu, likelihood.var

  coefs = is_affine(state, mu, rv_par)
  if coefs is None:
    return None
  
  a, b = coefs

  mu01 = state.ex_add(state.ex_mul(a, mu0), b)
  var01 = state.ex_mul(state.ex_mul(a, a), var0)

  mu0_new = mu01
  var0_new = state.ex_add(var01, var)

  return Normal(mu0_new, var0_new)
    
def gaussian_posterior(state: SymState, prior: Normal, likelihood: Normal, 
                      rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Normal]:
  if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
    
  mu0, var0 = prior.mu, prior.var
  mu, var = likelihood.mu, likelihood.var

  coefs = is_affine(state, mu, rv_par)
  if coefs is None:
    return None
  
  a, b = coefs

  x = rv_child if obs is None else obs

  mu01 = state.ex_add(state.ex_mul(a, mu0), b)
  var01 = state.ex_mul(state.ex_mul(a, a), var0)

  denom = state.ex_add(state.ex_div(Const(1), var01), state.ex_div(Const(1), var))
  var02 = state.ex_div(Const(1), denom)

  sum1 = state.ex_add(state.ex_div(mu01, var01), state.ex_div(x, var))
  mu02 = state.ex_mul(sum1, var02)

  mu1_new = state.ex_div(state.ex_add(mu02, state.ex_mul(Const(-1), b)), a)
  var1_new = state.ex_div(var02, state.ex_mul(a, a))

  return Normal(mu1_new, var1_new)

# Returns (marginal, posterior) distributions
def gaussian_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[Normal, Normal]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Normal(mu0, var0), Normal(mu, var):
      marginal = gaussian_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gaussian_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Bernoulli Conjugate ###
    
def bernoulli_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                              rv_par: RandomVar, rv_child: RandomVar) -> bool:
  match prior, likelihood:
    case Bernoulli(p1), Bernoulli(p2):
      return p2.depends_on(rv_par, False) and \
            not p1.depends_on(rv_child, True)
    case _:
      return False
    
def bernoulli_marginal(state: SymState, prior: Bernoulli, likelihood: Bernoulli, 
                       rv_par: RandomVar, rv_child: RandomVar) -> Optional[Bernoulli]:
  if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  p1, p2 = prior.p, likelihood.p
  p2_new = state.ex_add(state.ex_mul(p1, state.eval(p2.subst_rv(rv_par, Const(True)))),
                        state.ex_mul(state.ex_add(Const(1), state.ex_mul(Const(-1), p1)),
                                    state.eval(p2.subst_rv(rv_par, Const(False)))))
  return Bernoulli(p2_new)
    
def bernoulli_posterior(state: SymState, prior: Bernoulli, likelihood: Bernoulli, 
                        rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Bernoulli]:
  if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  x = rv_child if obs is None else obs
  
  p1, p2 = prior.p, likelihood.p
  p2_new = state.ex_add(state.ex_mul(p1, state.eval(p2.subst_rv(rv_par, Const(True)))),
                        state.ex_mul(state.ex_add(Const(1), state.ex_mul(Const(-1), p1)),
                                    state.eval(p2.subst_rv(rv_par, Const(False)))))
  
  p1_num_sub = state.ex_ite(x, p2, state.ex_add(Const(1), state.ex_mul(Const(-1), p2)))
  p1_num = state.ex_mul(p1, state.eval(p1_num_sub.subst_rv(rv_par, Const(True))))
  p1_denom = state.ex_ite(x, p2_new, state.ex_add(Const(1), state.ex_mul(Const(-1), p2_new)))
  p1_new = state.ex_div(p1_num, p1_denom)

  return Bernoulli(p1_new)
    
def bernoulli_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[Bernoulli, Bernoulli]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Bernoulli(p1), Bernoulli(p2):
      marginal = bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Beta Bernoulli Conjugate ###

def beta_bernoulli_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                    rv_par: RandomVar, rv_child: RandomVar) -> bool:
  
  match prior, likelihood:
    case Beta(a, b), Bernoulli(p):
      return rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False
    
def beta_bernoulli_marginal(state: SymState, prior: Beta, likelihood: Bernoulli,
                            rv_par: RandomVar, rv_child: RandomVar) -> Optional[Bernoulli]:
  if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  p_new = state.ex_div(a, state.ex_add(a, b))

  return Bernoulli(p_new)

def beta_bernoulli_posterior(state: SymState, prior: Beta, likelihood: Bernoulli,
                              rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Beta]:
  if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b

  x = rv_child if obs is None else obs

  a_new = state.ex_add(a, state.ex_ite(x, Const(1), Const(0)))
  b_new = state.ex_add(b, state.ex_ite(x, Const(0), Const(1)))

  return Beta(a_new, b_new)
    
def beta_bernoulli_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[Bernoulli, Beta]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Beta(a, b), Bernoulli(p):
      marginal = beta_bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = beta_bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Beta Binomial Conjugate ###

def beta_binomial_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                    rv_par: RandomVar, rv_child: RandomVar) -> bool:
  
  match prior, likelihood:
    case Beta(a, b), Binomial(n, p):
      return isinstance(n, Const) \
            and isinstance(a, Const)\
            and rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False
    
def beta_binomial_marginal(state: SymState, prior: Beta, likelihood: Binomial,
                            rv_par: RandomVar, rv_child: RandomVar) -> Optional[BetaBinomial]:
  if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  n, p = likelihood.n, likelihood.p

  return BetaBinomial(n, a, b)

def beta_binomial_posterior(state: SymState, prior: Beta, likelihood: Binomial, 
                              rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Beta]:
  if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  n, p = likelihood.n, likelihood.p

  assert isinstance(n, Const)

  a_new = state.ex_add(a, rv_child)
  b_new = state.ex_add(b, state.ex_add(Const(float(n.v)), state.ex_mul(Const(-1), rv_child)))

  return Beta(a_new, b_new)
    
def beta_binomial_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[BetaBinomial, Beta]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Beta(a, b), Binomial(n, p):
      marginal = beta_binomial_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = beta_binomial_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Gamma Poisson Conjugate ###

def gamma_poisson_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                    rv_par: RandomVar, rv_child: RandomVar) -> bool:
  match prior, likelihood:
    case Gamma(a, b), Poisson(l):
      return isinstance(a, Const) \
            and bool(np.isclose(round(a.v), a.v)) \
            and rv_par == l \
            and not b.depends_on(rv_child, True)
    case _:
      return False

def gamma_poisson_marginal(state: SymState, prior: Gamma, likelihood: Poisson,
                            rv_par: RandomVar, rv_child: RandomVar) -> Optional[NegativeBinomial]:
  if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  l = likelihood.l

  assert isinstance(a, Const)

  n_new = Const(int(a.v))
  p_new = state.ex_div(b, state.ex_add(Const(1), b))

  return NegativeBinomial(n_new, p_new)

def gamma_poisson_posterior(state: SymState, prior: Gamma, likelihood: Poisson,
                              rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Gamma]:
  if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b

  x = rv_child if obs is None else obs

  a_new = state.ex_add(a, x)
  b_new = state.ex_add(b, Const(1))

  return Gamma(a_new, b_new)
    
def gamma_poisson_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[NegativeBinomial, Gamma]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Gamma(a, b), Poisson(l):
      marginal = gamma_poisson_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gamma_poisson_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Gamma Normal Conjugate ###
# TODO: case of var being scaled invgamma

def gamma_normal_conjugate_check(state: SymState, prior: Gamma, likelihood: Normal,
                                    rv_par: RandomVar, rv_child: RandomVar) -> bool:
  
  a, b = prior.a, prior.b
  mu, var = likelihood.mu, likelihood.var

  return isinstance(mu, Const) \
        and var == state.ex_div(Const(1), rv_par) \
        and not a.depends_on(rv_child, True) \
        and not b.depends_on(rv_child, True)

def gamma_normal_marginal(state: SymState, prior: Gamma, likelihood: Normal,
                            rv_par: RandomVar, rv_child: RandomVar) -> Optional[StudentT]:
  if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  mu, var = likelihood.mu, likelihood.var

  tau2 = state.ex_div(b, a)
  nu = state.ex_mul(Const(2), a)

  return StudentT(mu, tau2, nu)

def gamma_normal_posterior(state: SymState, prior: Gamma, likelihood: Normal,
                            rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Gamma]:
  if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  mu, var = likelihood.mu, likelihood.var

  x = rv_child if obs is None else obs

  assert isinstance(mu, Const)

  a_new = state.ex_add(a, Const(0.5))
  b_inner = state.ex_add(x, Const(-mu.v))
  b_new = state.ex_add(b, state.ex_mul(Const(0.5), state.ex_mul(b_inner, b_inner)))

  return Gamma(a_new, b_new)
    
def gamma_normal_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[StudentT, Gamma]]:
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Gamma(a, b), Normal(mu, var):
      marginal = gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Normal-Inverse-Gamma Normal Conjugate ###

def normal_inverse_gamma_normal_conjugate_check(state: SymState, prior: SymDistr, likelihood: SymDistr,
                                    rv_par: RandomVar, rv_child: RandomVar) -> bool:
  
  match prior, likelihood:
    case Normal(mu0, var1), Normal(mu, var2):
      match var2:
        case Div(Const(1), var2_inner):
          if isinstance(var2_inner, RandomVar):
            # TODO: should this be passed in
            match state.get_entry(var2_inner, 'distribution'):
              case Gamma(a, b):
                # var1 should be the invgamma scaled by 1/lambda
                k = is_scaled(state, var1, var2)
                if k is None:
                  return False
                else:
                  match state.eval(k):
                    case Const(0):
                      return False
                    case Const(_):
                      return isinstance(mu0, Const) \
                              and mu == rv_par \
                              and not mu0.depends_on(rv_child, True) \
                              and not var1.depends_on(rv_child, True)
                    case _:
                      return False
              case _:
                return False
          else:
            return False
        case _:
          return False
    case _:
      return False
    
def normal_inverse_gamma_normal_marginal(state: SymState, prior: Normal, likelihood: Normal,
                            rv_par: RandomVar, rv_child: RandomVar) -> Optional[StudentT]:
  if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var1 = prior.mu, prior.var
  mu, var2 = likelihood.mu, likelihood.var

  match var2:
    case Div(Const(1), var2_inner):
      if isinstance(var2_inner, RandomVar):
        match state.get_entry(var2_inner, 'distribution'):
          case Gamma(a, b):
            k = is_scaled(state, var1, var2)
            assert k is not None
            assert isinstance(mu0, Const)

            lam = state.ex_div(Const(1), k)

            a_new = state.ex_add(a, Const(0.5))
            b_inner = state.ex_add(rv_child, Const(-mu0.v))
            b_new = state.ex_add(b, state.ex_mul(state.ex_div(lam, state.ex_div(lam, Const(1))), state.ex_div(state.ex_mul(b_inner, b_inner), Const(2))))

            state.set_entry(var2_inner, distribution=Gamma(a_new, b_new))


            mu_new = mu0
            tau2_new = state.ex_div(state.ex_mul(b, state.ex_add(lam, Const(1))), state.ex_mul(a, lam))
            nu_new = state.ex_mul(Const(2), a)

            return StudentT(mu_new, tau2_new, nu_new)
          case _:
            return None
      else:
        return None
    case _:
      return None
    
def normal_inverse_gamma_normal_posterior(state: SymState, prior: Normal, likelihood: Normal,
                            rv_par: RandomVar, rv_child: RandomVar, obs: Optional[SymExpr]=None) -> Optional[Normal]:
  if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var1 = prior.mu, prior.var
  mu, var2 = likelihood.mu, likelihood.var

  x = rv_child if obs is None else obs

  match var2:
    case Div(Const(1), var2_inner):
      # var2 must be a random variable of invgamma
      if isinstance(var2_inner, RandomVar):
        match state.get_entry(var2_inner, 'distribution'):
          case Gamma(a, b):
            k = is_scaled(state, var1, var2)
            assert k is not None
            assert isinstance(mu0, Const)

            lam = state.ex_div(Const(1), k)

            mu0_new = state.ex_div(state.ex_add(state.ex_mul(lam, mu0), x), state.ex_add(lam, Const(1)))
            lam_new = state.ex_add(lam, Const(1))
            
            a_new = state.ex_add(a, Const(0.5))
            b_inner = state.ex_add(x, Const(-mu0.v))
            b_new = state.ex_add(b, state.ex_mul(state.ex_div(lam, state.ex_div(lam, Const(1))), state.ex_div(state.ex_mul(b_inner, b_inner), Const(2))))

            state.set_entry(var2_inner, distribution=Gamma(a_new, b_new))

            var_new = state.ex_div(Const(1), state.ex_mul(lam_new, var2_inner))

            return Normal(mu0_new, var_new)
          case _:
            return None
      else:
        return None
    case _:
      return None
            
    
def normal_inverse_gamma_normal_conjugate(state: SymState, rv_par: RandomVar, rv_child: RandomVar) -> Optional[Tuple[StudentT, Normal]]:
  # print('normal_inverse_gamma_normal_conjugate', rv_par, rv_child)
  prior, likelihood = state.get_entry(rv_par, 'distribution'), state.get_entry(rv_child, 'distribution')
  match prior, likelihood:
    case Normal(mu0, var1), Normal(mu, var2):
      marginal = normal_inverse_gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = normal_inverse_gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None