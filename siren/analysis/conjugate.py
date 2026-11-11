from typing import Optional, Tuple

from siren.grammar import *
from siren.analysis.interface import AbsSymState

# If expr scales rv, returns the scaling factor
# Not complete
def is_scaled(state: AbsSymState, expr: AbsSymExpr, e: AbsSymExpr) -> Optional[AbsSymExpr]:
  if expr == e:
    return AbsConst(1)
  
  match expr:
    case AbsConst(_):
      return None
    case AbsRandomVar(_):
      return None
    case AbsAdd(e1, e2):
      s1 = is_scaled(state, e1, e)
      s2 = is_scaled(state, e2, e)
      if s1 is None or s2 is None:
        return None
      else:
        return state.ex_add(s1, s2)
    case AbsMul(e1, e2):
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
    case AbsDiv(e1, e2):
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
    case AbsIte(_):
      return None
    case AbsEq(_):
      return None
    case AbsLt(_):
      return None
    case AbsLst(_):
      return None
    case AbsPair(_):
      return None
    case UnkE(s):
      return None
    case TopE():
      return None
    case _:
      raise ValueError(expr)

def abs_is_affine(state: AbsSymState, expr: AbsSymExpr, rv: AbsRandomVar) -> Optional[Tuple[AbsSymExpr, AbsSymExpr]]:
  match expr:
    case AbsConst(_):
      return (AbsConst(0), expr)
    case AbsRandomVar(_):
      if expr == rv:
        return (AbsConst(1), AbsConst(0))
      else:
        return (AbsConst(0), expr)
    case AbsAdd(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        return (state.ex_add(a1, a2), state.ex_add(b1, b2))
    case AbsMul(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.eval(a1), state.eval(a2):
          case AbsConst(0), AbsConst(0):
            return (AbsConst(0), state.ex_mul(b1, b2))
          case a1, AbsConst(0):
            return (state.ex_mul(a1, b2), state.ex_mul(b1, b2))
          case AbsConst(0), a2:
            return (state.ex_mul(b1, a2), state.ex_mul(b1, b2))
          case _:
            return None
    case AbsDiv(e1, e2):
      coefs1 = abs_is_affine(state, e1, rv)
      coefs2 = abs_is_affine(state, e2, rv)
      if coefs1 is None or coefs2 is None:
        return None
      else:
        a1, b1 = coefs1
        a2, b2 = coefs2
        match state.eval(a2):
          case AbsConst(0):
            return (state.ex_div(a1, b2), state.ex_div(b1, b2))
          case _:
            return None
    case AbsIte(_):
      return None
    case AbsEq(_):
      return None
    case AbsLt(_):
      return None
    case AbsLst(_):
      return None
    case AbsPair(_):
      return None
    case UnkE(s):
      # Doesn't depend on rv so it's affine
      if len(s) == 0:
        return (AbsConst(0), expr)
      else:
        return None
    case TopE():
      return None
    case _:
      raise ValueError(expr)
    
### Affine Gaussian Conjugate ###

def gaussian_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr, 
                             rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsNormal(mu0, var0), AbsNormal(mu, var):
      coefs = abs_is_affine(state, mu, rv_par)
      if coefs is None:
        return False

      return not mu0.depends_on(rv_child, True) \
            and not var0.depends_on(rv_child, True) \
            and not var.depends_on(rv_par, True)
    case _:
      return False
    
def gaussian_marginal(state: AbsSymState, prior: AbsNormal, likelihood: AbsNormal, 
                      rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsNormal]:
  
  if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var0 = prior.mu, prior.var
  mu, var = likelihood.mu, likelihood.var

  coefs = abs_is_affine(state, mu, rv_par)
  if coefs is None:
    return None
  
  a, b = coefs

  mu01 = state.ex_add(state.ex_mul(a, mu0), b)
  var01 = state.ex_mul(state.ex_mul(a, a), var0)

  mu0_new = mu01
  var0_new = state.ex_add(var01, var)

  return AbsNormal(mu0_new, var0_new)
    
def gaussian_posterior(state: AbsSymState, prior: AbsNormal, likelihood: AbsNormal, 
                      rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsNormal]:
  if not gaussian_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
    
  mu0, var0 = prior.mu, prior.var
  mu, var = likelihood.mu, likelihood.var

  coefs = abs_is_affine(state, mu, rv_par)
  if coefs is None:
    return None
  
  a, b = coefs

  x = rv_child if obs is None else obs

  mu01 = state.ex_add(state.ex_mul(a, mu0), b)
  var01 = state.ex_mul(state.ex_mul(a, a), var0)

  denom = state.ex_add(state.ex_div(AbsConst(1), var01), state.ex_div(AbsConst(1), var))
  var02 = state.ex_div(AbsConst(1), denom)

  sum1 = state.ex_add(state.ex_div(mu01, var01), state.ex_div(x, var))
  mu02 = state.ex_mul(sum1, var02)

  mu1_new = state.ex_div(state.ex_add(mu02, state.ex_mul(AbsConst(-1), b)), a)
  var1_new = state.ex_div(var02, state.ex_mul(a, a))

  return AbsNormal(mu1_new, var1_new)

# Returns (marginal, posterior) distributions
def gaussian_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsNormal, AbsNormal]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsNormal(mu0, var0), AbsNormal(mu, var):
      marginal = gaussian_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gaussian_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Bernoulli Conjugate ###
    
def bernoulli_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                              rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBernoulli(p1), AbsBernoulli(p2):
      return p2.depends_on(rv_par, False) and \
            not p1.depends_on(rv_child, True)
    case _:
      return False
    
def bernoulli_marginal(state: AbsSymState, prior: AbsBernoulli, likelihood: AbsBernoulli, 
                       rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBernoulli]:
  if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  p1, p2 = prior.p, likelihood.p
  p2_new = state.ex_add(state.ex_mul(p1, state.eval(p2.subst_rv(rv_par, AbsConst(True)))),
                        state.ex_mul(state.ex_add(AbsConst(1), state.ex_mul(AbsConst(-1), p1)),
                                    state.eval(p2.subst_rv(rv_par, AbsConst(False)))))
  return AbsBernoulli(p2_new)
    
def bernoulli_posterior(state: AbsSymState, prior: AbsBernoulli, likelihood: AbsBernoulli, 
                        rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsBernoulli]:
  if not bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  x = rv_child if obs is None else obs
  
  p1, p2 = prior.p, likelihood.p
  p2_new = state.ex_add(state.ex_mul(p1, state.eval(p2.subst_rv(rv_par, AbsConst(True)))),
                        state.ex_mul(state.ex_add(AbsConst(1), state.ex_mul(AbsConst(-1), p1)),
                                    state.eval(p2.subst_rv(rv_par, AbsConst(False)))))
  
  p1_num_sub = state.ex_ite(x, p2, state.ex_add(AbsConst(1), state.ex_mul(AbsConst(-1), p2)))
  p1_num = state.ex_mul(p1, state.eval(p1_num_sub.subst_rv(rv_par, AbsConst(True))))
  p1_denom = state.ex_ite(x, p2_new, state.ex_add(AbsConst(1), state.ex_mul(AbsConst(-1), p2_new)))
  p1_new = state.ex_div(p1_num, p1_denom)

  return AbsBernoulli(p1_new)
    
def bernoulli_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBernoulli, AbsBernoulli]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsBernoulli(p1), AbsBernoulli(p2):
      marginal = bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Beta Bernoulli Conjugate ###

def beta_bernoulli_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBeta(a, b), AbsBernoulli(p):
      return rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False
    
def beta_bernoulli_marginal(state: AbsSymState, prior: AbsBeta, likelihood: AbsBernoulli,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBernoulli]:
  if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  p_new = state.ex_div(a, state.ex_add(a, b))

  return AbsBernoulli(p_new)

def beta_bernoulli_posterior(state: AbsSymState, prior: AbsBeta, likelihood: AbsBernoulli,
                              rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsBeta]:
  if not beta_bernoulli_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b

  x = rv_child if obs is None else obs

  a_new = state.ex_add(a, state.ex_ite(x, AbsConst(1), AbsConst(0)))
  b_new = state.ex_add(b, state.ex_ite(x, AbsConst(0), AbsConst(1)))

  return AbsBeta(a_new, b_new)
    
def beta_bernoulli_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBernoulli, AbsBeta]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsBeta(a, b), AbsBernoulli(p):
      marginal = beta_bernoulli_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = beta_bernoulli_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Beta Binomial Conjugate ###

def beta_binomial_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsBeta(a, b), AbsBinomial(n, p):
      return isinstance(n, AbsConst) \
            and isinstance(a, AbsConst)\
            and rv_par == p \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False
    
def beta_binomial_marginal(state: AbsSymState, prior: AbsBeta, likelihood: AbsBinomial,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsBetaBinomial]:
  if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  n, p = likelihood.n, likelihood.p

  return AbsBetaBinomial(n, a, b)

def beta_binomial_posterior(state: AbsSymState, prior: AbsBeta, likelihood: AbsBinomial, 
                              rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsBeta]:
  if not beta_binomial_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  n, p = likelihood.n, likelihood.p

  assert isinstance(n, AbsConst)

  a_new = state.ex_add(a, rv_child)
  n_new = UnkC() if isinstance(n.v, UnkC) else float(n.v)
  b_new = state.ex_add(b, state.ex_add(AbsConst(n_new), state.ex_mul(AbsConst(-1), rv_child)))

  return AbsBeta(a_new, b_new)
    
def beta_binomial_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsBetaBinomial, AbsBeta]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsBeta(a, b), AbsBinomial(n, p):
      marginal = beta_binomial_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = beta_binomial_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Gamma Poisson Conjugate ###

def gamma_poisson_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  match prior, likelihood:
    case AbsGamma(a, b), AbsPoisson(l):
      return isinstance(a, AbsConst) \
            and (isinstance(a.v, UnkC) or bool(np.isclose(round(a.v), a.v))) \
            and rv_par == l \
            and not b.depends_on(rv_child, True)
    case _:
      return False

def gamma_poisson_marginal(state: AbsSymState, prior: AbsGamma, likelihood: AbsPoisson,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsNegativeBinomial]:
  if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  l = likelihood.l

  assert isinstance(a, AbsConst)

  n_new = AbsConst(UnkC()) if isinstance(a.v, UnkC) else AbsConst(int(a.v))
  p_new = state.ex_div(b, state.ex_add(AbsConst(1), b))

  return AbsNegativeBinomial(n_new, p_new)

def gamma_poisson_posterior(state: AbsSymState, prior: AbsGamma, likelihood: AbsPoisson,
                              rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsGamma]:
  if not gamma_poisson_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b

  x = rv_child if obs is None else obs

  a_new = state.ex_add(a, x)
  b_new = state.ex_add(b, AbsConst(1))

  return AbsGamma(a_new, b_new)
    
def gamma_poisson_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsNegativeBinomial, AbsGamma]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsGamma(a, b), AbsPoisson(l):
      marginal = gamma_poisson_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gamma_poisson_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Gamma Normal Conjugate ###
# TODO: case of var being scaled invgamma

def gamma_normal_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  
  match prior, likelihood:
    case AbsGamma(a, b), AbsNormal(mu, var):
      return isinstance(mu, AbsConst) \
            and var == state.ex_div(AbsConst(1), rv_par) \
            and not a.depends_on(rv_child, True) \
            and not b.depends_on(rv_child, True)
    case _:
      return False

def gamma_normal_marginal(state: AbsSymState, prior: AbsGamma, likelihood: AbsNormal,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsStudentT]:
  if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  mu, var = likelihood.mu, likelihood.var

  tau2 = state.ex_div(b, a)
  nu = state.ex_mul(AbsConst(2), a)

  return AbsStudentT(mu, tau2, nu)

def gamma_normal_posterior(state: AbsSymState, prior: AbsGamma, likelihood: AbsNormal,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsGamma]:
  if not gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  a, b = prior.a, prior.b
  mu, var = likelihood.mu, likelihood.var

  x = rv_child if obs is None else obs

  assert isinstance(mu, AbsConst)

  a_new = state.ex_add(a, AbsConst(0.5))
  b_inner = state.ex_add(x, AbsConst(-mu.v))
  b_new = state.ex_add(b, state.ex_mul(AbsConst(0.5), state.ex_mul(b_inner, b_inner)))

  return AbsGamma(a_new, b_new)
    
def gamma_normal_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsStudentT, AbsGamma]]:
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsGamma(a, b), AbsNormal(mu, var):
      marginal = gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None
    
### Normal-Inverse-Gamma Normal Conjugate ###

def normal_inverse_gamma_normal_conjugate_check(state: AbsSymState, prior: AbsSymDistr, likelihood: AbsSymDistr,
                                    rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> bool:
  
  match prior, likelihood:
    case AbsNormal(mu0, var1), AbsNormal(mu, var2):
      match var2:
        case Div(AbsConst(1), var2_inner):
          if isinstance(var2_inner, AbsRandomVar):
            # TODO: should this be passed in
            match state.distr(var2_inner):
              case AbsGamma(a, b):
                # var1 should be the invgamma scaled by 1/lambda
                k = is_scaled(state, var1, var2)
                if k is None:
                  return False
                else:
                  match state.eval(k):
                    case AbsConst(0):
                      return False
                    case AbsConst(_):
                      return isinstance(mu0, AbsConst) \
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
    
def normal_inverse_gamma_normal_marginal(state: AbsSymState, prior: AbsNormal, likelihood: AbsNormal,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[AbsStudentT]:
  if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var1 = prior.mu, prior.var
  mu, var2 = likelihood.mu, likelihood.var

  match var2:
    case Div(AbsConst(1), var2_inner):
      if isinstance(var2_inner, AbsRandomVar):
        match state.distr(var2_inner):
          case AbsGamma(a, b):
            k = is_scaled(state, var1, var2)
            assert k is not None
            assert isinstance(mu0, AbsConst)

            lam = state.ex_div(AbsConst(1), k)

            a_new = state.ex_add(a, AbsConst(0.5))
            b_inner = state.ex_add(rv_child, AbsConst(-mu0.v))
            b_new = state.ex_add(b, state.ex_mul(state.ex_div(lam, state.ex_div(lam, AbsConst(1))), state.ex_div(state.ex_mul(b_inner, b_inner), AbsConst(2))))

            state.set_distr(var2_inner, AbsGamma(a_new, b_new))


            mu_new = mu0
            tau2_new = state.ex_div(state.ex_mul(b, state.ex_add(lam, AbsConst(1))), state.ex_mul(a, lam))
            nu_new = state.ex_mul(AbsConst(2), a)

            return AbsStudentT(mu_new, tau2_new, nu_new)
          case _:
            return None
      else:
        return None
    case _:
      return None
    
def normal_inverse_gamma_normal_posterior(state: AbsSymState, prior: AbsNormal, likelihood: AbsNormal,
                            rv_par: AbsRandomVar, rv_child: AbsRandomVar, obs: Optional[AbsSymExpr]=None) -> Optional[AbsNormal]:
  if not normal_inverse_gamma_normal_conjugate_check(state, prior, likelihood, rv_par, rv_child):
    return None
  
  mu0, var1 = prior.mu, prior.var
  mu, var2 = likelihood.mu, likelihood.var

  x = rv_child if obs is None else obs

  match var2:
    case Div(AbsConst(1), var2_inner):
      # var2 must be a random variable of invgamma
      if isinstance(var2_inner, AbsRandomVar):
        match state.distr(var2_inner):
          case AbsGamma(a, b):
            k = is_scaled(state, var1, var2)
            assert k is not None
            assert isinstance(mu0, AbsConst)

            lam = state.ex_div(AbsConst(1), k)

            mu0_new = state.ex_div(state.ex_add(state.ex_mul(lam, mu0), x), state.ex_add(lam, AbsConst(1)))
            lam_new = state.ex_add(lam, AbsConst(1))
            
            a_new = state.ex_add(a, AbsConst(0.5))
            b_inner = state.ex_add(x, AbsConst(-mu0.v))
            b_new = state.ex_add(b, state.ex_mul(state.ex_div(lam, state.ex_div(lam, AbsConst(1))), state.ex_div(state.ex_mul(b_inner, b_inner), AbsConst(2))))

            state.set_distr(var2_inner, AbsGamma(a_new, b_new))

            var_new = state.ex_div(AbsConst(1), state.ex_mul(lam_new, var2_inner))

            return AbsNormal(mu0_new, var_new)
          case _:
            return None
      else:
        return None
    case _:
      return None
            
    
def normal_inverse_gamma_normal_conjugate(state: AbsSymState, rv_par: AbsRandomVar, rv_child: AbsRandomVar) -> Optional[Tuple[AbsStudentT, AbsNormal]]:
  # print('normal_inverse_gamma_normal_conjugate', rv_par, rv_child)
  prior, likelihood = state.distr(rv_par), state.distr(rv_child)
  match prior, likelihood:
    case AbsNormal(mu0, var1), AbsNormal(mu, var2):
      marginal = normal_inverse_gamma_normal_marginal(state, prior, likelihood, rv_par, rv_child)
      posterior = normal_inverse_gamma_normal_posterior(state, prior, likelihood, rv_par, rv_child)
      if marginal is None or posterior is None:
        return None
      return (marginal, posterior)
    case _:
      return None