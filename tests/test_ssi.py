import pytest

from siren.grammar import *
from siren.evaluate import SMC
import siren.inference.conjugate as conj
from siren.inference.ssi import SSIState
from siren.inference.interface import Particle

def test_beta_bernoulli():
  state = SSIState(lambda state: state.value_impl)

  rv1 = state.new_var()
  rv1 = state.assume(rv1, Identifier(None, "beta_prior"), None, Beta(Const(1), Const(1)))
  assert isinstance(rv1, RandomVar)
  
  rv2 = state.assume(state.new_var(), Identifier(None, "bernoulli"), None, Bernoulli(rv1))
  assert isinstance(rv2, RandomVar)
  state.observe(rv2, Const(True))
  rv3 = state.assume(state.new_var(), Identifier(None, "bernoulli"), None, Bernoulli(rv1))
  assert isinstance(rv3, RandomVar)
  state.observe(rv3, Const(True))

  match state.distr(rv1):
    case Beta(a, b):
      match state.eval(a), state.eval(b):
        case Const(a), Const(b):
          print(f'Beta-Bernoulli posterior: ({a} {b})')
          assert a == 3
          assert b == 1
        case _:
          assert False
    case _:
      assert False

def affine_init():
  state = SSIState(lambda state: state.value_impl)

  rv = state.new_var()
  rv = state.assume(rv, Identifier(None, "var"), None, Normal(Const(0), Const(1)))
  assert isinstance(rv, RandomVar)
  
  return state, rv

def test_affine1():
  state, rv = affine_init()

  expr1 = Add(Mul(Const(2), rv), Const(3))

  match conj.is_affine(state, expr1, rv):
    case (a, b):
      assert state.eval(a) == Const(2)
      assert state.eval(b) == Const(3)
    case None:
      assert False

def test_affine2():
  state, rv = affine_init()

  expr2 = Add(rv, rv)

  match conj.is_affine(state, expr2, rv):
    case (a, b):
      assert state.eval(a) == Const(2)
      assert state.eval(b) == Const(0)
    case _:
      assert False

def test_affine3():
  state, rv = affine_init()

  expr3 = Add(Mul(rv, Const(2)), Add(Mul(rv, Const(4)), Const(5)))

  match conj.is_affine(state, expr3, rv): 
    case (a, b):
      assert state.eval(a) == Const(6)
      assert state.eval(b) == Const(5)
    case _:
      assert False

def test_gaussian_conjugate():
  state = SSIState(lambda state: state.value_impl)

  rv1 = state.new_var()
  rv1 = state.assume(rv1, Identifier(None, "gaussian_rv1"), None, Normal(Const(1), Const(100)))
  assert isinstance(rv1, RandomVar)

  rv2 = state.new_var()
  rv2 = state.assume(rv2, Identifier(None, "gaussian_rv2"), None, Normal(Mul(Const(3), rv1), Const(1)))
  assert isinstance(rv2, RandomVar)

  match conj.gaussian_conjugate(state, rv1, rv2):
    case (marginal, posterior):
      print(state.eval_distr(marginal), state.eval_distr(posterior))
      match marginal, posterior:
        case Normal(mu1, sigma1), Normal(mu2, sigma2):
          assert state.eval(mu1) == Const(3)
          assert state.eval(sigma1) == Const(901)

          mu_new, var_new = 1 * 3, 100 * (3 ** 2)
          var_new2 = 1. / ((1./var_new) + (1 / 1))
          a2, b2 = var_new2 * (1 / 1), var_new2 * (mu_new / var_new)
          a3, b3, var_new3 = a2 / 3, b2/ 3, var_new2 / (3 ** 2)

          assert state.eval(sigma2) == Const(var_new3)
          
          match conj.is_affine(state, mu2, rv2):
            case (a, b):
              assert state.eval(a) == Const(a3)
              assert state.eval(b) == Const(b3)

              print(f'Gaussian prior -- variance = {var_new3}, a = {a3}, b = {b3}')

            case _:
              assert False
        case _:
          assert False
    case _:
      assert False

if __name__ == '__main__':
  pytest.main()