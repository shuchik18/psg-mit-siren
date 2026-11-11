from typing import Dict, List, Tuple, Any, Set
from copy import copy, deepcopy

from siren.grammar import *

# Fast copies a dictionary of dictionaries or lists
# May not work for all symbolic states
def fast_copy(d: Dict[Any, Any]) -> Dict[Any, Any]:
  new_d = {}
  for k, v in d.items():
    if isinstance(v, dict):
      new_d[k] = fast_copy(v)
    elif isinstance(v, list):
      new_d[k] = copy(v)
    else:
      new_d[k] = v
  return new_d

def get_lst(l: Expr[SymExpr]) -> List[SymExpr]:
  match l:
    case Lst(exprs):
      return exprs
    case Const(exprs):
      if isinstance(exprs, list):
        return [Const(e) for e in exprs]
      else:
        raise ValueError(exprs)
    case _:
      raise ValueError(l)
    
def is_lst(expr: SymExpr) -> bool:
  match expr:
    case Lst(_):
      return True
    case Const(value):
      return isinstance(value, list)
    case _:
      return False
    
def get_pair(p: Expr[SymExpr]) -> Tuple[SymExpr, SymExpr]:
  match p:
    case Pair(fst, snd):
      return fst, snd
    case Const(exprs):
      if isinstance(exprs, Tuple):
        a, b = exprs
        return Const(a), Const(b)
      else:
        raise ValueError(exprs)
    case _:
      raise ValueError(p)

def is_pair(expr: SymExpr) -> bool:
  match expr:
    case Pair(_, _):
      return True
    case Const(value):
      return isinstance(value, tuple)
    case _:
      return False
    
def is_abs_lst(expr: AbsSymExpr) -> bool:
  match expr:
    case AbsLst(_):
      return True
    case AbsConst(value):
      return isinstance(value, list)
    case UnkE(_):
      return True
    case _:
      return False

def get_abs_lst(l: Expr[AbsSymExpr]) -> List[AbsSymExpr] | UnkE:
  match l:
    case AbsLst(exprs):
      return exprs
    case AbsConst(exprs):
      if isinstance(exprs, list):
        return [AbsConst(e) for e in exprs]
      else:
        raise ValueError(exprs)
    case UnkE(parents):
      return UnkE(parents)
    case _:
      raise ValueError(l)
    
def is_abs_pair(expr: AbsSymExpr) -> bool:
  match expr:
    case AbsPair(_, _):
      return True
    case AbsConst(value):
      return isinstance(value, tuple)
    case UnkE(_):
      return True
    case _:
      return False
    
def get_abs_pair(p: Expr[AbsSymExpr]) -> Tuple[AbsSymExpr, AbsSymExpr]:
  match p:
    case AbsPair(fst, snd):
      return fst, snd
    case AbsConst(exprs):
      if isinstance(exprs, Tuple):
        a, b = exprs
        return AbsConst(a), AbsConst(b)
      else:
        raise ValueError(exprs)
    case UnkE(parents):
      return UnkE(parents), UnkE(parents)
    case _:
      raise ValueError(p)
    
def match_rvs(unify_mapping: Dict[AbsRandomVar, Set[AbsRandomVar]], 
                e1: AbsSymExpr, e2: AbsSymExpr) -> Dict[AbsRandomVar, Set[AbsRandomVar]]:
  match e1, e2:
    case AbsRandomVar(_), AbsRandomVar(_):
      # e2 is already renamed 
      if e2 in unify_mapping:
        unify_mapping[e2].add(e1)
      else:
        unify_mapping[e2] = {e1}
      return unify_mapping
    case AbsAdd(e11, e12), AbsAdd(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case AbsMul(e11, e12), AbsMul(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case AbsDiv(e11, e12), AbsDiv(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case AbsIte(cond1, true1, false1), AbsIte(cond2, true2, false2):
      unify_mapping = match_rvs(unify_mapping, cond1, cond2)
      unify_mapping = match_rvs(unify_mapping, true1, true2)
      return match_rvs(unify_mapping, false1, false2)
    case AbsEq(e11, e12), AbsEq(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case AbsLt(e11, e12), AbsLt(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case AbsLst(es1), AbsLst(es2):
      for e1, e2 in zip(es1, es2):
        unify_mapping = match_rvs(unify_mapping, e1, e2)
      return unify_mapping
    case AbsPair(e11, e12), AbsPair(e21, e22):
      unify_mapping = match_rvs(unify_mapping, e11, e21)
      return match_rvs(unify_mapping, e12, e22)
    case _, _:
      return unify_mapping
