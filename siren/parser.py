from typing import Any, Optional, List
import lark

from siren.grammar import *

parser = lark.Lark(r'''
start: func* expression -> program
                   
func: "val" NAME "=" "fun" patternlist "->" expression "in"
                   
expression: 
  | "true" -> true
  | "false" -> false
  | "(" ")" -> nil
  | NUMBER -> number
  | INTEGER -> integer
  | STRING -> string
  | NAME -> variable
  | "(" expression ("," expression)* ")" -> pair
  | "let" patternlist "=" expression "in" expression -> let
  | ops -> ops
  | identifier args -> apply
  | "if" expression "then" expression "else" expression -> ifelse
  | "fold" "(" identifier "," expression "," expression ")" -> fold
  | "let" rvpattern "<-" expression "in" expression -> letrv
  | "observe" "(" expression "," expression ")" -> observe
  | "resample" "(" ")" -> resample
  | list -> list
                   
args:
  | "(" ")" -> nil
  | "(" expression ("," expression)* ")" -> expressionlist
                   
list: 
  | "[" "]" -> nil
  | "nil" -> nil
  | "[" expression ("," expression)* "]" -> list

rvpattern:
  | "sample" identifier -> sample
  | "symbolic" identifier -> symbolic
  | identifier -> identifier
                   
ops:
  | expression "+" expression -> add
  | expression "-" expression -> sub
  | expression "*" expression -> mul
  | expression "/" expression -> div
  | expression "::" expression -> cons
  | expression "=" expression -> eq
  | expression "<" expression -> lt

patternlist:
  | "(" ")" -> nil
  | "(" patternlist ")" -> paren
  | pattern -> pattern
  | pattern ("," pattern)+ -> patternlist
                   
pattern:
  | NAME -> identifier
  | "_" -> wildcard
  | "(" ")" -> nil
  | "(" pattern ")" -> paren
  | "(" pattern ("," pattern)+ ")" -> patternlist
                   
identifier: 
  | NAME -> ident
  | NAME "." NAME -> module
                   
%import common.SH_COMMENT -> COMMENT        
%ignore COMMENT

BLOCK_COMMENT: "(*" /((?!\*\)).|\n)+/ "*)"
%ignore BLOCK_COMMENT

%import common.CNAME -> NAME
%import common.ESCAPED_STRING -> STRING
%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> INTEGER
%import common.WS
%ignore WS
''')

# Parse Siren program (string) into AST
def parse_program(program: str) -> Program:
  parse_tree = parser.parse(program + '\n')

  # Makes an identifier, which can have a Module name
  def _make_identifier(x: Any) -> Identifier:
    if x.data == "ident":
      return Identifier(None, str(x.children[0].value))
    elif x.data == "module":
      module, name = x.children
      return Identifier(str(module), str(name))
    else:
      raise ValueError(x)

  # Parse Annotation and Identifier
  def _make_annot_ident(x: Any) -> Tuple[Optional[Annotation], Identifier]:
    if x.data == "sample":
      return Annotation.sample, _make_identifier(x.children[0])
    elif x.data == "symbolic":
      return Annotation.symbolic, _make_identifier(x.children[0])
    elif x.data == "identifier":
      return None, _make_identifier(x.children[0])
    else:
      raise ValueError(x)

  # Parse a pattern for variable binding as a list of identifiers
  def _make_pattern(x: Any) -> List[Any]:
    if x.data == "paren":
      return _make_pattern(x.children[0])
    elif x.data == "patternlist":
      return [
        _make_pattern(e) for e in x.children
      ]
    elif x.data == "nil":
      return []
    elif x.data == "identifier":
      return [Identifier(None, str(x.children[0].value))]
    elif x.data == "wildcard":
      return [Identifier(None, None)]
    elif x.data == "pattern":
      return _make_pattern(x.children[0])
    else:
      raise ValueError(x)
    
  # Prase arguments into list of expressions
  def _make_args(x: Any) -> List[Expr]:
    if x.data == "expressionlist":
      return [_make_expression(e) for e in x.children]
    elif x.data == "nil":
      return []
    else:
      raise ValueError(x)
    
  # Parse Pair expressions, which is allowed to be flattened
  def _make_pairs(x: List[Any]) -> Expr:
    if len(x) == 0:
      raise ValueError(x)
    elif len(x) == 1:
      return _make_expression(x[0])
    else:
      return GenericOp(Operator.pair, [
        _make_expression(x[0]),
        _make_pairs(x[1:]),
      ])

  def _make_list(x: Any) -> Expr:
    if x.data == "list":
      def _make_list_helper(x: List[Any]) -> Expr:
        if len(x) == 0:
          return GenericOp(Operator.lst, [])
        else:
          return GenericOp(Operator.cons, [
            _make_expression(x[0]),
            _make_list_helper(x[1:]),
          ])
      return _make_list_helper(x.children)
    elif x.data == "nil":
      return GenericOp(Operator.lst, [])
    else:
      raise ValueError(x.data)

  # Parse built in operators
  def _make_ops(x: Any) -> Op:
    # We represent as GenericOp first. Will be converted to the 
    # correct operator when we evaluate the program
    if x.data == "add":
      left, right = x.children
      return GenericOp(Operator.add, [_make_expression(left), _make_expression(right)])
    elif x.data == "sub":
      left, right = x.children
      return GenericOp(Operator.sub, [_make_expression(left), _make_expression(right)])
    elif x.data == "mul":
      left, right = x.children
      return GenericOp(Operator.mul, [_make_expression(left), _make_expression(right)])
    elif x.data == "div":
      left, right = x.children
      return GenericOp(Operator.div, [_make_expression(left), _make_expression(right)])
    elif x.data == "eq":
      left, right = x.children
      return GenericOp(Operator.eq, [_make_expression(left), _make_expression(right)])
    elif x.data == "lt":
      left, right = x.children
      return GenericOp(Operator.lt, [_make_expression(left), _make_expression(right)])
    elif x.data == "apply":
      identifier, args = x.children
      identifier = _make_identifier(identifier)
      if identifier.module is None and identifier.name in Operator.__members__:
        assert identifier.name is not None
        return GenericOp(Operator[identifier.name], _make_args(args))
      raise ValueError(x)
    else:
      raise ValueError(x.data)

  # Parse expressions
  def _make_expression(x: Any) -> Expr:
    if x.data == "number":
      return Const(float(x.children[0].value))
    elif x.data == "integer":
      return Const(int(x.children[0].value))
    elif x.data == "string":
      return Const(str(x.children[0].value.strip('"')))
    elif x.data == "nil":
      return Const(None)
    elif x.data == "true":
      return Const(True)
    elif x.data == "false":
      return Const(False)
    elif x.data == 'variable':
      identifier, = x.children
      return Identifier(None, str(identifier))
    elif x.data == "pair":
      return _make_pairs(x.children)
    elif x.data == "ops":
      return _make_ops(x.children[0])
    elif x.data == "apply":
      identifier, args = x.children
      identifier = _make_identifier(identifier)
      # Check if the identifier is a built-in operator, otherwise it is a function application
      if identifier.module is None and identifier.name in Operator.__members__:
        assert identifier.name is not None
        return GenericOp(Operator[identifier.name], _make_args(args))
      return Apply(identifier, _make_args(args))
    elif x.data == "ifelse":
      cond, true, false = x.children
      return IfElse(
          _make_expression(cond),
          _make_expression(true),
          _make_expression(false),
      )
    elif x.data == "let":
      pattern, value, body = x.children
      return Let(_make_pattern(pattern), _make_expression(value), _make_expression(body))
    elif x.data == "fold":
      identifier, init, expression = x.children
      return Fold(
          _make_identifier(identifier),
          _make_expression(init),
          _make_expression(expression),
      )
    elif x.data == "letrv":
      rv_pattern, distribution, expression = x.children
      annotation, identifier = _make_annot_ident(rv_pattern)
      return LetRV(
          identifier,
          annotation,
          _make_ops(distribution),
          _make_expression(expression),
      )
    elif x.data == "observe":
      distribution, value = x.children
      # Observe has to take a distribution (a built in operator) as the first argument
      return Observe(_make_ops(distribution), _make_expression(value))
    elif x.data == "resample":
      return Resample()
    elif x.data == "list":
      return _make_list(x.children[0])
    elif x.data == "expression":
      return _make_expression(x.children[0])
    else:
      raise ValueError(x.data)

  # parse functions
  functions = []

  for i in range(len(parse_tree.children) - 1):
    func = parse_tree.children[i]

    assert isinstance(func, lark.Tree)
    name, pattern, expression = func.children

    # Functions do not have module names
    functions.append(Function(
        Identifier(None, str(name)),
        _make_pattern(pattern),
        _make_expression(expression)
    ))

  expression = _make_expression(parse_tree.children[-1])

  return Program(functions, expression)
