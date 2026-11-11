# System Description

This file provides an overview of the source files in this system, as well as instructions on how to extend Siren with other hybrid inference algorithms.

## Overview
Here is a summary of subdirectories and important source files.

| Path                                        | Description |
| --------                                    | ------- |
| [`siren/siren.py`](siren/siren.py)          | Entrypoint for the Siren compiler.    |
| [`siren/parser.py`](siren/parser.py)        | Parses a Siren program into Siren AST. |
| [`siren/grammar.py`](siren/grammar.py)      | Defines Siren grammar for the AST. |
| [`siren/evaluate.py`](siren/evaluate.py)    | Evaluates a parsed Siren AST using a specified hybrid inference algorithm with the specified approximate inference algorithm. |
| [`siren/analyze.py`](siren/analyze.py)      | Analyzes a parsed Siren AST for a specified inference algorithm.    |
| [`siren/inference`](siren/inference/)       | This subdirectory contains shared code for inference and the algorithm-specific implementations of the hybrid inference interface. |
| [`siren/inference/interface.py`](siren/inference/interface.py)       | Defines shared code used for inference, and the `SymState` interface each hybrid inference algorithm must subclass and implement. |
| [`siren/analysis`](siren/analysis/)       | This subdirectory contains shared code for the inference plan satisfiability analysis and the algorithm-specific implementations of the abstract hybrid interface. |
| [`siren/analysis/interface.py`](siren/analysis/interface.py)       | Defines shared code used for inference, and the `AbsSymState` interface that must subclassed and implemented for each hybrd inference algorithm.    |
| [`benchmarks/`](benchmarks/)       | Contains benchmarks, benchmarking harness, and visualization code.  |

## Extending Siren with Other Hybrid Inference Algorithm

Siren can be extended with other hybrid inference algorithms that can implement the hybrid inference interface. 

### Implement the hybrid inference interface
First, the `SymState` class in [`siren/inference/interface.py`](siren/inference/interface.py) must be subclassed in a file within the [`siren/inference/`](siren/inference/) subdirectory. The following functions must be implemented by the subclass:
- `assume`: Inserts a random variable into the symbolic state with the given distribution and annotation.
- `observe`: Conditions the given random variable on the given constant value, returning the score.
- `value_impl`: Implements the Value operation. Samples the given random variable, returning the resulting constant value.
- `marginalize`: Marginalizes the given random variable.
See other files in the subdirectory for reference, such as [`siren/inference/ssi.py`](siren/inference/ssi.py) which defines the `SSISymState` class that implements semi-symbolic inference.

If the inference algorithm uses the implementation-specific data field in the symbolic state, the field can be accessed and mutated with `self.get_entry(rv, FIELDNAME)`, where FIELDNAME is the named alias for the field. See [`siren/inference/ds.py`](siren/inference/ds.py) or [`siren/inference/bp.py`](siren/inference/bp.py) for examples of this.

The subclass must be added to [`siren/inference/__init__.py`](siren/inference/__init__.py):
```python
from .subclass import SubState
```

### Implement the abstract inference interface
Similar to the previous step, the `AbsSymState` class in [`siren/analysis/interface.py`](siren/analysis/interface.py) must be subclassed in a file within the [`siren/analysis/`](siren/analysis/) subdirectory. The following functions must be implemented by the subclass:
- `assume`: Inserts an abstract random variable into the abstract symbolic state with the given abstract distribution and annotation.
- `observe`: Simulates conditioning the given abstract random variable on the given abstract constant value. Does not score. 
- `inner_value`: Simulates sampling the given abstract random variable, returning the `AbsConst(UnkC())`. Does not need to implement checking distribution encoding annotation violation, which is handled by shared code in `AbsSymState` in `value_impl`.
- `marginalize`: Marginalizes the given abstract random variable.
See other files in the subdirectory for reference, such as [`siren/analysis/ssi.py`](siren/analysis/ssi.py) which defines the `AbsSSISymState` class that implements abstract semi-symbolic inference.

The subclass may need to override the following functions, due to the implementation-specific data field, but must call the superclass function:
- `entry_referenced_rvs`: Returns the random variables referenced in the symbolic state of the given random variables.
- `entry_rename_rv`: Renames the given random variable to a different random variable name. 
- `entry_join`: Joins the entries of the given random variable in the two states.
- `set_dynamic`: Makes the given random variable have *top* entries, throwing `AnalysisViolatedAnnotationError` if any affected variable has an annotations.

See the other files for reference.

The subclass must be added to [`siren/analysis/__init__.py`](siren/analysis/__init__.py):
```python
from .subclass import AbsSubState
```

### Expose the inference algorithm as an inference option
Finally, in [`siren/siren.py`](siren/siren.py), import the classes implemented in the previous two steps.
```python
from .inference import SubState
from .analysis import AbsSubState
```

Add the subclasses to the `method_states` dictionary on [Line 15](siren/siren.py#L15), giving it an alias for the command line:
```python
method_states = {
  "ssi": (SSIState, AbsSSIState),
  "ds": (DSState, AbsDSState),
  "bp": (BPState, AbsBPState),
  "new_alg": (SubState, AbsSubState),
}
```
and add the alias as an option to the `method` argument on [Line 38](siren/siren.py#L38):
```python
p.add_argument(
  "--method",
  "-m",
  type=str,
  default="ssi",
  choices=["ssi", "ds", "bp", "new_alg"],
)
```
and add it to the match-case on [Line 70](siren/siren.py#L70) for printing out the selected method:
```python
match args.method:
  case "ssi":
    print("SSI")
  case "ds":
    print("DS")
  case "bp":
    print("BP")
  case "new_alg":
    print("New Algorithm")
  case _:
    raise ValueError("Invalid method")
```

## Extending Siren with Other Approximate Inference Algorithm

Siren can also be extended with other Monte Carlo methods to be combined with the hybrid inference algorithms.

### Implement the probabilistic handler operations
First, the `Handler` class in [`siren/evaluate.py`](siren/evaluate.py) must be subclassed in that file. The following functions must be implemented by the subclass:
- `assume`: Gets a new random variable name and uses `Symstate.assume` to create a new random variable, calling `Symstate.value` if the variable is sampled.
- `observe`: Creates a new random variable using `Symstate.assume`, calling `Symstate.value_expr` on the conditioned value, and call `Symstate.observe` to condition the model, returning the score.
- `value`: Implements the `Symstate.value` function which returns the function that implements the `Value` function of the hybrid inference interface using `Symstate.value_impl`.
- `infer`: Implements the evaluation of the program using `evaluate_particle`.

### Implement the abstract probabilistic handler operations
Similar to the previous step, the `AbsHandler` class in [`siren/analyze.py`](siren/analyze.py) must be subclassed in that file. The following functions must be implemented by the subclass:
- `assume`: Gets a new random variable name and uses `Symstate.assume` to create a new random variable, calling `Symstate.value` if the variable is sampled.
- `observe`: Creates a new random variable using `Symstate.assume`, calling `Symstate.value_expr` on the conditioned value, and call `Symstate.observe` to condition the model.
- `value`: Implements the `Symstate.value` function which returns the function that implements the `Value` function of the hybrid inference interface. 

### Expose the handlers as options
Finally, in [`siren/siren.py`](siren/siren.py), import the classes implemented in the previous two steps.
```python
from .evaluate import SubHandler
from .analyze import AbsSubHandler
```

Add the subclasses to the `handlers` dictionary on [Line 21](siren/siren.py#L21), giving it an alias for the command line:
```python
handlers = {
  'smc': (SMC, AbsSMC),
  'mh': (MH, AbsMH),
  "new_alg": (SubHandler, AbsSubHandler),
}
```
and add the alias as an option to the `handler` argument on [Line 37](siren/siren.py#L37):
```python
p.add_argument(
  "--handler",
  "-l",
  type=str,
  default="smc",
  choices=["smc", "mh", "new_alg"],
)
```
and add it to the match-case on [Line 57](siren/siren.py#L57) for printing out the selected method:
```python
match args.handler:
  case "smc":
    print("SMC")
  case "mh":
    print("MH")
  case "new_alg":
    print("New Algorithm")
  case _:
    raise ValueError("Invalid handler")
```