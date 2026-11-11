import argparse
import os
import json
import subprocess
import time
from math import log10, log
import glob
import itertools
import numpy as np
import csv
import sys
import pandas as pd
import ast
import tqdm
import wandb
import re

BENCHMARK_DIR = 'benchmarks'

DEFAULT_BENCHMARKS = [
  'envnoise',
  'noise',
  'outlier',
  'gtree',
  'outlierheavy',
  'slds',
  'runner',
  'radar',
  'wheels',
  'slam',
  'aircraft',
]

DEFAULT_METHODS = [
  'ssi',
  'ds',
  'bp',
]

DEFAULT_HANDLERS = [
  'smc',
  'mh',
]

N_INTERVALS = 30

INC = 1

CWD = '..'

LOGGING = False

def squared_error(true_x, x):
  return (true_x - x) ** 2

def cross_entropy(true_x, x):
  return -true_x * log(int(x) + 1e-10) - (1 - true_x) * log(1 - int(x) + 1e-10)

def close_to_target(target, value):
  if value == 0:
    return True
  if log10(value) - log10(target) <= 0.5:
    return True

  return False
  
def close_to_target_error(target_error, program_output):
  checks = []
  for var, error in program_output.items():
    if not close_to_target(target_error[var], error):
      checks.append(False)
    else:
      checks.append(True)
    
  return checks

def close_to_target_runtime(target_runtime, runtime):
  if log10(runtime) - log10(target_runtime) <= 0.1:
    return True

  return False

def is_satisfiable(annotation, distr_enc):
  if annotation == 'dynamic':
    return True
  elif annotation == 'symbolic':
    return distr_enc == 'symbolic'
  elif annotation == 'sample':
    return distr_enc == 'sample'
  else:
    raise Exception(f'Invalid annotation: {annotation}')

def get_plan_id(file):
  if os.path.basename(file) == 'baseline.si':
    return '-1'
  try:
    return str(int(os.path.basename(file)[4:-3]))
  except Exception:
    return None
  
def flatten_nested(structure):
  flattened_list = []
  for element in structure:
    if isinstance(element, tuple):
      flattened_list.extend(flatten_nested(element))
    elif isinstance(element, list):
      flattened_list.append(element)
    else:
      flattened_list.append([element])
  return flattened_list

# Runs benchmark executable and computes the absolute error of each variable
def run_siren(benchmark, file, handler, method, true_vars, error_func, data_file, raw, timeout, kwargs):
  # replace the file read with the correct datafile
  # File.read("data/processed_data.csv"))
  # string could be anything
  pattern = re.compile(r'File\.read\("(.*?)"\)')
  with open(os.path.join(CWD, file), 'r') as f:
    data = f.read()
    original_datafile = pattern.search(data).group(1)
    data = data.replace(f'File.read("{original_datafile}")', f'File.read("{data_file}")')
  with open(os.path.join(CWD, file), 'w') as f:
    f.write(data)
  
  # run siren file
  cmd = f'siren {file} -m {method} -l {handler}'
  for k, v in kwargs.items():
    if v is not None:
      cmd += f' --{k} {v}'
  
  tqdm.tqdm.write('>' + cmd)

  try:
    out = subprocess.check_output(cmd, cwd=CWD, shell=True, stderr=subprocess.STDOUT, timeout=timeout).decode("utf-8")
  except subprocess.TimeoutExpired as e:
    tqdm.tqdm.write(f'Timeout: {file}')
    return None
  except subprocess.CalledProcessError as e:
    output = e.output.decode("utf-8")
    tqdm.tqdm.write(output)
    return None
  
  # restore original datafile
  with open(os.path.join(CWD, file), 'w') as f:
    f.write(data.replace(f'File.read("{data_file}")', f'File.read("{original_datafile}")'))

  program_output = {}

  # parse output
  lines = out.strip().split('\n')

  # get execution time
  eval_time = None
  for i, line in enumerate(lines):
    line = line.strip()
    if line == '===== Evaluation Time =====':
      eval_time = float(lines[i + 1])
      break
  if eval_time is None:
    raise RuntimeError('Evaluation time not found')

  # get outputs between ===== Result ===== and ===== Runtime Inference Plan =====
  output = ''
  for i, line in enumerate(lines):
    line = line.strip()
    if line == '===== Result =====':
      output = lines[i + 1]
      break

  # eg format (0.1782178217821782, (-11.323677393073108, 8.861234052059762))
  # or (0.1782178217821782, (-11.323677393073108, [0, 1, 2]))
  split_output = ast.literal_eval(output)

  # parse line into dict
  split_output = flatten_nested(split_output)

  original_error_fun = error_func

  for true_var, out in zip(true_vars, split_output):
    var, true_vals = true_var

    # exception, it's easier to do this here
    if benchmark == 'slds' and var == 's':
      error_func = 'ce'
    else:
      error_func = original_error_fun

    if error_func == 'mse':
      # compute MSE error
      error = 0
      for true_val, val in zip(true_vals, out):
        error += squared_error(true_val, val)
      error /= len(true_vals)
    elif error_func == 'ce':
      # compute cross entropy error
      error = 0
      for true_val, val in zip(true_vals, out):
        error += cross_entropy(true_val, val)
      error /= len(true_vals)
    else:
      raise ValueError('Invalid error function')

    program_output[var] = error

    if raw:
      program_output[var + '_raw'] = out

  return eval_time, program_output

# Run experiments for each given number of particles
def run_particles(benchmark, files, n, handlers, methods, plans, true_vars, results_file, error_func, data_file, raw, kwargs):
  if len(files) == 0:
    # If no files specified, get all files in programs directory
    files = []
    for file in os.listdir(os.path.join(benchmark, 'programs')):
      if file.endswith('.si'):
        files.append(file)

    # harness in benchmarks directory already
    for file in files:
      if not os.path.exists(os.path.join(benchmark, 'programs', os.path.basename(file))):
        raise Exception(f'File not found: {file}')

    files = sorted(files, key=lambda x: int(os.path.basename(x)[4:-3]))
    files = map(lambda x: os.path.join(BENCHMARK_DIR, benchmark, 'programs', os.path.basename(x)), files)
    files = list(files)

  all_files = files
  
  for handler in handlers:
    print(f'Running with {handler}...')
    for method in methods:
      print(f'Running {method}...')

      files = filter(lambda x: plans[get_plan_id(x)]["satisfiable"][handler][method], all_files)
      files = list(files)
      # print(files)

      for file in tqdm.tqdm(files, desc=f"Files", position=0, leave=True, total=len(files)):
        plan_id = get_plan_id(file)
        if plan_id is None:
          print(f'Invalid file: {file}')
          continue

        particles = kwargs.get('particles', kwargs.get('samples', None))
        seed = kwargs.get('seed', None)
        if particles is None:
          raise ValueError('Must specify particles or samples')
        
        for p in tqdm.tqdm(particles, desc=f"Particles/Samples", position=1, leave=False, total=len(particles)):
          # print(f'Running with {p} particles')
          for i in tqdm.tqdm(range(n), desc=f"Iteration", position=2, leave=False, total=n):
            # print(f'{plan_id} {method} - {p} particles - Run {i}')

            siren_kwargs = {
              'particles': p,
              'samples': p,
              'warmup': kwargs.get('warmup', 0),
              'seed': seed,
            }

            run_outputs = run_siren(benchmark, file, handler, method, true_vars[handler], error_func, data_file[handler], raw, kwargs.get('timeout', 300), siren_kwargs)
            if run_outputs is None:
              # timeout
              tqdm.tqdm.write(f'Timed out: {plan_id} {handler} {method} - {p} particles')
              t = -1
              program_output = {var[0]: -1 for var in true_vars[handler]}
              if raw:
                program_output = {var[0] + '_raw': -1 for var in true_vars[handler]}
            else:
              t, program_output = run_outputs

            if LOGGING:
              logging_output = {
                f'{handler}-{method}-plan{plan_id}-{p}-it': i,
              }

              wandb.log(logging_output)

            true_vars_list = [var[0] for var in true_vars[handler]]

            row = [
              plan_id,
              handler,
              method,
              p,
              t,
            ]
            row += [program_output[var] for var in true_vars_list]
            if raw:
              row += [program_output[var + '_raw'] for var in true_vars_list]
            
            # write results to csv
            with open(results_file, 'a') as f:
              writer = csv.writer(f)
              writer.writerow(row)
            if t == -1:
              break

def find_satisfiable_plans(benchmark, files, handlers, methods, plans, knowns, timeout):
  # print(plans)
  if len(files) == 0:
    # If no files specified, get all files in programs directory
    files = []
    for file in os.listdir(os.path.join(benchmark, 'programs')):
      if file.endswith('.si'):
        files.append(file)

    # harness in benchmarks directory already
    for file in files:
      if not os.path.exists(os.path.join(benchmark, 'programs', os.path.basename(file))):
        raise Exception(f'File not found: {file}')

    files = sorted(files, key=lambda x: int(os.path.basename(x)[4:-3]))
    files = map(lambda x: os.path.join(BENCHMARK_DIR, benchmark, 'programs', os.path.basename(x)), files)
    files = list(files)

  satisfiable_plans = {}

  # Get runtime inference plan
  print('MAKE SURE TO MANUALLY CHECK RUNTIME INFERENCE PLAN!!!\n')
  for handler in handlers:
    print(f'Running with {handler}...')
    satisfiable_plans[handler] = {}
    for method in methods:
      print(f'For {method}...')
      satisfiable_plans[handler][method] = []

      for file in files:
        plan_id = get_plan_id(file)

        print(f'Checking {file}...')

        # intractible cases
        if benchmark == 'slam' and method == 'ssi' and plan_id in ['0', '2']:
          satisfiable_plans[handler][method].append(plan_id)
          print('> Satisfiable')
          continue

        if knowns is not None:
          # Use known checks
          pre_check = False
          for var in knowns:
            if not is_satisfiable(plans[plan_id]['plan'][var], knowns[var]):
              print('> Not satisfiable')
              pre_check = True
              break
          if pre_check:
            continue

        # get analysis output
        cmd = f'siren {file} -p 10 --samples 10 -l {handler} -m {method}'
        print('>', cmd)
        try:
          out = subprocess.check_output(cmd, cwd=CWD, shell=True, stderr=subprocess.STDOUT, timeout=timeout).decode("utf-8")
        except subprocess.CalledProcessError as e:
          output = e.output.decode("utf-8")
          if 'RuntimeViolatedAnnotationError' in output:
            print('> Not satisfiable')
            continue
          else:
            print(output)
            raise RuntimeError()

        satisfiable_plans[handler][method].append(plan_id)
        print('> Satisfiable')    


        # parse output
        lines = out.strip().split('\n')

        # get outputs after ===== Runtime Inference Plan =====
        true_plan = {}
        start = False
        for line in lines:
          line = line.strip()
          if line == '===== Runtime Inference Plan =====':
            start = True
          elif start:
            if line != '':
              var, enc = line.split(': ')[:2]
              true_plan[var] = enc.strip()

        # compare annotated and real plans
        # only for the ones that show up
        for var in plans[plan_id]['plan']:
          if var in true_plan:
            if not is_satisfiable(plans[plan_id]['plan'][var], true_plan[var]):
              print()
              print(f'ERROR: {file}')
              print('Annotated:', plans[plan_id]['plan'])
              print('Real:', true_plan)
              print()
              break

  return satisfiable_plans

def analyze(benchmark, files, handlers, methods, variables, plans, results):
  if len(files) == 0:
    # If no files specified, get all files in programs directory
    files = []
    for file in os.listdir(os.path.join(benchmark, 'programs')):
      if file.endswith('.si'):
        files.append(file)

    # harness in benchmarks directory already
    for file in files:
      if not os.path.exists(os.path.join(benchmark, 'programs', os.path.basename(file))):
        raise Exception(f'File not found: {file}')

    files = sorted(files, key=lambda x: int(os.path.basename(x)[4:-3]))
    files = map(lambda x: os.path.join(BENCHMARK_DIR, benchmark, 'programs', os.path.basename(x)), files)
    files = list(files)

  # Number of plans
  results['n_plans'] = len(files)
  # Number of variables
  results['n_vars'] = len(variables)

  for handler in handlers:
    print(f'Running with {handler}...')
    handler_results = {}

    for method in methods:
      print(f'For {method}...')
      method_results = {}
      method_results['plan'] = {}

      satisfied_plans = {}
      for plan_id, plan_data in plans.items():
        if plan_data['satisfiable'][handler][method]:
          satisfied_plans[plan_id] = plan_data['plan']

      # print(f'Satisfied plans: {list(satisfied_plans.keys())}')
      # print(satisfied_plans)

      n_true_satisfied = len(satisfied_plans.keys())
      n_inferred_satisfied = 0

      for file in tqdm.tqdm(files, desc=f"Files", position=0, leave=True, total=len(files)):
        plan_id = get_plan_id(file)
        method_results['plan'][plan_id] = {}

        method_results['plan'][plan_id]['true_satisfied'] = plan_id in satisfied_plans

        # print(f'Analyzing {file}...')

        # get analysis output
        cmd = f'siren {file} -l {handler} -m {method} --analyze-only'
        # print('>', cmd)
        t1 = time.time()
        try:
          out = subprocess.check_output(cmd, cwd=CWD, shell=True, stderr=subprocess.STDOUT).decode("utf-8")
        except subprocess.CalledProcessError as e:
          t2 = time.time()
          output = e.output.decode("utf-8")
          method_results['plan'][plan_id]['infer_satisfied'] = False
          method_results['plan'][plan_id]['analysis_time'] = t2 - t1
          continue

        t2 = time.time()
        method_results['plan'][plan_id]['analysis_time'] = t2 - t1

        method_results['plan'][plan_id]['infer_satisfied'] = True
        n_inferred_satisfied += 1

        # # parse output
        lines = out.strip().split('\n')

        # analysis_time = -1
        # for i, line in enumerate(lines):
        #   line = line.strip()
        #   if line == '===== Analysis Time =====':
        #     analysis_time = float(lines[i + 1])
        #     break

        # if analysis_time == -1:
        #   raise RuntimeError('Analysis time not found')
        
        # method_results['plan'][plan_id]['analysis_time'] = analysis_time

        # get outputs after ===== Inferred Inference Plan =====
        inferred_plan = {}
        start = False
        for line in lines:
          line = line.strip()
          if line == '===== Inferred Inference Plan =====':
            start = True
          elif start:
            if '=====' in line:
              break
            if line != '':
              var, enc = line.split(': ')
              inferred_plan[var] = enc.strip()

        # double check annotated and real plans match since satisfied
        for var in variables:
          if var not in inferred_plan:
            print()
            print(f'ERROR: {file} missing {var}')
            print('Inferred:', inferred_plan)
          elif not is_satisfiable(satisfied_plans[plan_id][var], inferred_plan[var]):
            print()
            print(f'ERROR: {file}')
            print('Annotated:', satisfied_plans[plan_id])
            print('Inferred:', inferred_plan)
            print()
            break

      method_results['n_true_satisfied'] = n_true_satisfied
      method_results['n_inferred_satisfied'] = n_inferred_satisfied

      n_satisfied_tp = 0
      n_satisfied_fp = 0
      n_satisfied_tn = 0
      n_satisfied_fn = 0

      for file in files:
        plan_id = get_plan_id(file)
        inferred_satisfied = method_results['plan'][plan_id]['infer_satisfied']
        true_satisfied = plan_id in satisfied_plans

        if inferred_satisfied and true_satisfied:
          n_satisfied_tp += 1
        elif inferred_satisfied and not true_satisfied:
          n_satisfied_fp += 1
        elif not inferred_satisfied and true_satisfied:
          n_satisfied_fn += 1
        else:
          n_satisfied_tn += 1

      method_results['n_satisfied_tp'] = n_satisfied_tp
      method_results['n_satisfied_fp'] = n_satisfied_fp
      method_results['n_satisfied_fn'] = n_satisfied_fn
      method_results['n_satisfied_tn'] = n_satisfied_tn

      handler_results[method] = method_results

    results[handler] = handler_results
  return results

def run_benchmark(benchmark, output, n, handlers, methods, files, error_func, raw, kwargs):
  outdir = os.path.join(benchmark)
  os.makedirs(outdir, exist_ok=True)

  with open(os.path.join(benchmark, 'config.json')) as f:
    config = json.load(f)  
  
  true_vars = config['true_vars']
  data_file = config['data_file']

  results_file = os.path.join(benchmark, output, 'results.csv')
  if not os.path.exists(results_file):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
      writer = csv.writer(f)
      fieldnames = ['plan_id', 'handler', 'method', 'particles', 'time']
      fieldnames += [var[0] for var in true_vars['smc']]
      if raw:
        fieldnames += [var[0]+'_raw' for var in true_vars['smc']]
      writer.writerow(fieldnames)

  run_particles(benchmark, files, n, handlers, methods, config['plans'], true_vars, results_file, error_func, data_file, raw, kwargs)

  if LOGGING:
    artifact = wandb.Artifact(f'{benchmark}_results.csv', type='results')
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

def analyze_benchmark(benchmark, files, output, handlers, methods):
  with open(os.path.join(benchmark, 'config.json')) as f:
    config = json.load(f)

  filename = os.path.join(benchmark, output, 'statistics.json')

  results = {}
  if os.path.exists(filename):
    with open(os.path.join(filename)) as f:
      results = json.load(f)

  variables = config['variables']
  plans = config['plans']

  results = analyze(benchmark, files, handlers, methods, variables, plans, results)

  os.makedirs(os.path.dirname(filename), exist_ok=True)

  # write statistics
  with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

def evaluation(particles, n, args, timeout=300):  
  handlers = ['smc']
  if args.subparser_name == 'artifact-eval':
    n = 10
  print(f"Running the benchmark for Section 2 example with n={n} for Figure 4")
  benchmark = 'examplegood'
  kwargs = {
    'particles': particles,
    'seed': args.seed,
    'timeout': timeout,
  }
  files = [
    'benchmarks/examplegood/programs/plan1.si',
    'benchmarks/examplegood/programs/plan3.si',
    'benchmarks/examplegood/programs/plan4.si',
  ]
  run_benchmark(benchmark, args.output, n, handlers, ['ssi'], files, 'mse', True, kwargs)

  print(f"Running the benchmark for Section 2 example with n={n} for Figure 5")
  benchmark = 'examplebad'
  files = [
    'benchmarks/examplebad/programs/plan3.si',
  ]
  run_benchmark(benchmark, args.output, n, handlers, ['ssi'], files, 'mse', True, {
    'particles': [16],
    'seed': args.seed,
    'timeout': timeout,
  })

  files = [
    'benchmarks/examplebad/programs/plan4.si',
  ]
  run_benchmark(benchmark, args.output, n, handlers, ['ssi'], files, 'mse', True, {
    'particles': [8],
    'seed': args.seed,
    'timeout': timeout,
  })

  print("Running the analysis for Section 5 Table 1")
  for benchmark in DEFAULT_BENCHMARKS:
    print('Benchmark: {}'.format(benchmark))
    analyze_benchmark(benchmark, [], args.output, handlers, DEFAULT_METHODS)

  if args.subparser_name == 'artifact-eval':
    n = 5

  print(f"Running the benchmarks for Section 5 with n={n} Figure 16")
  for benchmark in ['outlier', 'noise']:
    print('Benchmark: {}'.format(benchmark))
    run_benchmark(benchmark, args.output, n, handlers, ['ssi'], [], 'mse', False, kwargs)

  if args.subparser_name == 'full-replication':
    # skip appendix due to time constraints
    print(f"Running rest of the benchmarks with n={n} for Appendix F")
    for benchmark in DEFAULT_BENCHMARKS:
      print('Benchmark: {}'.format(benchmark))
      if benchmark == 'slds':
        files = [
          'benchmarks/slds/programs/plan67.si',
          'benchmarks/slds/programs/plan81.si', 
          'benchmarks/slds/programs/plan98.si', 
          'benchmarks/slds/programs/plan112.si', 
          'benchmarks/slds/programs/plan115.si',
          'benchmarks/slds/programs/plan127.si',
        ]
        run_benchmark(benchmark, args.output, n, handlers, ['ssi'], files, 'mse', False, kwargs)
        files = [
          'benchmarks/slds/programs/plan112.si', 
          'benchmarks/slds/programs/plan113.si', 
          'benchmarks/slds/programs/plan114.si', 
          'benchmarks/slds/programs/plan115.si', 
          'benchmarks/slds/programs/plan116.si', 
          'benchmarks/slds/programs/plan120.si', 
          'benchmarks/slds/programs/plan127.si',
        ]
        run_benchmark(benchmark, args.output, n, handlers, ['ds'], files, 'mse', False, kwargs)
        run_benchmark(benchmark, args.output, n, handlers, ['bp'], [], 'mse', False, kwargs)
      else:
        if benchmark in ['outlier', 'noise']:
          methods = ['ds', 'bp']
        else:
          methods = DEFAULT_METHODS

        run_benchmark(benchmark, args.output, n, handlers, methods, [], 'mse', False, kwargs)

    handlers = ['mh']
    print(f"Running the benchmark with n={n} for Appendix H")
    for benchmark in DEFAULT_BENCHMARKS:
      print('Benchmark: {}'.format(benchmark))
      if benchmark == 'slds':
        files = [
          'benchmarks/slds/programs/plan67.si',
          'benchmarks/slds/programs/plan81.si', 
          'benchmarks/slds/programs/plan98.si', 
          'benchmarks/slds/programs/plan112.si', 
          'benchmarks/slds/programs/plan115.si', 
          'benchmarks/slds/programs/plan127.si',
        ]
        run_benchmark(benchmark, args.output, n, handlers, ['ssi'], files, 'mse', False, kwargs)
        files = [
          'benchmarks/slds/programs/plan112.si', 
          'benchmarks/slds/programs/plan113.si', 
          'benchmarks/slds/programs/plan114.si', 
          'benchmarks/slds/programs/plan115.si', 
          'benchmarks/slds/programs/plan116.si', 
          'benchmarks/slds/programs/plan120.si', 
          'benchmarks/slds/programs/plan127.si',
        ]
        run_benchmark(benchmark, args.output, n, handlers, ['ds'], files, 'mse', False, kwargs)
        run_benchmark(benchmark, args.output, n, handlers, ['bp'], [], 'mse', False, kwargs)
      else:
        methods = DEFAULT_METHODS

        run_benchmark(benchmark, args.output, n, handlers, methods, [], 'mse', False, kwargs)

    print("Running the analysis for Appendix H")
    for benchmark in DEFAULT_BENCHMARKS:
      print('Benchmark: {}'.format(benchmark))
      analyze_benchmark(benchmark, [], args.output, handlers, DEFAULT_METHODS)
  
if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--benchmark', '-b', type=str, required=False)
  p.add_argument('--output', '-o', type=str, required=False, default='output')
  p.add_argument('--files', '-f', type=str, required=False)
  p.add_argument('--handlers', type=str, required=False)
  p.add_argument('--methods', '-m', type=str, required=False)
  p.add_argument('--logging', '-l', action='store_true')
  p.add_argument('--seed', type=int, required=False)

  sp = p.add_subparsers(dest='subparser_name')

  rp = sp.add_parser('run')
  rp.add_argument('--particles', '-p', type=int, required=False, nargs='+')
  rp.add_argument('--samples', type=int, required=False, nargs='+')
  rp.add_argument('--warmup', type=int, default=0)
  rp.add_argument('--n', '-n', type=int, required=False, default=100)
  rp.add_argument('--error-func', '-ef', type=str, required=False, default='mse')
  rp.add_argument('--raw', '-r', action='store_true')
  
  ap = sp.add_parser('analyze')

  rp = sp.add_parser('check')

  kp = sp.add_parser('kicktires')
  aep = sp.add_parser('artifact-eval')

  args = p.parse_args()

  print('Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
  start_time = time.time()

  if args.logging:
    wandb.init(project=f'siren-{args.subparser_name}')
    LOGGING = True

  if args.subparser_name == 'kicktires':
    n = 1
    particles = [1, 2, 4, 8, 16]
    
    evaluation(particles, n, args, timeout=15)
  elif args.subparser_name == 'artifact-eval':
    n = 5
    particles = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    evaluation(particles, n, args, timeout=300)
  elif args.subparser_name == 'full-replication':
    n = 100
    particles = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    evaluation(particles, n, args, timeout=300)
  else:
    benchmarks = [b.strip() for b in args.benchmark.split(',')] if args.benchmark is not None else DEFAULT_BENCHMARKS
    methods = [m.strip() for m in args.methods.split(',')] if args.methods is not None else DEFAULT_METHODS
    handlers = [h.strip() for h in args.handlers.split(',')] if args.handlers is not None else DEFAULT_HANDLERS

    for benchmark in benchmarks:
      print('Benchmark: {}'.format(benchmark))

      # If no files specified, get all files in programs directory
      if args.files is None:
        files = []
      else: 
        files = [f.strip() for f in args.files.split(',')]

      for file in files:
        if not os.path.exists(file):
          raise Exception(f'File not found: {file}')

      methods = [method for method in methods if method in DEFAULT_METHODS]

      if args.subparser_name == 'run':
        # Get list of particles
        if args.particles is None:
          particles = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
          # particles = sorted([int(x) for x in np.unique(np.logspace(
          #                                         np.log10(args.prange[0]), 
          #                                         np.log10(args.prange[1]), 
          #                                         N_INTERVALS, 
          #                                         dtype=int
          #                                       ))])
          print('Particles:', particles)
        else:
          particles = args.particles

        kwargs = {
          'particles': particles,
          'warmup': args.warmup,
          'samples': args.samples,
          'seed': args.seed,
        }

        run_benchmark(benchmark, args.output, args.n, handlers, methods, files, args.error_func, args.raw, kwargs)
      elif args.subparser_name == 'analyze':
        analyze_benchmark(benchmark, files, args.output, handlers, methods)

      elif args.subparser_name == 'check':
        with open(os.path.join(benchmark, 'config.json')) as f:
          config = json.load(f)

        knowns = config['known_enc'] if 'known_enc' in config else None
        satisfied_plan_ids = find_satisfiable_plans(benchmark, files, handlers, methods, config['plans'], knowns, timeout=300)
        
        for plan_id, plan_data in config['plans'].items():
          plan_data['satisfiable'] = {} if 'satisfiable' not in plan_data else plan_data['satisfiable']
          for handler in handlers:
            if handler not in plan_data['satisfiable']:
              plan_data['satisfiable'][handler] = {}
            for method in methods:
              plan_data['satisfiable'][handler][method] = (plan_id in satisfied_plan_ids[handler][method])

        with open(os.path.join(benchmark, 'config.json'), 'w') as f:
          json.dump(config, f, indent=2)

      else:
        print('Invalid subcommand')
        exit(1)

  end_time = time.time()
  print('End time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
  print('Elapsed time: {}'.format(end_time - start_time))

  if args.logging:
    wandb.finish()