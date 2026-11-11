import re
import os
import itertools
import json
import argparse

DEFAULT_BENCHMARKS = [
  'noise',
  'radar',
  'envnoise',
  'outlier',
  'outlierheavy',
  'gtree',
  'slds',
  'runner',
]

ANNOTATIONS = ['symbolic', 'sample']

def main(benchmarks):
  for benchmark in benchmarks:
    print(f'Generating plans for {benchmark}...')

    # delete all program files
    for file in os.listdir(os.path.join(benchmark, 'programs')):
      if file.endswith('.si'):
        os.remove(os.path.join(benchmark, 'programs', file))

    # Get placeholders and variables
    with open(os.path.join(benchmark, 'template.si')) as template_file:
      # Use regex to find the words PLACEHOLDER and the word following it
      # (which is the variable name)
      template = template_file.read()
      variables = re.findall(r'PLACEHOLDER (\w+)', template)
      
      print('  ', variables)

      plans = {}

      # Generate plans
      for i, annotations in enumerate(itertools.product(ANNOTATIONS, repeat=len(variables))):
        plan = {}
        for variable, annotation in zip(variables, annotations):
          if annotation == '':
            plan[variable] = 'dynamic'
          else:
            plan[variable] = annotation

        print(f'plan {i}: {plan}')

        plans[i] = {
          'plan': plan,
        }

        # Comment string
        plan_comment = '\n'.join(f'{k} - {v}' for k, v in plan.items())
        comment = f'(*\n{plan_comment}\n*)'

        program = template

        # Replace placeholders with annotations
        for variable, annotation in plan.items():
          print(f'  {variable} - {annotation}')
          if annotation == 'dynamic':
            program = re.sub(rf'\bPLACEHOLDER {variable}\b', f'{variable}', program)
          else:
            program = re.sub(rf'\bPLACEHOLDER {variable}\b', f'{annotation} {variable}', program)

        # Write program
        os.makedirs(os.path.join(benchmark, 'programs'), exist_ok=True)

        with open(os.path.join(benchmark, 'programs', f'plan{i}.si'), 'w') as plan_file:
          plan_file.write(f'{comment}\n\n{program}')

    # Write config
    with open(os.path.join(benchmark, 'config.json'), 'r') as configfile:
      config = json.load(configfile)

      config['plans'] = plans
      config['variables'] = variables

    with open(os.path.join(benchmark, 'config.json'), 'w') as configfile:
      json.dump(config, configfile, indent=2)

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--benchmark', '-b', type=str, required=False, nargs="+", default=DEFAULT_BENCHMARKS)

  args = p.parse_args()
  main(args.benchmark)