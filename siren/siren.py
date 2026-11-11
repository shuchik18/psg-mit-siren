import argparse
import os
import time
import cProfile

from . import parser
from .analyze import AbsSMC, AbsMH, AnalysisExit
from .evaluate import SMC, MH
from .inference import SSIState, DSState, BPState
from .analysis import AbsSSIState, AbsDSState, AbsBPState
from .inference_plan import runtime_inference_plan
import sys

# Maps the command line arguments to the symbolic state classes
method_states = {
    "ssi": (SSIState, AbsSSIState),
    "ds": (DSState, AbsDSState),
    "bp": (BPState, AbsBPState),
}

handlers = {
    'smc': (SMC, AbsSMC),
    'mh': (MH, AbsMH),
}

def main():
    sys.setrecursionlimit(5000)
    p = argparse.ArgumentParser()
    p.add_argument("filename", type=str)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--particles", "-p", type=int, default=100, help="Number of particles to use during SMC inference")
    p.add_argument("--samples", type=int, default=100, help="Number of samples to use during MH inference")
    p.add_argument("--warmup", "-w", type=int, default=0, help="Number of warmup samples to use during MH inference")
    p.add_argument("--thinning", "-t", type=int, default=1, help="Thining factor to use during MH inference")
    p.add_argument("--analyze", "-a", action="store_true", help="Apply the inference plan satisfiability analysis during compilation")
    p.add_argument("--analyze-only", "-ao", action="store_true", help="Only apply the inference plan satisfiability analysis, does not run the program")
    p.add_argument("--handler", "-l", type=str, default="smc", choices=["smc", "mh"])
    p.add_argument(
        "--method",
        "-m",
        type=str,
        default="ssi",
        choices=["ssi", "ds", "bp"],
    )
    p.add_argument("--seed", "-s", type=int, default=None)
    p.add_argument("--max-rvs", type=int, default=4, help="Maximum number of random variables to track in abstract expression types")
    p.add_argument("--profile", action="store_true", help="Profile the program")
    args = p.parse_args()

    with open(args.filename, "r") as f:
        program = parser.parse_program(f.read())
        # print(program)

    # Get the handler for the selected inference method
    (handler, analysis_handler) = handlers[args.handler]

    print("===== Handler =====")
    match args.handler:
        case "smc":
            print("SMC")
        case "mh":
            print("MH")
        case _:
            raise ValueError("Invalid handler")

    # Get the symbolic state classes for the selected inference method
    (inference_method, analysis_method) = method_states[args.method]

    print("===== Inference Algorithm =====")
    match args.method:
        case "ssi":
            print("SSI")
        case "ds":
            print("DS")
        case "bp":
            print("BP")
        case _:
            raise ValueError("Invalid method")

    # Run the inference plan satisfiability analysis
    if args.analyze or args.analyze_only:
        print("===== Inferred Inference Plan =====")
        t1 = time.time()
        try:
            inferred_plan = analysis_handler().infer(
                program, analysis_method, args.max_rvs
            )
        except AnalysisExit as e:
            inferred_plan = e.plan
        t2 = time.time()
        print(inferred_plan)
        print("===== Analysis Time =====")
        print(f"{t2 - t1}")

    # If the user only wants to analyze the program, we can stop here
    if not args.analyze_only:
        file_dir = os.path.dirname(os.path.realpath(args.filename))
        if args.profile:
            cProfile.runctx(
                "handler.infer(program, inference_method, file_dir, args.seed, n_particles=args.particles, n_samples=args.samples, n_warmups=args.warmup, n_thinning=args.thinning)",
                globals(),
                locals(),
                sort="cumtime",
            )
            return
        t1 = time.time()
        # try:
        res, probstate = handler().infer(
            program,
            inference_method,
            file_dir,
            args.seed,
            n_particles=args.particles,
            n_samples=args.samples,
            n_warmups=args.warmup,
            n_thinning=args.thinning,
        )
        t2 = time.time()
        print("===== Evaluation Time =====")
        print(f"{t2 - t1}")

        print("===== Result =====")
        print(res)

        # Get the runtime inference plan by inspecting the particles
        plan = runtime_inference_plan(probstate)

        if args.verbose:
            # Only for debugging to reduce the expressions
            probstate.simplify()
            print("===== Mixture =====")
            print(probstate.mixture())
            print("===== Prob State =====")
            print(probstate)

        print("===== Runtime Inference Plan =====")
        print(plan)

if __name__ == "__main__":
    main()
