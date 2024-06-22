from pathlib import Path
import subprocess
from evaluate import GENERATION, MLIR_OPT, Experiment, ExperimentResult, generate

N_GENERATE = 100

PRINT_GENERIC_ARGS = ["-mlir-print-op-generic"]
GENERIC_PRINTING_FOLDER = GENERATION/"generic-printing"
TMP_FOLDER_NAME = "tmp"
GENERIC_PRINTING_TMP = GENERIC_PRINTING_FOLDER/TMP_FOLDER_NAME

PROJECT_PATH = Path("../../")
REF_INTERPRETER = PROJECT_PATH/("mlir-quickcheck/dist-newstyle/build"
                                "/x86_64-linux/ghc-9.2.5/mlir-quickcheck-0.1.0.0"
                                "/x/ref-interpreter/build/"
                                "ref-interpreter/ref-interpreter")

def init_generic_printing_folder():
    GENERIC_PRINTING_FOLDER.mkdir(exist_ok=True, parents=True)
    (GENERIC_PRINTING_TMP).mkdir(exist_ok=True)

if __name__ == '__main__':
    # call generate
    init_generic_printing_folder()
    generate(GENERIC_PRINTING_TMP, N_GENERATE)

    generated = 0
    successfully_executed = []
    # loop through the files and run mlir-opt on PRINT_GENERIC_ARGS
    for fn in GENERIC_PRINTING_TMP.glob("*.mlir"):
        fp_generic = GENERIC_PRINTING_FOLDER/fn.name
        print(f"converting: {fp_generic}")

        # print into generic form
        outs = subprocess.run([MLIR_OPT.absolute(), *PRINT_GENERIC_ARGS, str(fn)], capture_output=True)
        with open(fp_generic, "w") as f:
            generic_file_raw = outs.stdout.decode("utf-8")
            legalized_args = generic_file_raw.replace(r"%arg", r"%99999")
            f.write(legalized_args)
        print(f"written to: {fp_generic}")
        generated += 1

        # run using the interpreter
        outs = subprocess.run([REF_INTERPRETER.absolute(), "-f", str(fp_generic), "-m", "func1"], capture_output=True)
        # see execution result
        if outs.returncode == 0:
            successfully_executed.append(fp_generic)
        else:
            # print(outs.stderr.decode("utf-8"))
            pass
        
    print(f"successfully executed: {len(successfully_executed)}/{generated}")
    print(f"- {successfully_executed}")