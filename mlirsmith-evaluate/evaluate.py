from dataclasses import dataclass, asdict
from enum import Enum, auto
import json
from pathlib import Path
import pprint as pp
import subprocess
import time
from typing import Optional
import pandas as pd

# make an enum class of experiment, with elements MlirSmith, Arith, Linalg, Tensor
class Experiment(Enum):
    MlirSmith = auto()
    Arith = auto()
    Linalg = auto()
    Tensor = auto()

# REMEMBER: first -
# cmake --build ../build --target mlirsmith
EXPERIMENT = Experiment.MlirSmith
N_TO_GENERATE = 100

GENERATION = Path("./generated")
MLIRSMITH = Path("../build/bin/mlirsmith")
MLIR_OPT = Path("../build/bin/mlir-opt")

COMPILE_MLIRSMITH_ARGS = "--canonicalize"

# from mlir/test/Integration/Dialect/Arith/CPU/test-wide-int-emulation-addi-i16.mlir
COMPILE_ARITH_ARGS = ("--convert-scf-to-cf --convert-cf-to-llvm "
                     "--convert-vector-to-llvm --convert-func-to-llvm "
                     "--convert-arith-to-llvm")

# from mlir/test/Integration/Dialect/Linalg/CPU/test-tensor-matmul.mlir
COMPILE_LINALG_ARGS = ("-linalg-bufferize -arith-bufferize"
                      "-tensor-bufferize -func-bufferize -finalizing-bufferize"
                      "-buffer-deallocation-pipeline -convert-bufferization-to-memref"
                      "-convert-linalg-to-loops -convert-scf-to-cf"
                      "-expand-strided-metadata -lower-affine -convert-arith-to-llvm "
                      "-convert-scf-to-cf --finalize-memref-to-llvm -convert-func-to-llvm "
                      "-reconcile-unrealized-casts")

# from mlir/test/Integration/Dialect/Linalg/CPU/test-tensor-e2e.mlir
COMPILE_TENSOR_ARGS = ("-arith-bufferize -linalg-bufferize"
                      "-tensor-bufferize -func-bufferize -finalizing-bufferize"
                      "-buffer-deallocation-pipeline -convert-bufferization-to-memref"
                      "-convert-linalg-to-loops -convert-arith-to-llvm -convert-scf-to-cf"
                      "-convert-cf-to-llvm --finalize-memref-to-llvm -convert-func-to-llvm "
                      "-reconcile-unrealized-casts")

def get_folder(experiment : Experiment) -> Path:
    if experiment == Experiment.MlirSmith:
        return GENERATION/"mlirsmith"
    if experiment == Experiment.Arith:
        return GENERATION/"arith"
    if experiment == Experiment.Linalg:
        return GENERATION/"linalg"
    if experiment == Experiment.Tensor:
        return GENERATION/"tensor"
    
def get_compile_args(experiment : Experiment) -> str:
    if experiment == Experiment.MlirSmith:
        return COMPILE_MLIRSMITH_ARGS
    if experiment == Experiment.Arith:
        return COMPILE_ARITH_ARGS
    if experiment == Experiment.Linalg:
        return COMPILE_LINALG_ARGS
    if experiment == Experiment.Tensor:
        return COMPILE_TENSOR_ARGS

def init_generation_folder() -> None:
    GENERATION.mkdir(exist_ok=True)
    (GENERATION/"mlirsmith").mkdir(exist_ok=True)
    (GENERATION/"arith").mkdir(exist_ok=True)
    (GENERATION/"linalg").mkdir(exist_ok=True)
    (GENERATION/"tensor").mkdir(exist_ok=True)

def write_to(filepath : Path, content : str) -> None:
    with open(filepath, "w") as f:
        f.write(content)

def generate(folder : Path, n : int) -> None:
    for i in range(n):
        outs = subprocess.run(MLIRSMITH.absolute(), capture_output=True)
        generated = outs.stderr.decode("utf-8")
        i_filename = folder/f"{i}.mlir"
        write_to(i_filename, generated)
        print(f"done {i_filename}")

def generate_everything(n : int) -> None:
    init_generation_folder()
    generate(get_folder(Experiment.MlirSmith), n)
    generate(get_folder(Experiment.Arith), n)
    generate(get_folder(Experiment.Linalg), n)
    generate(get_folder(Experiment.Tensor), n)

@dataclass
class ExperimentResult:
    successfully_generated : bool
    file_length : int
    successfully_compiled : bool
    file : Optional[str] = None
    compiler_errors : Optional[str] = None

def count_file_lines(file : Path) -> int:
    return sum(1 for _ in file.open())

def evaluate_generated_files(e : Experiment) -> dict[int, ExperimentResult]:
    folder = get_folder(e)
    compile_args = get_compile_args(e)
    results = []
    for i in range(N_TO_GENERATE):
        file = folder/f"{i}.mlir"

        # file doesn't exist. record results and try next one.
        if not file.exists():
            results[i] = ExperimentResult(
                successfully_generated = False,
                file_length = 0,
                successfully_compiled = False,
                file = None,
                compiler_errors = None
            )
            continue

        # file does exist. now try to compile it
        file_lines = count_file_lines(file)
        outs = subprocess.run([MLIR_OPT.absolute(), compile_args, file.absolute()], capture_output=True)
        out_errors = outs.stderr.decode("utf-8")
        results.append(ExperimentResult(
            successfully_generated = True,
            file_length = file_lines,
            successfully_compiled = outs.returncode == 0,
            file = file.absolute().as_posix(),
            compiler_errors = out_errors
        ))
    return results

def evaluate_result(ress : list[ExperimentResult]) -> dict[str, int]:
    total = len(ress)
    generated = sum(1 for res in ress if res.successfully_generated)
    compiled = sum(1 for res in ress if res.successfully_compiled)

    # save ress into a dataframe and compute summary statistics
    ress_dicts = [asdict(er) for er in ress]
    df = pd.DataFrame(ress_dicts)
    summary_stats = df.describe()
    summary_stats_dict = summary_stats.to_dict()
    compilation_stats = {
        "total" : total,
        "generated" : generated,
        "compiled" : compiled
    }
    return compilation_stats | summary_stats_dict

def save_result(ress : list[ExperimentResult], e : Experiment) -> None:
    folder = get_folder(e)
    ress_dicts = [asdict(r) for r in ress]
    with open(folder/"_results.json", "w") as f:
        f.write(json.dumps(ress_dicts))

if __name__ == "__main__":
    init_generation_folder()

    start_time = time.time()
    generate(get_folder(EXPERIMENT), N_TO_GENERATE)
    res = evaluate_generated_files(EXPERIMENT)
    end_time = time.time()

    save_result(res, EXPERIMENT)

    print(f"{EXPERIMENT} results:")
    pp.pprint(evaluate_result(res))
    print(f"Time elapsed: {end_time - start_time}s")