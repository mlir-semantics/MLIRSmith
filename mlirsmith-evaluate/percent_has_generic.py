import json
from pathlib import Path
import pprint as pp

RESULTS_PATH = Path("generated/linalg/_results.json")

def count_generics(ress):
    n_generics = [Path(r["file"]).read_text().count("linalg.generic") for r in ress]
    return n_generics

if __name__ == '__main__':
    ress = json.load(RESULTS_PATH.open())

    # find all of ress which have "successfully_compiled": true
    compiled_entries = [r for r in ress if r["successfully_compiled"]]

    # grep for "linalg.generic" within compiled_entries["file"]
    compiled_generic_files = [r for r in compiled_entries if "linalg.generic" in Path(r["file"]).read_text()]
    print(f"{len(compiled_generic_files)}/{len(compiled_entries)} of compiled files have 'linalg.generic' operation")
    print("these are, for instance (first 3):")
    pp.pprint([r["file"] for r in compiled_generic_files[:3]])

    # count the average number of linalg.generic operations generated
    n_generics = count_generics(ress)                       # number of linalg.generic operations in all files
    n_generics_compiled = count_generics(compiled_entries)  # number of linalg.generic operations compiled files 
    avg_n_generics = sum(n_generics) / len(n_generics)
    avg_n_generics_compiled = sum(n_generics_compiled) / len(n_generics_compiled)
    print(f"average number of 'linalg.generic' operations generated in all files: {avg_n_generics:.2f}")
    print(f"average number of 'linalg.generic' operations generated in compiled files: {avg_n_generics_compiled:.2f}")