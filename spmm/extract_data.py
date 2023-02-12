import os
import numpy as np
from typing import Any, List


def geomean_speedup(baseline: List, x: List) -> Any:
    return np.exp((np.log(np.array(baseline)) - np.log(np.array(x))).mean())


def extract_data():
    with open("spmm.dat", "w") as fout:
        fout.write(" ".join([
            "Dataset", "cuSPARSE", "Sputnik", "dgSPARSE", "TACO",
            "SparseTIR(no-hyb)", "SparseTIR(hyb)"
        ]) + "\n")
        datasets = [
            "cora", "citeseer", "pubmed", "ppi", "arxiv", "proteins", "reddit"
        ]
        display_names = [
            "Cora", "Citeseer", "Pubmed", "PPI", "ogbn-arxiv", "ogbn-proteins",
            "Reddit"
        ]
        for display_name, dataset in zip(display_names, datasets):
            # extract sparsetir hyb
            exec_times = {}
            with open("sparsetir_{}_hyb.log".format(dataset), "r") as f:
                exec_times["sparsetir_hyb"] = []
                for line in f:
                    if line.startswith("OOM"):
                        exec_times["sparsetir_hyb"].append("OOM")
                    elif line.startswith("tir hyb"):
                        exec_times["sparsetir_hyb"].append(
                            float(line.split()[-2]))

            # extract sparsetir naive
            with open("sparsetir_{}_naive.log".format(dataset), "r") as f:
                exec_times["sparsetir_naive"] = []
                for line in f:
                    if line.startswith("OOM"):
                        exec_times["sparsetir_naive"].append("OOM")
                    elif line.startswith("tir naive"):
                        exec_times["sparsetir_naive"].append(
                            float(line.split()[-2]))

            # extract taco
            exec_times["taco"] = []
            for feat_size in [32, 64, 128, 256, 512]:
                with open("taco_{}_{}.log".format(dataset, feat_size),
                          "r") as f:
                    min_exec_time = 1e9
                    OOM = True
                    for line in f:
                        if line.startswith("nnz_per_warp"):
                            exec_time = float(line.split()[-2])
                            min_exec_time = min(min_exec_time, exec_time)
                            OOM = False
                    if OOM:
                        exec_times["taco"].append("OOM")
                    else:
                        exec_times["taco"].append(min_exec_time)

            # extract dgsparse
            exec_times["cusparse"] = []
            exec_times["dgsparse"] = []
            for feat_size in [32, 64, 128, 256, 512]:
                with open("dgsparse_{}_{}.log".format(dataset, feat_size),
                          "r") as f:
                    lines = f.readlines()
                    if lines[2].startswith("[Cusparse]"):
                        exec_times["cusparse"].append(
                            float(lines[3].split()[1]))
                    else:
                        exec_times["cusparse"].append("OOM")
                    if lines[-1].startswith("Best:"):
                        exec_times["dgsparse"].append(
                            float(lines[-1].split()[1]))
                    else:
                        exec_times["dgsparse"].append("OOM")

            # extract sputnik
            exec_times["sputnik"] = []
            for feat_size in [32, 64, 128, 256, 512]:
                with open("sputnik_{}_{}.log".format(dataset, feat_size),
                          "r") as f:
                    lines = f.readlines()
                    if lines[2].startswith("[Sputnik]"):
                        exec_times["sputnik"].append(float(
                            lines[3].split()[1]))

            # export dat
            fout.write(display_name + " ")
            oom_cusparse = "OOM" in exec_times["cusparse"]
            if oom_cusparse:
                fout.write(" " + " ".join(["0"] * 6) + "\n")
                break
            else:
                for backend in [
                        "cusparse", "sputnik", "dgsparse", "taco",
                        "sparsetir_naive", "sparsetir_hyb"
                ]:
                    if "OOM" in exec_times[backend]:
                        fout.write("0 ")
                    else:
                        fout.write(
                            str(
                                geomean_speedup(exec_times["cusparse"],
                                                exec_times[backend])) + " ")
                fout.write("\n")


if __name__ == "__main__":
    extract_data()
