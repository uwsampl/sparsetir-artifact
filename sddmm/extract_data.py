import os
import numpy as np
from typing import Any, List


def geomean_speedup(baseline: List, x: List) -> Any:
    return np.exp((np.log(np.array(baseline)) - np.log(np.array(x))).mean())


def extract_data():
    with open("sddmm.dat", "w") as fout:
        fout.write(" ".join([
            "Dataset", "cuSPARSE", "Sputnik", "dgl", "dgSPARSE-csr",
            "dgSPARSE-coo", "TACO", "SparseTIR"
        ]) + "\n")
        datasets = [
            "cora", "citeseer", "pubmed", "ppi", "arxiv", "proteins", "reddit"
        ]
        display_names = [
            "Cora", "Citeseer", "Pubmed", "PPI", "ogbn-arxiv", "ogbn-proteins",
            "Reddit"
        ]
        for display_name, dataset in zip(display_names, datasets):
            # extract dgl
            exec_times = {}
            with open("dgl_{}.log".format(dataset), "r") as f:
                exec_times["dgl"] = []
                for line in f:
                    if line.startswith("OOM"):
                        exec_times["dgl"].append("OOM")
                    elif line.startswith("dgl"):
                        exec_times["dgl"].append(float(line.split()[-2]))

            # extract sparsetir
            with open("sparsetir_{}.log".format(dataset), "r") as f:
                exec_times["sparsetir"] = []
                for line in f:
                    if line.startswith("OOM"):
                        exec_times["sparsetir"].append("OOM")
                    elif line.startswith("sparse tir"):
                        exec_times["sparsetir"].append(float(line.split()[-2]))

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
            exec_times["dgsparse_csr"] = []
            exec_times["dgsparse_coo"] = []
            for feat_size in [32, 64, 128, 256, 512]:
                with open("dgsparse_{}_{}.log".format(dataset, feat_size),
                          "r") as f:
                    lines = f.readlines()
                    if lines[3].startswith("[cuSPARSE]"):
                        exec_times["cusparse"].append(
                            float(lines[4].split()[1]))
                    else:
                        exec_times["cusparse"].append("OOM")
                    if lines[5].startswith("[SDDMM-csr]"):
                        exec_times["dgsparse_csr"].append(
                            float(lines[6].split()[1]))
                    else:
                        exec_times["dgsparse_csr"].append("OOM")
                    if lines[7].startswith("[SDDMM-coo]"):
                        exec_times["dgsparse_coo"].append(
                            float(lines[8].split()[1]))
                    else:
                        exec_times["dgsparse_coo"].append("OOM")

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
            oom_dgl = "OOM" in exec_times["dgl"]
            if oom_dgl:
                fout.write(" " + " ".join(["0"] * 7) + "\n")
                break
            else:
                for backend in [
                        "cusparse", "sputnik", "dgl", "dgsparse_csr",
                        "dgsparse_coo", "taco", "sparsetir"
                ]:
                    if "OOM" in exec_times[backend]:
                        fout.write("0 ")
                    else:
                        fout.write(
                            str(
                                geomean_speedup(exec_times["dgl"],
                                                exec_times[backend])) + " ")
                fout.write("\n")


if __name__ == "__main__":
    extract_data()
