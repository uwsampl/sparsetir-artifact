import os
import numpy as np


def extract_data():
    with open("bsr-spmm.dat", "w") as fout_spmm:
        with open("bsr-sddmm.dat", "w") as fout_sddmm:
            first_line = (
                " ".join(["Dataset", "Triton", "SparseTIR-CSR", "SparseTIR-BSR"]) + "\n"
            )
            fout_spmm.write(first_line)
            fout_sddmm.write(first_line)
            fo = {"spmm": fout_spmm, "sddmm": fout_sddmm}
            datasets = ["pixelfly", "longformer"]
            display_names = ["Butterfly", "Longformer"]
            for display_name, dataset in zip(display_names, datasets):
                for op in ["spmm", "sddmm"]:
                    fo[op].write(display_name + " ")
                    exec_times = {}
                    for backend in ["triton_blocksparse", "csr", "sparse_tir"]:
                        with open(
                            "{}_{}_{}.log".format(backend, op, dataset), "r"
                        ) as f:
                            lines = f.readlines()
                            exec_times[backend] = float(lines[-1].split()[-2])

                    for backend in ["triton_blocksparse", "csr", "sparse_tir"]:
                        fo[op].write(
                            " "
                            + str(
                                exec_times["triton_blocksparse"] / exec_times[backend]
                            )
                        )
                    fo[op].write("\n")


if __name__ == "__main__":
    extract_data()
