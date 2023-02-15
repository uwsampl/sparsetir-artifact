import os
import numpy as np


def extract_data():
    datasets = ["aifb", "mutag", "bgs", "biokg", "am"]
    display_names = ["AIFB", "MUTAG", "BGS", "ogbn-biokg", "AM"]
    with open("rgcn-e2e.dat", "w") as fout:
        with open("rgcn-e2e-mem.dat", "w") as fout_mem:
            current_dataset = None
            current_model = None
            exec_times = {}
            mem = {}
            with open("rgcn.log", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("benchmarking on:"):
                        current_dataset = line.split()[-1]
                    if line.startswith("5-Graphiler"):
                        current_model = "graphiler"
                    if line.startswith("0-DGL-UDF"):
                        current_model = "dgl-udf"
                    if line.startswith("3-DGL-bmm"):
                        current_model = "dgl-bmm"
                    if line.startswith("1-DGL-slice"):
                        current_model = "dgl-slice"
                    if line.startswith("4-PyG-bmm"):
                        current_model = "pyg-bmm"
                    if line.startswith("2-PyG-slice"):
                        current_model = "pyg-slice"
                    if line.startswith("sparse_tir_naive"):
                        current_model = "sparsetir-naive"
                    if line.startswith("sparse_tir_composable"):
                        current_model = "sparsetir-composable"
                    if line.startswith("sparse_tir_tensorcore"):
                        current_model = "sparsetir-tensorcore"
                    if line.endswith("ms/infer"):
                        exec_times[(current_dataset, current_model)] = float(line.split()[-2])
                    if line.endswith("MB"):
                        mem[(current_dataset, current_model)] = float(line.split()[-2])
        
            fout.write("Dataset	PyG	DGL	Graphiler	SparseTIR(naive)	SparseTIR(hyb)	SparseTIR(hyb+TC)\n")
            fout_mem.write("Dataset	PyG	DGL	Graphiler	SparseTIR(naive)	SparseTIR(hyb)	SparseTIR(hyb+TC)\n")

            for display_name, dataset in zip(display_names, datasets):
                speed_str = display_name
                mem_str = display_name
                assert (dataset, "graphiler") in exec_times
                graphiler_dur = exec_times[(dataset, "graphiler")]
                graphiler_mem = mem[(dataset, "graphiler")]

                if not (dataset, "pyg-bmm") in mem and not (dataset, "pyg-slice") in mem:
                    speed_str += " 0"
                    mem_str += " 0"
                else:
                    pyg_bmm_dur = exec_times.get((dataset, "pyg-bmm"), 1e9)
                    pyg_slice_dur = exec_times.get((dataset, "pyg-slice"), 1e9)
                    pyg_bmm_mem = mem.get((dataset, "pyg-bmm"), 0)
                    pyg_slice_mem = mem.get((dataset, "pyg-slice"), 0)
                    spd_arr = [pyg_bmm_dur, pyg_slice_dur]
                    mem_arr = [pyg_bmm_mem, pyg_slice_mem]
                    
                    indices = np.argsort(spd_arr)
                    speed_str += " " + str(graphiler_dur / spd_arr[indices[0]])
                    mem_str += " " + str(mem_arr[indices[0]])
                
                if not (dataset, "dgl-udf") in mem and not (dataset, "dgl-slice") in mem and not (dataset, "dgl-bmm") in mem:
                    speed_str += " 0"
                    mem_str += " 0"
                else:
                    dgl_udf_dur = exec_times.get((dataset, "dgl-udf"), 1e9)
                    dgl_bmm_dur = exec_times.get((dataset, "dgl-bmm"), 1e9)
                    dgl_slice_dur = exec_times.get((dataset, "dgl-slice"), 1e9)
                    dgl_udf_mem = mem.get((dataset, "dgl-udf"), 1e9)
                    dgl_bmm_mem = mem.get((dataset, "dgl-bmm"), 1e9)
                    dgl_slice_mem = mem.get((dataset, "dgl-slice"), 1e9)

                    spd_arr = [dgl_udf_dur, dgl_bmm_dur, dgl_slice_dur]
                    mem_arr = [dgl_udf_mem, dgl_bmm_mem, dgl_slice_mem]
                    
                    indices = np.argsort(spd_arr)
                    speed_str += " " + str(graphiler_dur / spd_arr[indices[0]])
                    mem_str += " " + str(mem_arr[indices[0]])
                
                speed_str += " 1"
                mem_str += " {}".format(graphiler_mem)

                assert (dataset, "sparsetir-naive") in exec_times
                assert (dataset, "sparsetir-composable") in exec_times
                assert (dataset, "sparsetir-tensorcore") in exec_times
                sparsetir_naive_dur = exec_times[(dataset, "sparsetir-naive")]
                sparsetir_naive_mem = mem[(dataset, "sparsetir-naive")]
                sparsetir_composable_dur = exec_times[(dataset, "sparsetir-composable")]
                sparsetir_composable_mem = mem[(dataset, "sparsetir-composable")]
                sparsetir_tensorcore_dur = exec_times[(dataset, "sparsetir-tensorcore")]
                sparsetir_tensorcore_mem = mem[(dataset, "sparsetir-tensorcore")]

                speed_str += " " + str(graphiler_dur / sparsetir_naive_dur)
                mem_str += " " + str(sparsetir_naive_mem)
                speed_str += " " + str(graphiler_dur / sparsetir_composable_dur)
                mem_str += " " + str(sparsetir_composable_mem)
                speed_str += " " + str(graphiler_dur / sparsetir_tensorcore_dur)
                mem_str += " " + str(sparsetir_tensorcore_mem)
                fout.write(speed_str + "\n")
                fout_mem.write(mem_str + "\n")


if __name__ == "__main__":
    extract_data()
