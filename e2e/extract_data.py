import os


def extract_data():
    with open("graphsage-e2e.dat", "w") as fout:
        fout.write("Dataset         PyTorch+SparseTIR-SpMM\n")
        datasets = ["cora", "citeseer", "pubmed", "ppi", "arxiv", "reddit"]
        display_names = ["Cora", "Citeseer", "Pubmed", "PPI", "ogbn-arxiv", "Reddit"]
        for display_name, dataset in zip(display_names, datasets):
            with open("dgl_{}.log".format(dataset), "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1].split()
                    if last_line[-1] == "ms/epoch":
                        dgl_time = float(last_line[-2])
                    else:
                        dgl_time = 0
                else:
                    dgl_time = 0
            with open("sparsetir_{}.log".format(dataset), "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1].split()
                    if last_line[-1] == "ms/epoch":
                        sparsetir_time = float(lines[-1].split()[-2])
                    else:
                        sparsetir_time = 1e9
                else:
                    sparsetir_time = 1e9
            fout.write("{} {}\n".format(display_name, dgl_time / sparsetir_time))


if __name__ == "__main__":
    extract_data()
