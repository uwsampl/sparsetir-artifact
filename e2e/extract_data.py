import os


def extract_data():
    with open("graphsage-e2e.dat", "w") as fout:
        fout.write("Dataset         PyTorch+SparseTIR-SpMM\n")
        datasets = [
            "cora", "citeseer", "pubmed", "ppi", "arxiv", "proteins", "reddit"
        ]
        display_names = [
            "Cora", "Citeseer", "Pubmed", "PPI", "ogbn-arxiv", "ogbn-proteins",
            "Reddit"
        ]
        for display_name, dataset in zip(display_names, datasets):
            with open("dgl_{}.log".format(dataset), "r") as f:
                lines = f.readlines()
                dgl_time = float(lines[-1].split()[-2])
            with open("sparsetir_{}.log".format(dataset), "r") as f:
                lines = f.readlines()
                sparsetir_time = float(lines[-1].split()[-2])
            fout.write("{} {}\n".format(display_name,
                                        dgl_time / sparsetir_time))


if __name__ == "__main__":
    extract_data()
