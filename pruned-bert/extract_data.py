import os
import pandas


def extract_data():
    tbl_structured = pandas.read_csv("structured_single_op.csv")
    density = tbl_structured['density'].to_numpy()
    dbsrmm_dur = tbl_structured["sparsetir_dbsrmm_dur"].to_numpy()
    bsrmm_dur = tbl_structured["sparsetir_dur"].to_numpy()
    cublas_dur = tbl_structured["cublas_dur"].to_numpy()
    triton_dur = tbl_structured["triton_dur"].to_numpy()

    dbsrmm_speedup = cublas_dur / dbsrmm_dur
    bsrmm_speedup = cublas_dur / bsrmm_dur
    triton_speedup = cublas_dur / triton_dur
    with open("structured.dat", "w") as f:
        f.write("density	SparseTIR(BSR)	SparseTIR(DBSR)	cuBLAS	Triton\n")
        for i in range(len(density)):
            f.write("{} {} {} {} {}\n".format(density[i], bsrmm_speedup[i], dbsrmm_speedup[i], 1, triton_speedup[i]))
    
    tbl_unstructured = pandas.read_csv("unstructured_single_op.csv")
    density = tbl_unstructured['density'].to_numpy()
    sr_bcrs_dur = tbl_unstructured["sparsetir_sr_bcrs_dur"].to_numpy()
    bsr_dur = tbl_unstructured["sparsetir_bsrmm_dur"].to_numpy()
    cublas_dur = tbl_unstructured["cublas_dur"].to_numpy()
    cusparse_dur = tbl_unstructured["cusparse_dur"].to_numpy()

    sr_bcrs_speedup = cublas_dur / sr_bcrs_dur
    bsr_speedup = cublas_dur / bsr_dur
    cusparse_speedup = cublas_dur / cusparse_dur
    with open("unstructured.dat", "w") as f:
        f.write("density	    SparseTIR(BSR) SparseTIR(SR-BCRS)	    cuSPARSE cuBLAS\n")
        for i in range(len(density)):
            f.write("{} {} {} {} {}\n".format(density[i], bsr_speedup[i], sr_bcrs_speedup[i], cusparse_speedup[i], 1))


if __name__ == "__main__":
    extract_data()
