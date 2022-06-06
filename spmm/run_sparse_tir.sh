for dataset in proteins reddit
do
    echo "---------------dataset:" ${dataset} "------------------"
    python sparse_tir_spmm.py -d ${dataset}
done

