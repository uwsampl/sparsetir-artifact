for dataset in arxiv pubmed ppi proteins reddit
do
    echo "---------------dataset:" ${dataset} "------------------"
    python3 bench_sddmm.py -d ${dataset}
done

