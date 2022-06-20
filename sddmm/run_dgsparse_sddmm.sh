cd dgsparse
mkdir -p build
cd build
cmake ..
make -j16

cd example
cd sddmm
for dataset in arxiv pubmed ppi proteins reddit
do
    echo "---------------dataset:" ${dataset} "------------------"
    for feat_size in 32 64 128 256 512
    do
        echo "-----feat size:" ${feat_size} "-----"
        ./sddmm ../../../../data/${dataset}.npz ${feat_size}
    done
done
