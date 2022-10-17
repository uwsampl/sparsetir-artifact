cd ../3rdparty/sputnik
mkdir -p build
cd build
# Use 75 for turing architecture
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="80"
make -j16

cd sputnik
for dataset in cora citeseer pubmed ppi arxiv # proteins reddit
do
    echo "---------------dataset:" ${dataset} "------------------"
    for feat_size in 32 64 128 256 512
    do
        echo "-----feat size:" ${feat_size} "-----"
        ./spmm_benchmark ../../../../spmm/data/${dataset}.npz ${feat_size}
    done
done

