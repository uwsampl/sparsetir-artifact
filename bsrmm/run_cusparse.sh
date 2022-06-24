cd src
mkdir -p build
cd build
cmake ..
make -j16
cd ../../


for batch_size in 1
do
  ./src/build/bsrmm data/butterfly.npz ${batch_size} 64
done