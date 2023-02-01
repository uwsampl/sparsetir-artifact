# download sparseconv data
if [ -d layers/ ];
then
  echo "sparseconv data already downloaded."
else
  echo "Downloading sparseconv data..."
  wget https://homes.cs.washington.edu/~zhye/files/sparseconv.tar.gz
  tar -xvzf sparseconv.tar.gz
fi
