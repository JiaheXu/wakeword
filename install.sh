# Clone with submodules
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

# Build
mkdir build && cd build
cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release ..

make -j$(nproc)
sudo make install

##### ???????????????????????????
pip install pybind11

cd ~/CTranslate2/python
pip install .
