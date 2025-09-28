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

pip3 install openwakeword==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


sudo apt install portaudio19-dev alsa-utils

sudo apt update
sudo apt install -y ffmpeg libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev

pip install av -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --no-deps faster-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
