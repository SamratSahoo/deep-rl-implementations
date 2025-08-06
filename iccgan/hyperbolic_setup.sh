if ! command -v conda &> /dev/null
then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/.bashrc
    source ~/miniconda3/bin/activate
    conda init --all
fi

conda create -n iccgan python=3.10 -y
source activate base
conda activate iccgan

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
pip install isaacsim[all]==4.5.0.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0.0 --extra-index-url https://pypi.nvidia.com

echo "export OMNI_KIT_ACCEPT_EULA=YES" >> ~/.bashrc
# echo "PUBLIC_IP=$(curl -s ifconfig.me)" >> ~/.bashrc
# sudo ufw allow 47998/udp
# sudo ufw allow 49100/tcp
source ~/.bashrc
conda activate iccgan

cd ~/
git clone https://github.com/isaac-sim/IsaacLab.git
sudo apt update
sudo apt install cmake build-essential -y
sudo apt-get install libglu1 -y
sudo apt install clang -y
sudo apt install libstdc++-12-dev -y
cd ~/IsaacLab
conda activate iccgan
./isaaclab.sh --install
