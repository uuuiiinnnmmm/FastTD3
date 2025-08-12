Reproducing FastTD3 on a Laptop with Ubuntu 22.04
====
This document details the step-by-step process of setting up and running the FastTD3 project on a personal laptop. The process involves navigating several challenges related to NVIDIA drivers, CUDA setup, conflicting Python dependencies from its environment backend humanoid-bench, and resource limitations of a non-server-grade machine.

1. System & Hardware Prerequisites
---
OS: Ubuntu 22.04

GPU: NVIDIA GPU (e.g., RTX series)

Key Software: miniconda for environment management.

2. Environment Setup: A Tale of Two Projects
---
The core challenge is that FastTD3 (the algorithm) relies on humanoid-bench (the simulation environment), but the latter has several configuration issues. Our strategy is to set up a single, clean environment for FastTD3 and then carefully install humanoid-bench into it as a dependency.


###Step 2.1: NVIDIA Driver & CUDA Toolkit Installation & miniconda

A correct GPU driver and CUDA setup is the foundation for everything.

Install NVIDIA Driver: Download and install a compatible driver (version 575.xx or newer is recommended) from the NVIDIA Official Driver Download Page ( It's too new and hasn't been included in Ubuntu's official “app store” yet.)
So here we download and install it from PPA.
```
sudo add-apt-repository ppa:graphics-drivers/ppa
```
```
sudo apt update
```
```
sudo apt install nvidia-driver-575
```
After installation, reboot your system and verify the installation by running nvidia-smi in the terminal.
```
sudo reboot
```
Install CUDA Toolkit: Download and install the CUDA Toolkit. This guide uses CUDA 12.9.

Download from the NVIDIA CUDA Toolkit Archive.（We strongly recommend that you choose deb (network).）

During installation, when prompted "Do you want to install the NVIDIA driver?", select NO as you have already installed a newer driver in the previous step.

Configure Environment Variables: Add the CUDA path to your shell configuration file (e.g., ~/.bashrc).

```
nano ~/.bashrc
```
Add the following lines at the end of the file (adjust the version number if necessary):

```
export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Apply the changes by running 
```
source ~/.bashrc
```
Verify CUDA Installation:

```
nvcc --version
```
This command should display the version of the CUDA compiler, confirming a successful installation.

Then you can download and install Miniconda: Go to the Miniconda official website and download the latest bash installation script for your system (Linux x86_64).

During the installation process, when prompted with “Do you wish the installer to initialize Miniconda3 by running conda init?”, enter yes.

Restart the terminal: After installation is complete, close the current terminal and reopen it to apply the conda initialization. You will notice that the command prompt now includes the (base) prefix, indicating that you are in the conda base environment.

###Step 2.2: Creating the Master Conda Environment

We will create a single, dedicated environment for our final goal: running FastTD3.
Accept Anaconda's Terms of Service
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
```
# The FastTD3 README recommends Python 3.10
conda create -n fasttd3_hb -y python=3.10
conda activate fasttd3_hb
```
###Step 2.3: Installing Dependencies (The Correct Way)

This is the most critical part, where we navigate the dependency conflicts.

Clone the FastTD3 Repository:
```
# Make sure you are in your desired workspace directory, e.g., ~/
```
git clone https://github.com/younggyoseo/FastTD3 
cd FastTD3
```
Install FastTD3's Core Dependencies FIRST: This establishes the correct versions for core libraries like PyTorch, as defined by the main project's author. We use --trusted-host to bypass potential network/SSL issues.

```
pip install --trusted-host pypi.org --trusted-host download.pytorch.org -r requirements/requirements.txt
```
Install the humanoid-bench Backend: Now, we install the environment backend. humanoid-bench has a problematic setup.py file.

First, clone humanoid-bench to a separate directory:
```
# Go to your source code directory, e.g., ~/src/
cd ~/src
git clone https://github.com/carlosferrazza/humanoid-bench.git
```
Then, Correct humanoid-bench's setup.py

# --- 优化后的核心依赖列表 ---
# 只保留项目运行绝对必需的、通用的库，并且放宽版本限制
core_requirements = [
    "gymnasium>=0.29.0", # 核心API，允许小的次版本更新
    "numpy",             # 基础数学库，不限制版本
    "mujoco>=3.1.6",     # 核心模拟器
    "dm_control>=1.0.20", # 核心控制库
    "opencv-python",     # 通用图像处理
    "rich",              # 用于美观的终端输出
    "tqdm",              # 进度条
    "imageio",           # 读写图片/视频
    "natsort",           # 自然排序
    "ipdb",              # 调试工具 (也可放入 extras_require)
]

# --- (可选，但非常推荐) 定义可选依赖 ---
extras_require = {
    "jax": [
        "mujoco-mjx>=3.1.0",
        "gymnax>=0.0.8",
        "brax>=0.9.0",
        "jax",         # JAX 本身让用户自己选择CPU/GPU版本
        "jaxlib",
        "flax",
        "orbax-checkpoint", # JAX 生态的重要部分
        "ml_collections",
        "distrax",
    ],
    "torch": [
        "torch",
    ],
    "tensorflow": [
        "tensorflow",
        "tf-keras", # 新的独立 Keras 包
        "dm-reverb", # DreamerV3 可能需要
        "embodied",  # DreamerV3 的核心库
    ],
}

setup(
    name="humanoid_bench",
    version="0.2",
    author="RLL at UC Berkeley",
    url="https://github.com/carlosferrazza/humanoid-bench",
    description="Humanoid Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8", # 建议更精确的下限
    install_requires=core_requirements,
    extras_require=extras_require, # 添加可选依赖
```
# Activate the environment again to be sure
conda activate fasttd3_hb
# Navigate into the cloned humanoid-bench directory
cd humanoid-bench
# Use this command to install it into the fasttd3_hb environment
pip install -e
```
Install Final Shared Dependencies: The develop command doesn't install dependencies listed in setup.py. We must install them manually.

```
pip install jax jaxlib
```
Verify the Installation: Check that both projects are now recognized by pip.

```
pip list | grep "fast-td3"
pip list | grep "humanoid-bench"
```
3. Running Experiments: Adapting to Laptop Hardware
---
The default hyperparameters in FastTD3 are tuned for a high-end NVIDIA A100 80GB GPU. Running them on a laptop will instantly cause CUDA out of memory errors or system crashes due to RAM exhaustion.

###Step 3.1: Finding a Stable Baseline

Your first goal is to find a set of parameters that runs without crashing. This requires drastically reducing resource consumption.

Start with this "Survival" configuration:

```
python fast_td3/train.py \
    --env_name h1hand-hurdle-v0 \
    --exp_name FastTD3_survival_test \
    --seed 1 \
    --no-use_wandb \
    --num_envs 8 \
    --buffer_size 2000 \
    --batch_size 1024 \
    --critic_hidden_dim 256 \
    --actor_hidden_dim 256
```
--no-use_wandb: Crucial for initial tests. This disables online logging, preventing network timeouts from crashing the run.

The other parameters drastically reduce RAM (num_envs) and VRAM (buffer_size, batch_size, network dimensions) usage.

###Step 3.2: Systematic Tuning to Find Your Hardware's Limit

Once you have a stable baseline, you can systematically increase parameters to maximize performance.

Monitor Your Resources: In separate terminals, keep these commands running during training:

GPU Monitoring: watch -n 0.5 nvidia-smi

RAM Monitoring: watch -n 1 free -h

Tuning Strategy:

Goal 1: Maximize GPU Utilization (GPU-Util): Your GPU is often waiting for the CPU.

Slowly increase --num_envs (e.g., 8 -> 12 -> 16) until your System RAM is nearly full. This will better "feed" the GPU.

Goal 2: Fill VRAM: Your GPU VRAM is likely underutilized.

Slowly increase --buffer_size and --batch_size until your GPU Memory-Usage in nvidia-smi is around 90-95%. This generally leads to more stable training.

Goal 3: Improve Learning: Once resources are maximized, start tuning algorithmic parameters.

Increase learning_rate: The default might be too low. Try 3e-5 or 1e-4.

Increase num_updates: Try 4 or 8 to improve sample efficiency.

Introduce learning_starts: Add --learning_starts 10000 to prevent unstable early learning and catastrophic forgetting.

4. Visualizing Results
---
The training_notebook.ipynb provided in the FastTD3 repo is the best tool for visualization.

Address a System Proxy Bug: Jupyter Lab may fail to start if you have a system-wide proxy (e.g., for VPNs). Launch it from a clean terminal:

```
unset http_proxy https_proxy all_proxy
jupyter lab
```
Create a New Notebook: To avoid modifying the original, create a new file (e.g., visualize.ipynb).

Load and Visualize: Copy the necessary code blocks from the original notebook into your new one. The key steps are:

Import all necessary packages (add import sys if missing).

Set the checkpoint_path to the absolute path of your trained model file (.pt).

Load the checkpoint, making sure to convert the loaded args dictionary back into a HumanoidBenchArgs object:

```
loaded_args_dict = torch_checkpoint["args"]
args = HumanoidBenchArgs(**loaded_args_dict)
```
Use the "Record Video" method from our troubleshooting sessions to render a video of your agent's performance, as this is more memory-safe than live rendering.
