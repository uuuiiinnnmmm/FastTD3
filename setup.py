from pathlib import Path
from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text()

# --- 优化后的核心依赖列表 ---
# 只保留项目运行绝对必需的、通用的库，并且放宽版本限制
core_requirements = [
    "gymnasium==0.29.1", # 核心API，允许小的次版本更新
    "numpy",             # 基础数学库，不限制版本
    "mujoco==3.1.6",     # 核心模拟器
    "dm_control==1.0.20", # 核心控制库
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
        "mujoco-mjx==3.1.6",
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
    extras_require=extras_require, # ★★★ 添加可选依赖
)
