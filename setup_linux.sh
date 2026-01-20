#!/bin/bash
# ============================================================
# DOPtools Tg 预测模型 - Linux 服务器部署脚本
# 使用国内镜像源加速 (适用于中国大陆网络环境)
# ============================================================
# 使用方法:
#   1. 将整个 DOPtools 项目目录上传到服务器
#   2. 运行: bash setup_linux.sh
#   3. 运行模型: bash run_model.sh
# ============================================================

set -e  # 遇到错误立即退出

echo "============================================================"
echo "DOPtools 环境安装脚本 (Linux - 国内镜像加速版)"
echo "============================================================"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda 未安装！"
    echo "请先安装 Miniconda (使用清华镜像):"
    echo "  wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 初始化 conda
eval "$(conda shell.bash hook)"

# ============================================================
# 配置国内镜像源
# ============================================================
echo ""
echo "[0/4] 配置国内镜像源 (清华大学源)..."

# 配置 conda 镜像 (清华大学)
cat > ~/.condarc << 'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

echo "Conda 镜像配置完成!"

# 配置 pip 镜像 (清华大学)
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "Pip 镜像配置完成!"

# 清除 conda 缓存
conda clean -i -y 2>/dev/null || true

# ============================================================
# 创建 conda 环境
# ============================================================
echo ""
echo "[1/4] 创建 Conda 环境 (doptools_env, Python 3.10)..."
conda create -n doptools_env python=3.10 -y

# 激活环境
echo ""
echo "[2/4] 激活环境..."
conda activate doptools_env

# ============================================================
# 安装依赖
# ============================================================
echo ""
echo "[3/4] 安装 DOPtools 及依赖 (使用清华镜像)..."

# 安装 DOPtools (从本地目录)
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Mordred 描述符库
echo "安装 Mordred 描述符库..."
pip install mordred -i https://pypi.tuna.tsinghua.edu.cn/simple

# ============================================================
# 验证安装
# ============================================================
echo ""
echo "[4/4] 验证安装..."
python -c "import doptools; print('DOPtools 安装成功!')"
python -c "from chython import smiles; print('Chython OK')"
python -c "import optuna; print('Optuna version:', optuna.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
python -c "import mordred; print('Mordred OK')"
python -c "from rdkit import Chem; from rdkit.Chem import Descriptors; print('RDKit Descriptors OK')"

echo ""
echo "============================================================"
echo "安装完成！"
echo ""
echo "镜像源配置:"
echo "  Conda: 清华大学镜像 (https://mirrors.tuna.tsinghua.edu.cn)"
echo "  Pip:   清华大学镜像 (https://pypi.tuna.tsinghua.edu.cn)"
echo ""
echo "使用方法:"
echo "  conda activate doptools_env"
echo "  python build_tg_model.py"
echo ""
echo "或者直接运行:"
echo "  bash run_model.sh"
echo "============================================================"
