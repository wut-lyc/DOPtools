#!/bin/bash
# ============================================================
# DOPtools Tg 预测模型 - Linux 服务器部署脚本
# ============================================================
# 使用方法:
#   1. 将整个 DOPtools 项目目录上传到服务器
#   2. 运行: bash setup_linux.sh
#   3. 运行模型: bash run_model.sh
# ============================================================

set -e  # 遇到错误立即退出

echo "============================================================"
echo "DOPtools 环境安装脚本 (Linux)"
echo "============================================================"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda 未安装！"
    echo "请先安装 Miniconda 或 Anaconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 初始化 conda (如果需要)
eval "$(conda shell.bash hook)"

# 创建 conda 环境
echo ""
echo "[1/3] 创建 Conda 环境 (doptools_env, Python 3.10)..."
conda create -n doptools_env python=3.10 -y

# 激活环境
echo ""
echo "[2/3] 激活环境并安装 DOPtools..."
conda activate doptools_env

# 安装 DOPtools (从本地目录)
pip install -e .

# 验证安装
echo ""
echo "[3/3] 验证安装..."
python -c "import doptools; print('DOPtools 安装成功!')"
python -c "from chython import smiles; print('Chython OK')"
python -c "import optuna; print('Optuna version:', optuna.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

echo ""
echo "============================================================"
echo "安装完成！"
echo ""
echo "使用方法:"
echo "  conda activate doptools_env"
echo "  python build_tg_model.py"
echo ""
echo "或者直接运行:"
echo "  bash run_model.sh"
echo "============================================================"
