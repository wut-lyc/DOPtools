#!/bin/bash
# ============================================================
# 运行 Tg 预测模型
# ============================================================

set -e

# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate doptools_env

# 运行模型
echo "开始运行 Tg 预测模型..."
echo "数据文件: processed_data.csv"
echo ""

# 使用 nohup 后台运行 (可选，去掉注释使用)
# nohup python build_tg_model.py > output.log 2>&1 &
# echo "模型正在后台运行，日志保存到 output.log"
# echo "使用 'tail -f output.log' 查看进度"

# 前台运行
python build_tg_model.py
