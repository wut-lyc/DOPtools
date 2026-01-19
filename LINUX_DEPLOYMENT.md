# DOPtools Tg 预测模型 - Linux 服务器部署指南

## 快速开始

### 1. 上传项目到服务器

```bash
# 方法1: 使用 scp
scp -r DOPtools user@server:/path/to/destination/

# 方法2: 使用 rsync (推荐，支持断点续传)
rsync -avz --progress DOPtools/ user@server:/path/to/DOPtools/

# 方法3: 打包后上传
tar -czvf DOPtools.tar.gz DOPtools/
scp DOPtools.tar.gz user@server:/path/to/
# 在服务器上解压
ssh user@server "cd /path/to/ && tar -xzvf DOPtools.tar.gz"
```

### 2. 在服务器上安装环境

```bash
# SSH 登录服务器
ssh user@server

# 进入项目目录
cd /path/to/DOPtools

# 运行安装脚本
bash setup_linux.sh
```

### 3. 运行模型

```bash
# 方法1: 使用运行脚本
bash run_model.sh

# 方法2: 手动运行
conda activate doptools_env
python build_tg_model.py

# 方法3: 后台运行 (推荐长时间任务)
nohup python build_tg_model.py > output.log 2>&1 &
tail -f output.log  # 查看实时输出
```

---

## 详细步骤

### 前提条件

确保服务器已安装：
- **Conda** (Miniconda 或 Anaconda)
- **Python 3.9+**

如果没有 Conda，安装 Miniconda：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 手动安装依赖

如果自动安装失败，可以手动安装：

```bash
# 创建环境
conda create -n doptools_env python=3.10 -y
conda activate doptools_env

# 安装核心依赖
pip install pandas>=2.1 numpy>=1.25 scipy>=1.7
pip install scikit-learn>=1.5 matplotlib>=3.4
pip install chython>=1.78 rdkit>=2023.09.02
pip install optuna>=3.5 xgboost>=2.0
pip install tqdm>=4.66.3 timeout-decorator==0.5
pip install xlwt>=1.3 xlrd>=2.0 openpyxl>=3.1 pillow>=11.2.1
pip install ipython>=7.22

# 安装 DOPtools
pip install -e /path/to/DOPtools
```

### 使用 Screen/Tmux 运行 (推荐)

对于长时间运行的任务，使用 screen 或 tmux：

```bash
# 使用 screen
screen -S tg_model
conda activate doptools_env
python build_tg_model.py
# 按 Ctrl+A, D 分离会话
# 重新连接: screen -r tg_model

# 使用 tmux
tmux new -s tg_model
conda activate doptools_env
python build_tg_model.py
# 按 Ctrl+B, D 分离会话
# 重新连接: tmux attach -t tg_model
```

---

## 重要文件说明

| 文件 | 说明 |
|------|------|
| `processed_data.csv` | 你的数据集 (879条记录) |
| `build_tg_model.py` | 主脚本 - 构建 Tg 预测模型 |
| `setup_linux.sh` | 环境安装脚本 |
| `run_model.sh` | 模型运行脚本 |
| `output_tg_model/` | 输出目录 (运行后生成) |

---

## 调整参数

在 `build_tg_model.py` 中可以调整以下参数：

```python
results = launch_study(
    {"circus": descriptors},
    data[["logTg"]],
    "output_tg_model",
    "SVR",          # 方法: SVR, RFR, XGBR
    100,            # 试验次数 (增加以获得更好结果)
    5,              # K-fold 折数
    3,              # 交叉验证重复次数
    192,            # CPU 数量 (Linux 使用 192 核)
    120,            # 超时时间 (秒)
    (0, 0),         # 早停
    True            # 写入文件
)
```

**在 Linux 服务器上，可以将 CPU 数量设置为更高的值（如 4 或 8）以加速优化。**

---

## 问题排查

### 1. Conda 命令找不到
```bash
source ~/.bashrc
# 或
export PATH="$HOME/miniconda3/bin:$PATH"
```

### 2. RDKit 安装失败
```bash
# 尝试从 conda-forge 安装
conda install -c conda-forge rdkit
```

### 3. 内存不足
减少试验次数或 CPU 数量，或使用更大内存的服务器。
