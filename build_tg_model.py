#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tg (玻璃化转变温度) 预测模型搭建脚本

基于 DOPtools 教程流程:
1. 加载数据并解析 SMILES
2. 计算 CircuS 描述符
3. 使用 Optuna 优化 SVR 模型超参数
4. 保存结果并可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from chython import smiles
from doptools import ChythonCircus
from doptools.optimizer import launch_study
from doptools.cli.plotter import make_regression_plot

# 多进程设置
import multiprocessing
import platform

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Linux 使用 fork (更快), Windows 使用 spawn
    if platform.system() == "Linux":
        multiprocessing.set_start_method("fork", force=True)
        N_JOBS = 192  # Linux 上使用 192 核
    else:
        N_JOBS = 1  # Windows 建议用 1 避免问题

# ============================================================
# Step 1: 加载数据
# ============================================================
print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

data = pd.read_csv("processed_data.csv")
print(f"加载了 {len(data)} 条记录")
print(f"列名: {data.columns.tolist()}")
print(f"logTg 范围: {data['logTg'].min():.4f} ~ {data['logTg'].max():.4f}")

# ============================================================
# Step 2: 解析 SMILES 并转换为 Chython 分子对象
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 解析 SMILES")
print("=" * 60)

mols = []
failed_idx = []
for i, smi in enumerate(data.SMILES):
    try:
        mol = smiles(smi)
        mol.canonicalize()
        mols.append(mol)
    except Exception as e:
        print(f"警告: 第 {i} 行 SMILES 解析失败: {smi[:50]}... - {e}")
        failed_idx.append(i)
        mols.append(None)

# 过滤失败的分子
if failed_idx:
    print(f"共 {len(failed_idx)} 个分子解析失败，将被移除")
    valid_mask = [m is not None for m in mols]
    data = data[valid_mask].reset_index(drop=True)
    mols = [m for m in mols if m is not None]

print(f"成功解析 {len(mols)} 个分子")

# ============================================================
# Step 3: 计算 CircuS 描述符
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 计算 CircuS 描述符 (radius 0-4)")
print("=" * 60)

circus = ChythonCircus(0, 4)
circus.fit(mols)
descriptors = circus.transform(mols)

print(f"描述符矩阵形状: {descriptors.shape}")
print(f"非零描述符数量: {(descriptors.sum(axis=0) > 0).sum()}")

# 保存描述符
os.makedirs("output_tg_model", exist_ok=True)
descriptors.to_csv("output_tg_model/circus_descriptors.csv")
with open("output_tg_model/circus_fragmentor.pkl", "wb") as f:
    pickle.dump(circus, f)
print("描述符已保存到 output_tg_model/")

# ============================================================
# Step 4: 模型优化 (使用 Optuna + SVR)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 开始模型超参数优化")
print("=" * 60)

if __name__ == "__main__":
    print("使用 SVR 进行回归建模...")
    print("优化参数: descriptor space, scaling, C, kernel, coef0")
    print("这可能需要几分钟时间...")
    
    # 运行优化
    # 参数说明:
    # - 描述符空间字典
    # - 目标变量 (DataFrame 格式)
    # - 输出目录
    # - 方法 (SVR/RFR/XGBR)
    # - 试验次数 (减少以加快速度，正式运行可增加到 500+)
    # - K-fold 折数
    # - 重复次数
    # - CPU 数量 (Windows 建议设为 1 避免多进程问题)
    # - 超时时间 (秒)
    # - 早停参数
    # - 是否写入文件
    
    results = launch_study(
        {"circus": descriptors},      # 描述符空间
        data[["logTg"]],              # 目标变量
        "output_tg_model",            # 输出目录
        "SVR",                        # 方法
        100,                          # 试验次数 (可调整)
        5,                            # K-fold 折数
        3,                            # 重复次数
        N_JOBS,                       # CPU 数量 (Linux=4, Windows=1)
        120,                          # 超时时间
        (0, 0),                       # 早停 (禁用)
        True                          # 写入文件
    )
    
    # ============================================================
    # Step 5: 分析结果
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 分析优化结果")
    print("=" * 60)
    
    trials_table = results[0]
    predictions_dict = results[1]
    
    # 排序获取最佳结果
    best_trials = trials_table.sort_values(by="score", ascending=False)
    print("\n最佳 5 个试验:")
    print(best_trials.head())
    
    best_trial = best_trials.iloc[0]
    print(f"\n最佳模型:")
    print(f"  Trial: {best_trial['trial']}")
    print(f"  R² Score: {best_trial['score']:.4f}")
    print(f"  描述符: {best_trial['desc']}")
    print(f"  缩放: {best_trial['scaling']}")
    print(f"  方法: {best_trial['method']}")
    print(f"  C: {best_trial['C']:.4f}")
    print(f"  Kernel: {best_trial['kernel']}")
    
    # ============================================================
    # Step 6: 生成回归图
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6: 生成回归图")
    print("=" * 60)
    
    # 获取最佳模型的预测结果
    best_predictions = predictions_dict[best_trial['trial']]["predictions"]
    
    # 计算平均预测值
    pred_cols = [c for c in best_predictions.columns if "predicted" in c]
    best_predictions["logTg.predicted.avg"] = best_predictions[pred_cols].mean(axis=1)
    
    # 绘制回归图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    observed = best_predictions["logTg.observed"]
    predicted = best_predictions["logTg.predicted.avg"]
    
    ax.scatter(observed, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # 添加对角线
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    # 计算统计指标
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(observed, predicted)
    mae = mean_absolute_error(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    
    ax.set_xlabel("Observed logTg", fontsize=12)
    ax.set_ylabel("Predicted logTg", fontsize=12)
    ax.set_title(f"Tg Prediction Model\nR² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("output_tg_model/regression_plot.png", dpi=150)
    print("回归图已保存到 output_tg_model/regression_plot.png")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("完成！所有结果已保存到 output_tg_model/ 目录")
    print("=" * 60)
