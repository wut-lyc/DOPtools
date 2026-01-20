#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tg (玻璃化转变温度) 预测模型搭建脚本

基于 DOPtools 教程流程:
1. 加载数据并解析 SMILES
2. 计算 CircuS 描述符
3. 使用 Optuna 优化 SVR/RFR/XGBR 模型超参数 (比较三种方法)
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
# 配置参数
# ============================================================
N_TRIALS = 5000         # 每种方法的试验次数
N_FOLDS = 5             # K-fold 折数
N_REPEATS = 3           # 交叉验证重复次数
TIMEOUT = 300           # 超时时间 (秒)
METHODS = ["SVR", "RFR", "XGBR"]  # 要比较的方法

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
# Step 4: 模型优化 - 比较三种方法
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Step 4: 开始模型超参数优化 (比较三种方法)")
    print("=" * 60)
    print(f"配置: {N_TRIALS} 试验/方法, {N_FOLDS}-fold CV, {N_REPEATS} 重复")
    print(f"使用 {N_JOBS} 个 CPU")
    print(f"方法: {METHODS}")
    
    # 存储各方法的最佳结果
    best_results = {}
    
    for method in METHODS:
        print("\n" + "-" * 60)
        print(f"优化 {method} 模型...")
        print("-" * 60)
        
        output_dir = f"output_tg_model/{method}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行优化
        launch_study(
            {"circus": descriptors},      # 描述符空间
            data[["logTg"]],              # 目标变量
            output_dir,                   # 输出目录
            method,                       # 方法
            N_TRIALS,                     # 试验次数
            N_FOLDS,                      # K-fold 折数
            N_REPEATS,                    # 重复次数
            N_JOBS,                       # CPU 数量
            TIMEOUT,                      # 超时时间
            (0, 0),                       # 早停
            True                          # 写入文件
        )
        
        # 读取最佳结果
        trials_file = f"{output_dir}/trials.all"
        if os.path.exists(trials_file):
            trials = pd.read_table(trials_file, sep=" ")
            best = trials.sort_values(by="score", ascending=False).iloc[0]
            best_results[method] = {
                "score": best["score"],
                "trial": best["trial"],
                "desc": best["desc"],
                "scaling": best["scaling"]
            }
            print(f"{method} 最佳 R²: {best['score']:.4f}")
    
    # ============================================================
    # Step 5: 比较结果
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 方法比较结果")
    print("=" * 60)
    
    print("\n各方法最佳 R² 分数:")
    print("-" * 40)
    for method, result in sorted(best_results.items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"{method:6s}: R² = {result['score']:.4f}")
    
    # 找出最佳方法
    if best_results:
        best_method = max(best_results.items(), key=lambda x: x[1]["score"])
        print(f"\n🏆 最佳方法: {best_method[0]} (R² = {best_method[1]['score']:.4f})")
        
        # ============================================================
        # Step 6: 为最佳方法生成回归图
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Step 6: 生成最佳方法 ({best_method[0]}) 的回归图")
        print("=" * 60)
        
        best_method_name = best_method[0]
        best_trial_num = int(best_method[1]["trial"])
        pred_file = f"output_tg_model/{best_method_name}/trial.{best_trial_num}/predictions"
        
        if os.path.exists(pred_file):
            best_predictions = pd.read_table(pred_file, sep=" ")
            
            # 找到观测值和预测值列
            obs_col = [c for c in best_predictions.columns if "observed" in c][0]
            pred_cols = [c for c in best_predictions.columns if "predicted" in c]
            
            # 计算平均预测值
            best_predictions["predicted_avg"] = best_predictions[pred_cols].mean(axis=1)
            
            # 绘制回归图
            fig, ax = plt.subplots(figsize=(8, 8))
            
            observed = best_predictions[obs_col]
            predicted = best_predictions["predicted_avg"]
            
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
            ax.set_title(f"Tg Prediction Model ({best_method_name})\nR² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("output_tg_model/best_model_regression_plot.png", dpi=150)
            print(f"回归图已保存到 output_tg_model/best_model_regression_plot.png")
            print(f"\n最佳模型性能:")
            print(f"  方法: {best_method_name}")
            print(f"  R² = {r2:.4f}")
            print(f"  MAE = {mae:.4f}")
            print(f"  RMSE = {rmse:.4f}")
    
    # ============================================================
    # 输出汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("完成！结果目录结构:")
    print("=" * 60)
    print("output_tg_model/")
    print("├── SVR/           # SVR 优化结果")
    print("├── RFR/           # 随机森林优化结果")
    print("├── XGBR/          # XGBoost 优化结果")
    print("├── circus_descriptors.csv")
    print("├── circus_fragmentor.pkl")
    print("└── best_model_regression_plot.png")
    print("=" * 60)
