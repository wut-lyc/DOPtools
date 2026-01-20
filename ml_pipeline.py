#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整机器学习Pipeline - 分子描述符计算与模型训练
=============================================
从描述符计算到模型评估的完整workflow

Pipeline流程：
1. 描述符计算(RDKit2D, Mordred2D, CircuS, ChyLine)
2. 缺失值处理(均值/中位数/删除)
3. 标准化/归一化
4. 数据清洗(去方差0/同值95%/高相关>0.95)
5. 特征选择(无/RF-15,20,50/RFE-15,20,50/RFECV)
6. 模型训练(7种模型+网格搜索)
7. 模型评估(R2, Q_LOO2, Q_LMO2, y-randomization)
8. 结果可视化

使用方法：
    python ml_pipeline.py \\
        --input processed_data.csv \\
        --smiles-col SMILES \\
        --target-col logTg \\
        --output-dir results \\
        --n-jobs 192
"""

import argparse
import os
import pickle
import warnings
import itertools
from pathlib import Path
from datetime import datetime
from rdkit import Chem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, GridSearchCV, LeaveOneOut, cross_val_score, KFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from chython import smiles

from doptools.chem.chem_features import (
    ChythonCircus, ChythonLinear,
    Mordred2DCalculator, RDKit2DCalculator
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class MLPipeline:
    """完整机器学习Pipeline"""
    
    def __init__(self, output_dir, n_jobs=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        
        # 存储结果
        self.results = []
        self.best_models = []
        
    def calculate_descriptors(self, data, smiles_col):
        """步骤1: 计算描述符"""
        print("\\n" + "="*70)
        print("步骤1: 计算分子描述符")
        print("="*70)
        
        # 解析分子
        print("\\n解析SMILES...")
        mols = []
        smiles_list = data[smiles_col].tolist()
        
        for i, smi in enumerate(smiles_list):
            try:
                mol = smiles(smi)
                mol.canonicalize()
                # 验证RDKit兼容性
                if Chem.MolFromSmiles(str(mol)) is None:
                    print(f"警告: 第 {i} 行 SMILES 被RDKit拒绝: {smi[:50]}...")
                    mols.append(None)
                else:
                    mols.append(mol)
            except:
                mols.append(None)
        
        valid_mask = [m is not None for m in mols]
        data_clean = data[valid_mask].reset_index(drop=True)
        mols_clean = [m for m in mols if m is not None]
        smiles_clean = [s for s, v in zip(smiles_list, valid_mask) if v]
        
        # 创建RDKit分子列表 (用于Mordred和RDKit计算器)
        mols_rdkit = [Chem.MolFromSmiles(str(m)) for m in mols_clean]
        
        print(f"成功解析 {len(mols_clean)}/{len(mols)} 个分子")
        
        # 计算描述符
        all_descriptors = []
        
        # RDKit2D
        print("\\n[1/4] RDKit 2D...")
        rdkit_calc = RDKit2DCalculator(fmt="mol")
        rdkit_calc.fit(mols_rdkit)
        rdkit_desc = rdkit_calc.transform(mols_rdkit)
        rdkit_desc.columns = ["RDKit_" + c for c in rdkit_desc.columns]
        all_descriptors.append(rdkit_desc)
        print(f"  ✓ {rdkit_desc.shape[1]} 个描述符")
        
        # Mordred2D
        print("\\n[2/4] Mordred 2D...")
        mordred_calc = Mordred2DCalculator(fmt="mol")
        mordred_calc.fit(mols_rdkit)
        mordred_desc = mordred_calc.transform(mols_rdkit)
        mordred_desc.columns = ["Mordred_" + c for c in mordred_desc.columns]
        all_descriptors.append(mordred_desc)
        print(f"  ✓ {mordred_desc.shape[1]} 个描述符")
        
        # CircuS
        print("\\n[3/4] ChythonCircus...")
        circus_calc = ChythonCircus(lower=0, upper=4, fmt="mol")
        circus_calc.fit(mols_clean)
        circus_desc = circus_calc.transform(mols_clean)
        circus_desc.columns = ["CircuS_" + str(c) for c in circus_desc.columns]
        all_descriptors.append(circus_desc)
        print(f"  ✓ {circus_desc.shape[1]} 个描述符")
        
        # ChyLine
        print("\\n[4/4] ChythonLinear...")
        linear_calc = ChythonLinear(lower=0, upper=4, fmt="mol")
        linear_calc.fit(mols_clean)
        linear_desc = linear_calc.transform(mols_clean)
        linear_desc.columns = ["ChyLine_" + str(c) for c in linear_desc.columns]
        all_descriptors.append(linear_desc)
        print(f"  ✓ {linear_desc.shape[1]} 个描述符")
        
        # 合并
        X = pd.concat(all_descriptors, axis=1)
        print(f"\\n总计: {X.shape[1]} 个描述符")
        
        return data_clean, X
    
    def handle_missing_values(self, X, method):
        """步骤2: 缺失值处理"""
        print(f"\\n缺失值处理方法: {method}")
        
        if method == 'drop':
            X_filled = X.dropna(axis=1)
            print(f"  删除 {X.shape[1] - X_filled.shape[1]} 个含缺失值的列")
        elif method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            X_filled = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print(f"  使用均值填充")
        else:  # median
            imputer = SimpleImputer(strategy='median')
            X_filled = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print(f"  使用中位数填充")
        
        return X_filled
    
    def scale_data(self, X, method):
        """步骤3: 标准化/归一化"""
        print(f"\\n数据缩放方法: {method}")
        
        if method == 'standard':
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print("  使用标准化 (零均值, 单位方差)")
        else:  # minmax
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print("  使用归一化 [0, 1]")
        
        return X_scaled, scaler
    
    def clean_features(self, X, y):
        """步骤4: 数据清洗"""
        print("\\n" + "="*70)
        print("步骤4: 数据清洗")
        print("="*70)
        
        initial_n = X.shape[1]
        
        # 1. 去除方差为0的特征
        print("\\n[1/3] 去除方差为0的特征...")
        zero_var = X.std() == 0
        X_clean = X.loc[:, ~zero_var]
        print(f"  删除 {zero_var.sum()} 个零方差特征")
        
        # 2. 去除95%样本取同一值的特征
        print("\\n[2/3] 去除95%样本同值的特征...")
        to_drop = []
        for col in X_clean.columns:
            value_counts = X_clean[col].value_counts()
            if len(value_counts) > 0:
                max_freq = value_counts.iloc[0] / len(X_clean)
                if max_freq >= 0.95:
                    to_drop.append(col)
        X_clean = X_clean.drop(columns=to_drop)
        print(f"  删除 {len(to_drop)} 个高同值特征")
        
        # 3. 去除高相关性特征 (>0.95)
        print("\\n[3/3] 去除高相关性特征 (>0.95)...")
        corr_matrix = X_clean.corr().abs()
        target_corr = X_clean.corrwith(y).abs()
        
        to_drop_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                
                if corr_matrix.iloc[i, j] > 0.95:
                    if target_corr[col_i] < target_corr[col_j]:
                        to_drop_corr.add(col_i)
                    else:
                        to_drop_corr.add(col_j)
        
        X_clean = X_clean.drop(columns=list(to_drop_corr))
        print(f"  删除 {len(to_drop_corr)} 个高相关特征")
        
        print(f"\\n清洗结果: {initial_n} → {X_clean.shape[1]} 特征")
        return X_clean
    
    def select_features(self, X_train, y_train, X_test, method, n_features=None):
        """步骤5: 特征选择"""
        print(f"\\n特征选择: {method}", end="")
        if n_features:
            print(f" (保留{n_features}个)")
        else:
            print()
        
        if method == 'none':
            return X_train, X_test, None
        
        elif method.startswith('rf'):  # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            selected_cols = X_train.columns[indices]
            
            return X_train[selected_cols], X_test[selected_cols], selected_cols.tolist()
        
        elif method.startswith('rfe'):  # RFE
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=self.n_jobs)
            rfe = RFE(rf, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)
            selected_cols = X_train.columns[rfe.support_]
            
            return X_train[selected_cols], X_test[selected_cols], selected_cols.tolist()
        
        elif method == 'rfecv':  # RFECV
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=self.n_jobs)
            rfecv = RFECV(rf, step=1, cv=5, scoring='r2', n_jobs=self.n_jobs, min_features_to_select=10)
            rfecv.fit(X_train, y_train)
            selected_cols = X_train.columns[rfecv.support_]
            print(f"  (自动选择{len(selected_cols)}个特征)")
            
            return X_train[selected_cols], X_test[selected_cols], selected_cols.tolist()
    
    def get_model_configs(self):
        """获取所有模型及其参数网格"""
        configs = {
            'DecisionTree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'ccp_alpha': [0.0, 0.01, 0.02, 0.05]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=self.n_jobs),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'SVR': {
                'model': SVR(cache_size=2000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsRegressor(n_jobs=self.n_jobs),
                'params': {
                    'n_neighbors': [3, 5, 7, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }
            },
            'GaussianProcess': {
                'model': GaussianProcessRegressor(random_state=42, n_restarts_optimizer=2),
                'params': {
                    'kernel': [
                        ConstantKernel() * RBF(),
                        ConstantKernel() * RBF() + WhiteKernel(),
                    ],
                    'alpha': [1e-10, 1e-8, 1e-6]
                }
            },
            'KernelRidge': {
                'model': KernelRidge(),
                'params': {
                    'alpha': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': [None, 0.1, 1]
                }
            }
        }
        
        return configs
    
    def calculate_q2_loo(self, model, X, y):
        """计算Q_LOO^2"""
        loo = LeaveOneOut()
        y_pred = []
        y_true = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test.iloc[0])
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        q2_loo = 1 - (ss_res / ss_tot)
        
        return q2_loo
    
    def calculate_q2_lmo(self, model, X, y, test_size=0.2, n_iterations=100):
        """计算Q_LMO^2 (Leave-Many-Out)"""
        q2_scores = []
        
        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
            q2 = 1 - (ss_res / ss_tot)
            q2_scores.append(q2)
        
        return np.mean(q2_scores)
    
    def y_randomization_test(self, model, X, y, n_iterations=100):
        """Y随机化检验"""
        print(f"\\n  Y随机化检验 ({n_iterations}次)...")
        random_r2_scores = []
        
        for i in range(n_iterations):
            y_shuffled = y.sample(frac=1, random_state=i).reset_index(drop=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_shuffled, test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_random = r2_score(y_test, y_pred)
            random_r2_scores.append(r2_random)
        
        return random_r2_scores
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, 
                          pipeline_config, X_full, y_full, config_id, total_configs):
        """步骤6-7: 模型训练与评估"""
        configs = self.get_model_configs()
        results_list = []
        
        model_idx = 0
        total_models = len(configs)
        
        for model_name, config in configs.items():
            model_idx += 1
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\\n[{now_str}] [配置 {config_id}/{total_configs}] [模型 {model_idx}/{total_models}] 训练 {model_name}...", flush=True)
            
            # 网格搜索
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='r2',
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # 预测
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # 基本指标
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            print(f"  > R²(test)={r2_test:.4f} | R²(train)={r2_train:.4f} | RMSE={rmse_test:.4f}")
            print(f"  > 最佳参数: {grid_search.best_params_}")
            
            # Q² 计算
            print(f"  > 计算 Q² (LOO & LMO)...", flush=True)
            q2_loo = self.calculate_q2_loo(best_model, X_full, y_full)
            q2_lmo = self.calculate_q2_lmo(best_model, X_full, y_full)
            print(f"  > Q²_LOO={q2_loo:.4f} | Q²_LMO={q2_lmo:.4f}")
            
            # Y随机化
            y_random_r2 = self.y_randomization_test(best_model, X_full, y_full, n_iterations=50)
            y_random_mean = np.mean(y_random_r2)
            print(f"Y随机化R²均值: {y_random_mean:.4f}")
            
            # 保存结果
            result = {
                'model_name': model_name,
                'pipeline_config': pipeline_config,
                'best_params': grid_search.best_params_,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'mae_test': mae_test,
                'q2_loo': q2_loo,
                'q2_lmo': q2_lmo,
                'y_random_r2_mean': y_random_mean,
                'y_random_r2_std': np.std(y_random_r2),
                'best_model': best_model,
                'y_pred_test': y_pred_test,
                'y_test': y_test
            }
            
            results_list.append(result)
        
        return results_list
    
    def run_pipeline(self, data, smiles_col, target_col):
        """运行完整Pipeline"""
        y = data[target_col]
        
        # 步骤1: 计算描述符
        data_clean, X_raw = self.calculate_descriptors(data, smiles_col)
        y = data_clean[target_col]
       
        # 定义pipeline配置
        missing_methods = ['drop', 'mean', 'median']
        scale_methods = ['standard', 'minmax']
        feature_methods = [
            ('none', None),
            ('rf', 15), ('rf', 20), ('rf', 50),
            ('rfe', 15), ('rfe', 20), ('rfe', 50),
            ('rfecv', None)
        ]
        
        all_results = []
        config_id = 0
        
        # 遍历所有组合
        total_configs = len(missing_methods) * len(scale_methods) * len(feature_methods)
        
        for missing_method in missing_methods:
            # 步骤2: 缺失值处理
            X_filled = self.handle_missing_values(X_raw, missing_method)
            
            for scale_method in scale_methods:
                # 步骤3: 缩放
                X_scaled, scaler = self.scale_data(X_filled, scale_method)
                
                # 步骤4: 清洗
                X_clean = self.clean_features(X_scaled, y)
                
                for feat_method, n_features in feature_methods:
                    config_id += 1
                    print(f"\\n{'#'*80}")
                    print(f"[进度: {config_id}/{total_configs}] 处理配置")
                    print(f"  - 缺失值处理: {missing_method}")
                    print(f"  - 数据缩放:   {scale_method}")
                    print(f"  - 特征选择:   {feat_method} (n={n_features})")
                    print(f"{'#'*80}")
                    
                    # 划分训练测试集
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_clean, y, test_size=0.2, random_state=42
                    )
                    
                    # 步骤5: 特征选择
                    X_train_sel, X_test_sel, selected_features = self.select_features(
                        X_train, y_train, X_test, feat_method, n_features
                    )
                    
                    # 重建完整数据集用于交叉验证
                    if selected_features:
                        X_full_sel = X_clean[selected_features]
                    else:
                        X_full_sel = X_clean
                    
                    pipeline_config = {
                        'missing': missing_method,
                        'scaling': scale_method,
                        'feature_selection': feat_method,
                        'n_features': len(selected_features) if selected_features else X_clean.shape[1]
                    }
                    
                    # 步骤6-7: 训练与评估
                    results = self.train_and_evaluate(
                        X_train_sel, X_test_sel, y_train, y_test,
                        pipeline_config, X_full_sel, y,
                        config_id, total_configs
                    )
                    
                    all_results.extend(results)
        
        self.results = all_results
        return all_results
    
    def plot_results(self):
        """绘制结果对比图"""
        print("\\n" + "="*70)
        print("生成可视化结果")
        print("="*70)
        
        df_results = pd.DataFrame([{
            'Model': r['model_name'],
            'Config': f"{r['pipeline_config']['missing']}-{r['pipeline_config']['scaling']}-{r['pipeline_config']['feature_selection']}",
            'R2_test': r['r2_test'],
            'Q2_LOO': r['q2_loo'],
            'Q2_LMO': r['q2_lmo'],
            'RMSE': r['rmse_test']
        } for r in self.results])
        
        # 1. Top 10 模型对比
        top_10 = df_results.nlargest(10, 'R2_test')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R2对比
        axes[0, 0].barh(range(10), top_10['R2_test'])
        axes[0, 0].set_yticks(range(10))
        axes[0, 0].set_yticklabels([f"{m}-{c[:20]}" for m, c in zip(top_10['Model'], top_10['Config'])])
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Top 10 Models - R² Test')
        axes[0, 0].invert_yaxis()
        
        # Q2_LOO对比
        axes[0, 1].barh(range(10), top_10['Q2_LOO'])
        axes[0, 1].set_yticks(range(10))
        axes[0, 1].set_yticklabels([f"{m}-{c[:20]}" for m, c in zip(top_10['Model'], top_10['Config'])])
        axes[0, 1].set_xlabel('Q²_LOO Score')
        axes[0, 1].set_title('Top 10 Models - Q²_LOO')
        axes[0, 1].invert_yaxis()
        
        # Q2_LMO对比
        axes[1, 0].barh(range(10), top_10['Q2_LMO'])
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([f"{m}-{c[:20]}" for m, c in zip(top_10['Model'], top_10['Config'])])
        axes[1, 0].set_xlabel('Q²_LMO Score')
        axes[1, 0].set_title('Top 10 Models - Q²_LMO')
        axes[1, 0].invert_yaxis()
        
        # RMSE对比
        axes[1, 1].barh(range(10), top_10['RMSE'])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([f"{m}-{c[:20]}" for m, c in zip(top_10['Model'], top_10['Config'])])
        axes[1, 1].set_xlabel('RMSE')
        axes[1, 1].set_title('Top 10 Models - RMSE')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_10_models_comparison.png', dpi=300)
        print(f"✓ 已保存: {self.output_dir / 'top_10_models_comparison.png'}")
        
        # 2. 最佳模型预测图
        best_result = max(self.results, key=lambda x: x['r2_test'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(best_result['y_test'], best_result['y_pred_test'], alpha=0.6, edgecolors='k')
        
        min_val = min(best_result['y_test'].min(), best_result['y_pred_test'].min())
        max_val = max(best_result['y_test'].max(), best_result['y_pred_test'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        ax.set_xlabel('Observed', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f"Best Model: {best_result['model_name']}\\nR² = {best_result['r2_test']:.4f}, Q²_LOO = {best_result['q2_loo']:.4f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_model_prediction.png', dpi=300)
        print(f"✓ 已保存: {self.output_dir / 'best_model_prediction.png'}")
        
        # 3. 保存CSV结果
        df_results.to_csv(self.output_dir / 'all_results.csv', index=False)
        print(f"✓ 已保存: {self.output_dir / 'all_results.csv'}")
        
        return df_results


def main():
    parser = argparse.ArgumentParser(description='完整ML Pipeline')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件')
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES列名')
    parser.add_argument('--target-col', required=True, help='目标变量列名')
    parser.add_argument('--output-dir', default='ml_results', help='输出目录')
    parser.add_argument('--n-jobs', type=int, default=1, help='并行核心数')
    
    args = parser.parse_args()
    
    # 加载数据
    print("="*70)
    print("完整机器学习Pipeline")
    print("="*70)
    print(f"输入文件: {args.input}")
    print(f"目标变量: {args.target_col}")
    print(f"输出目录: {args.output_dir}")
    print(f"并行核心数: {args.n_jobs}")
    
    data = pd.read_csv(args.input)
    print(f"\\n加载了 {len(data)} 条记录")
    
    # 运行pipeline
    pipeline = MLPipeline(args.output_dir, args.n_jobs)
    results = pipeline.run_pipeline(data, args.smiles_col, args.target_col)
    
    # 可视化
    df_results = pipeline.plot_results()
    
    # 输出最佳模型
    best_result = max(results, key=lambda x: x['r2_test'])
    print("\\n" + "="*70)
    print("最佳模型")
    print("="*70)
    print(f"模型: {best_result['model_name']}")
    print(f"配置: {best_result['pipeline_config']}")
    print(f"最佳参数: {best_result['best_params']}")
    print(f"\\nR² (test):  {best_result['r2_test']:.4f}")
    print(f"Q²_LOO:     {best_result['q2_loo']:.4f}")
    print(f"Q²_LMO:     {best_result['q2_lmo']:.4f}")
    print(f"RMSE:       {best_result['rmse_test']:.4f}")
    print(f"\\nY随机化R² (mean±std): {best_result['y_random_r2_mean']:.4f} ± {best_result['y_random_r2_std']:.4f}")
    
    # 保存最佳模型
    with open(Path(args.output_dir) / 'best_model.pkl', 'wb') as f:
        pickle.dump(best_result['best_model'], f)
    print(f"\\n✓ 最佳模型已保存")
    
    print("\\n" + "="*70)
    print("Pipeline完成！")
    print("="*70)


if __name__ == "__main__":
    main()
