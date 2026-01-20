#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子描述符计算脚本
=================
计算多种分子描述符（RDKit, Mordred, ChythonCircus, ChythonLinear）
并自动进行数据清洗和共线性处理。

使用方法:
    python calculate_descriptors.py \\
        --input data.csv \\
        --smiles-col SMILES \\
        --target-col logTg \\
        --output descriptors_clean.csv \\
        --variance-threshold 0.03 \\
        --collinearity-threshold 0.8 \\
        --n-jobs 192
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from chython import smiles
import warnings
warnings.filterwarnings('ignore')

from doptools.chem.chem_features import (
    ChythonCircus,
    ChythonLinear,
    Mordred2DCalculator,
    RDKit2DCalculator
)


def parse_molecules(smiles_list):
    """解析SMILES为分子对象"""
    mols = []
    failed_idx = []
    
    for i, smi in enumerate(smiles_list):
        try:
            mol = smiles(smi)
            mol.canonicalize()
            mols.append(mol)
        except Exception as e:
            print(f"警告: 第 {i} 行 SMILES 解析失败: {smi[:50]}... - {e}")
            failed_idx.append(i)
            mols.append(None)
    
    return mols, failed_idx


def calculate_descriptors(mols, smiles_list, args):
    """计算所有描述符"""
    all_descriptors = []
    descriptor_names = []
    
    print("\\n" + "="*60)
    print("计算分子描述符")
    print("="*60)
    
    # 1. RDKit 2D描述符
    if not args.skip_rdkit:
        print("\\n[1/4] 计算 RDKit 2D 描述符...")
        try:
            rdkit_calc = RDKit2DCalculator(fmt="mol")
            rdkit_calc.fit(mols)
            rdkit_desc = rdkit_calc.transform(mols)
            all_descriptors.append(rdkit_desc)
            descriptor_names.append("RDKit2D")
            print(f"  ✓ RDKit: {rdkit_desc.shape[1]} 个描述符")
        except Exception as e:
            print(f"  ✗ RDKit 计算失败: {e}")
    
    # 2. Mordred 2D描述符
    if not args.skip_mordred:
        print("\\n[2/4] 计算 Mordred 2D 描述符...")
        try:
            mordred_calc = Mordred2DCalculator(fmt="smiles")
            mordred_calc.fit(smiles_list)
            mordred_desc = mordred_calc.transform(smiles_list)
            all_descriptors.append(mordred_desc)
            descriptor_names.append("Mordred2D")
            print(f"  ✓ Mordred: {mordred_desc.shape[1]} 个描述符")
        except Exception as  e:
            print(f"  ✗ Mordred 计算失败: {e}")
    
    # 3. ChythonCircus描述符
    if not args.skip_circus:
        print("\\n[3/4] 计算 ChythonCircus 描述符...")
        try:
            circus_calc = ChythonCircus(
                lower=args.circus_lower,
                upper=args.circus_upper,
                fmt="mol"
            )
            circus_calc.fit(mols)
            circus_desc = circus_calc.transform(mols)
            all_descriptors.append(circus_desc)
            descriptor_names.append("CircuS")
            print(f"  ✓ CircuS (radius {args.circus_lower}-{args.circus_upper}): {circus_desc.shape[1]} 个描述符")
        except Exception as e:
            print(f"  ✗ CircuS 计算失败: {e}")
    
    # 4. ChythonLinear描述符
    if not args.skip_linear:
        print("\\n[4/4] 计算 ChythonLinear 描述符...")
        try:
            linear_calc = ChythonLinear(
                lower=args.linear_lower,
                upper=args.linear_upper,
                fmt="mol"
            )
            linear_calc.fit(mols)
            linear_desc = linear_calc.transform(mols)
            all_descriptors.append(linear_desc)
            descriptor_names.append("ChyLine")
            print(f"  ✓ ChyLine (length {args.linear_lower}-{args.linear_upper}): {linear_desc.shape[1]} 个描述符")
        except Exception as e:
            print(f"  ✗ ChyLine 计算失败: {e}")
    
    # 合并所有描述符
    if all_descriptors:
        combined = pd.concat(all_descriptors, axis=1)
        print(f"\\n合并后总计: {combined.shape[1]} 个描述符")
        return combined
    else:
        raise ValueError("没有成功计算任何描述符！")


def clean_descriptors(X, variance_threshold=0.03):
    """数据清洗：删除缺失值、常数列、低方差列"""
    print("\\n" + "="*60)
    print("数据清洗")
    print("="*60)
    
    initial_shape = X.shape
    print(f"初始形状: {initial_shape}")
    
    # 1. 删除含缺失值的列
    print("\\n[1/3] 删除含缺失值的列...")
    n_missing = X.isna().sum().sum()
    X_clean = X.dropna(axis=1)
    removed = initial_shape[1] - X_clean.shape[1]
    print(f"  - 发现 {n_missing} 个缺失值")
    print(f"  - 删除 {removed} 列")
    
    # 2. 删除常数列
    print("\\n[2/3] 删除常数列（方差=0）...")
    constant_cols = (X_clean.std() == 0).sum()
    X_clean = X_clean.loc[:, X_clean.std() != 0]
    print(f"  - 删除 {constant_cols} 个常数列")
    
    # 3. 删除低方差列
    print(f"\\n[3/3] 删除低方差列（方差 < {variance_threshold}）...")
    low_var_cols = (X_clean.std() < variance_threshold).sum()
    X_clean = X_clean.loc[:, X_clean.std() >= variance_threshold]
    print(f"  - 删除 {low_var_cols} 个低方差列")
    
    print(f"\\n清洗后形状: {X_clean.shape}")
    print(f"保留率: {X_clean.shape[1] / initial_shape[1] * 100:.1f}%")
    
    return X_clean


def handle_collinearity(X, y, threshold=0.8):
    """处理共线性：删除高度相关的描述符"""
    print("\\n" + "="*60)
    print(f"共线性处理 (R² > {threshold})")
    print("="*60)
    
    initial_cols = X.shape[1]
    
    # 计算描述符之间的相关矩阵
    print("\\n计算相关矩阵...")
    corr_matrix = X.corr()
    
    # 计算R²矩阵
    r_squared = corr_matrix ** 2
    np.fill_diagonal(r_squared.values, 0)  # 对角线设为0
    
    # 计算每个描述符与目标变量的相关性
    print("计算与目标变量的相关性...")
    y_corr = X.corrwith(y).abs()
    
    # 找出需要删除的列
    to_drop = set()
    n_pairs = 0
    
    print(f"\\n识别共线性描述符对...")
    for i in range(len(r_squared.columns)):
        if r_squared.columns[i] in to_drop:
            continue
            
        for j in range(i+1, len(r_squared.columns)):
            if r_squared.columns[j] in to_drop:
                continue
                
            if r_squared.iloc[i, j] > threshold:
                n_pairs += 1
                col_i = r_squared.columns[i]
                col_j = r_squared.columns[j]
                
                # 保留与y相关性更大的那个
                if y_corr[col_i] >= y_corr[col_j]:
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)
    
    print(f"  - 发现 {n_pairs} 对高度相关的描述符")
    print(f"  - 标记删除 {len(to_drop)} 个描述符")
    
    # 删除列
    X_reduced = X.drop(columns=list(to_drop))
    
    print(f"\\n处理后形状: {X_reduced.shape}")
    print(f"保留率: {X_reduced.shape[1] / initial_cols * 100:.1f}%")
    
    return X_reduced


def main():
    parser = argparse.ArgumentParser(
        description='计算分子描述符并进行数据清洗和共线性处理',
        epilog='示例: python calculate_descriptors.py -i data.csv --smiles-col SMILES --target-col logTg -o output.csv'
    )
    
    # 输入输出参数
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--smiles-col', default='SMILES', help='SMILES列名 (默认: SMILES)')
    parser.add_argument('--target-col', required=True, help='目标变量列名 (如: logTg)')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件路径')
    
    # 描述符计算参数
    parser.add_argument('--skip-rdkit', action='store_true', help='跳过RDKit描述符')
    parser.add_argument('--skip-mordred', action='store_true', help='跳过Mordred描述符')
    parser.add_argument('--skip-circus', action='store_true', help='跳过CircuS描述符')
    parser.add_argument('--skip-linear', action='store_true', help='跳过ChyLine描述符')
    
    parser.add_argument('--circus-lower', type=int, default=0, help='CircuS最小半径 (默认: 0)')
    parser.add_argument('--circus-upper', type=int, default=4, help='CircuS最大半径 (默认: 4)')
    parser.add_argument('--linear-lower', type=int, default=0, help='ChyLine最小长度 (默认: 0)')
    parser.add_argument('--linear-upper', type=int, default=4, help='ChyLine最大长度 (默认: 4)')
    
    # 数据清洗参数
    parser.add_argument('--variance-threshold', type=float, default=0.03, 
                       help='低方差阈值 (默认: 0.03)')
    parser.add_argument('--collinearity-threshold', type=float, default=0.8,
                       help='共线性R²阈值 (默认: 0.8)')
    
    #并行计算参数
    parser.add_argument('--n-jobs', type=int, default=1, help='并行核心数 (默认: 1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("分子描述符计算工具")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"SMILES列: {args.smiles_col}")
    print(f"目标列: {args.target_col}")
    print(f"输出文件: {args.output}")
    print(f"方差阈值: {args.variance_threshold}")
    print(f"共线性阈值: R² > {args.collinearity_threshold}")
    
    # 1. 加载数据
    print("\\n" + "="*60)
    print("加载数据")
    print("="*60)
    data = pd.read_csv(args.input)
    print(f"读取 {len(data)} 条记录")
    print(f"列名: {list(data.columns)}")
    
    #2. 解析SMILES
    print("\\n" + "="*60)
    print("解析SMILES")
    print("="*60)
    mols, failed_idx = parse_molecules(data[args.smiles_col])
    
    # 过滤失败的分子
    if failed_idx:
        print(f"\\n移除 {len(failed_idx)} 个解析失败的分子")
        valid_mask = [m is not None for m in mols]
        data = data[valid_mask].reset_index(drop=True)
        mols = [m for m in mols if m is not None]
    
    print(f"成功解析 {len(mols)} 个分子")
    
    # 提取目标变量
    y = data[args.target_col]
    smiles_list = data[args.smiles_col].tolist()
    
    # 3. 计算描述符
    X = calculate_descriptors(mols, smiles_list, args)
    
    # 4. 数据清洗
    X_clean = clean_descriptors(X, args.variance_threshold)
    
    # 5. 共线性处理
    X_final = handle_collinearity(X_clean, y, args.collinearity_threshold)
    
    # 6. 保存结果
    print("\\n" + "="*60)
    print("保存结果")
    print("="*60)
    
    # 合并ID、SMILES、目标变量和描述符
    result = pd.concat([
        data[['ID', args.smiles_col, args.target_col]] if 'ID' in data.columns else data[[args.smiles_col, args.target_col]],
        X_final
    ], axis=1)
    
    result.to_csv(args.output, index=False)
    print(f"✓ 已保存到: {args.output}")
    print(f"  - 行数: {result.shape[0]}")
    print(f"  - 描述符数: {X_final.shape[1]}")
    print(f"  - 总列数: {result.shape[1]}")
    
    print("\\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
