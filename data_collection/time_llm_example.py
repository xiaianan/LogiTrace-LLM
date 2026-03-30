"""
Time-LLM 数据加载示例
演示如何使用收集的数据集进行模型训练
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_time_llm_data(filepath='data_collection/storage/processed/time_llm_dataset.csv'):
    """加载Time-LLM格式的数据"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def prepare_features(df):
    """准备特征矩阵"""
    # 选择特征列
    feature_cols = [
        'ddr4_spot_price', 'ddr4_contract_price',
        'nand_spot_price', 'nand_contract_price',
        'dxi_index',
        'ddr4_spot_contract_gap', 'nand_spot_contract_gap'
    ]

    # 移除有缺失值的行
    df_clean = df[feature_cols].dropna()

    return df_clean, feature_cols


def split_train_test(df, train_ratio=0.8):
    """时间序列数据划分 (不能随机划分)"""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def normalize_features(train_df, test_df, feature_cols):
    """标准化特征"""
    scaler = StandardScaler()

    # 训练集拟合
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    # 测试集转换
    test_scaled = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, scaler


def visualize_data(df):
    """可视化数据"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 价格走势
    axes[0].plot(df['date'], df['ddr4_spot_price'], label='DDR4 Spot', alpha=0.8)
    axes[0].plot(df['date'], df['ddr4_contract_price'], label='DDR4 Contract', alpha=0.8)
    axes[0].set_title('DRAM Price Trends')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # NAND价格
    axes[1].plot(df['date'], df['nand_spot_price'], label='NAND Spot', color='orange', alpha=0.8)
    axes[1].plot(df['date'], df['nand_contract_price'], label='NAND Contract', color='red', alpha=0.8)
    axes[1].set_title('NAND Flash Price Trends')
    axes[1].set_ylabel('Price (USD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # DXI指数
    axes[2].plot(df['date'], df['dxi_index'], label='DXI Index', color='green', alpha=0.8)
    axes[2].set_title('DXI Index Trend')
    axes[2].set_ylabel('DXI Index')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_trends.png', dpi=150)
    plt.close()
    print("可视化图表已保存: price_trends.png")


def main():
    """主函数"""
    print("=" * 60)
    print("Time-LLM 数据加载示例")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载数据集...")
    df = load_time_llm_data()
    print(f"   总记录数: {len(df)}")
    print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")

    # 查看数据
    print("\n2. 数据预览:")
    print(df.head())

    print("\n3. 数据统计:")
    print(df.describe())

    # 准备特征
    print("\n4. 准备特征...")
    df_features, feature_cols = prepare_features(df)
    print(f"   特征列: {feature_cols}")
    print(f"   有效记录数: {len(df_features)}")

    # 划分训练集和测试集
    print("\n5. 划分训练集和测试集...")
    train_df, test_df = split_train_test(df_features, train_ratio=0.8)
    print(f"   训练集: {len(train_df)} 条")
    print(f"   测试集: {len(test_df)} 条")

    # 标准化
    print("\n6. 标准化特征...")
    train_scaled, test_scaled, scaler = normalize_features(train_df, test_df, feature_cols)
    print(f"   训练集形状: {train_scaled.shape}")
    print(f"   测试集形状: {test_scaled.shape}")

    # 可视化
    print("\n7. 生成可视化图表...")
    visualize_data(df)

    # 保存处理后的数据
    print("\n8. 保存处理后的数据...")
    np.save('train_data.npy', train_scaled)
    np.save('test_data.npy', test_scaled)
    print("   训练数据已保存: train_data.npy")
    print("   测试数据已保存: test_data.npy")

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)

    return train_scaled, test_scaled


if __name__ == '__main__':
    main()
