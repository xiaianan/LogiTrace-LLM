"""
数据清洗与预处理模块
整合、清洗多数据源，生成Time-LLM可用的格式
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""

    def __init__(self, config_path='config/settings.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_raw_data(self, spot_file, contract_file, dxi_file):
        """加载原始数据文件"""
        spot_df = pd.read_csv(spot_file) if spot_file else None
        contract_df = pd.read_csv(contract_file) if contract_file else None
        dxi_df = pd.read_csv(dxi_file) if dxi_file else None

        return spot_df, contract_df, dxi_df

    def clean_spot_prices(self, df):
        """清洗现货价格数据"""
        logger.info("清洗现货价格数据...")

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 移除无效日期
        df = df.dropna(subset=['date'])

        # 标准化价格列
        if 'price_usd' in df.columns:
            df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')

        # 按产品类型和规格标准化
        df['product_type'] = df['product_type'].str.upper().str.strip()

        # 过滤目标规格
        target_specs = ['DDR4', 'DDR5', 'NAND']
        df = df[df['product_type'].isin(target_specs)]

        # 移除异常值 (使用IQR方法)
        for product in target_specs:
            mask = df['product_type'] == product
            if mask.sum() > 0:
                q1 = df.loc[mask, 'price_usd'].quantile(0.25)
                q3 = df.loc[mask, 'price_usd'].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = mask & ((df['price_usd'] < lower) | (df['price_usd'] > upper))
                df = df[~outliers]

        # 按日期排序
        df = df.sort_values(['product_type', 'date'])

        logger.info(f"清洗后现货数据: {len(df)} 条记录")
        return df

    def clean_contract_prices(self, df):
        """清洗合约价格数据"""
        logger.info("清洗合约价格数据...")

        # 转换价格
        df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')

        # 转换合约期间为日期
        df['contract_start_date'] = df['contract_period'].apply(self._parse_contract_period)

        return df

    def _parse_contract_period(self, period_str):
        """解析合约期间为日期"""
        try:
            # 常见格式: "2024 Q1", "Mar 2024", "2024-03"
            if 'Q' in period_str:
                year, quarter = period_str.split()
                year = int(year)
                q = int(quarter[1])
                month = (q - 1) * 3 + 1
                return pd.Timestamp(f"{year}-{month:02d}-01")
            else:
                return pd.to_datetime(period_str, errors='coerce')
        except:
            return None

    def clean_dxi_data(self, df):
        """清洗DXI指数数据"""
        logger.info("清洗DXI数据...")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # 确保dxi_index是数值
        df['dxi_index'] = pd.to_numeric(df['dxi_index'], errors='coerce')

        # 按日期排序
        df = df.sort_values('date')

        return df

    def align_time_series(self, spot_df, contract_df, dxi_df):
        """对齐时间序列数据"""
        logger.info("对齐时间序列...")

        # 获取日期范围
        all_dates = pd.date_range(
            start=self.config['time_range']['start_date'],
            end=self.config['time_range']['end_date'],
            freq='D'
        )

        # 创建完整日期骨架
        date_skeleton = pd.DataFrame({'date': all_dates})

        # 处理现货价格 - 按产品类型透视
        if spot_df is not None:
            spot_pivot = spot_df.pivot_table(
                index='date',
                columns='product_type',
                values='price_usd',
                aggfunc='mean'
            ).reset_index()

            # 重命名列
            spot_pivot.columns = ['date'] + [f'{col.lower()}_spot_price' for col in spot_pivot.columns[1:]]

            # 合并到骨架
            date_skeleton = date_skeleton.merge(spot_pivot, on='date', how='left')

        # 处理合约价格
        if contract_df is not None:
            contract_pivot = contract_df.pivot_table(
                index='contract_start_date',
                columns='product_type',
                values='price_usd',
                aggfunc='mean'
            ).reset_index()

            contract_pivot.columns = ['date'] + [f'{col.lower()}_contract_price' for col in contract_pivot.columns[1:]]
            date_skeleton = date_skeleton.merge(contract_pivot, on='date', how='left')

        # 处理DXI指数
        if dxi_df is not None:
            dxi_df = dxi_df[['date', 'dxi_index']]
            date_skeleton = date_skeleton.merge(dxi_df, on='date', how='left')

        # 填充缺失值 (前向填充，因为价格是连续的)
        date_skeleton = date_skeleton.fillna(method='ffill')

        return date_skeleton

    def calculate_features(self, df):
        """计算特征工程"""
        logger.info("计算特征...")

        # 对每种产品计算技术指标
        price_cols = [col for col in df.columns if 'price' in col]

        for col in price_cols:
            # 日收益率
            df[f'{col}_returns'] = df[col].pct_change()

            # 移动平均线
            df[f'{col}_ma7'] = df[col].rolling(window=7).mean()
            df[f'{col}_ma30'] = df[col].rolling(window=30).mean()

            # 波动率
            df[f'{col}_volatility'] = df[col].rolling(window=30).std()

            # 价格动量
            df[f'{col}_momentum'] = df[col] - df[col].shift(7)

        # 计算现货/合约价差
        spot_cols = [col for col in df.columns if 'spot_price' in col]
        contract_cols = [col for col in df.columns if 'contract_price' in col]

        for spot_col in spot_cols:
            product = spot_col.replace('_spot_price', '')
            contract_col = f'{product}_contract_price'

            if contract_col in df.columns:
                gap_col = f'{product}_spot_contract_gap'
                df[gap_col] = df[spot_col] - df[contract_col]
                df[f'{gap_col}_pct'] = (df[gap_col] / df[contract_col]) * 100

        return df

    def generate_time_llm_format(self, df):
        """
        生成Time-LLM可用的格式
        Time-LLM通常需要: [date, feature1, feature2, ..., target]
        """
        logger.info("生成Time-LLM格式...")

        # 确定目标变量 (这里用DXI指数作为市场趋势指标)
        if 'dxi_index' in df.columns:
            target_col = 'dxi_index'
        else:
            # 使用第一个价格列作为目标
            price_cols = [col for col in df.columns if 'price' in col]
            target_col = price_cols[0] if price_cols else None

        if target_col:
            # 创建滞后特征 (Time-LLM通常需要历史窗口)
            for lag in [1, 7, 30]:
                df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

            # 创建未来目标 (用于监督学习)
            df[f'{target_col}_future7'] = df[target_col].shift(-7)
            df[f'{target_col}_future30'] = df[target_col].shift(-30)

        # 移除完全缺失的行
        df = df.dropna(how='all', subset=df.columns[1:])

        return df

    def validate_data_quality(self, df):
        """验证数据质量"""
        logger.info("验证数据质量...")

        report = {
            'total_rows': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'completeness': (1 - df.isnull().sum() / len(df)).to_dict()
        }

        # 检查数据完整性
        for col in df.columns:
            if col != 'date' and df[col].dtype in ['float64', 'int64']:
                report[f'{col}_stats'] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }

        return report

    def save_processed_data(self, df, filename='processed_data.csv'):
        """保存处理后的数据"""
        output_dir = self.config['output']['processed_dir']
        import os
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, encoding=self.config['output']['encoding'])

        logger.info(f"处理后的数据已保存到: {filepath}")
        return filepath


if __name__ == '__main__':
    processor = DataProcessor()

    # 示例：加载并处理数据
    # spot_df, contract_df, dxi_df = processor.load_raw_data(
    #     'storage/raw/spot_prices.csv',
    #     'storage/raw/contract_prices.csv',
    #     'storage/raw/dxi_index.csv'
    # )
    # ...
