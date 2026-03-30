"""
备用数据源和模拟数据生成器
当主要数据源无法访问时提供备选方案
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可选导入yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Using mock data only.")


class AlternativeDataSource:
    """备用数据源 - 使用Yahoo Finance的半导体ETF数据"""

    def __init__(self):
        # 半导体相关ETF代码
        self.tickers = {
            'SOXX': 'iShares Semiconductor ETF',  # 主要半导体ETF
            'SMH': 'VanEck Semiconductor ETF',
            'XSD': 'SPDR S&P Semiconductor ETF',
            '^SOX': 'PHLX Semiconductor Index',  # 费城半导体指数
        }

    def fetch_etf_data(self, ticker='SOXX', period='2y'):
        """
        获取ETF历史数据作为存储市场的代理指标
        半导体ETF走势与存储颗粒价格高度相关
        """
        try:
            logger.info(f"获取 {ticker} ETF数据...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # 计算技术指标
            df['returns'] = df['close'].pct_change()
            df['ma_7'] = df['close'].rolling(7).mean()
            df['ma_30'] = df['close'].rolling(30).mean()
            df['volatility'] = df['returns'].rolling(30).std() * np.sqrt(252)

            logger.info(f"✓ 获取到 {len(df)} 条ETF记录")
            return df

        except Exception as e:
            logger.error(f"获取ETF数据失败: {e}")
            return None

    def fetch_all_tickers(self):
        """获取所有相关ETF数据"""
        all_data = {}
        for ticker in self.tickers:
            df = self.fetch_etf_data(ticker)
            if df is not None:
                all_data[ticker] = df
        return all_data


class MockDataGenerator:
    """
    模拟数据生成器
    基于真实存储市场的统计特征生成模拟数据
    用于测试和演示
    """

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_dram_prices(self, months=24):
        """
        生成DRAM价格模拟数据
        基于2022-2024年存储市场的真实波动特征
        """
        logger.info(f"生成 {months} 个月的DRAM模拟数据...")

        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # DRAM价格的统计特征 (基于历史数据)
        # DDR4 8Gb 价格在 $2.0 - $5.0 区间波动
        base_price = 3.5
        volatility = 0.02  # 日波动率 2%

        # 生成价格序列 (几何布朗运动 + 周期性)
        returns = np.random.normal(0, volatility, len(dates))

        # 添加周期性 (存储市场约3-4年一个周期)
        cycle_period = 365 * 3  # 3年周期
        days_from_start = np.arange(len(dates))
        cycle_component = 0.3 * np.sin(2 * np.pi * days_from_start / cycle_period)

        # 添加趋势成分
        trend = np.linspace(0, -0.2, len(dates))  # 假设轻微下降趋势

        # 组合收益率
        total_returns = returns + cycle_component / 100 + trend / len(dates)

        # 计算价格
        prices = base_price * np.exp(np.cumsum(total_returns))

        # 生成合约价格 (滞后且平滑)
        contract_prices = pd.Series(prices).rolling(30).mean().values
        contract_prices = np.roll(contract_prices, 30)  # 滞后一个月

        df = pd.DataFrame({
            'date': dates,
            'ddr4_spot_price': prices,
            'ddr4_contract_price': contract_prices,
            'product_type': 'DDR4',
            'specification': '8Gb 1Gx8 3200MT/s',
            'source': 'mock_data'
        })

        # 添加噪声使价格更真实
        df['ddr4_spot_price'] = df['ddr4_spot_price'].round(2)
        df['ddr4_contract_price'] = df['ddr4_contract_price'].round(2)

        logger.info(f"✓ 生成 {len(df)} 条DRAM记录")
        return df

    def generate_nand_prices(self, months=24):
        """生成NAND Flash价格模拟数据"""
        logger.info(f"生成 {months} 个月的NAND模拟数据...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # NAND价格通常比DRAM波动更大
        base_price = 2.8
        volatility = 0.025

        returns = np.random.normal(0, volatility, len(dates))
        cycle_period = 365 * 3
        days_from_start = np.arange(len(dates))
        cycle_component = 0.35 * np.sin(2 * np.pi * days_from_start / cycle_period + np.pi/4)

        total_returns = returns + cycle_component / 100
        prices = base_price * np.exp(np.cumsum(total_returns))

        contract_prices = pd.Series(prices).rolling(30).mean().values
        contract_prices = np.roll(contract_prices, 30)

        df = pd.DataFrame({
            'date': dates,
            'nand_spot_price': prices,
            'nand_contract_price': contract_prices,
            'product_type': 'NAND_TLC',
            'specification': '128Gb TLC',
            'source': 'mock_data'
        })

        df['nand_spot_price'] = df['nand_spot_price'].round(2)
        df['nand_contract_price'] = df['nand_contract_price'].round(2)

        logger.info(f"✓ 生成 {len(df)} 条NAND记录")
        return df

    def generate_dxi_index(self, months=24):
        """生成DXI指数模拟数据"""
        logger.info(f"生成 {months} 个月的DXI模拟数据...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # DXI指数基准 (2024年约20000点左右)
        base_index = 20000
        volatility = 0.015

        returns = np.random.normal(0, volatility, len(dates))
        cycle_period = 365 * 3
        days_from_start = np.arange(len(dates))
        cycle_component = 0.25 * np.sin(2 * np.pi * days_from_start / cycle_period)

        total_returns = returns + cycle_component / 100
        index_values = base_index * np.exp(np.cumsum(total_returns))

        df = pd.DataFrame({
            'date': dates,
            'dxi_index': index_values.round(0),
            'change': np.diff(index_values, prepend=index_values[0]).round(2),
            'source': 'mock_data'
        })

        logger.info(f"✓ 生成 {len(df)} 条DXI记录")
        return df

    def generate_full_dataset(self, months=24):
        """生成完整数据集"""
        logger.info(f"生成完整模拟数据集 ({months}个月)...")

        dram_df = self.generate_dram_prices(months)
        nand_df = self.generate_nand_prices(months)
        dxi_df = self.generate_dxi_index(months)

        # 合并数据
        full_df = dram_df[['date', 'ddr4_spot_price', 'ddr4_contract_price']].merge(
            nand_df[['date', 'nand_spot_price', 'nand_contract_price']],
            on='date',
            how='outer'
        ).merge(
            dxi_df[['date', 'dxi_index']],
            on='date',
            how='outer'
        )

        # 排序
        full_df = full_df.sort_values('date').reset_index(drop=True)

        # 填充缺失值
        full_df = full_df.fillna(method='ffill')

        # 计算特征
        for col in ['ddr4_spot_price', 'nand_spot_price']:
            if col in full_df.columns:
                full_df[f'{col}_returns'] = full_df[col].pct_change()
                full_df[f'{col}_ma7'] = full_df[col].rolling(7).mean()
                full_df[f'{col}_ma30'] = full_df[col].rolling(30).mean()

        # 现货合约价差
        full_df['ddr4_spot_contract_gap'] = full_df['ddr4_spot_price'] - full_df['ddr4_contract_price']
        full_df['nand_spot_contract_gap'] = full_df['nand_spot_price'] - full_df['nand_contract_price']

        logger.info(f"✓ 完整数据集生成完成: {len(full_df)} 条记录")
        logger.info(f"  日期范围: {full_df['date'].min()} 至 {full_df['date'].max()}")

        return full_df


if __name__ == '__main__':
    # 测试备用数据源
    print("测试备用数据源...")

    # 生成模拟数据
    generator = MockDataGenerator()
    dataset = generator.generate_full_dataset(months=24)

    print(f"\n数据集预览:")
    print(dataset.head(10))
    print(f"\n数据形状: {dataset.shape}")
    print(f"\n数据统计:")
    print(dataset.describe())

    # 保存测试数据
    dataset.to_csv('test_mock_data.csv', index=False)
    print("\n测试数据已保存到 test_mock_data.csv")
