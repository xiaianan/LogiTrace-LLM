#!/usr/bin/env python
"""
快速启动脚本
立即生成Time-LLM可用的存储价格数据集
"""
import os
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_start():
    """快速启动 - 生成模拟数据集"""
    logger.info("=" * 60)
    logger.info("存储颗粒价格数据 - 快速启动")
    logger.info("=" * 60)

    # 导入模拟数据生成器
    from crawlers.alternative_sources import MockDataGenerator

    # 创建输出目录
    os.makedirs('storage/raw', exist_ok=True)
    os.makedirs('storage/processed', exist_ok=True)

    # 生成模拟数据
    logger.info("\n正在生成24个月的历史数据...")
    generator = MockDataGenerator(seed=42)
    dataset = generator.generate_full_dataset(months=24)

    # 保存原始数据
    raw_file = 'storage/raw/spot_prices_raw.csv'
    dataset.to_csv(raw_file, index=False)
    logger.info(f"✓ 原始数据已保存: {raw_file}")

    # 保存处理后的数据 (Time-LLM格式)
    processed_file = 'storage/processed/time_llm_dataset.csv'
    dataset.to_csv(processed_file, index=False)
    logger.info(f"✓ 处理后数据已保存: {processed_file}")

    # 生成数据摘要
    logger.info("\n" + "=" * 60)
    logger.info("数据摘要")
    logger.info("=" * 60)
    logger.info(f"总记录数: {len(dataset)}")
    logger.info(f"日期范围: {dataset['date'].min()} 至 {dataset['date'].max()}")
    logger.info(f"\n价格统计:")
    logger.info(f"  DDR4现货均价: ${dataset['ddr4_spot_price'].mean():.2f}")
    logger.info(f"  DDR4合约均价: ${dataset['ddr4_contract_price'].mean():.2f}")
    logger.info(f"  NAND现货均价: ${dataset['nand_spot_price'].mean():.2f}")
    logger.info(f"  DXI指数均值: {dataset['dxi_index'].mean():.0f}")

    logger.info("\n" + "=" * 60)
    logger.info("数据集已准备就绪!")
    logger.info("文件位置:")
    logger.info(f"  1. {processed_file}")
    logger.info(f"  2. {raw_file}")
    logger.info("=" * 60)

    return processed_file


def run_real_crawler():
    """运行真实爬虫 (需要较长时间)"""
    logger.info("启动真实数据爬虫...")
    logger.info("注意: 这可能需要10-30分钟完成")

    import subprocess
    result = subprocess.run([sys.executable, 'main.py', '--mode', 'full'])
    return result.returncode == 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='存储价格数据快速启动')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                        help='quick=快速生成模拟数据, full=运行真实爬虫')

    args = parser.parse_args()

    if args.mode == 'quick':
        quick_start()
    else:
        run_real_crawler()
