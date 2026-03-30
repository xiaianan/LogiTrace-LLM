"""
主入口脚本
协调所有爬虫模块和数据处理
"""
import os
import sys
import yaml
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawlers.dramexchange_spot import DRAMeXchangeSpotCrawler
from crawlers.contract_price import ContractPriceCrawler
from crawlers.dxi_index import DXICrawler
from processors.cleaner import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_full_pipeline(use_selenium=False):
    """
    运行完整的数据收集流程

    Args:
        use_selenium: 是否使用Selenium处理动态页面
    """
    logger.info("=" * 60)
    logger.info("存储颗粒价格数据收集 - 完整流程")
    logger.info("=" * 60)

    # 步骤1: 收集现货价格
    logger.info("\n[步骤 1/4] 收集现货价格...")
    try:
        spot_crawler = DRAMeXchangeSpotCrawler()
        spot_data = spot_crawler.fetch_current_spot_prices()
        spot_file = spot_crawler.save_to_csv(spot_data, 'spot_prices_raw.csv')
        logger.info(f"✓ 现货价格收集完成: {len(spot_data)} 条记录")
    except Exception as e:
        logger.error(f"✗ 现货价格收集失败: {e}")
        spot_file = None
        spot_data = []

    # 步骤2: 收集合约价格
    logger.info("\n[步骤 2/4] 收集合约价格...")
    try:
        contract_crawler = ContractPriceCrawler()
        contract_data = contract_crawler.fetch_contract_prices()
        # 保存合约价格
        import pandas as pd
        contract_df = pd.DataFrame(contract_data)
        contract_file = os.path.join('storage/raw', 'contract_prices_raw.csv')
        os.makedirs('storage/raw', exist_ok=True)
        contract_df.to_csv(contract_file, index=False)
        logger.info(f"✓ 合约价格收集完成: {len(contract_data)} 条记录")
    except Exception as e:
        logger.error(f"✗ 合约价格收集失败: {e}")
        contract_file = None

    # 步骤3: 收集DXI指数
    logger.info("\n[步骤 3/4] 收集DXI指数...")
    try:
        dxi_crawler = DXICrawler()
        dxi_data = dxi_crawler.fetch_dxi_data(days=730)
        dxi_df = pd.DataFrame(dxi_data)
        dxi_file = os.path.join('storage/raw', 'dxi_index_raw.csv')
        dxi_df.to_csv(dxi_file, index=False)
        logger.info(f"✓ DXI指数收集完成: {len(dxi_data)} 条记录")
    except Exception as e:
        logger.error(f"✗ DXI指数收集失败: {e}")
        dxi_file = None

    # 步骤4: 数据处理和整合
    logger.info("\n[步骤 4/4] 数据处理...")
    try:
        processor = DataProcessor()

        # 加载数据
        spot_df = pd.read_csv(spot_file) if spot_file and os.path.exists(spot_file) else None
        contract_df = pd.read_csv(contract_file) if contract_file and os.path.exists(contract_file) else None
        dxi_df = pd.read_csv(dxi_file) if dxi_file and os.path.exists(dxi_file) else None

        # 清洗数据
        if spot_df is not None:
            spot_df = processor.clean_spot_prices(spot_df)
        if contract_df is not None:
            contract_df = processor.clean_contract_prices(contract_df)
        if dxi_df is not None:
            dxi_df = processor.clean_dxi_data(dxi_df)

        # 对齐时间序列
        merged_df = processor.align_time_series(spot_df, contract_df, dxi_df)

        # 计算特征
        featured_df = processor.calculate_features(merged_df)

        # 生成Time-LLM格式
        final_df = processor.generate_time_llm_format(featured_df)

        # 验证数据质量
        quality_report = processor.validate_data_quality(final_df)

        # 保存最终数据
        output_file = processor.save_processed_data(final_df, 'time_llm_dataset.csv')

        # 保存质量报告
        import json
        report_file = os.path.join('storage/processed', 'data_quality_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, default=str)

        logger.info(f"✓ 数据处理完成")
        logger.info(f"✓ 最终数据已保存: {output_file}")
        logger.info(f"✓ 质量报告已保存: {report_file}")

    except Exception as e:
        logger.error(f"✗ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("数据收集流程完成")
    logger.info("=" * 60)


def run_quick_mode():
    """快速模式：使用备用数据源或生成模拟数据（用于测试）"""
    logger.info("运行快速模式（使用备用方案）...")

    # 这里可以集成第三方API或备用数据源
    # 例如: Yahoo Finance的半导体ETF数据作为替代

    logger.info("快速模式完成")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='存储颗粒价格数据收集')
    parser.add_argument('--mode', choices=['full', 'quick'], default='full',
                        help='运行模式: full=完整流程, quick=快速模式')
    parser.add_argument('--selenium', action='store_true',
                        help='使用Selenium处理动态页面')

    args = parser.parse_args()

    if args.mode == 'full':
        run_full_pipeline(use_selenium=args.selenium)
    else:
        run_quick_mode()
