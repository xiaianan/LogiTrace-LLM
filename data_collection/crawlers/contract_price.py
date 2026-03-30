"""
TrendForce 合约价格爬虫
抓取DRAM/NAND Flash的月度合约价格
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import yaml
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContractPriceCrawler:
    """合约价格爬虫"""

    def __init__(self, config_path='config/settings.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.source_config = self.config['sources']['dramexchange']
        self.session = requests.Session()

    def fetch_contract_prices(self, months=24):
        """
        获取合约价格数据
        合约价格通常是月度更新
        """
        url = f"{self.source_config['base_url']}/ContractPrice/"

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }

            response = self.session.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 解析合约价格表格
            contract_data = self._parse_contract_table(soup)

            logger.info(f"成功抓取合约价格数据: {len(contract_data)} 条记录")
            return contract_data

        except Exception as e:
            logger.error(f"抓取合约价格失败: {e}")
            return []

    def _parse_contract_table(self, soup):
        """
        解析合约价格表格
        合约价格通常按季度或月度分类
        """
        prices = []

        # 查找合约价格表格
        tables = soup.find_all('table', {'class': re.compile(r'contract|price', re.I)})

        for table in tables:
            # 获取表格标题以确定时间范围
            caption = table.find('caption')
            period = caption.get_text(strip=True) if caption else 'Unknown'

            rows = table.find_all('tr')
            headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                product = cells[0].get_text(strip=True)

                # 提取各季度/月度的价格
                for i, cell in enumerate(cells[1:], start=1):
                    if i < len(headers):
                        time_period = headers[i]
                        price_text = cell.get_text(strip=True)

                        # 解析价格值
                        price_value = self._extract_price(price_text)

                        if price_value:
                            record = {
                                'product_type': self._categorize_product(product),
                                'specification': product,
                                'contract_period': time_period,
                                'price_usd': price_value,
                                'price_change': self._extract_change(price_text),
                                'source': 'DRAMeXchange',
                                'collection_date': datetime.now().strftime('%Y-%m-%d')
                            }
                            prices.append(record)

        return prices

    def _categorize_product(self, product_name):
        """对产品进行分类"""
        product_lower = product_name.lower()

        if 'ddr5' in product_lower:
            return 'DDR5'
        elif 'ddr4' in product_lower:
            return 'DDR4'
        elif 'ddr3' in product_lower:
            return 'DDR3'
        elif 'nand' in product_lower or 'flash' in product_lower:
            if 'tlc' in product_lower:
                return 'NAND_TLC'
            elif 'qlc' in product_lower:
                return 'NAND_QLC'
            else:
                return 'NAND_Flash'
        elif 'ssd' in product_lower:
            return 'SSD'

        return 'Other'

    def _extract_price(self, text):
        """从文本中提取价格数值"""
        # 匹配价格格式: $3.25, 3.25 USD, etc.
        patterns = [
            r'\$?([0-9]+\.[0-9]+)',
            r'([0-9]+\.[0-9]+)\s*(?:USD|\$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _extract_change(self, text):
        """提取价格变动"""
        # 匹配变动格式: +0.05, -0.05, ↑0.05, ↓0.05
        patterns = [
            r'([\+\-])([0-9]+\.[0-9]+)',
            r'([↑↓])\s*([0-9]+\.[0-9]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                direction = match.group(1)
                value = float(match.group(2))
                return value if direction in ['+', '↑'] else -value

        return None

    def calculate_spot_contract_gap(self, spot_df, contract_df):
        """
        计算现货/合约价差
        这是重要的预测特征
        """
        # 合并现货和合约价格
        merged = pd.merge(
            spot_df,
            contract_df,
            on=['date', 'product_type'],
            how='outer',
            suffixes=('_spot', '_contract')
        )

        # 计算价差
        merged['spot_contract_gap'] = merged['price_usd_spot'] - merged['price_usd_contract']
        merged['spot_contract_gap_pct'] = (merged['spot_contract_gap'] / merged['price_usd_contract']) * 100

        # 价差扩大信号 (>20% 被认为是强信号)
        merged['gap_signal'] = merged['spot_contract_gap_pct'].apply(
            lambda x: 'strong_up' if x > 20 else ('strong_down' if x < -20 else 'normal')
        )

        return merged


if __name__ == '__main__':
    crawler = ContractPriceCrawler()
    contract_data = crawler.fetch_contract_prices(months=24)

    if contract_data:
        df = pd.DataFrame(contract_data)
        print(f"获取到 {len(df)} 条合约价格记录")
        print(df.head())
