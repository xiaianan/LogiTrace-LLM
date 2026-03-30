"""
TrendForce DXI指数爬虫
DXI (DRAMeXchange Index) 是存储市场的综合指数
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import yaml
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DXICrawler:
    """DXI指数爬虫"""

    def __init__(self, config_path='config/settings.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.source_config = self.config['sources']['dramexchange']
        self.session = requests.Session()

    def fetch_dxi_data(self, days=730):
        """
        获取DXI指数历史数据
        默认抓取过去730天(约24个月)的数据
        """
        url = self.source_config['dxi_url']

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'X-Requested-With': 'XMLHttpRequest',
            }

            # 某些网站通过AJAX加载图表数据
            # 需要找到数据API端点
            api_url = f"{self.source_config['base_url']}/api/dxi/history"

            params = {
                'start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d'),
                'period': 'daily'
            }

            response = self.session.get(api_url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                return self._parse_dxi_json(data)
            else:
                # 如果API访问失败，尝试从HTML页面抓取
                return self._fetch_dxi_from_html(url)

        except Exception as e:
            logger.error(f"获取DXI数据失败: {e}")
            return self._fetch_dxi_from_html(url)

    def _parse_dxi_json(self, json_data):
        """解析DXI JSON数据"""
        dxi_records = []

        # 根据实际API响应结构调整
        if 'data' in json_data:
            for item in json_data['data']:
                record = {
                    'date': item.get('date'),
                    'dxi_index': item.get('value'),
                    'change': item.get('change'),
                    'change_percent': item.get('changePercent'),
                    'source': 'DRAMeXchange'
                }
                dxi_records.append(record)

        return dxi_records

    def _fetch_dxi_from_html(self, url):
        """从HTML页面抓取DXI数据"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }

            response = self.session.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 查找包含DXI数据的脚本或表格
            # 通常图表数据会嵌入在JavaScript中
            scripts = soup.find_all('script')

            for script in scripts:
                if script.string and 'dxi' in script.string.lower():
                    data = self._extract_data_from_script(script.string)
                    if data:
                        return data

            # 如果没有找到脚本数据，查找表格
            return self._extract_data_from_table(soup)

        except Exception as e:
            logger.error(f"从HTML抓取DXI失败: {e}")
            return []

    def _extract_data_from_script(self, script_content):
        """从JavaScript脚本中提取DXI数据"""
        try:
            # 使用正则表达式查找数据数组
            patterns = [
                r'data\s*:\s*(\[.*?\])',
                r'series\s*:\s*(\[.*?\])',
                r'DXI\s*:\s*(\[.*?\])'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, script_content, re.DOTALL)
                if matches:
                    data = json.loads(matches[0])
                    return self._convert_to_records(data)

        except Exception as e:
            logger.error(f"解析脚本数据失败: {e}")
            return None

    def _convert_to_records(self, data):
        """将图表数据转换为记录格式"""
        records = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    records.append({
                        'date': item.get('x') or item.get('date'),
                        'dxi_index': item.get('y') or item.get('value'),
                        'source': 'DRAMeXchange'
                    })

        return records

    def _extract_data_from_table(self, soup):
        """从表格中提取DXI数据"""
        records = []

        # 查找包含DXI数据的表格
        tables = soup.find_all('table', {'class': re.compile(r'dxi|index', re.I)})

        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # 跳过表头
                cells = row.find_all('td')
                if len(cells) >= 2:
                    record = {
                        'date': cells[0].get_text(strip=True),
                        'dxi_index': cells[1].get_text(strip=True),
                        'source': 'DRAMeXchange'
                    }
                    records.append(record)

        return records

    def calculate_technical_indicators(self, df):
        """
        计算技术指标
        - 移动平均线 (MA)
        - 相对强弱指数 (RSI)
        - 波动率
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 移动平均线
        df['ma_7'] = df['dxi_index'].rolling(window=7).mean()
        df['ma_30'] = df['dxi_index'].rolling(window=30).mean()

        # 价格波动率
        df['price_change'] = df['dxi_index'].pct_change()
        df['volatility_30d'] = df['price_change'].rolling(window=30).std() * (252 ** 0.5)

        return df


if __name__ == '__main__':
    crawler = DXICrawler()
    dxi_data = crawler.fetch_dxi_data(days=730)

    if dxi_data:
        df = pd.DataFrame(dxi_data)
        df_with_indicators = crawler.calculate_technical_indicators(df)
        print(f"获取到 {len(df_with_indicators)} 条DXI记录")
        print(df_with_indicators.head())
