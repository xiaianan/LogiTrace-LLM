"""
DRAMeXchange 现货价格爬虫
抓取DDR4、DDR5、NAND Flash的每日现货价格
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DRAMeXchangeSpotCrawler:
    """DRAMeXchange现货价格爬虫"""

    def __init__(self, config_path='config/settings.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.source_config = self.config['sources']['dramexchange']
        self.base_url = self.source_config['base_url']
        self.session = requests.Session()
        self.data = []

    def fetch_current_spot_prices(self):
        """
        抓取当前现货价格
        目标页面: https://www.dramexchange.com/#
        """
        url = self.source_config['spot_price_url']

        try:
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }

            response = self.session.get(url, headers=headers, timeout=self.source_config['timeout'])
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # 解析价格表格
            # 注意：实际选择器需要根据网站结构调整
            price_data = self._parse_price_table(soup)

            logger.info(f"成功抓取现货价格数据: {len(price_data)} 条记录")
            return price_data

        except Exception as e:
            logger.error(f"抓取现货价格失败: {e}")
            return []

    def _parse_price_table(self, soup):
        """
        解析价格表格
        需要根据实际网页结构调整CSS选择器
        """
        prices = []

        # 示例选择器（需要根据实际网页修改）
        # 查找包含DRAM价格的表格
        tables = soup.find_all('table', class_='price-table')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # 跳过表头
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    price_record = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'product_type': cells[0].get_text(strip=True),
                        'specification': cells[1].get_text(strip=True),
                        'price_usd': cells[2].get_text(strip=True),
                        'change': cells[3].get_text(strip=True) if len(cells) > 3 else None,
                        'source': 'DRAMeXchange'
                    }
                    prices.append(price_record)

        return prices

    def fetch_historical_from_news(self, months=24):
        """
        通过新闻文章抓取历史价格数据
        TrendForce经常发布Weekly Price Update文章
        """
        url = self.source_config['news_url']
        historical_data = []

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }

            # 计算需要抓取的文章页数
            pages = (months // 4) + 1  # 假设每周一篇文章

            for page in range(1, pages + 1):
                page_url = f"{url}?page={page}"
                response = self.session.get(page_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')

                # 查找价格更新相关文章
                articles = soup.find_all('article')
                for article in articles:
                    title = article.get_text()
                    if 'price' in title.lower() or 'price update' in title.lower():
                        article_data = self._parse_price_article(article)
                        historical_data.extend(article_data)

                logger.info(f"已处理第 {page}/{pages} 页新闻")

                # 礼貌性延迟
                import time
                time.sleep(self.source_config['rate_limit'])

            return historical_data

        except Exception as e:
            logger.error(f"抓取历史数据失败: {e}")
            return []

    def _parse_price_article(self, article):
        """解析价格文章中的价格信息"""
        # 需要从文章内容中提取价格变动百分比和关键节点价格
        # 这需要根据实际文章结构调整
        return []

    def save_to_csv(self, data, filename=None):
        """保存数据到CSV文件"""
        if not data:
            logger.warning("没有数据可保存")
            return

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"spot_price_{timestamp}.csv"

        output_dir = self.config['output']['raw_dir']
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding=self.config['output']['encoding'])
        logger.info(f"数据已保存到: {filepath}")

        return filepath


if __name__ == '__main__':
    crawler = DRAMeXchangeSpotCrawler()

    # 抓取当前价格
    current_prices = crawler.fetch_current_spot_prices()

    # 抓取历史数据
    historical_prices = crawler.fetch_historical_from_news(months=24)

    # 合并数据
    all_prices = current_prices + historical_prices

    # 保存
    crawler.save_to_csv(all_prices, 'dram_spot_prices.csv')
