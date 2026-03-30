"""
反检测工具模块
提供User-Agent轮换、代理池、请求延迟等功能
"""
import random
import time
from fake_useragent import UserAgent


class AntiDetectionManager:
    """反爬虫检测管理器"""

    def __init__(self, config=None):
        self.config = config or {}
        self.ua = UserAgent()
        self.used_agents = []

    def get_random_headers(self):
        """获取随机请求头"""
        headers = {
            'User-Agent': self.ua.random if self.config.get('user_agent_rotation', True) else self.ua.chrome,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        return headers

    def random_delay(self, min_seconds=None, max_seconds=None):
        """随机延迟"""
        min_s = min_seconds or self.config.get('request_delay_min', 2)
        max_s = max_seconds or self.config.get('request_delay_max', 5)
        delay = random.uniform(min_s, max_s)
        time.sleep(delay)
        return delay

    def get_proxy(self):
        """获取代理（如配置了代理池）"""
        if not self.config.get('use_proxy', False):
            return None
        proxies = self.config.get('proxy_list', [])
        return random.choice(proxies) if proxies else None


class RateLimiter:
    """请求频率限制器"""

    def __init__(self, min_interval=3):
        self.min_interval = min_interval
        self.last_request_time = 0

    def wait(self):
        """等待到可以发送下一个请求"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()
