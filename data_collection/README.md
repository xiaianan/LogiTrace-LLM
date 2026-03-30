# 存储颗粒价格数据收集系统

## 项目简介

本系统用于收集存储颗粒(DRAM/NAND Flash)的价格数据，为Time-LLM模型提供训练和预测数据。

**主要数据源**: DRAMeXchange (TrendForce)
**数据范围**: 过去24个月
**关键指标**: 现货价、合约价、DXI指数、现货/合约价差

---

## 项目结构

```
data_collection/
├── config/
│   └── settings.yaml           # 配置文件
├── crawlers/
│   ├── dramexchange_spot.py    # 现货价格爬虫
│   ├── contract_price.py       # 合约价格爬虫
│   ├── dxi_index.py           # DXI指数爬虫
│   └── alternative_sources.py  # 备用数据源(模拟数据)
├── processors/
│   └── cleaner.py              # 数据清洗和预处理
├── utils/
│   └── anti_detection.py       # 反检测工具
├── storage/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
├── main.py                     # 主入口
├── quick_start.py              # 快速启动脚本
├── requirements.txt            # 依赖包
├── data_dictionary.md          # 数据字典
└── README.md                   # 本文件
```

---

## 快速开始

### 1. 安装依赖

```bash
cd data_collection
pip install -r requirements.txt
```

### 2. 快速生成数据集 (推荐)

由于项目截止日期临近，使用模拟数据快速生成完整数据集:

```bash
python quick_start.py
```

这将在 `storage/processed/time_llm_dataset.csv` 生成24个月的完整数据集。

### 3. 运行真实爬虫 (可选)

如需抓取DRAMeXchange真实数据:

```bash
python main.py --mode full
```

或使用Selenium处理动态页面:

```bash
python main.py --mode full --selenium
```

---

## 数据字段说明

### 核心字段
- `date`: 日期
- `ddr4_spot_price`: DDR4现货价格
- `ddr4_contract_price`: DDR4合约价格
- `nand_spot_price`: NAND现货价格
- `dxi_index`: DXI综合指数

### 衍生特征
- `*_returns`: 日收益率
- `*_ma7/ma30`: 移动平均线
- `*_spot_contract_gap`: 现货合约价差
- `*_lag*/future*`: 滞后/前瞻特征

详见 [data_dictionary.md](data_dictionary.md)

---

## 配置说明

编辑 `config/settings.yaml` 调整:
- 数据源URL
- 目标存储颗粒规格
- 时间范围
- 反检测设置

---

## 输出文件

运行后将生成以下文件:

1. **原始数据** (`storage/raw/`)
   - `spot_prices_raw.csv`: 现货价格
   - `contract_prices_raw.csv`: 合约价格
   - `dxi_index_raw.csv`: DXI指数

2. **处理后数据** (`storage/processed/`)
   - `time_llm_dataset.csv`: Time-LLM标准格式
   - `data_quality_report.json`: 数据质量报告

---

## 使用示例

```python
import pandas as pd

# 加载数据
df = pd.read_csv('storage/processed/time_llm_dataset.csv')

# 查看数据
print(df.head())
print(df.describe())

# 用于Time-LLM的特征列
features = [
    'ddr4_spot_price',
    'ddr4_contract_price',
    'dxi_index',
    'ddr4_spot_contract_gap_pct'
]

# 目标列 (预测未来7天DXI指数)
target = 'dxi_index_future7'
```

---

## 注意事项

1. **网站反爬**: DRAMeXchange有反爬机制，请:
   - 设置合理的请求间隔 (默认3秒)
   - 使用User-Agent轮换
   - 避免在高峰时段大量请求

2. **数据质量**: 如遇数据缺失，系统会:
   - 使用前向填充
   - 使用备用数据源
   - 生成标记提示

3. **模拟数据**: 快速模式生成的数据基于:
   - 存储市场的真实统计特征
   - 3-4年周期性规律
   - 几何布朗运动模型

---

## PolyU数据库资源

作为PolyU学生，可尝试使用以下学校资源:
- Bloomberg Terminal (商学院)
- Wind (万得) 金融终端
- Refinitiv Eikon

联系图书馆确认访问权限: `lb-db@polyu.edu.hk`

---

## 技术栈

- **Python 3.8+**
- **requests**: HTTP请求
- **BeautifulSoup**: HTML解析
- **Selenium**: 动态页面爬取
- **pandas**: 数据处理
- **numpy**: 数值计算

---

## 故障排除

### 问题1: 无法连接DRAMeXchange
**解决**: 使用备用数据源
```bash
python quick_start.py --mode quick
```

### 问题2: 缺少依赖包
**解决**: 重新安装
```bash
pip install -r requirements.txt
```

### 问题3: 编码错误
**解决**: 数据文件使用UTF-8编码，Windows下可能需要指定:
```python
pd.read_csv('file.csv', encoding='utf-8-sig')
```

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

**课程**: ISE5334 - Industrial Prompt Engineering for Generative AI
**学校**: The Hong Kong Polytechnic University
**截止日期**: 2026年3月31日
