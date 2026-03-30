# 存储颗粒价格数据字典

## 数据集概述

**数据集名称**: Time-LLM Storage Price Dataset
**数据来源**: DRAMeXchange / TrendForce (模拟数据)
**时间范围**: 过去24个月 (2024-03 至 2026-03)
**更新频率**: 每日
**记录总数**: ~730条 (每日一条)

---

## 字段说明

### 基础字段

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|----------|------|------|
| `date` | datetime | 日期 | 2025-01-15 |
| `ddr4_spot_price` | float | DDR4 8Gb 现货价格 (USD) | 3.25 |
| `ddr4_contract_price` | float | DDR4 合约价格 (USD) | 3.50 |
| `nand_spot_price` | float | NAND Flash 128Gb 现货价格 (USD) | 2.80 |
| `nand_contract_price` | float | NAND Flash 合约价格 (USD) | 3.00 |
| `dxi_index` | int | DRAMeXchange综合指数 | 18500 |

### 技术指标字段

| 字段名 | 数据类型 | 说明 | 计算公式 |
|--------|----------|------|----------|
| `ddr4_spot_price_returns` | float | 日收益率 | `(今日价格-昨日价格)/昨日价格` |
| `ddr4_spot_price_ma7` | float | 7日移动平均 | `MA(price, 7)` |
| `ddr4_spot_price_ma30` | float | 30日移动平均 | `MA(price, 30)` |
| `ddr4_spot_price_volatility` | float | 30日波动率 | `Std(returns, 30) * sqrt(252)` |
| `nand_spot_price_returns` | float | NAND日收益率 | 同上 |
| `nand_spot_price_ma7` | float | NAND 7日均线 | 同上 |
| `nand_spot_price_ma30` | float | NAND 30日均线 | 同上 |

### 价差字段 (重要预测特征)

| 字段名 | 数据类型 | 说明 | 解读 |
|--------|----------|------|------|
| `ddr4_spot_contract_gap` | float | 现货-合约价差 (USD) | `spot - contract` |
| `ddr4_spot_contract_gap_pct` | float | 价差百分比 | `(gap / contract) * 100` |
| `nand_spot_contract_gap` | float | NAND现货-合约价差 | 同上 |
| `nand_spot_contract_gap_pct` | float | NAND价差百分比 | 同上 |

### 预测目标字段

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| `dxi_index_lag1` | int | DXI指数滞后1天 |
| `dxi_index_lag7` | int | DXI指数滞后7天 |
| `dxi_index_lag30` | int | DXI指数滞后30天 |
| `dxi_index_future7` | int | 未来7天DXI指数 (目标) |
| `dxi_index_future30` | int | 未来30天DXI指数 (目标) |

---

## 数据质量统计

### 完整性
- 日期范围: 完整无缺失
- 价格数据: 99.5%完整 (节假日可能缺失)
- 技术指标: 前30天为NaN (因计算窗口)

### 数值范围
- DDR4现货价格: $1.50 - $6.00
- NAND现货价格: $1.20 - $4.50
- DXI指数: 15,000 - 25,000

---

## 使用说明

### 1. 用于Time-LLM模型

```python
import pandas as pd

# 加载数据
df = pd.read_csv('processed_data/time_llm_dataset.csv')

# 特征列 (输入)
feature_cols = [
    'ddr4_spot_price', 'ddr4_contract_price',
    'nand_spot_price', 'nand_contract_price',
    'dxi_index',
    'ddr4_spot_contract_gap_pct',
    'nand_spot_contract_gap_pct'
]

# 目标列 (输出)
target_col = 'dxi_index_future7'  # 或 'dxi_index_future30'

X = df[feature_cols].values
y = df[target_col].values
```

### 2. 数据划分

```python
# 时间序列划分 (不能随机划分!)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
```

### 3. 标准化

```python
from sklearn.preprocessing import StandardScaler

# 训练集拟合，测试集转换
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
```

---

## 数据来源说明

### 主要来源
1. **DRAMeXchange (TrendForce)**
   - 全球存储市场领先研究机构
   - 数据被视为行业"金标准"
   - URL: https://www.dramexchange.com

### 备用来源
1. **Yahoo Finance**
   - SOXX ETF (iShares Semiconductor ETF)
   - 费城半导体指数 (^SOX)

---

## 数据更新

### 自动更新流程
1. 每日定时运行爬虫脚本
2. 抓取最新价格数据
3. 重新计算技术指标
4. 更新数据集文件

### 手动更新
```bash
cd data_collection
python main.py --mode full
```

---

## 引用格式

```bibtex
@misc{storage_price_dataset_2025,
  title={Storage Price Dataset for Time-LLM},
  author={ISE5334 Group Project},
  year={2025},
  publisher={PolyU},
  source={DRAMeXchange/TrendForce}
}
```

---

## 注意事项

1. **数据滞后**: 合约价格通常滞后现货价格30天
2. **节假日**: 周末和节假日无交易数据
3. **异常值**: 已使用IQR方法处理
4. **预测风险**: 存储市场价格受多种因素影响，模型预测仅供参考

---

## 联系方式

如有数据问题，请联系项目团队：
- 课程: ISE5334 - Industrial Prompt Engineering for Generative AI
- 学校: The Hong Kong Polytechnic University
