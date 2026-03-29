# Time-LLM × GLM workflow — core export

本压缩包仅包含**接入 Streamlit / `workflow_orchestrator` 工作流**所需的 **Time-LLM（Qwen2-7B 4-bit + LoRA）** 代码与数据管线，便于上传 GitHub 或归档。

## 包内目录说明

| 路径 | 作用 |
|------|------|
| `workflow_orchestrator.py` | 量化快照、逻辑链、GLM 决策、默认 RISEN/V3 提示词 |
| `app.py` | Streamlit 前端 |
| `timellm_runtime.py` | Time-LLM 在线推理与缓存 |
| `train.py` / `eval_and_report.py` | 面板数据微调与测试集评估 |
| `scripts/fine_tune.sh` | AutoDL/单卡微调入口示例 |
| `scripts/run_prompt_ablation*.py` | 提示词消融（需 `GLM_API_KEY`） |
| `data_provider/` | `Dataset_Custom_Cleaned` 等 |
| `layers/`、`models/TimeLLM.py` | Time-LLM 模型实现（工作流所用版本） |
| `timellm-dataset/storage/processed/time_llm_data_cleaned.csv` | 微调/推理对齐用清洗表 |
| `timellm-dataset/` 下部分源码与文档 | 数据字典、清洗/对齐脚本参考 |
| `*.md`（用例、提示词迭代、评估、dataset_report 等） | 实验与报告文档 |

## 刻意未打入包（请自备或 Git LFS）

- **`checkpoints/*.pth`**：体积大；需自行训练生成或单独传输。
- **`.env`**：含密钥；仅提供 **`.env.example`**，复制后填写。
- **Qwen2-7B-Instruct 权重**：通过环境变量 **`TIME_LLM_MODEL_PATH`** 指向本地目录或 HF；参见 `timellm_runtime.py` 头部说明。
- **`models/gpt2_local/`**：旧版/对照用 GPT-2 本地文件，工作流主线为 Qwen，未收录以减小体积。

## 最小运行提示

```bash
pip install -r requirements.txt
cp .env.example .env   # 填写 GLM_API_KEY、TIME_LLM_MODEL_PATH 等
export PYTHONPATH="$PWD:$PYTHONPATH"
# 微调（可选）
bash scripts/fine_tune.sh
# 前端
streamlit run app.py
```

## 许可证

上游 **Time-LLM** 及所用模型权重请遵循各自原项目与模型协议；本工作流改动以你方仓库声明为准。
