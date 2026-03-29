from math import sqrt

import torch
import torch.nn as nn

# PyTorch <2.3 lacks this; transformers 4.40+ bitsandbytes integration expects it.
import torch.compiler as _torch_compiler

if not hasattr(_torch_compiler, "is_compiling"):
    _torch_compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
)

try:
    from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
except Exception:  # pragma: no cover
    LlamaConfig, LlamaModel, LlamaTokenizer = None, None, None
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig, TaskType, get_peft_model = None, None, None

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.prompt_max_length = int(getattr(configs, "prompt_max_length", 2048))
        self.num_tokens = int(getattr(configs, "num_tokens", 1000))
        self.max_vocab_for_mapping = int(getattr(configs, "max_vocab_for_mapping", 64000))
        self.use_lora = bool(getattr(configs, "use_lora", False))
        self.lora_r = int(getattr(configs, "lora_r", 8))
        self.lora_alpha = int(getattr(configs, "lora_alpha", 16))
        self.lora_dropout = float(getattr(configs, "lora_dropout", 0.05))
        self.train_patch_embedding = bool(getattr(configs, "train_patch_embedding", False))
        self.train_reprogramming = bool(getattr(configs, "train_reprogramming", True))
        self.model_path = getattr(configs, "model_path", "")
        self.reprogram_alpha = float(getattr(configs, "reprogram_alpha", 1.0))
        self.patch_embed_scale = float(getattr(configs, "patch_embed_scale", 0.1))

        if configs.llm_model == 'LLAMA':
            if LlamaConfig is None or LlamaModel is None or LlamaTokenizer is None:
                raise ImportError(
                    "Your transformers version does not include Llama* classes. "
                    "Please upgrade transformers or avoid configs.llm_model='LLAMA'."
                )
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model in {"LLAMA3_8B_GPTQ", "QWEN2_7B_GPTQ_INT4", "QWEN2_7B_BNB4"}:
            default_paths = {
                # 这里把“下载路径”交给 HF_HOME（缓存目录），而不是硬编码为本地文件夹。
                # 只要模型已在 HF cache 中存在，from_pretrained 就能直接命中。
                "LLAMA3_8B_GPTQ": "Llama-3-8B-Instruct-GPTQ",
                "QWEN2_7B_GPTQ_INT4": "Qwen2-7B-Instruct-GPTQ-Int4",
                "QWEN2_7B_BNB4": "Qwen/Qwen2-7B-Instruct",
            }
            model_name_or_path = self.model_path or default_paths[configs.llm_model]
            model_config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            model_config.output_attentions = True
            model_config.output_hidden_states = True
            model_kwargs = dict(
                trust_remote_code=True,
                local_files_only=False,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                config=model_config,
            )
            if bool(getattr(configs, "force_single_gpu_map", False)):
                model_kwargs["device_map"] = {"": 0}
            if configs.llm_model == "QWEN2_7B_BNB4":
                # transformers==4.40.0 下优先走旧参数路径，兼容性更好。
                model_kwargs.pop("torch_dtype", None)
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                model_kwargs["bnb_4bit_use_double_quant"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            if configs.llm_model == "QWEN2_7B_BNB4":
                # Qwen2 4-bit 在 AutoModel + dispatch_model 下可能触发内部 .to()；
                # 用 CausalLM 包装再取 base model 可规避该冲突。
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, **model_kwargs
                ).model
            else:
                self.llm_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            # 8GB 显存场景：量化主干 + LoRA 低秩适配
            self.use_lora = True
        elif configs.llm_model == "QWEN2_7B":
            # AutoDL 本地或 HF：Qwen2-7B 全精度/FP16 底座（显存充足时使用，与 patch/reprogram 联合微调）
            model_name_or_path = self.model_path or "Qwen/Qwen2-7B-Instruct"
            model_config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            model_config.output_attentions = True
            model_config.output_hidden_states = True
            model_kwargs = dict(
                trust_remote_code=True,
                local_files_only=False,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                config=model_config,
            )
            if bool(getattr(configs, "force_single_gpu_map", False)):
                model_kwargs["device_map"] = {"": 0}
            self.llm_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.use_lora = bool(getattr(configs, "use_lora", False))
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if self.use_lora:
            if get_peft_model is None:
                raise ImportError("peft is required for LoRA. Please install: pip install peft")
            lora_target_modules = list(
                getattr(
                    configs,
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"],
                )
            )
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_target_modules,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_cfg)
            for name, param in self.llm_model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

        if configs.prompt_domain:
            content_max_len = getattr(configs, "content_max_length", None)
            if content_max_len is not None and isinstance(configs.content, str):
                self.description = configs.content[: int(content_max_len)]
            else:
                self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.use_vocab_mapping = self.vocab_size <= self.max_vocab_for_mapping
        if self.use_vocab_mapping:
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        else:
            self.mapping_layer = None
            token_idx = torch.linspace(0, self.vocab_size - 1, self.num_tokens, dtype=torch.long)
            self.register_buffer("token_bank_indices", token_idx, persistent=False)

        # 使用真实 backbone hidden size，避免 configs.llm_dim 与模型配置不一致导致维度错误
        hidden_size = None
        if hasattr(self.llm_model, "config"):
            hidden_size = getattr(self.llm_model.config, "hidden_size", None) or getattr(
                self.llm_model.config, "n_embd", None
            )
        if hidden_size is None:
            # fallback：保持原始配置
            hidden_size = self.d_llm
        self.d_llm = int(hidden_size)

        _repro_drop = float(getattr(configs, "reprogramming_dropout", configs.dropout))
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model,
            configs.n_heads,
            self.d_ff,
            self.d_llm,
            attention_dropout=_repro_drop,
        )
        # 对于 Qwen2-7B，hidden_size=3584；这里显式记录并对齐重编程层输出维度。
        self.reprogramming_layer.out_projection = nn.Linear(
            self.reprogramming_layer.out_projection.in_features, self.d_llm
        )

        # 合约换月示性变量：分类嵌入叠加到对应变量的 patch token 上，参与 Reprogramming 前的线性表征
        self.contract_roll_var_idx = getattr(configs, "contract_roll_var_idx", None)
        if self.contract_roll_var_idx is not None:
            self.roll_embed = nn.Embedding(2, configs.d_model)
        else:
            self.roll_embed = None

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError
        self.output_norm = nn.LayerNorm(configs.enc_in)
        self.output_smoother = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # LoRA 场景：仅训练指定模块，冻结其余 Time-LLM 非 backbone 参数。
        if self.use_lora:
            for p in self.patch_embedding.parameters():
                p.requires_grad = self.train_patch_embedding
            if self.mapping_layer is not None:
                for p in self.mapping_layer.parameters():
                    p.requires_grad = False
            for p in self.output_projection.parameters():
                p.requires_grad = False
            for p in self.normalize_layers.parameters():
                p.requires_grad = False
            # 确保 reprogramming 可按配置训练
            for p in self.reprogramming_layer.parameters():
                p.requires_grad = self.train_reprogramming
            if self.roll_embed is not None:
                for p in self.roll_embed.parameters():
                    p.requires_grad = True

        self._init_reprogramming_weights(std=0.002)

    def _init_reprogramming_weights(self, std=0.002):
        for module in [
            self.reprogramming_layer.query_projection,
            self.reprogramming_layer.key_projection,
            self.reprogramming_layer.value_projection,
            self.reprogramming_layer.out_projection,
        ]:
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        roll_raw = None
        if self.roll_embed is not None and self.contract_roll_var_idx is not None:
            vi = int(self.contract_roll_var_idx)
            roll_raw = (x_enc[:, -1, vi] > 0.5).long().clamp(0, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
        ).input_ids
        embed_layer = self.llm_model.get_input_embeddings()
        embed_device = embed_layer.weight.device
        prompt_embeddings = embed_layer(prompt.to(embed_device))  # (batch, prompt_token, dim)
        prompt_embeddings = torch.nan_to_num(prompt_embeddings, nan=0.0, posinf=1e4, neginf=-1e4)

        word_embeddings = embed_layer.weight
        if self.use_vocab_mapping:
            source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        else:
            token_idx = self.token_bank_indices.to(word_embeddings.device)
            source_embeddings = word_embeddings.index_select(0, token_idx)
        repro_dtype = next(self.reprogramming_layer.parameters()).dtype
        source_embeddings = source_embeddings.to(repro_dtype)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        patch_dtype = next(self.patch_embedding.parameters()).dtype
        enc_out, n_vars = self.patch_embedding(x_enc.to(patch_dtype))
        enc_out = enc_out * self.patch_embed_scale
        if self.roll_embed is not None and roll_raw is not None:
            B_batch = roll_raw.shape[0]
            Lp, D = enc_out.shape[1], enc_out.shape[2]
            vi = int(self.contract_roll_var_idx)
            enc_out = enc_out.view(B_batch, n_vars, Lp, D)
            enc_out[:, vi, :, :] = enc_out[:, vi, :, :] + self.roll_embed(roll_raw).unsqueeze(1).to(
                dtype=enc_out.dtype, device=enc_out.device
            )
            enc_out = enc_out.reshape(B_batch * n_vars, Lp, D)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        enc_out = enc_out * self.reprogram_alpha
        enc_out = torch.nan_to_num(enc_out, nan=0.0, posinf=1e4, neginf=-1e4)
        enc_out = enc_out.to(embed_device)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e4, neginf=-1e4)
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = torch.tanh(dec_out)
        dec_out = self.output_norm(dec_out)
        dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e4, neginf=-1e4)

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = self.output_smoother(dec_out.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        dec_out = torch.nan_to_num(dec_out, nan=0.0, posinf=1e4, neginf=-1e4)

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


# Backbones that use bitsandbytes / GPTQ: do not call ``Model.float()`` or ``Model.to()`` on the full
# tree — that corrupts quantized params (e.g. matmul Float vs Byte). Move only Time-LLM heads.
QUANTIZED_LLM_BACKBONES = frozenset({"QWEN2_7B_BNB4", "LLAMA3_8B_GPTQ", "QWEN2_7B_GPTQ_INT4"})


def align_timellm_auxiliary_modules(model: Model) -> None:
    """Place patch / reprogram / output heads on the same device and dtype as the LLM embeddings."""
    emb = model.llm_model.get_input_embeddings()
    dev = emb.weight.device
    dtyp = emb.weight.dtype
    modules = [
        model.dropout,
        model.patch_embedding,
        model.reprogramming_layer,
        model.output_projection,
        model.output_norm,
        model.normalize_layers,
        model.output_smoother,
    ]
    if model.mapping_layer is not None:
        modules.append(model.mapping_layer)
    if model.roll_embed is not None:
        modules.append(model.roll_embed)
    for m in modules:
        m.to(device=dev, dtype=dtyp)


def uses_quantized_llm_backbone(llm_model: str) -> bool:
    return str(llm_model or "") in QUANTIZED_LLM_BACKBONES
