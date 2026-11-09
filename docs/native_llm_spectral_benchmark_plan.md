# 原生 LLM 光谱直接分类对比实验：完整实施方案

> 项目代号：**native_llm_spectral_benchmark**  
> 文档版本：v1.0（2026-11-09）  
> 应用范围：红外（FTIR/NIR）、拉曼及其他数值型光谱的食品/药材等样本分类  
> 目标：建立能够与光谱智能体比较的、**不借助外部工具的原生 LLM 直接分类基线**。  
> 状态：**方案与本地演示脚本已准备；真实数据、老师确认的 Prompt、正式模型与 API Key 尚未提供，尚未开展真实分类实验。**

---

## 0. 先读：实验边界和必须遵守的约定

本项目研究的是：“将某样本的原始光谱数值与固定 Prompt 交给 LLM，模型是否能在**不训练、不调用外部工具**的情况下直接判断预先定义的类别？”

- **允许：** Python 读取 CSV/Excel、检查数据格式、将数值序列化为文本、填入固定 Prompt、调用 LLM API、解析返回的类别、写入日志、计算评价指标。
- **禁止：** 对输入光谱进行 SNV/MSC、平滑、求导、归一化、PCA、人工峰提取或模型训练；用 SVM/RF/PLS-DA/CNN/Transformer 分类器取代 LLM；让 LLM 使用工具、检索、代码执行、外部数据库来完成判断。
- **标签隔离：** 真实标签只供 Python **在模型响应以后**评分，不能出现在该测试样本的 Prompt 中。样本文件名、元数据和类别名称中也不能无意透露答案。
- **提示词固定：** 用户已有完整固定 Prompt。正式运行时必须**逐字使用已审批的 Prompt**；只允许在已经确认的“光谱数据插入位置”替换数据。示例文件中的演示 Prompt **不能直接用于论文实验**。
- **任务类型：** 若 Prompt 中没有带标签参考样本，则称 zero-shot direct classification；若包含少量标注示例，称 few-shot direct classification。两者必须分别标注，不得混称。
- **不得偷换比较对象：** 本方案评价的是原生 LLM 数值光谱直接分类，不等同于“LLM 生成代码完成机器学习分类”。

**最终判断标准：** 不只是运行成功，而是每个结果都能追溯到数据版本、Prompt 版本、模型标识、请求参数、原始回答、解析规则和评分方式。

## 1. 科研问题、实验目标与交付物

### 1.1 科研问题

1. 在光谱种类和类别数既定的情况下，纯 LLM 能获得怎样的分类性能？
2. 对相同输入、不同模型及重复调用，分类性能和无效输出率是否稳定？
3. 智能体工具链相对于“直接输入光谱 + 固定 Prompt”的收益是多少？此问题待智能体结果汇总后回答。

### 1.2 主实验与可选实验

- **主实验 E1：** 每个测试光谱对应一次无工具、无训练的 API 调用；每个候选模型使用相同测试样本与同一版本 Prompt，得到 Accuracy、Macro-F1 等。
- **重复实验 E2：** 在参数与样本固定时，每个模型重复调用 3 次，分别汇总均值和标准差。重复调用不等于独立数据重采样。
- **扩展实验 E3（可选）：** 分别按 FTIR、拉曼、NIR、食品类别数或样本来源报告结果，不随意合并不同物理单位的数据。
- **对照 E4（后续）：** 同一个独立测试集上，对比原生 LLM 与完整光谱智能体；同时报告方法差异，**不能**将不同输入信息量造成的优势全部归因于“智能体推理能力”。

### 1.3 最终交付

正式完成后至少应有：原始数据清单、测试集样本清单、冻结的 Prompt、冻结的模型/参数配置、每次 API 的原始响应 JSONL、准确率和 Macro-F1 表、按类报告、混淆矩阵、异常/失败记录、实验环境和 Git commit 编号、论文 Methods/Results 所需统计表。

## 2. 数据准备与质量检查（第一天）

### 2.1 推荐字段

建立两个独立文件：

**① spectra.csv / spectra.xlsx：输入给 LLM 的纯光谱数据。**

支持两种布局。**long 格式**（每行一个光谱点）：

~~~csv
sample_id,axis,intensity
S001,400,0.125
S001,402,0.132
S001,404,0.141
S002,400,0.218
S002,402,0.224
~~~

**wide 格式**（每行一个样本，列名为光谱坐标）：

~~~csv
sample_id,400,402,404
S001,0.125,0.132,0.141
S002,0.218,0.224,0.229
~~~

以上数字为虚构结构样例，**不代表真实食品或药材的谱峰特征**。

**② metadata.csv / metadata.xlsx：测试样本标签，仅本地评分使用。**

~~~csv
sample_id,true_label,batch_id,split
S001,A,Batch01,test
S002,B,Batch02,test
S003,C,Batch03,test
~~~

字段解释：

| 字段 | 是否必要 | 说明 |
|---|---|---|
| sample_id | 必须 | 唯一、稳定、不暗示类别的样本 ID |
| axis | long 格式必须 | 波数（cm⁻¹）或波长（nm），记录单位 |
| intensity | long 格式必须 | 光谱强度，记录吸光度/透过率/计数等物理定义 |
| true_label | 必须 | 独立获取的真实分类标签 |
| batch_id | 强烈建议 | 原料批次、测量批次或样本来源 |
| split | 建议 | train / validation / test，**仅用于评估集选取** |
| modality | 建议 | FTIR / Raman / NIR / HSI 等 |

### 2.2 必须确认的科学元数据

记录仪器与测试条件：厂家/设备型号（可公开时）、光谱技术、x 轴单位、强度单位、光谱范围、采样点数、分辨率、采样方式、每样本重复扫描次数、标签获得方式、类别定义、参考标准、同一样本/批次的重复测量关系、文件编码、是否已有仪器自动处理。**未经实验设计批准，程序不再对这些数值做额外光谱处理。**

### 2.3 数据质量控制清单

- 一个 sample_id 只能有一个真实类别；同一文件中的重复坐标、空坐标、NaN/Inf、非数值列应报错或人工核验，不要静默填补或合并。
- 光谱文件的横轴方向与单位要一致；不要为“方便模型”偷偷重排坐标、截断波段、插值、填充或压缩数据。
- 每个样本坐标点数量及 token 消耗要记录。太长时**先确认模型上下文容量**；不能未经审批只取前 100 个峰或删除不利区段。
- Excel 多工作表、合并单元格、附加标题行时，先由研究者确认使用哪个工作表及标题行；示例脚本默认读取第一张表。
- CSV 数值转文本属于必需序列化；该操作可能改变数字显示精度，需采用固定格式并在论文中披露。示例采用约 15 位有效数字的浮点文本输出。
- 检查样本 ID、文件名、标签表是否有“AA / 牛奶 / 掺假”之类泄漏真实类别的信息。送入 Prompt 的仅为经批准的光谱数据。

### 2.4 测试集划分与泄漏防范

**纯 zero-shot 的原生 LLM 直接分类不需要用训练集拟合模型。** 但如果要与已训练的智能体内部机器学习模型公平比较，仍然需要一个**双方共同且未参与建模的独立测试集**。

推荐优先按真实材料来源、制备批次、供货商/生产批次或独立样本分组划分，避免同一样本的重复扫描同时出现在训练集和测试集。在类别与批次数量允许的情况下，先按 batch_id 分组，再确定测试集。**60/20/20 只是可选经验比例，不是这类 zero-shot 实验的强制要求。** 小样本与强相关重复测量情况需另行设计。测试集一旦冻结，不再据其表现修改 Prompt 或筛选模型。

示例脚本**默认对 metadata.csv 中的全部样本评分**。实际使用时必须事先将 metadata.csv 与 spectra.csv 整理成**仅包含独立测试样本**的版本；若项目以后加入自动 split 过滤，也必须留下过滤规则和测试样本清单。

## 3. 固定 Prompt 与数值输入协议（第一天）

### 3.1 只替换数据占位符

将已经审批的完整提示词原文放入 `prompts/fixed_prompt.txt`。在老师确认的**原始数据输入位置**放置唯一占位符 `{{SPECTRUM}}`，模型调用前由 Python 替换为当前样本的“axis,intensity”文本。若原文没有合适的数据插入位置，先确认插入方案并对 Prompt 重新定版，不要擅自附加任务指令。

示例：

~~~text
【此处粘贴已确认的完整固定 Prompt，不要照抄演示文本】

{{SPECTRUM}}
~~~

要求 Prompt 中的候选类别及返回格式与评价规则一致。用于测试样本的 Prompt 不得同时出现 `true_label` 列、该样本的真实类别或其同批次已知答案。

### 3.2 发送给 LLM 的结构

Chat Completions 示例仅发送**一条 user 消息**，正文为“固定 Prompt + 当前光谱数值”，不配置任何 tools，不允许 function calling，不额外联网、检索或运行分类器。具体服务商有不同的参数限制，必须在首次付费调用前根据官方文档核验。

**不要**要求模型生成 Python、调用 scikit-learn 或训练算法；只保存它直接给出的类别。若你的固定 Prompt 要求 JSON，解析方式选 JSON；若只输出类别，选 exact。

### 3.3 Prompt 版本管理

正式实验前将 Prompt 文件做 SHA-256 校验；任何空格、类别定义、分隔符、示例数量和输出格式的改动均视为新实验版本。保留配置、数据、代码和 Prompt 的版本号，以防“测试后微调 Prompt”造成评估集信息泄漏。

## 4. 模型选型及公平对比（第二天）

### 4.1 先跑通一个模型，再做横向比较

建议采用分阶段策略：

1. **首选联调模型：** 有稳定 OpenAI-compatible Chat Completions API、可按量计费、文档完整的文本模型；当前演示配置选 **DeepSeek deepseek-flash（非思考模式）**，仅用来打通流程，并非性能推荐。
2. **主对比：** 从 DeepSeek、Qwen、GPT 等系列中选 2–4 个实际可用的文本模型，模型的具体 API ID **以实验当天服务商控制台为准**。
3. **记录差异：** 基础/推理模型、思考模式、是否支持 temperature=0、上下文限制、是否存在结构化输出强制模式，都会影响公平性。**不要**把“是否启用思考”与“模型家族”混为同一因素。

### 4.2 固定和报告的变量

| 类型 | 字段/做法 |
|---|---|
| 数据 | 同一测试 sample_id 列表、类别定义、光谱数值及坐标顺序 |
| Prompt | 同一版本的文本及序列化协议；必要的提供商角色映射须记录 |
| 推理 | 每个样本独立新会话；不带历史上下文；无工具 |
| 参数 | 温度尽可能取 0（仅当服务商支持）；记录 top_p、max_tokens、thinking 参数等 |
| 模型身份 | 供应商、产品名称、实际 model ID、响应 model 字段、日期、可用快照或 revision |
| 数据容量 | 完整数值光谱须能放入上下文；未容纳的样本**不允许静默裁剪** |
| 资源统计 | 每次输入/输出 tokens、调用耗时、失败率、实际费用（若可取） |

若某模型不支持其他模型的相同参数，就采用该模型**官方有效的最近似配置**并在方法中注明差异。temperature=0 也**不等于绝对确定性**，所以需要重复测试观察稳定性。

### 4.3 可用的官方 API 说明

- DeepSeek Chat Completions：<https://api-docs.deepseek.com/api/create-chat-completion/>（2026-11 核验时列出 deepseek-flash、deepseek-v4-pro；需以运行时文档为准）
- 阿里云百炼 Qwen OpenAI-compatible Chat：<https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>（部分地域使用带 WorkspaceId 的专属 base_url）
- OpenAI Chat Completions：<https://platform.openai.com/docs/api-reference/chat/create>
- GitHub 文档：<https://docs.github.com/en/repositories>

**注意：** API 地址、可用模型名、思考参数和计费可能变化；以上链接是接口说明，不是模型排名，也不意味着账号已经开通对应权限。

## 5. Windows + Miniconda：从零执行（第二天）

你已经装有 Miniconda，**无需重新安装 Miniforge**。在 Windows PowerShell 或 Anaconda Prompt 中执行：

### 第 1 步：获取示例代码

从本对话下载随文提供的 `native_llm_spectral_benchmark_starter.zip`，解压至例如 `D:\research\native_llm_spectral_benchmark`；项目内文件包括：

~~~text
native_llm_spectral_benchmark/
├── benchmark.py                  # 完整的最小实验执行器
├── config.json                   # 数据、模型、评价配置
├── .env.example                  # API 密钥环境变量范例
├── .gitignore
├── requirements.txt
├── prompts/
│   └── fixed_prompt.txt          # 仅演示！正式实验必须换成已审批原文
├── data/
│   ├── spectra.csv               # 合成演示值，非真实科学数据
│   └── metadata.csv
├── tests/
│   └── test_benchmark.py
└── results/                      # 运行时生成，勿提交真实数据与原始日志
~~~

### 第 2 步：创建环境并安装依赖

~~~powershell
conda create -n native_spectra python=3.11 -y
conda activate native_spectra

cd D:\research\native_llm_spectral_benchmark
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

依赖：pandas、openpyxl、python-dotenv、openai、scikit-learn。注意：**scikit-learn 仅用于计算 Accuracy/F1/混淆矩阵，不是分类推断模型。**

### 第 3 步：替换数据和 Prompt

- 用你自己的**独立测试集**替换 `data/spectra.csv` 和 `data/metadata.csv`。若是 Excel，将配置文件对应路径改成 .xlsx。
- long 与 wide 的选择修改 `config.json` 中的 `data.format`，分别填 `long` / `wide`。
- 对照真实数据列名修改 `sample_id_col`、`axis_col`、`value_col`、`label_col`；wide 格式可通过 `feature_columns` 明确列顺序。
- 将 `allowed_labels` 改为项目的真实候选类别，**不要**保留演示用 A、B、C。
- 将完整固定 Prompt 填入 `prompts/fixed_prompt.txt`，且 `{{SPECTRUM}}` 恰好出现一次。
- 重要：将不同光谱数据集分开运行，记录各自配置和数据哈希；不要直接混合 FTIR 的 cm⁻¹ 与其他技术的 nm 数据。

### 第 4 步：设置 API Key

复制 `.env.example` 为 `.env`，按对应厂商在**本机**填写密钥：

~~~powershell
Copy-Item .env.example .env
notepad .env
~~~

示例内容：

~~~text
DEEPSEEK_API_KEY=这里填写你的密钥
~~~

**不要把真实 API Key 发给他人，不要写到 Markdown、GitHub、截图或提交日志中。** 如改用 Qwen / GPT，请修改 `api_key_env`、`base_url`、`model` 以及厂商支持的可选参数，并在 .env 中添加对应环境变量。

### 第 5 步：修改 config.json

示例配置如下，具体字段可参考随附文件：

~~~json
{
  "dataset_id": "FOOD_FTIR_TEST_V1",
  "modality": "FTIR",
  "data": {
    "format": "long",
    "spectra_file": "data/spectra.csv",
    "metadata_file": "data/metadata.csv",
    "sample_id_col": "sample_id",
    "axis_col": "axis",
    "value_col": "intensity",
    "label_col": "true_label"
  },
  "prompt_file": "prompts/fixed_prompt.txt",
  "allowed_labels": ["A", "B", "C"],
  "parse": {"mode": "exact"},
  "repeats": 3,
  "results_file": "results/predictions.jsonl",
  "api": {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0,
    "max_output_tokens": 64,
    "output_token_param": "max_tokens",
    "timeout_seconds": 120,
    "max_retries": 2,
    "extra_body": {"thinking": {"type": "disabled"}}
  }
}
~~~

这仅是 2026 年 11 月针对 DeepSeek **非思考模式**的联调示例。其他服务商不要原样复制 `extra_body.thinking`；若模型不支持 temperature 或 max_tokens 参数，应设为 null 或按官方要求调整，且写入实验报告。

### 第 6 步：不花 API 费用进行检查（dry-run）

~~~powershell
python benchmark.py --dry-run
~~~

检查 `results/prompt_preview.txt`：它必须**只包含已确认的 Prompt 与该测试样本的完整光谱数值**；不能泄漏真实标签，不得少点、错单位或改变坐标顺序。若预览不对，**先停止，不调用 API**。

### 第 7 步：小批量联调

~~~powershell
python benchmark.py --run --limit 3
~~~

这会对**前三个测试样本**按 `repeats` 次数调用真实 API，并产生 `results/predictions.jsonl`。请人工检查模型是否真实收到数值光谱、回答格式是否匹配以及 tokens/延迟/费用是否符合预期。

**警告：** 演示数据与演示 Prompt 没有真实科学意义。不要把联调得到的数值当作论文分类精度。若正式测试数据与 Prompt 尚未替换，只运行 dry-run，不执行付费 API。

### 第 8 步：正式全量运行与失败补跑

~~~powershell
python benchmark.py --run
python benchmark.py --run --retry-errors
python benchmark.py --evaluate
~~~

运行器采用追加式 JSONL 日志；相同的配置、Prompt、光谱和标签文件会形成同一实验哈希，便于断点续跑。若更新模型、Prompt、数据或配置，会形成新哈希并与旧实验隔离。**不要手动把不同哈希的结果合并。** API 错误记录不会当作模型分类错误直接算入最终分数；必须补跑或单独报告不可恢复失败。格式错误的模型响应保留为无效预测，不可静默删除。

### 第 9 步：改变模型

为每个模型复制一份不同的配置文件，例如 `configs/deepseek_ftir.json`、`configs/qwen_ftir.json`；但**注意脚本使用配置文件所在目录作为相对路径根目录**，因此最简单方式是将多个配置文件放在**项目根目录**，例如：

~~~powershell
Copy-Item config.json config.deepseek.json
Copy-Item config.json config.qwen.json

python benchmark.py --config config.deepseek.json --dry-run
python benchmark.py --config config.deepseek.json --run
python benchmark.py --config config.deepseek.json --evaluate

python benchmark.py --config config.qwen.json --dry-run
python benchmark.py --config config.qwen.json --run
python benchmark.py --config config.qwen.json --evaluate
~~~

每份配置应指向同一冻结测试集和 Prompt，只更换模型有关参数。不要让两个模型在**完全相同的 results_file 路径下同时并行写入**；推荐分别配置 `results/deepseek_predictions.jsonl`、`results/qwen_predictions.jsonl`。模型间使用相同测试样本 ID 进行配对比较。

## 6. 输出、错误处理与统计口径（第三天）

### 6.1 逐次预测 JSONL

每条记录至少包括：

`timestamp_utc, experiment_hash, dataset_id, modality, sample_id, repeat, model_name_config, response_model, true_label, raw_response, prediction, status, prompt_sha256, spectrum_sha256, spectral_points, response_id, prompt_tokens, completion_tokens, finish_reason, latency_seconds`。

`true_label` 仅写在**调用结束后**的本地结果，不输入给模型。日志中可能包含受限研究数据和模型响应，**默认不上传 GitHub**。

状态包括：

- `ok`：可唯一解析为允许类别。
- `invalid_output`：API 成功，但文本不符合预定义解析规则，或无法唯一识别类别。
- `api_error`：网络、额度、超时、鉴权、限速等请求失败；保留错误类型与简要信息。

**禁止**把无效回答“人工猜测成最接近的类别”。若解析规则需调整，冻结新的解析版本并全数据一致重算。

### 6.2 指标定义与主报告口径

在**所有预定请求成功取得 API 响应**后才计算论文主表：

- **Accuracy_all** = 预测正确的样本数 / 所有预定样本数；`invalid_output` 记为预测错误；
- **Macro-Precision / Macro-Recall / Macro-F1**：仅针对预先声明的真实类别，按类均匀平均；无效输出作为额外预测类别，但不作为真实类别参与 macro 平均；
- **Invalid Output Rate** = 无法解析为合法类别的响应数 / 成功返回的预定响应数；
- **Valid-only Accuracy（辅助）** = 正确预测数 / 有效输出数；必须与覆盖率同时报告，**不得只报告这个看似较高的数**；
- **API Error Rate** 单独报告，不与模型分类准确率混淆；
- **混淆矩阵**：候选类别 + 一个 `__INVALID__` 列，保留错分与无效回答。

对于三次重复，分别统计每一轮 Accuracy / Macro-F1，再报告**均值 ± 样本标准差**。同一测试集多次推理的标准差是生成稳定性指标，**不是**独立采样误差；若论文需要置信区间，可另做以独立样本/批次为单位的 bootstrap，避免把同一样本的重复扫描视为独立观测。

### 6.3 论文结果表（空表，不填虚构分数）

| 数据集 | 光谱类型 | 模型及版本 | 测试样本数 | Accuracy ↑ | Macro-F1 ↑ | 无效输出率 ↓ | API 失败率 ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Dataset-1 | FTIR | 待确定 | — | — | — | — | — |
| Dataset-1 | FTIR | 待确定 | — | — | — | — | — |
| Dataset-2 | Raman | 待确定 | — | — | — | — | — |

另附每种模型 × 数据集的混淆矩阵，以及各类 Precision/Recall/F1。能合理配对时，可对分类正确/错误的样本进行配对统计检验；不可仅凭很小的百分点差异宣称显著优越。

## 7. 经费与性能边界

### 7.1 调用预算

单模型预期调用次数：

`N_calls = N_test_samples × N_repeats`

总调用量：

`N_total ≈ Σ(N_test_samples × N_repeats × N_models)`

大致费用由**输入 tokens × 输入计费单价 + 输出 tokens × 输出计费单价**决定，并应考虑缓存折扣、重试、思考 tokens 与不同供应商的计费策略。先拿 1 个样本做 dry-run，再用最多 3 个**真实测试样本**小额试跑，记录真实 tokens 和实付费用估计，再决定是否全量运行。不要提前假设长光谱“肯定放得下”。

### 7.2 数值光谱的先天问题

LLM 通过文本 token 读取光谱数值并不等于它具备专业化学光谱模型的测量归纳能力。序列化精度、光谱点数量、模型上下文与训练分布都会制约表现。主实验应保留这些事实，而不是为了好看的结果先擅自进行谱峰挑选或特征工程。

如需比较“原始完整光谱”“峰位摘要”“图像输入”等，应另设有明确协议的消融实验，并在 Methods 中单独定义，**不能替换原始 direct-LLM 基线**。

## 8. 质量保证、验收标准和研究日志

### 8.1 每次正式运行前的核查

- [ ] data、metadata、Prompt、config 已复制为冻结版本并生成 SHA-256。
- [ ] 主测试集与智能体使用的测试集相同，按独立材料/批次隔离，无训练泄漏。
- [ ] 每一行的 sample_id、axis、intensity、true_label 对应准确，且未将标签送给模型。
- [ ] 光谱点数量、坐标顺序、单位、输入文本的格式固定。
- [ ] Prompt 是老师确认的最终原文，{{SPECTRUM}} 插入位置经审核。
- [ ] 模型 API ID、思考模式、temperature、输出上限与解析方式已记录。
- [ ] dry-run 生成的 Prompt 预览人工核验通过。
- [ ] 小样本 API 联调的响应与费用检查通过。
- [ ] 所有样本、所有重复轮次、所有模型都有可追踪的响应或明确失败记录。
- [ ] 报告主表没有过滤无效响应，也没有把 API 失败混作模型错误。
- [ ] 原始数据和密钥未公开；仓库提交编号和脚本版本已记录。

### 8.2 验收标准

**阶段 A（流程完成）**：单元测试通过；CSV/Excel 的 long/wide 布局能正确导入；dry-run 精确预览请求；标签不会进入 Prompt。

**阶段 B（小规模有效）**：至少 3 个真实测试样本成功调用正式指定模型，原始响应、耗时、tokens、解析结果均落盘；可以安全断点续跑。

**阶段 C（完整基线）**：每个模型 × 数据集的所有预定请求已完成；无效输出率与模型表现可复算；结果表与混淆矩阵一致。

**阶段 D（可写论文）**：有冻结版本、硬件/软件/API 描述、数据切分依据、指标公式、失败处置、费用和限制性分析；必要时完成与智能体结果的配对比较。

### 8.3 建议工作顺序（可按样本规模调整）

| 顺序 | 工作 | 输出 |
|---|---|---|
| 1 | 盘点各数据集、单位、标签、批次、Prompt | dataset_manifest.csv、Prompt v1 |
| 2 | 整理独立测试集并核查泄漏 | frozen_test_spectra、metadata、样本列表 |
| 3 | 安装运行环境与执行 dry-run | 依赖清单、Prompt 预览、测试记录 |
| 4 | 小样本 API 联调与解析规则锁定 | 3 样本日志、预算预估 |
| 5 | 多模型全量分类与异常补跑 | JSONL 原始响应 |
| 6 | 计算指标、绘制混淆矩阵 | metrics、figures、结果总表 |
| 7 | 与智能体结果对齐并撰写论文 | Methods、Results、Discussion |

## 9. GitHub 管理、隐私与项目提交

本方案文档已存放在仓库 `ioopms/bio-chem-ml-assistant` 的 `docs/native_llm_spectral_benchmark_plan.md`。这是当前账户下现有的**科研助手仓库**，并非自动创建了一个同名新仓库。如果后续希望实验完全独立，可在 GitHub 新建 `native_llm_spectral_benchmark` 仓库，再迁移文档与代码。

建议实际实验项目单独保存于 `experiments/native_llm_spectral_benchmark/` 或单独仓库；将 `data/`、`results/`、`.env` 加入 `.gitignore`，只公开经过授权的演示样本和非涉密配置。项目上传前使用 `git status` 和 `git diff --cached` 核对，避免误提交样本数据和密钥。

Windows 下手工提交示例（假设已克隆目标仓库，并将方案文件复制至 docs/）：

~~~powershell
git status
git add docs/native_llm_spectral_benchmark_plan.md
git commit -m "docs: add reproducible native-LLM spectral classification plan"
git push origin main
~~~

**不要**执行 `git add .` 后不检查就提交，尤其当本地存放了未公开的食品/药材数据、API Key 或模型原始日志。

## 10. 下一步还需要补齐的真实实验信息

| 项目 | 当前情况 | 正式开跑前动作 |
|---|---|---|
| 真实 CSV/Excel | 未提供 | 检查列名、long/wide 形式、样本/批次关系 |
| 固定 Prompt 原文 | 用户称已确定，但未提供 | 放入 prompts/fixed_prompt.txt，人工确认插入位置 |
| 真实分类标签和类别数量 | 未提供 | 确认 allowed_labels、标签证据来源 |
| 测试集划分 | 未提供 | 冻结独立测试样本清单，避免与智能体训练集重叠 |
| LLM 模型 | 未确定 | 选择实际可访问的 API model ID，先做 3 样本联调 |
| API Key | 未提供且不需要发送 | 用户本机 .env 私密保存 |
| 模型数和重复次数 | 尚未最终确定 | 建议先 1 个模型联调，再扩展到 2–4 个模型、每模型 3 次 |

## 11. 可直接用于论文的 Methods 写作骨架

> **Direct native-LLM baseline.** To assess the classification ability of native large language models without external analytical tools, we constructed a direct-prediction baseline using [FTIR/Raman/...] spectra. For each held-out sample, the original spectral coordinate–intensity pairs were serialized into a fixed textual representation and inserted into a version-controlled, predefined prompt. Each request was sent independently to [model IDs, revisions, providers] through [API/protocol], without spectral preprocessing, feature extraction, model fine-tuning, retrieval, code execution, or external classifiers. The ground-truth label was withheld from the model and used only for post-hoc evaluation. Model settings included [temperature/thinking/output budget] and [repeats] independent calls per sample. Outputs were parsed with a predetermined rule; nonconforming model responses were counted as invalid classifications. Overall accuracy, macro-averaged precision, recall, F1, invalid-output rates, and confusion matrices were computed on the same frozen test set. API errors were logged and retried under the predefined protocol. Data, prompt, and configuration hashes were retained to support reproducibility.

上述英文段落**是待填充模板**，不代表实验已经完成。最终需按真实数据集、模型和处理细节修改。

---

### 附：关键决策记录

1. **Python 只负责实验调度与统计，LLM 自行给出分类标签。**
2. **未获得真实固定 Prompt 之前，只做格式检查，不得开展正式实验。**
3. **默认 zero-shot 不需要训练集拟合；独立测试集仍必需。**
4. **评估集和标签不进入模型请求。**
5. **无效输出与 API 错误分开统计，原始响应始终保留。**
6. **模型 ID、参数和接口信息按正式运行当天的官方文档确认。**
7. **论文主表只使用完成度已核验的实验结果，不使用演示数据分数。**
