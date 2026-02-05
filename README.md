# 药品检验标准智能分析系统

基于大语言模型的药品检验标准自动化分析系统，支持实体抽取、聚类分析、耗材定额提取等功能。

## 项目简介

本项目利用 AI 大语言模型（通义千问 Qwen）对药品检验标准进行智能化分析处理，实现：

- **实体抽取**：从药品标准文本中自动提取检验相关的仪器设备、试剂、检测步骤等实体信息
- **聚类分析**：对大量药品进行分类聚类，发现相似性
- **耗材定额提取**：批量提取药品检验过程中所需的耗材定额，支持并发处理
- **报告生成**：自动生成药品检验摘要报告

## 项目结构

```
drug-inspection-project/
├── drug_extraction/           # 药品检验标准实体抽取
│   ├── Clustering/            # 聚类分析模块
│   │   ├── cluster_single_drug.py
│   │   ├── cluster_test.py
│   │   ├── generate_entity_embeddings.py
│   │   └── multi_drug_clustering.py
│   ├── Extracting/            # 实体抽取模块
│   │   └── test_final.py
│   └── recluster/             # 重新聚类模块
│       └── 1.py
├── summary/                   # 摘要生成模块
│   ├── 1.py
│   └── 2.py
├── 性状抽取/                   # 药品性状抽取模块
│   ├── extract_properties.py
│   ├── process_drugs.py
│   └── stream_process_drugs.py
├── 消耗品预估/                 # 耗材定额提取模块（推荐使用）
│   ├── batch_extract_standards.py  # 批量提取（支持并发）
│   ├── extract_drug_standards.py   # 单个提取
│   └── retry_failed.py              # 失败重试
└── 摘要/                       # 报告生成模块
    └── generate_report.py
```

## 核心功能

### 1. 实体抽取 (`drug_extraction/Extracting/test_final.py`)

从药品检验标准文本中抽取以下类型的实体：

- **仪器设备**：HPLC、UV分光光度计等
- **试剂/溶液**：标准品、对照品、缓冲液等
- **色谱柱/填充剂**：C18、硅胶等
- **其他材料**：滤膜、离心管等
- **前处理过程**：粉碎、溶解、离心等
- **溶液配置过程**：标准溶液制备、缓冲液配制
- **具体测定操作**：进样、检测、数据分析
- **限度判定指标**：含量限度、杂质限度等

### 2. 耗材定额提取 (`消耗品预估/`)

批量处理药品标准，提取检验过程中的耗材定额：

- 支持多线程并发处理
- 断点续传，失败重试
- 按检验项目（性状、鉴别、检查、含量测定）分组
- 支持跨文本关联（通则引用）
- 识别 HPLC/GC 流动相配比

### 3. 聚类分析 (`drug_extraction/Clustering/`)

对大量药品进行聚类分析，发现相似药品，便于批量处理。

## 环境要求

- Python 3.8+
- 依赖包：
  - `openai`
  - `requests`
  - `tqdm`
  - `concurrent.futures`

## 配置说明

在使用前，需要配置 API 密钥：

```python
# 在各模块的配置文件顶部修改
API_KEY = "your-api-key-here"  # 通义千问 API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-flash"  # 或其他模型
```

**注意**：出于安全考虑，API_KEY 已设置为空值，使用前请填入你自己的 API 密钥。

## 使用方法

### 耗材定额批量提取（推荐）

```bash
cd 消耗品预估
python batch_extract_standards.py
```

### 单个药品耗材提取

```bash
cd 消耗品预估
python extract_drug_standards.py
```

### 实体抽取

```bash
cd drug_extraction/Extracting
python test_final.py
```

### 聚类分析

```bash
cd drug_extraction/Clustering
python multi_drug_clustering.py
```

## 输入输出

### 输入格式

系统期望的输入为 JSON 格式的药品标准数据，包含以下字段：

```json
{
  "名称": "药品名称",
  "性状": "性状描述...",
  "鉴别": "鉴别描述...",
  "检查": "检查描述...",
  "含量测定": "含量测定描述...",
  "通则引用": [
    {
      "number": "通则编号",
      "name": "通则名称",
      "content": "通则详细内容"
    }
  ]
}
```

### 输出格式

#### 实体抽取结果

```json
{
  "检验项目": "检验项目名称",
  "检测资源": {
    "仪器设备": [...],
    "试剂/溶液": [...],
    "色谱柱/填充剂": [...],
    "其他材料": [...]
  },
  "检测步骤": {...}
}
```

#### 耗材定额结果

```json
{
  "drug_name": "药品名称",
  "inspection_items": [
    {
      "item_name": "检验项目名称",
      "operations": [
        {
          "step_type": "sample_preparation",
          "base_basis": {
            "target_name": "取本品X g/ml",
            "amount": 1.0,
            "unit": "g"
          },
          "consumables": [
            {
              "name": "甲醇",
              "amount": 5.0,
              "unit": "ml"
            }
          ],
          "mobile_phase": "乙腈-水(50:50)"
        }
      ]
    }
  ]
}
```

## 技术特点

1. **高精度实体定义**：明确的实体类别定义，减少抽取歧义
2. **跨文本关联**：支持通则引用，自动关联检验通则内容
3. **容错机制**：API 调用失败自动重试，指数退避
4. **断点续传**：处理中断后可继续，支持覆盖模式
5. **中文友好**：保留原始中文描述，符合中文药典规范

## 注意事项

- API 密钥需要自行配置，项目中已清空
- 大数据文件（超过100MB）未上传至 GitHub
- 建议使用虚拟环境运行
- 批量处理时请注意 API 调用频率限制

## License

MIT License

## 贡献者

[@dazzlinng](https://github.com/dazzlinng)
