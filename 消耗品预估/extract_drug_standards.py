"""
药品检验标准耗材提取脚本
功能：从药品检验标准 JSON 文件中提取耗材定额信息
"""

import os
import sys
import json
from openai import OpenAI
from pathlib import Path

# ============================== Windows 编码设置 ==============================
# 设置 UTF-8 编码输出，解决 Windows 控制台 GBK 编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================== 配置部分 ==============================
# API 配置
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-flash"

# 文件夹配置
DATA_FOLDER = "data"
RESULTS_FOLDER = "results"

# ============================== System Prompt ==============================
SYSTEM_PROMPT = """你是一个制药工程领域的资深QA专家。你将接收到一份包含"药品具体标准"及其"引用通则"的文本。
你的任务是提取检验过程中所需的**耗材定额**。

请严格遵守以下规则，输出 JSON：

1. **结构化提取**：
   - 按"检验项目"（如【性状】【鉴别】【检查】【含量测定】）分组。
   - **跨文本关联（核心）**：当药品标准中提到"依法检查（通则XXXX）"时，你必须查阅文本后附的该通则内容，提取通则中规定的试剂和用量（例如通则中规定的"加硫氰酸铵溶液3ml"必须提取出来）。

2. **提取核心逻辑 - "Base Pair"（基准配比）**：
   - 找到"取本品 X g/ml"作为 `base_basis`。
   - 提取该步骤下（包括引用的通则步骤中）加入的所有试剂、溶剂、耗材，作为 `consumables`。
   - **不要计算比例**，保留原文数值。

3. **特殊场景**：
   - **HPLC/GC**：标记 `step_type` 为 "chromatography"，提取 `mobile_phase`（流动相）配比。
   - **对照品**：如果是"另取对照品..."，请单独列为一个 operation。

4. **语言规范**：
   - **Key (键名)**：必须使用英文 (如 drug_name, consumables)。
   - **Value (值)**：必须保留**原始中文**，严禁翻译 (如保留"甲醇"，不要变成"Methanol")。

输出 JSON 格式模板：
{
  "drug_name": "String",
  "inspection_items": [
    {
      "item_name": "String (如：铁盐检查)",
      "operations": [
        {
          "step_type": "sample_preparation" | "chromatography",
          "base_basis": { "target_name": "String", "amount": float, "unit": "String" },
          "consumables": [
            { "name": "String (中文)", "amount": float, "unit": "String" }
          ],
          "mobile_phase": "String"
        }
      ]
    }
  ]
}
"""


# ============================== 数据处理函数 ==============================

def build_input_text(drug_data: dict) -> str:
    """
    将药品 JSON 数据转换为大模型可读的长文本

    Args:
        drug_data: 药品标准数据字典

    Returns:
        构建好的文本字符串
    """
    text_parts = []

    # 步骤 A: 提取药品名称
    drug_name = drug_data.get("名称", "未知药品")
    text_parts.append(f"【药品名称】\n{drug_name}\n")

    # 步骤 B: 提取检验项目内容
    inspection_fields = ["性状", "鉴别", "检查", "含量测定"]
    for field in inspection_fields:
        if field in drug_data and drug_data[field]:
            text_parts.append(f"【{field}】\n{drug_data[field]}\n")

    # 步骤 C（核心）: 提取通则引用的详细内容
    if "通则引用" in drug_data and drug_data["通则引用"]:
        text_parts.append("\n【通则引用详细内容】\n")
        text_parts.append("（以下为上述检验中引用的通则标准完整内容，请结合药品标准提取耗材）\n")

        for general_chapter in drug_data["通则引用"]:
            # 处理两种数据格式：字典或字符串
            if isinstance(general_chapter, dict):
                number = general_chapter.get("number", "")
                name = general_chapter.get("name", "")
                content = general_chapter.get("content", "")

                text_parts.append(f"\n--- {number} {name} ---\n")
                text_parts.append(f"{content}\n")
            elif isinstance(general_chapter, str):
                # 如果是字符串，直接显示（这种情况下没有详细内容）
                text_parts.append(f"\n--- {general_chapter} （未提供详细内容）---\n")

    return "\n".join(text_parts)


def load_json_files(data_folder: str) -> list:
    """
    加载指定文件夹下的所有 JSON 文件

    Args:
        data_folder: 数据文件夹路径

    Returns:
        包含 (文件名, 数据字典) 元组的列表
    """
    json_files = []
    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"❌ 错误：文件夹 {data_folder} 不存在")
        return json_files

    for file_path in data_path.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_files.append((file_path.name, data))
                print(f"✅ 已加载: {file_path.name}")
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path.name}: {str(e)}")

    return json_files


def call_qwen_model(input_text: str, client: OpenAI) -> dict:
    """
    调用阿里云 Qwen-Long 模型提取耗材信息

    Args:
        input_text: 输入文本
        client: OpenAI 客户端实例

    Returns:
        模型返回的 JSON 结果
    """
    try:
        print("🔄 正在调用 Qwen-Long 模型...")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,  # 降低温度以获得更稳定的结果
            response_format={"type": "json_object"}  # 强制返回 JSON 格式
        )

        result_text = response.choices[0].message.content
        print("✅ 模型调用成功")

        # 解析返回的 JSON
        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"❌ 模型调用失败: {str(e)}")
        return None


def save_result(filename: str, result: dict, results_folder: str):
    """
    保存提取结果到 JSON 文件

    Args:
        filename: 原始文件名
        result: 提取结果字典
        results_folder: 结果文件夹路径
    """
    try:
        # 创建结果文件夹
        results_path = Path(results_folder)
        results_path.mkdir(exist_ok=True)

        # 构建输出文件名（添加 _result 后缀）
        original_name = Path(filename).stem
        output_filename = f"{original_name}_result.json"
        output_path = results_path / output_filename

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ 结果已保存: {output_filename}")

    except Exception as e:
        print(f"❌ 保存结果失败 {filename}: {str(e)}")


# ============================== 主函数 ==============================

def main():
    """
    主函数：执行完整的提取流程
    """
    print("=" * 60)
    print("药品检验标准耗材提取工具")
    print("=" * 60)
    print()

    # 初始化 OpenAI 客户端
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        print(f"✅ 已连接到阿里云 DashScope (模型: {MODEL_NAME})")
        print()
    except Exception as e:
        print(f"❌ 初始化客户端失败: {str(e)}")
        return

    # 加载所有 JSON 文件
    json_files = load_json_files(DATA_FOLDER)

    if not json_files:
        print(f"❌ 在 {DATA_FOLDER} 文件夹中未找到 JSON 文件")
        return

    print(f"\n📊 共加载 {len(json_files)} 个文件")
    print()

    # 处理每个文件
    success_count = 0
    fail_count = 0

    for idx, (filename, drug_data) in enumerate(json_files, 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(json_files)}] 正在处理: {filename}")
        print(f"{'=' * 60}")

        # 构建输入文本
        input_text = build_input_text(drug_data)
        print(f"📝 输入文本长度: {len(input_text)} 字符")

        # 调用模型
        result = call_qwen_model(input_text, client)

        if result:
            # 保存结果
            save_result(filename, result, RESULTS_FOLDER)
            success_count += 1
        else:
            fail_count += 1

    # 输出统计信息
    print()
    print("=" * 60)
    print("📊 处理完成！")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {fail_count} 个")
    print(f"   结果保存在: {RESULTS_FOLDER}/ 文件夹")
    print("=" * 60)


if __name__ == "__main__":
    main()
