"""
重新抽取失败的 6 个药品
超时时间增加到 180 秒
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI

# API 配置
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-flash"

# 文件配置
INPUT_FOLDER = "data"
RESULTS_FOLDER = "results"

# System Prompt (从原脚本复制)
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
   - **特殊情况**：如果原文耗材用量为"适量"或"少许"或"足量"：
     * `amount` 必须填数字 -1（注意：是数字 -1，不是字符串）
     * `unit` 填字符串 "适量"

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


def sanitize_filename(filename: str) -> str:
    """清洗文件名"""
    import re
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', filename)
    return sanitized.strip() or "unnamed_drug"


def build_input_text(drug_data: dict) -> str:
    """构建输入文本"""
    text_parts = []

    # 药品名称
    drug_name = drug_data.get("名称", "未知药品")
    text_parts.append(f"【药品名称】\n{drug_name}\n")

    # 检验项目
    inspection_fields = ["性状", "鉴别", "检查", "含量测定"]
    for field in inspection_fields:
        if field in drug_data and drug_data[field]:
            text_parts.append(f"【{field}】\n{drug_data[field]}\n")

    # 通则引用
    if "通则引用" in drug_data and drug_data["通则引用"]:
        text_parts.append("\n【通则引用详细内容】\n")
        text_parts.append("（以下为上述检验中引用的通则标准完整内容，请结合药品标准提取耗材）\n")

        for general_chapter in drug_data["通则引用"]:
            # 防御性编程
            if not isinstance(general_chapter, dict):
                continue

            number = general_chapter.get("number", "")
            name = general_chapter.get("name", "")
            content = general_chapter.get("content", "")

            text_parts.append(f"\n--- {number} {name} ---\n")
            text_parts.append(f"{content}\n")

    return "\n".join(text_parts)


def postprocess_consumables(result: dict) -> dict:
    """后处理：修正"适量"等模糊用量为 -1"""
    fuzzy_keywords = ["适量", "少许", "足量", "数滴", "数ml", "数克"]

    if "inspection_items" not in result:
        return result

    for item in result["inspection_items"]:
        if "operations" not in item:
            continue

        for operation in item["operations"]:
            # 修正 consumables 中的 amount
            if "consumables" in operation:
                for consumable in operation["consumables"]:
                    amount = consumable.get("amount")
                    if isinstance(amount, str) and any(keyword in amount for keyword in fuzzy_keywords):
                        consumable["amount"] = -1
                        consumable["unit"] = "适量"

            # 修正 base_basis 中的 amount
            if "base_basis" in operation:
                base_amount = operation["base_basis"].get("amount")
                if isinstance(base_amount, str) and any(keyword in base_amount for keyword in fuzzy_keywords):
                    operation["base_basis"]["amount"] = -1
                    operation["base_basis"]["unit"] = "适量"

    return result


def call_qwen_model(input_text: str, client: OpenAI, max_retries: int = 5) -> dict:
    """调用模型（超时180秒，最多重试5次）"""
    for attempt in range(max_retries):
        try:
            print(f"    尝试 {attempt + 1}/{max_retries}...", end=" ")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.1,
                timeout=180  # 增加到 180 秒
            )

            result_text = response.choices[0].message.content

            # 清理 markdown 代码块标记
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            # 尝试修复常见 JSON 错误
            result_text = fix_common_json_errors(result_text)

            result = json.loads(result_text)
            result = postprocess_consumables(result)

            print("[OK]")
            return result

        except Exception as e:
            error_msg = str(e)
            is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()

            if is_timeout and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 递增等待：10秒、20秒、30秒...
                print(f"[Timeout] {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue

            print(f"[Failed] {error_msg}")
            raise Exception(f"模型调用失败: {error_msg}")

    raise Exception("模型调用失败: 达到最大重试次数")


def fix_common_json_errors(text: str) -> str:
    """尝试修复常见的 JSON 错误"""
    import re

    # 尝试找到最后一个完整的 JSON 对象
    # 如果字符串被截断，尝试找到最后一个完整的 }
    try:
        # 从后往前找，找到最后一个完整的对象
        brace_count = 0
        last_valid_pos = -1

        for i in range(len(text) - 1, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1

            if brace_count == 0 and text[i] in '}]':
                last_valid_pos = i + 1
                break

        if last_valid_pos > 0:
            text = text[:last_valid_pos]
    except:
        pass

    return text


def save_result(drug_name: str, result: dict, results_folder: str):
    """保存结果"""
    results_path = Path(results_folder)
    results_path.mkdir(exist_ok=True)

    safe_name = sanitize_filename(drug_name)
    output_filename = f"{safe_name}_result.json"
    output_path = results_path / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    print("=" * 80)
    print("重新抽取失败的 6 个药品")
    print("=" * 80)
    print()

    # 初始化客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("[OK] 已连接到阿里云 DashScope")
    print(f"    模型: {MODEL_NAME}")
    print(f"    超时设置: 180 秒")
    print(f"    最大重试: 5 次")
    print()

    # 读取所有文件
    input_folder = Path(INPUT_FOLDER)
    files = list(input_folder.glob("*.json"))

    print(f"[INFO] 找到 {len(files)} 个药品文件")
    print()

    # 处理每个药品
    success_count = 0
    failed_count = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 处理: {file_path.name}")

        try:
            # 读取数据
            with open(file_path, 'r', encoding='utf-8') as f:
                drug_data = json.load(f)

            drug_name = drug_data.get("名称", file_path.stem)
            print(f"    药品名称: {drug_name}")

            # 构建输入文本
            input_text = build_input_text(drug_data)

            # 调用模型
            print("    正在调用模型...", end=" ")
            result = call_qwen_model(input_text, client)

            # 保存结果
            output_path = save_result(drug_name, result, RESULTS_FOLDER)
            print(f"    结果已保存: {output_path.name}")

            success_count += 1
            print(f"    [DONE] 完成")

        except Exception as e:
            print(f"    [FAILED] {str(e)}")
            failed_count += 1

        print()

    # 总结
    print("=" * 80)
    print("[INFO] 处理完成！")
    print("=" * 80)
    print(f"   [SUCCESS] 成功: {success_count} 个")
    print(f"   [FAILED] 失败: {failed_count} 个")
    print(f"   [LOCATION] 结果保存在: {RESULTS_FOLDER}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
