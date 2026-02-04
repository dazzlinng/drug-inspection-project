"""
药品检验标准耗材批量提取脚本
功能：批量处理大量药品标准，支持并发、断点续传、错误处理
"""

import os
import json
import re
import threading
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ============================== 配置部分 ==============================
# API 配置
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-flash"  # 使用 Flash 模型，速度快成本低

# 文件配置
INPUT_FILE = "all_drugs_standards.json"
RESULTS_FOLDER = "results"
FAILED_LOG = "failed_drugs.log"

# 并发配置
MAX_WORKERS = 5  # 并发线程数，避免触发 QPS 限制

# 测试模式配置
TEST_MODE = False  # True=只处理前10个药品，False=处理全部
OVERWRITE = True  # True=强制覆盖已有结果，False=跳过已有结果

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


# ============================== 全局变量（线程安全） ==============================
# 用于线程安全的文件写入锁
file_lock = threading.Lock()
failed_counter = 0
success_counter = 0
skipped_counter = 0


# ============================== 辅助函数 ==============================

def sanitize_filename(filename: str) -> str:
    """
    清洗文件名，移除非法字符

    Args:
        filename: 原始文件名

    Returns:
        清洗后的安全文件名
    """
    # 移除或替换 Windows/Linux 不允许的字符
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', filename)

    # 移除前后空格
    sanitized = sanitized.strip()

    # 如果为空，返回默认名称
    if not sanitized:
        sanitized = "unnamed_drug"

    return sanitized


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
            # 防御性编程：检查是否为字典类型
            if not isinstance(general_chapter, dict):
                # 如果是字符串或其他类型，跳过
                continue

            number = general_chapter.get("number", "")
            name = general_chapter.get("name", "")
            content = general_chapter.get("content", "")

            text_parts.append(f"\n--- {number} {name} ---\n")
            text_parts.append(f"{content}\n")

    return "\n".join(text_parts)


def load_drug_data(input_file: str) -> list:
    """
    加载药品数据 JSON 文件

    Args:
        input_file: 输入文件路径

    Returns:
        药品数据列表
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[ERROR] 错误：输入文件应该是一个列表（List）格式")
            return []

        print(f"[OK] 成功加载 {len(data)} 个药品标准")
        return data

    except FileNotFoundError:
        print(f"[ERROR] 错误：文件 {input_file} 不存在")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析错误：{str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] 加载数据失败：{str(e)}")
        return []


def is_result_exists(drug_name: str, results_folder: str) -> bool:
    """
    检查结果文件是否已存在

    Args:
        drug_name: 药品名称
        results_folder: 结果文件夹路径

    Returns:
        True 如果文件已存在，False 否则
    """
    safe_name = sanitize_filename(drug_name)
    result_file = Path(results_folder) / f"{safe_name}_result.json"
    return result_file.exists()


def save_result(drug_name: str, result: dict, results_folder: str):
    """
    保存提取结果到 JSON 文件（线程安全）

    Args:
        drug_name: 药品名称
        result: 提取结果字典
        results_folder: 结果文件夹路径
    """
    global file_lock

    try:
        # 创建结果文件夹
        results_path = Path(results_folder)
        results_path.mkdir(exist_ok=True)

        # 构建输出文件名
        safe_name = sanitize_filename(drug_name)
        output_filename = f"{safe_name}_result.json"
        output_path = results_path / output_filename

        # 使用锁保护文件写入
        with file_lock:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise Exception(f"保存结果失败: {str(e)}")


def log_failed_drug(drug_name: str, error_msg: str):
    """
    将失败的药品记录到日志文件（线程安全）

    Args:
        drug_name: 药品名称
        error_msg: 错误信息
    """
    global file_lock

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {drug_name} - {error_msg}\n"

    with file_lock:
        with open(FAILED_LOG, 'a', encoding='utf-8') as f:
            f.write(log_entry)


def call_qwen_model(input_text: str, client: OpenAI, max_retries: int = 3) -> dict:
    """
    调用阿里云 Qwen-Flash 模型提取耗材信息（支持自动重试）

    Args:
        input_text: 输入文本
        client: OpenAI 客户端实例
        max_retries: 最大重试次数（默认3次）

    Returns:
        模型返回的 JSON 结果
    """
    import time

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.1,  # 降低温度以获得更稳定的结果
                timeout=120  # 增加超时时间到 120 秒
            )

            result_text = response.choices[0].message.content

            # 尝试解析 JSON（可能包含 markdown 代码块）
            # 移除可能的 markdown 代码块标记
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)

            # 后处理：修正"适量"等模糊用量为 -1
            result = postprocess_consumables(result)

            return result

        except Exception as e:
            error_msg = str(e)
            # 判断是否为超时错误
            is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()

            # 如果是超时错误且还有重试次数，等待后重试
            if is_timeout and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 递增等待时间：5秒、10秒、15秒
                print(f"\n[RETRY] 超时错误，{wait_time}秒后进行第{attempt + 2}次重试...")
                time.sleep(wait_time)
                continue

            # 如果不是超时错误，或已达到最大重试次数，抛出异常
            raise Exception(f"模型调用失败: {error_msg}")

    # 理论上不会执行到这里，但为了代码完整性
    raise Exception("模型调用失败: 达到最大重试次数")


def postprocess_consumables(result: dict) -> dict:
    """
    后处理：将所有"适量"、"少许"、"足量"等模糊用量转换为数字 -1

    Args:
        result: 模型返回的结果字典

    Returns:
        修正后的结果字典
    """
    # 定义需要转换为 -1 的模糊用量关键词
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
                    # 如果 amount 是字符串且包含模糊关键词，转换为 -1
                    if isinstance(amount, str) and any(keyword in amount for keyword in fuzzy_keywords):
                        consumable["amount"] = -1
                        consumable["unit"] = "适量"

            # 修正 base_basis 中的 amount（如果存在）
            if "base_basis" in operation:
                base_amount = operation["base_basis"].get("amount")
                if isinstance(base_amount, str) and any(keyword in base_amount for keyword in fuzzy_keywords):
                    operation["base_basis"]["amount"] = -1
                    operation["base_basis"]["unit"] = "适量"

    return result


# ============================== 处理单个药品的函数 ==============================

def process_single_drug(drug_data: dict, client: OpenAI, pbar: tqdm, overwrite: bool = False) -> tuple:
    """
    处理单个药品标准（线程工作函数）

    Args:
        drug_data: 药品数据字典
        client: OpenAI 客户端实例
        pbar: 进度条实例
        overwrite: 是否强制覆盖已有结果

    Returns:
        (成功状态, 药品名称, 错误信息)
    """
    global success_counter, failed_counter, skipped_counter

    drug_name = drug_data.get("名称", "未知药品")

    try:
        # 步骤 1: 检查是否已存在结果（断点续传）
        if not overwrite and is_result_exists(drug_name, RESULTS_FOLDER):
            with file_lock:
                skipped_counter += 1
                pbar.set_postfix({
                    "成功": success_counter,
                    "跳过": skipped_counter,
                    "失败": failed_counter
                })
            return (True, drug_name, "已存在，跳过")

        # 步骤 2: 构建输入文本
        input_text = build_input_text(drug_data)

        # 步骤 3: 调用模型
        result = call_qwen_model(input_text, client)

        # 步骤 4: 保存结果
        save_result(drug_name, result, RESULTS_FOLDER)

        # 更新成功计数
        with file_lock:
            success_counter += 1
            pbar.set_postfix({
                "成功": success_counter,
                "跳过": skipped_counter,
                "失败": failed_counter
            })

        return (True, drug_name, None)

    except Exception as e:
        error_msg = str(e)

        # 记录失败日志
        log_failed_drug(drug_name, error_msg)

        # 更新失败计数
        with file_lock:
            failed_counter += 1
            pbar.set_postfix({
                "成功": success_counter,
                "跳过": skipped_counter,
                "失败": failed_counter
            })

        # 打印红色错误信息（仅在终端支持时）
        print(f"\n[FAILED] 处理失败: {drug_name} - {error_msg}")

        return (False, drug_name, error_msg)


# ============================== 主函数 ==============================

def main():
    """
    主函数：执行批量处理流程
    """
    global success_counter, failed_counter, skipped_counter

    print("=" * 80)
    print("药品检验标准耗材批量提取工具")
    print("=" * 80)
    print()

    # 初始化 OpenAI 客户端
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        print(f"[OK] 已连接到阿里云 DashScope")
        print(f"   模型: {MODEL_NAME}")
        print(f"   并发数: {MAX_WORKERS}")
        print()
    except Exception as e:
        print(f"[ERROR] 初始化客户端失败: {str(e)}")
        return

    # 加载数据
    print(f"[INFO] 正在加载数据文件: {INPUT_FILE}")
    drug_list = load_drug_data(INPUT_FILE)

    if not drug_list:
        print("[ERROR] 没有可处理的数据")
        return

    print(f"[INFO] 共 {len(drug_list)} 个药品需要处理")
    print()

    # 测试模式：只处理前10个药品
    if TEST_MODE:
        print("[TEST MODE] 测试模式已启用，只处理前 10 个药品")
        drug_list = drug_list[:10]
        print(f"[TEST MODE] 实际将处理 {len(drug_list)} 个药品")
        print()

    # 创建结果文件夹
    Path(RESULTS_FOLDER).mkdir(exist_ok=True)

    # 清空失败日志（新建或覆盖）
    with open(FAILED_LOG, 'w', encoding='utf-8') as f:
        f.write(f"# 失败记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 输入文件: {INPUT_FILE}\n")
        f.write(f"# 总数: {len(drug_list)}\n\n")

    print("=" * 80)
    print("[INFO] 开始批量处理...")
    print("=" * 80)
    print()

    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建进度条
        with tqdm(total=len(drug_list), desc="处理进度", unit="个") as pbar:
            # 提交所有任务
            futures = {
                executor.submit(process_single_drug, drug_data, client, pbar, OVERWRITE): drug_data
                for drug_data in drug_list
            }

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    success, drug_name, error = future.result()
                except Exception as e:
                    # 这里的异常通常不会被触发，因为已经在 process_single_drug 中捕获
                    drug_name = futures[future].get("名称", "未知")
                    log_failed_drug(drug_name, f"未知错误: {str(e)}")
                    with file_lock:
                        failed_counter += 1

                # 更新进度条
                pbar.update(1)

    # 输出最终统计
    print()
    print("=" * 80)
    print("[INFO] 批量处理完成！")
    print("=" * 80)
    print(f"   [OK] 成功: {success_counter} 个")
    print(f"   [SKIP] 跳过: {skipped_counter} 个（已存在）")
    print(f"   [FAILED] 失败: {failed_counter} 个")
    print()
    print(f"[INFO] 结果保存在: {RESULTS_FOLDER}/")
    print(f"[INFO] 失败日志: {FAILED_LOG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
