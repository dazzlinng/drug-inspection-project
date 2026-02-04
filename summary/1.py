import json
import pandas as pd
from openai import OpenAI
import time
import re

# 初始化OpenAI客户端
client = OpenAI(
  
)

def read_files(json_path, csv_path):
    """读取JSON和CSV文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        drug_standards = json.load(f)
    
    cluster_df = pd.read_csv(csv_path)
    
    # 确保列名正确
    if '类名' not in cluster_df.columns:
        raise ValueError("CSV文件中缺少'类名'列")
    if '药品' not in cluster_df.columns:
        raise ValueError("CSV文件中缺少'药品'列")
    
    return drug_standards, cluster_df

def extract_check_info_for_drug(drug_data):
    """提取药品的检查信息"""
    check_info = ""
    
    # 处理不同格式的药品数据
    if isinstance(drug_data, dict) and "检验标准" in drug_data:
        for item in drug_data["检验标准"]:
            if item.get("检验项目") == "检查":
                # 提取检测资源信息
                resources = item.get("检测资源", {})
                instruments = resources.get("仪器设备", [])
                reagents = resources.get("试剂/溶液", [])
                columns = resources.get("色谱柱/填充剂", [])
                
                # 提取检测步骤信息
                steps = item.get("检测步骤", {})
                operations = steps.get("具体测定操作", [])
                
                # 构建检查信息字符串
                info_parts = []
                
                if instruments:
                    info_parts.append(f"仪器设备: {', '.join([i for i in instruments if i != '未找到'])}")
                
                if reagents:
                    info_parts.append(f"试剂/溶液: {', '.join([r for r in reagents if r != '未找到'])}")
                
                if columns and any(c != '未找到' for c in columns):
                    info_parts.append(f"色谱柱/填充剂: {', '.join([c for c in columns if c != '未找到'])}")
                
                if operations:
                    info_parts.append(f"检测操作: {'; '.join([op for op in operations if op != '未找到'])}")
                
                check_info = "; ".join(info_parts)
                break  # 只取第一个"检查"项目
    
    elif isinstance(drug_data, list):
        # 如果是旧格式的列表
        for item in drug_data:
            if item.get("检验项目") == "检查":
                resources = item.get("检测资源", {})
                instruments = resources.get("仪器设备", [])
                reagents = resources.get("试剂/溶液", [])
                columns = resources.get("色谱柱/填充剂", [])
                
                steps = item.get("检测步骤", {})
                operations = steps.get("具体测定操作", [])
                
                info_parts = []
                
                if instruments:
                    info_parts.append(f"仪器设备: {', '.join([i for i in instruments if i != '未找到'])}")
                
                if reagents:
                    info_parts.append(f"试剂/溶液: {', '.join([r for r in reagents if r != '未找到'])}")
                
                if columns and any(c != '未找到' for c in columns):
                    info_parts.append(f"色谱柱/填充剂: {', '.join([c for c in columns if c != '未找到'])}")
                
                if operations:
                    info_parts.append(f"检测操作: {'; '.join([op for op in operations if op != '未找到'])}")
                
                check_info = "; ".join(info_parts)
                break
    
    return check_info

def get_drug_check_info(drug_standards, drug_list):
    """获取药品的检查信息"""
    check_info = {}
    for drug in drug_list:
        if drug in drug_standards:
            # 提取检查部分的信息
            drug_data = drug_standards[drug]
            check_detail = extract_check_info_for_drug(drug_data)
            if check_detail:
                check_info[drug] = check_detail
    return check_info

def generate_summary_for_batch(cluster_name, batch_drugs, drug_standards):
    """为一批药品生成摘要"""
    # 获取这批药品的检查信息
    check_info = get_drug_check_info(drug_standards, batch_drugs)
    
    if not check_info:
        return ""
    
    # 构建输入文本，限制总长度
    input_text = f"聚类名称: {cluster_name}\n\n"
    input_text += f"药品列表: {', '.join(batch_drugs)}\n\n"
    input_text += "各药品检查信息:\n"
    
    total_length = len(input_text)
    for drug, info in check_info.items():
        # 确保总长度不超过1000字符
        if total_length + len(info) > 1000:
            remaining_chars = 1000 - total_length
            if remaining_chars > 0:
                input_text += f"{drug}: {info[:remaining_chars]}\n"
            break
        else:
            input_text += f"{drug}: {info}\n"
            total_length += len(info)
    
    # 调用API生成摘要
    prompt = f"""
请根据以下信息生成专业的药品检验摘要:

输入内容:
{input_text}

摘要生成要求:
1. 摘要结构：
   - 开头必须包含"类名与药品列表摘要"：药品检查中需要用到[聚类名称]的药品有：[药品列表]。
   - 后续3-5段专业总结性摘要，客观描述共性与差异

2. 内容要求：
   - 专业、客观、总结性强
   - 3-5段，包含共性分析、关键差异点及专业意义
   - 严格避免主观表述（如"我发现"、"有趣的是"）
   - 重点分析：检测方法共性、参数差异、药品特性与检测条件的关系

3. 输出格式：
   - [聚类名称]的药品检验摘要
   - 药品检查中需要用到[聚类名称]的药品有：[药品列表]。
   - [专业总结性摘要，3-5段，客观描述共性与差异]

请严格按照以上格式输出。
"""
    
    try:
        response = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "system", "content": "您是一位专业的药品检验专家，负责分析药品检验标准并生成专业摘要。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

def merge_cluster_summaries(cluster_name, cluster_drugs, batch_summaries):
    """合并聚类的所有批次摘要"""
    if not batch_summaries:
        return f"无法为聚类'{cluster_name}'生成摘要"
    
    all_drugs = ', '.join(cluster_drugs)
    
    # 如果只有一个摘要，直接返回
    if len(batch_summaries) == 1:
        return batch_summaries[0]
    
    # 否则整合多个摘要
    combined_content = "\n---分隔线---\n".join(batch_summaries)
    
    combined_prompt = f"""
以下是关于"{cluster_name}"聚类的多个药品检验摘要片段，请整合成一个完整的摘要：

{combined_content}

请按照以下格式输出完整摘要：

{cluster_name}的药品检验摘要

药品检查中需要用到{cluster_name}的药品有：{all_drugs}。

[整合后的专业总结性摘要，3-5段，客观描述共性与差异，涵盖所有批次的关键信息]

要求：
1. 保持专业、客观的表述风格
2. 整合所有关键信息点
3. 突出共性分析、关键差异点及专业意义
4. 避免主观表述
5. 确保内容连贯性和逻辑性
"""
    
    try:
        response = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "system", "content": "您是一位专业的药品检验专家，负责整合多份摘要为完整报告。"},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.1,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"整合摘要失败: {e}")
        # 如果整合失败，拼接原始摘要
        result = f"{cluster_name}的药品检验摘要\n\n"
        result += f"药品检查中需要用到{cluster_name}的药品有：{all_drugs}。\n\n"
        result += combined_content
        return result

def process_clusters(json_path, csv_path):
    """处理所有聚类"""
    drug_standards, cluster_df = read_files(json_path, csv_path)
    
    results = {}
    
    # 按类名分组
    for cluster_name in cluster_df['类名'].unique():
        print(f"正在处理聚类: {cluster_name}")
        
        # 获取该类的所有药品
        cluster_drugs = []
        for drugs_str in cluster_df[cluster_df['类名'] == cluster_name]['药品']:
            if pd.isna(drugs_str):
                continue
            # 将逗号分隔的药品列表拆分成单独的药品
            drugs_list = [drug.strip() for drug in str(drugs_str).split(',') if drug.strip()]
            cluster_drugs.extend(drugs_list)
        
        # 去除重复药品
        cluster_drugs = list(set(cluster_drugs))
        
        # 按5个一批拆分
        batches = [cluster_drugs[i:i+5] for i in range(0, len(cluster_drugs), 5)]
        
        cluster_summaries = []
        
        # 处理每个批次
        for i, batch in enumerate(batches):
            print(f"  批次 {i+1}/{len(batches)}, 药品数量: {len(batch)}")
            
            batch_summary = generate_summary_for_batch(cluster_name, batch, drug_standards)
            
            if batch_summary:
                cluster_summaries.append(batch_summary)
            else:
                # 如果某个批次生成失败，添加占位符
                cluster_summaries.append(f"批次{batch}摘要生成失败")
            
            # 添加延时避免API频率限制
            time.sleep(1)
        
        # 合并所有小部分摘要为完整摘要
        final_summary = merge_cluster_summaries(cluster_name, cluster_drugs, cluster_summaries)
        results[cluster_name] = final_summary
        
        print(f"  完成聚类: {cluster_name}")
    
    return results

def save_results(results, output_file):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for cluster_name, summary in results.items():
            f.write(f"{'='*60}\n")
            f.write(f"聚类: {cluster_name}\n")
            f.write(f"{'='*60}\n")
            f.write(summary)
            f.write(f"\n\n")

def main():
    """主函数"""
    json_path = "full_drug_standards.json"
    csv_path = "全局聚类索引表.csv"
    output_file = "聚类摘要结果.txt"
    
    print("开始处理药品检验标准聚类摘要...")
    print(f"读取JSON文件: {json_path}")
    print(f"读取CSV文件: {csv_path}")
    
    try:
        results = process_clusters(json_path, csv_path)
        
        save_results(results, output_file)
        
        print(f"处理完成！结果已保存至 {output_file}")
        
        # 打印统计信息
        print(f"总共处理了 {len(results)} 个聚类")
        for cluster_name in results.keys():
            print(f"- {cluster_name}")
            
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在于当前目录:")
        print("- full_drug_standards.json")
        print("- 全局聚类索引表.csv")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()