# -*- coding: utf-8 -*- 
import json 
import os 
 
# 创建输出目录 
output_dir = "药品性状" 
os.makedirs(output_dir, exist_ok=True) 
 
# 读取JSON文件 
with open('full_drug_standards.json', 'r', encoding='utf-8') as f: 
    data = json.load(f) 
 
# 抽取指定数量的药品 
extracted_count = 0 
max_count = 20 
 
for drug in data: 
    if extracted_count 
        break 
 
    # 获取药品名称和性状 
    name = drug.get("名称", "未知药品") 
    property_desc = drug.get("性状", "未找到") 
 
    # 创建单个药品的JSON数据 
    drug_property = { 
        "名称": name, 
        "性状": property_desc 
    } 
 
    # 生成安全的文件名（移除可能引起问题的字符） 
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip() 
    # 如果文件名为空，则使用索引 
    if not safe_name: 
        safe_name = f"drug_{extracted_count}" 
 
    # 写入单独的JSON文件 
    output_file = os.path.join(output_dir, f"{safe_name}.json") 
    with open(output_file, 'w', encoding='utf-8') as out_f: 
        json.dump(drug_property, out_f, ensure_ascii=False, indent=2) 
 
    print(f"已处理: {name}") 
    extracted_count += 1 
 
print(f"成功抽取{extracted_count}个药品的性状信息并保存到{output_dir}目录下") 
