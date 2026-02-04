# -*- coding: utf-8 -*-  
import json  
import os  
  
def extract_drug_properties(input_file, output_dir, count=20):  
    """ >> extract_properties.py & echo     从full_drug_standards.json中抽取指定数量药品的"性状"要素， >> extract_properties.py & echo     并为每个药品生成一个独立的JSON文件。 >> extract_properties.py & echo. >> extract_properties.py & echo     :param input_file: 输入的JSON文件路径 >> extract_properties.py & echo     :param output_dir: 输出目录路径 >> extract_properties.py & echo     :param count: 要抽取的药品数量，默认为20 >> extract_properties.py & echo     """  
    # 创建输出目录（如果不存在）  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
  
    # 读取JSON文件  
    with open(input_file, 'r', encoding='utf-8') as f:  
        drugs_data = json.load(f)  
  
    # 抽取指定数量的药品  
    extracted_count = 0  
    for drug in drugs_data:  
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
        with open(output_file, 'w', encoding='utf-8') as f:  
            json.dump(drug_property, f, ensure_ascii=False, indent=2)  
  
        extracted_count += 1  
  
    print(f"成功抽取{extracted_count}个药品的性状信息并保存到{output_dir}目录下")  
  
if __name__ == "__main__":  
    input_file = "full_drug_standards.json"  
    output_dir = "药品性状"  
  
    extract_drug_properties(input_file, output_dir, 20) 
