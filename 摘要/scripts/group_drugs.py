import json 
# 读取药品列表  
def read_drugs_from_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as f:  
        drugs = [line.strip() for line in f.readlines()]  
    return drugs 
  
# 将药品分组（每组5个）  
def group_drugs(drugs, group_size=5):  
    groups = []  
    for i in range(0, len(drugs), group_size):  
        groups.append(drugs[i:i+group_size])  
    return groups 
  
# 保存分组信息到JSON文件  
def save_groups_to_json(groups, output_file):  
    with open(output_file, 'w', encoding='utf-8') as f:  
        json.dump(groups, f, ensure_ascii=False, indent=2) 
  
# 主函数  
def main():  
    # 读取药品列表  
    drugs_file = r"d:\Mysystem\新建文件夹 (3)\摘要\extracted_drugs.txt"  
    drugs = read_drugs_from_file(drugs_file) 
    print(f"总共找到 {len(drugs)} 种药品")  
    # 分组药品  
    drug_groups = group_drugs(drugs)  
    print(f"药品已分为 {len(drug_groups)} 组，每组最多5个药品") 
    # 显示前几组作为示例  
    print("\n前5组药品:")  
    for i in range(min(5, len(drug_groups))):  
        print(f"组 {i+1}: {', '.join(drug_groups[i])}") 
    # 保存分组信息到文件  
    output_file = r"d:\Mysystem\新建文件夹 (3)\摘要\drug_groups.json"  
    save_groups_to_json(drug_groups, output_file)  
    print(f"\n分组信息已保存到 {output_file}") 
  
if __name__ == "__main__":  
    main() 
