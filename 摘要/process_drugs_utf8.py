# -*- coding: utf-8 -*-  
import json  
import os 
# 读取药品列表  
def read_drugs():  
    file_path = "d:/Mysystem/新建文件夹 (3)/摘要/extracted_drugs.txt"  
    with open(file_path, 'r', encoding='gbk') as f:  
        drugs = [line.strip() for line in f.readlines()]  
    return drugs 
# 分组药品  
def group_drugs(drugs, group_size=5):  
    groups = []  
    for i in range(0, len(drugs), group_size):  
        groups.append(drugs[i:i+group_size])  
    return groups 
# 保存分组到JSON文件  
def save_groups(groups):  
    output_path = "d:/Mysystem/新建文件夹 (3)/摘要/drug_groups.json"  
    with open(output_path, 'w', encoding='utf-8') as f:  
        json.dump(groups, f, ensure_ascii=False, indent=2)  
    print("分组信息已保存到 {}".format(output_path)) 
def main():  
    # 读取药品  
    drugs = read_drugs()  
    print("总共找到 {} 种药品".format(len(drugs))) 
    # 分组  
    groups = group_drugs(drugs)  
    print("药品已分为 {} 组，每组最多5个药品".format(len(groups))) 
    # 保存分组  
    save_groups(groups)  
    # 显示前几组作为示例  
    print("\n前5组药品:")  
    for i in range(min(5, len(groups))):  
        print("组 {}: {}".format(i+1, ", ".join(groups[i]))) 
if __name__ == "__main__":  
    main() 
