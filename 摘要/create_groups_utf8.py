# -*- coding: utf-8 -*-  
import json 
# 读取药品列表  
with open(r'd:\Mysystem\新建文件夹 (3)\摘要\extracted_drugs.txt', 'r', encoding='gbk') as f:  
    drugs = [line.strip() for line in f.readlines()] 
# 分组药品（每组5个）  
groups = [drugs[i:i+5] for i in range(0, len(drugs), 5)] 
# 保存分组信息到JSON文件  
with open(r'd:\Mysystem\新建文件夹 (3)\摘要\drug_groups.json', 'w', encoding='utf-8') as out_f:  
    json.dump(groups, out_f, ensure_ascii=False, indent=2) 
print('分组信息已保存到 drug_groups.json')  
print(f'总共 {len(drugs)} 种药品，分为 {len(groups)} 组') 
