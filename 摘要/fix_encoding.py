# -*- coding: utf-8 -*-  
import json 
f = open("d:/Mysystem/新建文件夹 (3)/摘要/extracted_drugs.txt", "r", encoding="gbk")  
drugs = [line.strip() for line in f.readlines()]  
f.close() 
print("总共找到 {} 种药品".format(len(drugs)))  
groups = [drugs[i:i+5] for i in range(0, len(drugs), 5)] 
with open("d:/Mysystem/新建文件夹 (3)/摘要/drug_groups_fixed.json", "w", encoding="utf-8") as out_f:  
    json.dump(groups, out_f, ensure_ascii=False, indent=2) 
