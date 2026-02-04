import json  
import codecs 
f=open(r'd:\Mysystem\新建文件夹 (3)\摘要\extracted_drugs.txt', 'r', encoding='gbk')  
drugs = [line.strip() for line in f.readlines()]  
f.close() 
groups = [drugs[i:i+5] for i in range(0, len(drugs), 5)]  
with open(r'd:\Mysystem\新建文件夹 (3)\摘要\drug_groups.json', 'w', encoding='utf-8') as out_f:  
    json.dump(groups, out_f, ensure_ascii=False, indent=2) 
print('分组信息已保存到 drug_groups.json') 
