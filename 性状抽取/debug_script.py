import json 
import os 
os.makedirs('药品性状', exist_ok=True) 
f = open('full_drug_standards.json', 'r', encoding='utf-8') 
data = json.load(f) 
f.close() 
drug = data[0] 
name = drug.get('名称', '未知药品') 
prop = drug.get('性状', '未找到') 
drug_prop = {'名称': name, '性状': prop} 
print(f'第一个药品: {name}') 
output_file = os.path.join('药品性状', 'first_drug.json') 
with open(output_file, 'w', encoding='utf-8') as out_f: 
    json.dump(drug_prop, out_f, ensure_ascii=False, indent=2) 
print('第一个药品文件已保存') 
