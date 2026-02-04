import numpy as np
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 重新加载阿苯达唑的实体（使用之前正确的路径）
folder_path = "extracted_results"
file_name = "extracted_entities_阿苯达唑.json"
file_path = os.path.join(folder_path, file_name)

def load_entities_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    drug_name = data['药品名称']
    entities = []
    
    # 遍历所有检验标准
    for standard in data['检验标准']:
        # 提取检测资源中的实体
        for resource_type, resource_list in standard['检测资源'].items():
            if resource_list != ["未找到"] and resource_list:
                for item in resource_list:
                    entities.append({
                        'text': item,
                        'type': resource_type,
                        'drug_name': drug_name,
                        'test_item': standard['检验项目']
                    })
        
        # 提取检测步骤中的实体
        for step_type, step_list in standard['检测步骤'].items():
            if step_type != "前处理过程" and step_list != ["未找到"] and step_list:
                for item in step_list:
                    entities.append({
                        'text': item,
                        'type': step_type,
                        'drug_name': drug_name,
                        'test_item': standard['检验项目']
                    })
    
    return entities

# 加载实体
entities = load_entities_from_json(file_path)
print(f"阿苯达唑共提取 {len(entities)} 个实体")

# 2. 加载中文向量化模型
print("正在加载中文向量化模型 BAAI/bge-large-zh-v1.5...")
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
print("模型加载完成！")

# 3. 为实体文本生成向量
print("正在为实体生成向量...")
entity_texts = [entity['text'] for entity in entities]
embeddings = model.encode(entity_texts, show_progress_bar=True, batch_size=32)
print(f"生成的向量形状: {embeddings.shape}")

# 4. 保存向量到文件（可选，但对后续处理很有用）
os.makedirs('embeddings', exist_ok=True)
np.save('embeddings/aben_dazuo_entities.npy', embeddings)
print("向量已保存到 embeddings/aben_dazuo_entities.npy")