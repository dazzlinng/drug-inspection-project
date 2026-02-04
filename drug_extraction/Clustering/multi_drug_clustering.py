import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sentence_transformers import SentenceTransformer
import re
import random
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================
# 改进的实体提取函数 - 增加容错性
# ======================

def extract_entities_from_drug(data, drug_name):
    """从单个药品数据中提取实体 - 改进版本"""
    entities = []
    
    # 首先确保数据是字典
    if not isinstance(data, dict):
        print(f"警告: {drug_name} 的数据不是字典格式")
        return entities
    
    # 提取所有可能的实体字段
    entity_fields = [
        '性状', '鉴别', '检查', '含量测定', '类别', '贮藏', '制剂'
    ]
    
    for field in entity_fields:
        if field in data:
            try:
                value = data[field]
                if isinstance(value, dict):
                    # 字典类型的字段（如性状）
                    for key, content in value.items():
                        if content and key != '标准依据' and not key.startswith('reference_'):
                            entities.append({
                                'text': str(content).strip(),
                                'type': field,
                                'subtype': key,
                                'drug_name': drug_name,
                                'source': field
                            })
                elif isinstance(value, list):
                    # 列表类型的字段（如鉴别、检查）
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if v and k != '标准依据' and not k.startswith('reference_'):
                                    entities.append({
                                        'text': str(v).strip(),
                                        'type': field,
                                        'subtype': k,
                                        'drug_name': drug_name,
                                        'source': field
                                    })
                        elif isinstance(item, str):
                            entities.append({
                                'text': str(item).strip(),
                                'type': field,
                                'subtype': '文本',
                                'drug_name': drug_name,
                                'source': field
                            })
                elif isinstance(value, str):
                    # 字符串类型的字段（如类别、贮藏）
                    if value.strip():
                        entities.append({
                            'text': value.strip(),
                            'type': field,
                            'subtype': '文本',
                            'drug_name': drug_name,
                            'source': field
                        })
            except Exception as e:
                print(f"处理字段 '{field}' 时出错: {str(e)}")
                continue
    
    # 如果没有找到任何实体，尝试直接搜索
    if len(entities) == 0:
        print(f"警告: 在 {drug_name} 中未找到实体，尝试直接搜索...")
        # 尝试在数据中搜索可能的实体文本
        for key, value in data.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                text = str(value).strip()
                # 过滤掉太短或只包含数字的文本
                if len(text) > 2 and not re.match(r'^[\d\s\p{P}]+$', text):
                    entities.append({
                        'text': text,
                        'type': '未知',
                        'subtype': key,
                        'drug_name': drug_name,
                        'source': '通用'
                    })
    
    return entities

def debug_json_structure(file_path):
    """调试JSON文件结构"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"\n=== 文件: {file_path} ===")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print("键值对:")
            for key, value in data.items():
                print(f"  {key}: {type(value)} -> {str(value)[:50]}...")
        else:
            print("数据不是字典格式")
            
        return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def load_all_entities(folder_path, selected_drugs=None):
    """加载所有或指定药品的实体"""
    all_entities = []
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个药品JSON文件")
    
    # 如果指定了药品，只加载这些药品
    if selected_drugs:
        # 创建一个药品名称到文件名的映射
        drug_to_file = {}
        for file_name in json_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if '药品名称' in data:
                        drug_to_file[data['药品名称']] = file_name
                    elif 'name' in data:
                        drug_to_file[data['name']] = file_name
                    elif 'drug_name' in data:
                        drug_to_file[data['drug_name']] = file_name
            except:
                continue
        
        # 选择指定的药品文件
        selected_files = []
        for drug in selected_drugs:
            if drug in drug_to_file:
                selected_files.append(drug_to_file[drug])
            else:
                print(f"警告: 未找到药品 '{drug}' 的数据文件")
        
        json_files = selected_files
        print(f"将加载 {len(json_files)} 个指定的药品数据")
    
    # 加载实体
    for file_name in tqdm(json_files, desc="加载药品实体"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding=':utf-8') as f:
                data = json.load(f)
                
                # 查找药品名称
                drug_name = None
                if '药品名称' in data:
                    drug_name = data['药品名称']
                elif 'name' in data:
                    drug_name = data['name']
                elif 'drug_name' in data:
                    drug_name = data['drug_name']
                elif 'title' in data:
                    drug_name = data['title']
                else:
                    # 从文件名提取药品名称
                    base_name = os.path.splitext(file_name)[0]
                    drug_name = base_name.replace('extracted_entities_', '').strip()
                    print(f"警告: 从文件名推断药品名称: {drug_name}")
                
                if not drug_name:
                    print(f"警告: 无法确定药品名称，跳过文件 {file_name}")
                    continue
                
                # 提取实体
                entities = extract_entities_from_drug(data, drug_name)
                all_entities.extend(entities)
                
                # 添加调试信息
                if len(entities) == 0:
                    print(f"警告: 在 {drug_name} 中未找到实体")
                    
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue
    
    # 预处理实体
    processed_entities = preprocess_entities(all_entities)
    
    print(f"共加载 {len(all_entities)} 个原始实体")
    print(f"预处理后剩余 {len(processed_entities)} 个有效实体")
    
    return processed_entities

def preprocess_entities(entities):
    """预处理实体，过滤低质量数据"""
    processed_entities = []
    
    for entity in entities:
        text = entity['text'].strip()
        
        # 过滤太短的实体
        if len(text) < 2:
            continue
        
        # 过滤只包含数字或标点的实体
        if re.match(r'^[\d\s\p{P}]+$', text):
            continue
        
        # 过滤重复的实体
        if any(e['text'] == text and e['drug_name'] == entity['drug_name'] for e in processed_entities):
            continue
        
        # 添加预处理后的实体
        processed_entity = entity.copy()
        processed_entity['text'] = text
        processed_entities.append(processed_entity)
    
    return processed_entities

# ======================
# 其余函数保持不变
# ======================

def determine_optimal_k(embeddings, max_k=20):
    """使用肘部法则和轮廓系数确定最佳聚类数"""
    print("正在确定最佳聚类数量...")
    
    # 采样减少计算量
    sample_size = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[indices]
    
    # 计算不同k值的SSE和轮廓系数
    sse = []
    silhouette_scores = []
    k_values = range(2, min(max_k, sample_size-1))
    
    for k in tqdm(k_values, desc="评估聚类数量"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(sample_embeddings)
        sse.append(kmeans.inertia_)
        
        # 计算轮廓系数
        if k > 1 and k < sample_size:
            labels = kmeans.labels_
            score = silhouette_score(sample_embeddings, labels, sample_size=1000)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, sse, 'b-', marker='o')
    plt.xlabel('聚类数量 k')
    plt.ylabel('SSE (误差平方和)')
    plt.title('肘部法则')
    
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'g-', marker='o')
    plt.xlabel('聚类数量 k')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数分析')
    
    plt.tight_layout()
    plt.savefig('clustering_evaluation.png')
    plt.close()
    
    # 基于轮廓系数选择最佳k (排除前几个k值，因为往往不稳定)
    best_k_idx = np.argmax(silhouette_scores[3:]) + 3
    best_k = k_values[best_k_idx]
    
    print(f"基于轮廓系数分析，最佳聚类数量: k = {best_k}")
    print(f"轮廓系数: {silhouette_scores[best_k_idx]:.4f}")
    
    user_input = input(f"是否接受推荐的聚类数量 {best_k}? (y/n) 或输入自定义数量: ").strip().lower()
    if user_input.isdigit():
        return int(user_input)
    elif user_input.startswith('y') or user_input == '':
        return best_k
    else:
        return min(50, max(10, len(embeddings) // 100))

def compute_embeddings(all_entities, batch_size=128, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """批量计算实体嵌入向量"""
    entity_texts = [entity['text'] for entity in all_entities]
    
    print(f"使用模型 '{model_name}' 计算 {len(entity_texts)} 个实体的嵌入向量")
    
    # 加载模型
    model = SentenceTransformer(model_name)
    model.max_seq_length = 128  # 限制序列长度，加速处理
    
    embeddings = []
    
    # 分批处理，避免内存溢出
    for i in tqdm(range(0, len(entity_texts), batch_size), desc="计算嵌入向量"):
        batch = entity_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    result = np.vstack(embeddings)
    print(f"嵌入向量形状: {result.shape}")
    
    return result

def perform_clustering(embeddings, max_clusters=200):
    """执行聚类，自动确定最佳聚类数量"""
    # 先用小样本估计最佳k值
    sample_size = min(5000, len(embeddings))
    if sample_size < 10:
        print("警告: 实体数量太少，无法进行有意义的聚类")
        return np.zeros(len(embeddings), dtype=int), 1, None
    
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # 肘部法则和轮廓系数确定k
    best_k = determine_optimal_k(sample_embeddings, max_k=min(50, max_clusters))
    
    print(f"在完整数据集上执行聚类 - 聚类数量: {best_k}")
    
    # 在完整数据集上执行聚类
    if len(embeddings) > 10000:
        # 大数据集使用MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=best_k, batch_size=1024, random_state=42, n_init=10)
    else:
        # 较小数据集使用标准KMeans
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 分析聚类分布
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print("聚类分布:")
    for cluster_id, count in sorted(cluster_distribution.items()):
        print(f"  聚类 {cluster_id}: {count} 个实体 ({count/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, best_k, kmeans

def analyze_clusters(all_entities, cluster_labels):
    """分析聚类结果"""
    # 按聚类分组
    clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(all_entities[i])
    
    # 分析每个聚类
    cluster_analysis = []
    for cluster_id, entities in clusters.items():
        # 获取该聚类涉及的药品
        drugs = list(set([entity['drug_name'] for entity in entities]))
        
        # 获取该聚类中的实体类型分布
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # 获取代表性实体
        representative_entities = []
        entity_texts = [entity['text'] for entity in entities]
        
        # 选择出现频率高的实体（去重后）
        from collections import Counter
        counter = Counter(entity_texts)
        top_entities = counter.most_common(5)
        representative_entities = [text for text, count in top_entities]
        
        cluster_analysis.append({
            'cluster_id': cluster_id,
            'entity_count': len(entities),
            'drug_count': len(drugs),
            'drugs': drugs[:10],  # 只显示前10个药品
            'type_distribution': type_counts,
            'representative_entities': representative_entities
        })
    
    return clusters, cluster_analysis

def save_results(all_entities, cluster_labels, cluster_analysis, clusters, output_dir):
    """保存聚类结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 保存完整的聚类结果
    results = []
    for i, entity in enumerate(all_entities):
        entity_result = entity.copy()
        entity_result['cluster_id'] = int(cluster_labels[i])
        results.append(entity_result)
    
    with open(os.path.join(output_dir, 'full_clustering_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 2. 保存聚类摘要
    cluster_summary = []
    for analysis in cluster_analysis:
        cluster_summary.append({
            'cluster_id': analysis['cluster_id'],
            'entity_count': analysis['entity_count'],
            'drug_count': analysis['drug_count'],
            'drugs_sample': analysis['drugs'],
            'type_distribution': analysis['type_distribution'],
            'representative_entities': analysis['representative_entities']
        })
    
    with open(os.path.join(output_dir, 'cluster_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(cluster_summary, f, ensure_ascii=False, indent=2)
    
    # 3. 保存CSV格式的聚类结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'clustering_results.csv'), index=False, encoding='utf_8_sig')
    
    print(f"聚类结果已保存至: {output_dir}")

def get_representative_drugs(all_drugs, num_drugs=30):
    """选择代表性的药品进行试点"""
    # 按药品名称排序
    sorted_drugs = sorted(all_drugs)
    
    # 选择代表性样本 - 确保覆盖不同首字母
    representative_drugs = []
    alphabet_coverage = set()
    
    for drug in sorted_drugs:
        first_char = drug[0].lower()
        if first_char not in alphabet_coverage or len(representative_drugs) < num_drugs // 2:
            representative_drugs.append(drug)
            alphabet_coverage.add(first_char)
            if len(representative_drugs) >= num_drugs:
                break
    
    # 如果还不够，随机选择补充
    if len(representative_drugs) < num_drugs:
        remaining_drugs = [d for d in sorted_drugs if d not in representative_drugs]
        random.shuffle(remaining_drugs)
        representative_drugs.extend(remaining_drugs[:num_drugs - len(representative_drugs)])
    
    return representative_drugs[:num_drugs]

def main():
    """主函数"""
    print("="*50)
    print("多药品实体聚类系统")
    print("="*50)
    
    # 配置参数
    data_folder = "extracted_results"  # 药品JSON文件所在文件夹
    output_base_dir = "clustering_results"
    pilot_drug_count = 30  # 试点药品数量
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    
    # 1. 获取所有药品名称
    print("\n1. 获取所有药品名称...")
    all_drug_names = []
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    for file_name in tqdm(json_files, desc="读取药品名称"):
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 尝试获取药品名称
                drug_name = None
                if '药品名称' in data:
                    drug_name = data['药品名称']
                elif 'name' in data:
                    drug_name = data['name']
                elif 'drug_name' in data:
                    drug_name = data['drug_name']
                elif 'title' in data:
                    drug_name = data['title']
                else:
                    # 从文件名提取
                    base_name = os.path.splitext(file_name)[0]
                    drug_name = base_name.replace('extracted_entities_', '').strip()
                
                if drug_name:
                    all_drug_names.append(drug_name)
        except Exception as e:
            print(f"读取文件 {file_name} 时出错: {str(e)}")
            continue
    
    print(f"共找到 {len(all_drug_names)} 个药品")
    
    # 2. 选择试点药品
    print(f"\n2. 选择 {pilot_drug_count} 个代表性药品进行试点...")
    selected_drugs = get_representative_drugs(all_drug_names, pilot_drug_count)
    print("选择的试点药品:")
    for i, drug in enumerate(selected_drugs, 1):
        print(f"  {i}. {drug}")
    
    user_confirm = input("\n是否确认使用上述药品进行试点聚类? (y/n): ").strip().lower()
    if user_confirm != 'y':
        print("用户取消操作。程序退出。")
        return
    
    # 3. 加载选定药品的实体
    print("\n3. 加载选定药品的实体...")
    entities = load_all_entities(data_folder, selected_drugs)
    
    if len(entities) == 0:
        print("错误: 没有加载到任何实体。请检查数据。")
        return
    
    # 4. 计算嵌入向量
    print("\n4. 计算实体嵌入向量...")
    embeddings = compute_embeddings(entities, batch_size=128)
    
    # 5. 执行聚类
    print("\n5. 执行聚类分析...")
    cluster_labels, best_k, kmeans_model = perform_clustering(embeddings)
    
    # 6. 分析聚类结果
    print("\n6. 分析聚类结果...")
    clusters, cluster_analysis = analyze_clusters(entities, cluster_labels)
    
    # 7. 保存结果
    print("\n8. 保存聚类结果...")
    save_results(entities, cluster_labels, cluster_analysis, clusters, output_dir)
    
    # 9. 显示摘要
    print("\n" + "="*50)
    print("聚类结果摘要")
    print("="*50)
    print(f"总实体数: {len(entities)}")
    print(f"总聚类数: {best_k}")
    print(f"涉及药品数: {len(set([e['drug_name'] for e in entities]))}")
    
    print("\n前10个聚类详情:")
    for i, analysis in enumerate(sorted(cluster_analysis, key=lambda x: x['entity_count'], reverse=True)[:10]):
        print(f"\n聚类 #{analysis['cluster_id']} (实体数: {analysis['entity_count']}, 药品数: {analysis['drug_count']}):")
        print(f"  代表性实体: {', '.join(analysis['representative_entities'][:3])}")
        print(f"  涉及药品: {', '.join(analysis['drugs'][:3])}{'...' if len(analysis['drugs']) > 3 else ''}")
        print(f"  类型分布: {analysis['type_distribution']}")
    
    print(f"\n所有结果已保存至: {output_dir}")
    print("\n下一步建议:")
    print("1. 检查聚类结果和可视化图像")
    print("2. 根据聚类结果生成每个聚类的摘要")
    print("3. 评估试点结果，准备扩展到全部药品")

if __name__ == "__main__":
    main()