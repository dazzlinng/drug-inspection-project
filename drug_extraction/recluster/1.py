import numpy as np
import json
import os
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 全局配置与中文显示 ==========
# 配置参数（根据硬件调整）
CONFIG = {
    "extracted_results_path": "D:\\drug_extraction\\extracted_results",  # 你的JSON文件路径
    "output_dir": "large_scale_cluster_results",
    "embedding_save_path": "embeddings/large_scale_embeddings.npy",
    "entity_meta_save_path": "embeddings/entity_meta.json",
    "batch_size": 256,  # 嵌入生成的批次大小
    "n_jobs": 8,  # 加载文件的线程数
    "pca_dim": 128,  # PCA降维后的维度（768→128）
    "max_k": 200,  # 最大聚类数（避免遍历过多）
    "min_cluster_size": 5,  # 最小聚类实体数（过滤小聚类）
    "gpu_device": 0,  # GPU设备号（-1为CPU）
}

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0

def get_chinese_font():
    try:
        font_prop = mpl.font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')  # Mac
    except:
        try:
            font_prop = mpl.font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows
        except:
            font_prop = None
    return font_prop

chinese_font = get_chinese_font()

# ========== 2. 大规模数据加载（多线程） ==========
def load_entity_meta(file_path):
    """单文件加载函数（供多线程调用）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_name = data['药品名称']
        valid_entities = []
        
        # 预处理规则（强化版）
        def is_valid_entity(text):
            text = text.strip()
            # 基础过滤
            if not text or len(text) > 80 or text in {"未找到", "无", "空", "N/A"}:
                return False
            # 纯数字/纯符号过滤
            if all(c in '0123456789.+-/%()（）、，：: ' for c in text) and len(text) < 10:
                return False
            # 无意义短文本过滤
            meaningless = {"的", "得", "地", "在", "为", "有", "与", "或", "和", "等", "其", "该", "此"}
            if len(text) == 1 and text not in {"H", "pH", "C", "N", "O", "S", "P", "K", "Na"}:
                return False
            if text in meaningless:
                return False
            return True
        
        # 提取所有实体
        for standard in data.get('检验标准', []):
            # 检测资源（仪器、试剂等）
            for res_type, res_list in standard.get('检测资源', {}).items():
                if isinstance(res_list, str):
                    res_list = [res_list]
                if res_list and res_list != ["未找到"]:
                    for item in res_list:
                        if is_valid_entity(item):
                            valid_entities.append({
                                "text": item.strip(),
                                "entity_type": res_type,
                                "drug_name": drug_name,
                                "test_item": standard.get('检验项目', '未知项目')
                            })
            # 检测步骤（判定指标等）
            for step_type, step_list in standard.get('检测步骤', {}).items():
                if step_type == "前处理过程":
                    continue
                if isinstance(step_list, str):
                    step_list = [step_list]
                if step_list and step_list != ["未找到"]:
                    for item in step_list:
                        if is_valid_entity(item):
                            valid_entities.append({
                                "text": item.strip(),
                                "entity_type": step_type,
                                "drug_name": drug_name,
                                "test_item": standard.get('检验项目', '未知项目')
                            })
        return valid_entities
    except Exception as e:
        print(f"加载文件失败 {file_path}: {str(e)[:50]}")
        return []

def load_all_entities():
    """多线程加载所有2728个文件的实体"""
    # 检查是否已缓存元数据
    if os.path.exists(CONFIG['entity_meta_save_path']):
        print("加载缓存的实体元数据...")
        with open(CONFIG['entity_meta_save_path'], 'r', encoding='utf-8') as f:
            all_entities = json.load(f)
        print(f"缓存加载完成，实体总数: {len(all_entities)}")
        return all_entities
    
    # 遍历所有JSON文件
    json_files = [os.path.join(CONFIG['extracted_results_path'], f) 
                  for f in os.listdir(CONFIG['extracted_results_path']) 
                  if f.endswith('.json')]
    print(f"发现 {len(json_files)} 个JSON文件（目标2728个）")
    
    # 多线程加载
    all_entities = []
    with ThreadPoolExecutor(max_workers=CONFIG['n_jobs']) as executor:
        futures = {executor.submit(load_entity_meta, fp): fp for fp in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="多线程加载实体"):
            all_entities.extend(future.result())
    
    # 全局去重（相同text+entity_type视为同一实体）
    seen = set()
    unique_entities = []
    for ent in all_entities:
        key = (ent['text'], ent['entity_type'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    # 缓存元数据到磁盘
    os.makedirs(os.path.dirname(CONFIG['entity_meta_save_path']), exist_ok=True)
    with open(CONFIG['entity_meta_save_path'], 'w', encoding='utf-8') as f:
        json.dump(unique_entities, f, ensure_ascii=False, indent=2)
    
    print(f"原始实体数: {len(all_entities)}, 去重后: {len(unique_entities)}")
    return unique_entities

# ========== 3. 大规模嵌入生成（GPU+批量） ==========
def generate_large_scale_embeddings(entities):
    """生成嵌入向量（支持GPU、批量、缓存）"""
    # 检查缓存
    if os.path.exists(CONFIG['embedding_save_path']):
        print("加载缓存的嵌入向量...")
        embeddings = np.load(CONFIG['embedding_save_path'])
        return embeddings
    
    # 加载BGE模型（GPU加速）
    print("加载BGE-large-zh-v1.5模型...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    if CONFIG['gpu_device'] >= 0:
        model = model.to(f'cuda:{CONFIG["gpu_device"]}')
    
    # 提取文本列表
    texts = [ent['text'] for ent in entities]
    
    # 批量生成嵌入
    embeddings = []
    for i in tqdm(range(0, len(texts), CONFIG['batch_size']), desc="批量生成嵌入"):
        batch_texts = texts[i:i+CONFIG['batch_size']]
        batch_emb = model.encode(
            batch_texts,
            batch_size=CONFIG['batch_size'],
            show_progress_bar=False,
            normalize_embeddings=True  # 归一化，提升聚类效果
        )
        embeddings.append(batch_emb)
    
    embeddings = np.vstack(embeddings)
    
    # 缓存嵌入向量
    np.save(CONFIG['embedding_save_path'], embeddings)
    print(f"嵌入向量生成完成，形状: {embeddings.shape}")
    
    # 释放内存
    del model
    gc.collect()
    return embeddings

# ========== 4. 大规模聚类核心函数（分层聚类） ==========
def cluster_large_scale_entities(entities, embeddings):
    """分层聚类：先按实体类型分组，再在组内聚类"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 按实体类型分组
    type_groups = defaultdict(list)
    type_embeddings = defaultdict(list)
    for i, ent in enumerate(entities):
        type_groups[ent['entity_type']].append(ent)
        type_embeddings[ent['entity_type']].append(embeddings[i])
    
    # 全局聚类索引表
    global_cluster_index = []
    
    # 遍历每个实体类型进行聚类
    for entity_type, type_ents in tqdm(type_groups.items(), desc="按实体类型聚类"):
        if len(type_ents) < CONFIG['min_cluster_size']:
            print(f"实体类型 {entity_type} 数量过少（{len(type_ents)}），跳过聚类")
            continue
        
        # 提取该类型的嵌入并降维
        type_emb = np.vstack(type_embeddings[entity_type])
        print(f"\n处理实体类型: {entity_type}，实体数: {len(type_ents)}，嵌入维度: {type_emb.shape}")
        
        # PCA降维（减少计算量）
        if type_emb.shape[1] > CONFIG['pca_dim']:
            pca = PCA(n_components=CONFIG['pca_dim'], random_state=42)
            type_emb_pca = pca.fit_transform(type_emb)
            print(f"PCA降维完成: {type_emb.shape} → {type_emb_pca.shape}")
        else:
            type_emb_pca = type_emb
        
        # 快速选择最佳k值（肘部法则）
        def select_best_k(emb, max_k):
            max_k = min(max_k, len(emb)//2)
            if max_k < 2:
                return 1
            inertias = []
            k_candidates = range(2, min(max_k, 50) + 1)  # 限定最多50个候选k值
            for k in k_candidates:
                mbk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
                mbk.fit(emb)
                inertias.append(mbk.inertia_)
            
            # 肘部法则找拐点
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) == 0:
                return 2
            elbow_idx = np.argmax(second_diffs) + 2
            best_k = k_candidates[max(0, elbow_idx-1)]
            return best_k
        
        best_k = select_best_k(type_emb_pca, CONFIG['max_k'])
        print(f"{entity_type} 最佳聚类数: {best_k}")
        
        # 用MiniBatchKMeans聚类（适配大规模数据）
        mbk = MiniBatchKMeans(
            n_clusters=best_k,
            random_state=42,
            batch_size=1024,
            max_iter=500
        )
        labels = mbk.fit_predict(type_emb_pca)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(type_ents[i])
        
        # 过滤小聚类
        filtered_clusters = {k: v for k, v in clusters.items() if len(v) >= CONFIG['min_cluster_size']}
        print(f"过滤小聚类后剩余: {len(filtered_clusters)} 个聚类")
        
        # 生成聚类类名（取聚类内最频繁的实体文本）
        cluster_names = {}
        for cluster_id, cluster_ents in filtered_clusters.items():
            text_counter = Counter([ent['text'] for ent in cluster_ents])
            top_text = text_counter.most_common(1)[0][0]
            cluster_names[cluster_id] = top_text
        
        # 生成该类型的聚类索引
        type_cluster_index = []
        for cluster_id, cluster_ents in filtered_clusters.items():
            # 收集药品列表（去重）
            drugs = list(set([ent['drug_name'] for ent in cluster_ents]))
            type_cluster_index.append({
                '实体类型': entity_type,
                '类名': cluster_names[cluster_id],
                '药品': ', '.join(drugs),
                '实体数量': len(cluster_ents),
                '涉及药品数': len(drugs)
            })
        
        # 保存该类型的聚类结果
        type_df = pd.DataFrame(type_cluster_index)
        type_df = type_df.sort_values('实体数量', ascending=False)
        type_save_path = os.path.join(CONFIG['output_dir'], f"{entity_type}_聚类索引表.csv")
        type_df.to_csv(type_save_path, index=False, encoding='utf-8-sig')
        print(f"{entity_type} 聚类表已保存: {type_save_path}")
        
        # 合并到全局索引表
        global_cluster_index.extend(type_cluster_index)
        
        # 释放内存
        del type_emb, type_emb_pca, mbk, labels
        gc.collect()
    
    # 保存全局聚类索引表
    global_df = pd.DataFrame(global_cluster_index)
    global_df = global_df.sort_values(['实体类型', '实体数量'], ascending=[True, False])
    global_save_path = os.path.join(CONFIG['output_dir'], "全局聚类索引表.csv")
    global_df.to_csv(global_save_path, index=False, encoding='utf-8-sig')
    print(f"\n全局聚类索引表已保存: {global_save_path}")
    
    # 生成统计报告
    stats = {
        "总实体类型数": len(type_groups),
        "总聚类数": len(global_cluster_index),
        "涉及药品总数": len(set([ent['drug_name'] for ent in entities])),
        "平均每个聚类实体数": np.mean([item['实体数量'] for item in global_cluster_index]),
        "平均每个聚类药品数": np.mean([item['涉及药品数'] for item in global_cluster_index])
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(CONFIG['output_dir'], "聚类统计报告.csv"), index=False, encoding='utf-8-sig')
    print("聚类统计报告已保存")
    
    return global_cluster_index

# ========== 5. 主函数 ==========
def main():
    # 1. 加载所有实体（多线程+缓存）
    entities = load_all_entities()
    if len(entities) == 0:
        print("未加载到有效实体，程序退出")
        return
    
    # 2. 生成嵌入向量（GPU+批量+缓存）
    embeddings = generate_large_scale_embeddings(entities)
    
    # 3. 分层聚类（按实体类型+MiniBatchKMeans）
    cluster_index = cluster_large_scale_entities(entities, embeddings)
    
    # 4. 最终统计
    print("\n===== 大规模聚类完成 =====")
    print(f"总有效实体数: {len(entities)}")
    print(f"总聚类数: {len(cluster_index)}")
    print(f"结果保存路径: {CONFIG['output_dir']}")
    print("关键文件：")
    print(f"  - 全局聚类索引表.csv: 所有类型的聚类汇总")
    print(f"  - [实体类型]_聚类索引表.csv: 各类型的详细聚类")
    print(f"  - 聚类统计报告.csv: 整体统计信息")

if __name__ == "__main__":
    main()