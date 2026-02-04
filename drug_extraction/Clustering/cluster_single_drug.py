import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ========== 1. 解决中文显示问题 ==========
# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试找到系统中可用的中文字体
def get_chinese_font():
    """尝试找到系统中可用的中文字体"""
    fontpaths = mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    chinese_fonts = []
    for fontpath in fontpaths:
        # 检查是否包含中文字体（简体中文或繁体中文）
        if 'simhei' in fontpath.lower() or 'simsun' in fontpath.lower() or 'msyh' in fontpath.lower() or 'arialuni' in fontpath.lower():
            chinese_fonts.append(fontpath)
    
    if chinese_fonts:
        font_prop = mpl.font_manager.FontProperties(fname=chinese_fonts[0])
        print(f"找到中文字体: {chinese_fonts[0]}")
        return font_prop
    return None

# 获取中文字体属性
chinese_font = get_chinese_font()

# ========== 2. 重新加载阿苯达唑的实体 ==========
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

print("正在加载实体数据...")
# 加载实体
entities = load_entities_from_json(file_path)
entity_texts = [entity['text'] for entity in entities]
print(f"阿苯达唑共提取 {len(entities)} 个实体")

# ========== 3. 加载之前保存的向量 ==========
embeddings_path = 'embeddings/aben_dazuo_entities.npy'
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
    print(f"成功加载保存的向量，形状: {embeddings.shape}")
else:
    print(f"错误：找不到向量文件 {embeddings_path}")
    exit()

# ========== 4. 确定最佳聚类数量 - 使用肘部法则和轮廓系数 ==========
print("\n" + "="*50)
print("正在确定最佳聚类数量...")
print("="*50)

# 定义k的范围
k_range = range(2, 16)  # 尝试2到15个聚类
inertia_values = []     # 存储每个k的inertia值（用于肘部法则）
silhouette_scores = []  # 存储每个k的轮廓系数

for k in tqdm(k_range, desc="计算不同k值的聚类指标"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    # 计算inertia（簇内平方和）
    inertia_values.append(kmeans.inertia_)
    
    # 计算轮廓系数
    score = silhouette_score(embeddings, kmeans.labels_)
    silhouette_scores.append(score)

# ========== 5. 可视化肘部法则和轮廓系数 ==========
plt.figure(figsize=(15, 6))

# 肘部法则图
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, 'bo-')
plt.xlabel('聚类数量 (k)', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('Inertia (簇内平方和)', fontproperties=chinese_font if chinese_font else None)
plt.title('肘部法则 (Elbow Method)', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)

# 轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数量 (k)', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('轮廓系数', fontproperties=chinese_font if chinese_font else None)
plt.title('轮廓系数 (Silhouette Score)', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)

plt.tight_layout()
os.makedirs('cluster_results', exist_ok=True)
plt.savefig('cluster_results/aben_dazuo_cluster_evaluation.png', bbox_inches='tight', dpi=300)
print("聚类评估图表已保存到: cluster_results/aben_dazuo_cluster_evaluation.png")
plt.close()  # 关闭图表，避免内存泄漏

# 显示图表
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, 'bo-')
plt.xlabel('聚类数量 (k)', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('Inertia (簇内平方和)', fontproperties=chinese_font if chinese_font else None)
plt.title('肘部法则 (Elbow Method)', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数量 (k)', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('轮廓系数', fontproperties=chinese_font if chinese_font else None)
plt.title('轮廓系数 (Silhouette Score)', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 6. 选择最佳k值 ==========
print("\n聚类评估结果:")
for i, k in enumerate(k_range):
    print(f"k={k}: Inertia={inertia_values[i]:.2f}, Silhouette Score={silhouette_scores[i]:.4f}")

# 改进：使用肘部法则的"拐点"和轮廓系数综合判断
# 寻找肘部法则的拐点（通过计算二阶导数）
inertia_changes = np.diff(inertia_values)
elbow_suggestions = []
for i in range(1, len(inertia_changes)-1):
    if inertia_changes[i] < inertia_changes[i-1] * 0.6:  # 如果下降幅度减少60%以上
        elbow_suggestions.append(k_range[i+1])

# 基于轮廓系数的最佳k
best_k_by_silhouette = k_range[np.argmax(silhouette_scores)]

print(f"\n肘部法则建议的k值: {elbow_suggestions}")
print(f"轮廓系数建议的最佳k值: {best_k_by_silhouette}")

# 自动选择：优先考虑轮廓系数最高的，但不超过肘部法则建议的范围
if elbow_suggestions:
    # 选择轮廓系数最高的k，但不超过肘部法则建议的最大值
    max_elbow_k = max(elbow_suggestions)
    valid_indices = [i for i, k in enumerate(k_range) if k <= max_elbow_k]
    if valid_indices:
        best_k = k_range[valid_indices[np.argmax([silhouette_scores[i] for i in valid_indices])]]
    else:
        best_k = best_k_by_silhouette
else:
    best_k = best_k_by_silhouette

print(f"综合建议的最佳聚类数量 k = {best_k}")

# 交互式选择k值（添加输入验证）
while True:
    user_input = input(f"\n接受建议的k值 {best_k} 吗？(y/n) 或输入自定义k值 (2-{max(k_range)}): ").strip().lower()
    
    if user_input == 'y' or user_input == '':
        k_to_use = best_k
        break
    elif user_input == 'n':
        try:
            custom_k = int(input(f"请输入自定义的k值 (2-{max(k_range)}): "))
            if 2 <= custom_k <= max(k_range):
                k_to_use = custom_k
                break
            else:
                print(f"错误：k值必须在2到{max(k_range)}之间")
        except ValueError:
            print("错误：请输入有效的整数")
    else:
        try:
            custom_k = int(user_input)
            if 2 <= custom_k <= max(k_range):
                k_to_use = custom_k
                break
            else:
                print(f"错误：k值必须在2到{max(k_range)}之间")
        except ValueError:
            print("错误：无效输入，请输入'y'、'n'或有效的整数")

print(f"将使用 k = {k_to_use} 进行聚类...")

# ========== 7. 执行k-means聚类 ==========
print("\n" + "="*50)
print(f"使用 k = {k_to_use} 执行k-means聚类...")
print("="*50)

kmeans = KMeans(n_clusters=k_to_use, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# ========== 8. 降维可视化（PCA降到2D） ==========
print("正在降维以可视化聚类结果...")
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

# 创建更清晰的可视化
plt.figure(figsize=(12, 8))

# 使用更好的颜色映射
colors = plt.cm.tab20(np.linspace(0, 1, k_to_use))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=cluster_labels, cmap='tab20', 
                     alpha=0.8, s=80, edgecolors='w', linewidth=0.5)

# 添加聚类中心
centers = kmeans.cluster_centers_
centers_2d = pca.transform(centers)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
           c='black', marker='X', s=200, linewidths=2, label='聚类中心')

# 为每个聚类添加标签
for i in range(k_to_use):
    # 找到该聚类中最近的点
    cluster_points = embeddings_2d[cluster_labels == i]
    if len(cluster_points) > 0:
        centroid = np.mean(cluster_points, axis=0)
        plt.text(centroid[0], centroid[1], f'聚类 {i}', 
                fontsize=10, fontproperties=chinese_font if chinese_font else None,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

plt.colorbar(scatter, label='聚类ID')
plt.title(f'阿苯达唑实体聚类可视化 (k={k_to_use})', fontproperties=chinese_font if chinese_font else None)
plt.xlabel('PCA Component 1', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('PCA Component 2', fontproperties=chinese_font if chinese_font else None)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()

# 保存高清图像
plt.savefig(f'cluster_results/aben_dazuo_clusters_k{k_to_use}.png', dpi=300, bbox_inches='tight')
print(f"聚类可视化已保存到: cluster_results/aben_dazuo_clusters_k{k_to_use}.png")
plt.show()

# ========== 9. 分析聚类结果 ==========
print("\n" + "="*50)
print("聚类结果分析")
print("="*50)

# 按聚类分组实体
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append({
        'text': entity_texts[i],
        'type': entities[i]['type'],
        'drug_name': entities[i]['drug_name'],
        'test_item': entities[i]['test_item']
    })

# 显示每个聚类的详细信息
for cluster_id in sorted(clusters.keys()):
    print(f"\n聚类 {cluster_id} (共 {len(clusters[cluster_id])} 个实体):")
    print("-" * 40)
    
    # 统计类型分布
    type_counts = {}
    for entity in clusters[cluster_id]:
        entity_type = entity['type']
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    print("类型分布:")
    for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count} 个")
    
    # 查找最频繁出现的实体
    text_counts = {}
    for entity in clusters[cluster_id]:
        text = entity['text']
        text_counts[text] = text_counts.get(text, 0) + 1
    
    most_common = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n最常见实体 ({len(most_common)} 个):")
    for i, (text, count) in enumerate(most_common):
        print(f"  {i+1}. {text} (出现 {count} 次)")
    
    # 检验项目分布
    test_items = set(entity['test_item'] for entity in clusters[cluster_id])
    print(f"\n涉及的检验项目 ({len(test_items)} 个):")
    for i, item in enumerate(list(test_items)[:5]):  # 只显示前5个
        print(f"  {item}")
    if len(test_items) > 5:
        print(f"  ... (还有 {len(test_items)-5} 个其他检验项目)")

# ========== 10. 保存聚类结果 ==========
cluster_results = {
    'drug_name': entities[0]['drug_name'],
    'total_entities': len(entities),
    'best_k': k_to_use,
    'clusters': {}
}

for cluster_id, entities_in_cluster in clusters.items():
    cluster_results['clusters'][int(cluster_id)] = [
        {
            'text': entity['text'],
            'type': entity['type'],
            'test_item': entity['test_item']
        } for entity in entities_in_cluster
    ]

# 保存为JSON文件
with open(f'cluster_results/aben_dazuo_clusters_k{k_to_use}.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_results, f, ensure_ascii=False, indent=2)

print(f"\n聚类结果已保存到: cluster_results/aben_dazuo_clusters_k{k_to_use}.json")

# ========== 11. 生成聚类索引表示例 ==========
print("\n" + "="*50)
print("聚类索引表示例")
print("="*50)

print("\n类名\t药品")
print("-" * 40)

# 为每个聚类生成一个代表性的类名
for cluster_id in sorted(clusters.keys()):
    # 取该聚类中出现频率最高的实体作为类名
    cluster_entities = clusters[cluster_id]
    text_counts = {}
    for entity in cluster_entities:
        text = entity['text']
        text_counts[text] = text_counts.get(text, 0) + 1
    
    if text_counts:
        # 选择出现频率最高的实体，或者最短的常见实体
        sorted_texts = sorted(text_counts.items(), key=lambda x: (-x[1], len(x[0])))
        class_name = sorted_texts[0][0]
    else:
        class_name = f"聚类_{cluster_id}"
    
    # 如果类名太长，截断
    if len(class_name) > 20:
        class_name = class_name[:20] + "..."
    
    # 药品列表
    drug_list = ", ".join(set([entity['drug_name'] for entity in cluster_entities]))
    
    print(f"{class_name}\t{drug_list}")

print("\n注意：这只是初步的聚类索引表示例。在完整数据集中，每个类名会包含多个药品。")