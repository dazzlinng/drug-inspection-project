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
import csv

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
    
    # 定义需要过滤的无意义字符和词
    meaningless_chars = {".", ",", "，", "。", "/", "\\", "(", ")", "（", "）", ":", ":", "、", 
                         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "o", "O", "l",
                         "滴", "g", "m", "L", "ml", "μl", "mg", "g", "kg", "%", "#", "$", "&"}
    
    # 辅助函数：检查实体是否有效
    def is_valid_entity(text):
        text = text.strip()
        
        # 1. 过滤空文本
        if not text:
            return False
            
        # 2. 过滤单字符（除非是特殊化学符号如"H", "Na", "Cl"等）
        if len(text) == 1:
            # 保留一些有意义的单字符化学符号
            meaningful_single_chars = {"H", "K", "Na", "Ca", "Mg", "Fe", "Cu", "Zn", "pH"}
            if text not in meaningful_single_chars:
                return False
        
        # 3. 过滤明显无意义的短文本
        meaningless_texts = {"未", "未找", "未找到", "不", "无", "的", "得", "找", "到", "见", "为", "在", "有", 
                            "与", "或", "和", "等", "其", "该", "此", "些", "上述", "如下"}
        if text in meaningless_texts or text.lower() in meaningless_texts:
            return False
            
        # 4. 过滤纯数字或简单符号组合
        if text.replace('.', '', 1).isdigit() or all(c in meaningless_chars for c in text):
            return False
            
        # 5. 过滤过长文本（可能是提取错误）
        if len(text) > 60:
            return False
            
        # 6. 过滤包含过多特殊字符的文本
        special_char_count = sum(1 for c in text if not c.isalnum() and c not in '()[]{}-_,. ')
        if special_char_count > len(text) * 0.5:  # 超过50%是特殊字符
            return False
            
        return True
    
    # 遍历所有检验标准
    for standard in data['检验标准']:
        # 提取检测资源中的实体
        for resource_type, resource_list in standard['检测资源'].items():
            if resource_list != ["未找到"] and resource_list:
                for item in resource_list:
                    # 添加过滤检查
                    if is_valid_entity(item):
                        entities.append({
                            'text': item,
                            'type': resource_type,
                            'drug_name': drug_name,
                            'test_item': standard['检验项目']
                        })
                    else:
                        # 可选：记录被过滤的实体用于调试
                        # print(f"已过滤无效实体 (检测资源): '{item}'")
                        pass
        
        # 提取检测步骤中的实体
        for step_type, step_list in standard['检测步骤'].items():
            if step_type != "前处理过程" and step_list != ["未找到"] and step_list:
                for item in step_list:
                    # 添加过滤检查
                    if is_valid_entity(item):
                        entities.append({
                            'text': item,
                            'type': step_type,
                            'drug_name': drug_name,
                            'test_item': standard['检验项目']
                        })
                    else:
                        # 可选：记录被过滤的实体用于调试
                        # print(f"已过滤无效实体 (检测步骤): '{item}'")
                        pass
    
    # 额外过滤：移除重复实体（相同text、type、test_item）
    unique_entities = []
    seen = set()
    for entity in entities:
        key = (entity['text'], entity['type'], entity['test_item'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    print(f"原始实体数量: {len(entities) + (len(entities) - len(unique_entities))}")
    print(f"过滤后实体数量: {len(unique_entities)}")
    print(f"已过滤无效/重复实体: {len(entities) + (len(entities) - len(unique_entities)) - len(unique_entities)}")
    
    return unique_entities

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

# ========== 9. 分析聚类结果并保存到文件 ==========
print("\n" + "="*50)
print("聚类结果分析 - 保存到文件")
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

# 创建保存聚类分析的文件
analysis_file = f'cluster_results/aben_dazuo_cluster_analysis_k{k_to_use}.txt'
with open(analysis_file, 'w', encoding='utf-8') as f:
    f.write(f"聚类分析报告 - 阿苯达唑 (k={k_to_use})\n")
    f.write("=" * 60 + "\n")
    f.write(f"总实体数: {len(entities)}\n")
    f.write(f"聚类数量: {k_to_use}\n")
    f.write("=" * 60 + "\n\n")
    
    # 显示每个聚类的详细信息
    for cluster_id in sorted(clusters.keys()):
        cluster_entities = clusters[cluster_id]
        f.write(f"聚类 {cluster_id} (共 {len(cluster_entities)} 个实体):\n")
        f.write("-" * 40 + "\n")
        
        # 统计类型分布
        type_counts = {}
        for entity in cluster_entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        f.write("类型分布:\n")
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(cluster_entities) * 100
            f.write(f"  {entity_type}: {count} 个 ({percentage:.1f}%)\n")
        
        # 查找最频繁出现的实体
        text_counts = {}
        for entity in cluster_entities:
            text = entity['text']
            text_counts[text] = text_counts.get(text, 0) + 1
        
        most_common = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        f.write(f"\n最常见实体 (前5个):\n")
        for i, (text, count) in enumerate(most_common):
            percentage = count / len(cluster_entities) * 100
            f.write(f"  {i+1}. {text} (出现 {count} 次, {percentage:.1f}%)\n")
        
        # 检验项目分布
        test_items = {}
        for entity in cluster_entities:
            item = entity['test_item']
            test_items[item] = test_items.get(item, 0) + 1
        
        f.write(f"\n检验项目分布:\n")
        for item, count in sorted(test_items.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(cluster_entities) * 100
            f.write(f"  {item}: {count} 个 ({percentage:.1f}%)\n")
        
        f.write("\n相关实体列表 (前20个):\n")
        for i, entity in enumerate(cluster_entities[:20]):
            f.write(f"  {i+1}. {entity['text']} ({entity['type']}) - {entity['test_item']}\n")
        if len(cluster_entities) > 20:
            f.write(f"  ... (还有 {len(cluster_entities)-20} 个实体)\n")
        
        f.write("\n" + "=" * 60 + "\n\n")

print(f"聚类结果分析已保存到: {analysis_file}")

# ========== 10. 生成聚类索引表并保存到文件 ==========
print("\n" + "="*50)
print("聚类索引表 - 保存到文件")
print("="*50)

# 保存为文本文件
index_txt_file = f'cluster_results/aben_dazuo_cluster_index_k{k_to_use}.txt'
with open(index_txt_file, 'w', encoding='utf-8') as f:
    f.write("聚类索引表\n")
    f.write("=" * 60 + "\n")
    f.write("类名\t实体数量\t主要类型\t药品\n")
    f.write("-" * 60 + "\n")
    
    # 为每个聚类生成一个代表性的类名
    for cluster_id in sorted(clusters.keys()):
        cluster_entities = clusters[cluster_id]
        
        # 1. 确定类名
        # 统计最常见实体
        text_counts = {}
        for entity in cluster_entities:
            text = entity['text']
            text_counts[text] = text_counts.get(text, 0) + 1
        
        if text_counts:
            # 选择出现频率最高的实体，或者最短的常见实体
            sorted_texts = sorted(text_counts.items(), key=lambda x: (-x[1], len(x[0])))
            # 如果最高频的实体太长，尝试找一个较短的
            class_name = sorted_texts[0][0]
            for text, count in sorted_texts:
                if len(text) <= 20 and count >= max(2, len(cluster_entities) * 0.2):
                    class_name = text
                    break
        else:
            class_name = f"聚类_{cluster_id}"
        
        # 2. 实体数量
        entity_count = len(cluster_entities)
        
        # 3. 主要类型
        type_counts = {}
        for entity in cluster_entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        main_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        main_types_str = ", ".join([f"{t}({c})" for t, c in main_types])
        
        # 4. 药品列表
        drug_list = entities[0]['drug_name']  # 这里只有一个药品
        
        f.write(f"{class_name}\t{entity_count}\t{main_types_str}\t{drug_list}\n")

print(f"聚类索引表(文本格式)已保存到: {index_txt_file}")

# 保存为CSV文件 (更适合后续处理)
index_csv_file = f'cluster_results/aben_dazuo_cluster_index_k{k_to_use}.csv'
with open(index_csv_file, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig 支持Excel打开
    writer = csv.writer(f)
    writer.writerow(['类名', '实体数量', '主要类型', '药品', '聚类ID', '示例实体'])
    
    for cluster_id in sorted(clusters.keys()):
        cluster_entities = clusters[cluster_id]
        
        # 1. 确定类名 (同上)
        text_counts = {}
        for entity in cluster_entities:
            text = entity['text']
            text_counts[text] = text_counts.get(text, 0) + 1
        
        if text_counts:
            sorted_texts = sorted(text_counts.items(), key=lambda x: (-x[1], len(x[0])))
            class_name = sorted_texts[0][0]
            for text, count in sorted_texts:
                if len(text) <= 20 and count >= max(2, len(cluster_entities) * 0.2):
                    class_name = text
                    break
        else:
            class_name = f"聚类_{cluster_id}"
        
        # 2. 实体数量
        entity_count = len(cluster_entities)
        
        # 3. 主要类型
        type_counts = {}
        for entity in cluster_entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        main_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        main_types_str = ", ".join([f"{t}({c})" for t, c in main_types])
        
        # 4. 药品
        drug_name = entities[0]['drug_name']
        
        # 5. 示例实体 (前3个)
        example_entities = "; ".join([entity['text'] for entity in cluster_entities[:3]])
        
        writer.writerow([class_name, entity_count, main_types_str, drug_name, cluster_id, example_entities])

print(f"聚类索引表(CSV格式)已保存到: {index_csv_file}")

# ========== 11. 保存聚类结果 ==========
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
json_file = f'cluster_results/aben_dazuo_clusters_k{k_to_use}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_results, f, ensure_ascii=False, indent=2)

print(f"\n聚类结果已保存到: {json_file}")

print("\n" + "="*60)
print("处理完成！文件汇总:")
print("="*60)
print(f"1. 聚类评估图表: cluster_results/aben_dazuo_cluster_evaluation.png")
print(f"2. 聚类可视化: cluster_results/aben_dazuo_clusters_k{k_to_use}.png")
print(f"3. 聚类结果分析: {analysis_file}")
print(f"4. 聚类索引表(文本): {index_txt_file}")
print(f"5. 聚类索引表(CSV): {index_csv_file}")
print(f"6. 聚类结果数据: {json_file}")
print("="*60)