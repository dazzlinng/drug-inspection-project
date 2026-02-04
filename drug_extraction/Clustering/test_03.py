import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import jieba
from tqdm import tqdm
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import sys
from typing import List, Dict, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clustering_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DrugEntityClustering")

class DrugEntityClusterAnalyzer:
    """药品实体聚类分析系统"""
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化聚类分析器
        
        参数:
            model_name: 用于生成文本嵌入的模型名称
        """
        self.model_name = model_name
        self.embedding_model = None
        self.stopwords = self._load_stopwords()
        self.chemical_symbol_fix = {
            "C": "碳元素含量", "H": "氢元素含量", "N": "氮元素含量", 
            "O": "氧元素含量", "S": "硫元素含量", "P": "磷元素含量",
            "Cl": "氯元素含量", "F": "氟元素含量", "Br": "溴元素含量"
        }
        self.reagent_standardization = self._get_reagent_standardization_rules()
        
        # 初始化嵌入模型
        self._init_embedding_model()
        
    def _load_stopwords(self):
        """加载停用词"""
        stopwords = set()
        try:
            # 常见中英文停用词
            common_stopwords = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '在', '上', '下', '中',
                              'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']
            stopwords.update(common_stopwords)
        except:
            pass
        return stopwords
    
    def _get_reagent_standardization_rules(self):
        """获取试剂标准化规则"""
        return [
            (r"稀\s*盐酸", "稀盐酸"),
            (r"浓\s*盐酸", "浓盐酸"),
            (r"稀\s*硫酸", "稀硫酸"),
            (r"稀\s*硝酸", "稀硝酸"),
            (r"冰\s*醋酸", "冰醋酸"),
            (r"标准\s*铁\s*溶液", "标准铁溶液"),
            (r"标准\s*铅\s*溶液", "标准铅溶液"),
            (r"标准\s*砷\s*溶液", "标准砷溶液"),
            (r"高氯酸\s*滴定液\s*\(0\.1\s*mol/L\)", "高氯酸滴定液(0.1mol/L)"),
            (r"高氯酸\s*滴定液\s*\(0\.05\s*mol/L\)", "高氯酸滴定液(0.05mol/L)"),
            (r"氢氧化钠\s*滴定液\s*\(0\.1\s*mol/L\)", "氢氧化钠滴定液(0.1mol/L)"),
            (r"盐酸\s*滴定液\s*\(0\.1\s*mol/L\)", "盐酸滴定液(0.1mol/L)"),
            (r"三氯甲烷\s*-\s*冰醋酸\s*\(9\s*:\s*1\)", "三氯甲烷-冰醋酸(9:1)"),
            (r"三氯甲烷\s*-\s*乙醚\s*-\s*冰醋酸\s*\(30\s*:\s*7\s*:\s*3\)", "三氯甲烷-乙醚-冰醋酸(30:7:3)"),
            (r"供试品\s*溶液\s*\(每1ml中约含\d+mg\)", "供试品溶液"),
            (r"对照\s*溶液\s*\(\d\)\s*\(每1ml中约含\d+μg\)", "对照溶液"),
            (r"对照\s*溶液\(A\)", "对照溶液A"),
            (r"对照\s*溶液\(B\)", "对照溶液B"),
            (r"乙醇制\s*氢氧化钾\s*试液", "乙醇制氢氧化钾试液"),
            (r"三硝基苯酚\s*试液", "三硝基苯酚试液"),
            (r"硫酸\s*肼\s*溶液", "硫酸肼溶液"),
            (r"无水\s*乙醇", "无水乙醇"),
            (r"蒸馏\s*水", "纯化水"),
            (r"纯\s*化\s*水", "纯化水"),
            (r"超\s*纯\s*水", "纯化水"),
        ]
    
    def _init_embedding_model(self):
        """初始化文本嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("嵌入模型加载成功")
        except ImportError:
            logger.error("未安装sentence-transformers库，请先安装: pip install sentence-transformers")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入向量"""
        if not text or not isinstance(text, str):
            return np.zeros(384)  # MiniLM的维度
            
        # 清理文本
        text = re.sub(r'\s+', ' ', text.strip())
        
        try:
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入时出错: {e}")
            return np.zeros(384)
    
    def load_entities_from_json(self, file_path: str, drug_name: str) -> List[Dict]:
        """
        从JSON文件加载实体
        
        参数:
            file_path: JSON文件路径
            drug_name: 药品名称
            
        返回:
            实体列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保数据格式正确
            if not isinstance(data, dict) or 'entities' not in data:
                logger.error(f"JSON格式错误，缺少'entities'键: {file_path}")
                return []
                
            entities = data['entities']
            logger.info(f"成功加载 {len(entities)} 个原始实体")
            
            # 清理和验证实体
            valid_entities = []
            for entity in entities:
                try:
                    # 确保必要字段存在
                    required_fields = ['text', 'test_item', 'type']
                    if not all(field in entity for field in required_fields):
                        continue
                        
                    # 基本验证
                    if not entity['text'].strip() or len(entity['text']) < 2:
                        continue
                        
                    valid_entities.append(entity)
                except Exception as e:
                    continue
            
            logger.info(f"验证后有效实体数量: {len(valid_entities)}")
            return valid_entities
            
        except Exception as e:
            logger.error(f"加载文件时出错 {file_path}: {e}")
            return []
    
    def preprocess_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        预处理实体，解决边界实体和标准化问题
        
        参数:
            entities: 原始实体列表
            
        返回:
            预处理后的实体列表
        """
        logger.info("开始预处理实体...")
        processed_entities = []
        chemical_fix_applied = 0
        reagent_standardized = 0
        
        for entity in entities:
            text = entity['text'].strip()
            original_text = text
            
            # 1. 修复分离的化学符号
            if text in self.chemical_symbol_fix and entity['type'] == '限度判定指标':
                text = self.chemical_symbol_fix[text]
                chemical_fix_applied += 1
                
            # 2. 标准化试剂命名
            for pattern, replacement in self.reagent_standardization:
                new_text = re.sub(pattern, replacement, text)
                if new_text != text:
                    text = new_text
                    reagent_standardized += 1
            
            # 3. 处理特殊边界情况
            # 合并孤立的化学式部分
            if re.match(r'^[A-Z][a-z]?$', text) and entity['type'] in ['限度判定指标', '计算公式']:
                # 尝试在相同检验项目中找到包含该化学符号的上下文
                related_entities = [
                    e for e in entities 
                    if e['test_item'] == entity['test_item'] 
                    and text in e['text']
                    and len(e['text']) > 5
                ]
                if related_entities:
                    # 使用最相关的上下文实体补充
                    context_text = related_entities[0]['text']
                    if len(context_text) > 20:
                        context_text = context_text[:20] + "..."
                    text = f"{text} ({context_text})"
            
            # 4. 过滤低质量实体
            if len(text) < 2 or text.lower() in self.stopwords or re.match(r'^[\d\s\W]+$', text):
                continue
            
            # 5. 处理过度截断的文本
            if text.endswith('...') and len(text) < 10:
                # 查找完整的原文
                full_texts = [e['text'] for e in entities if text[:-3] in e['text'] and len(e['text']) > len(text)]
                if full_texts:
                    text = sorted(full_texts, key=len)[0][:30]  # 取最短且包含该片段的文本
            
            processed_entity = entity.copy()
            processed_entity['text'] = text
            processed_entity['original_text'] = original_text
            
            processed_entities.append(processed_entity)
        
        logger.info(f"预处理完成。原始实体: {len(entities)}, 处理后实体: {len(processed_entities)}")
        logger.info(f"化学符号修复: {chemical_fix_applied} 处，试剂标准化: {reagent_standardized} 处")
        
        return processed_entities
    
    def build_entity_context(self, entities: List[Dict]) -> Dict[int, str]:
        """
        构建实体上下文特征，捕获同一检验步骤中的共现实体
        
        参数:
            entities: 预处理后的实体列表
            
        返回:
            实体ID到上下文文本的映射
        """
        logger.info("构建实体上下文...")
        context_map = {}
        
        # 按检验项目和类型分组
        grouped_by_test_item = defaultdict(list)
        grouped_by_type = defaultdict(list)
        
        for i, entity in enumerate(entities):
            grouped_by_test_item[entity['test_item']].append((i, entity))
            grouped_by_type[entity['type']].append((i, entity))
        
        # 为每个实体构建上下文
        for i, entity in enumerate(entities):
            context_parts = []
            
            # 1. 同一检验项目中的其他实体（类型多样）
            same_test_items = grouped_by_test_item[entity['test_item']]
            other_items = [e for idx, e in same_test_items if idx != i and e['type'] != entity['type']]
            
            # 选择最多3个其他类型的实体
            selected = other_items[:3]
            for _, e in selected:
                context_parts.append(f"{e['type']}: {e['text']}")
            
            # 2. 同一类型中的相似实体（同一检验项目）
            same_type_items = [e for idx, e in grouped_by_type[entity['type']] 
                             if idx != i and e['test_item'] == entity['test_item']]
            
            # 选择最多2个同类型实体
            selected = same_type_items[:2]
            for _, e in selected:
                context_parts.append(f"相似{entity['type']}: {e['text']}")
            
            # 3. 构建上下文文本
            context_text = "； ".join(context_parts[:5])  # 限制上下文长度
            context_map[i] = context_text
        
        logger.info("实体上下文构建完成")
        return context_map
    
    def create_enhanced_features(self, entities: List[Dict], context_map: Dict[int, str] = None) -> Tuple[np.ndarray, Dict]:
        """
        创建增强的多模态特征向量
        
        参数:
            entities: 预处理后的实体列表
            context_map: 实体上下文映射
            
        返回:
            增强的特征矩阵和特征信息
        """
        logger.info("创建增强特征...")
        n_samples = len(entities)
        
        # 1. 生成文本嵌入
        logger.info("  生成文本嵌入...")
        texts = [entity['text'] for entity in entities]
        text_embeddings = np.array([self.get_embedding(text) for text in tqdm(texts, desc="生成嵌入")])
        
        # 2. 生成上下文嵌入（如果有）
        context_embeddings = np.zeros_like(text_embeddings)
        if context_map:
            logger.info("  生成上下文嵌入...")
            context_texts = [context_map.get(i, "") for i in range(n_samples)]
            non_empty_contexts = [(i, text) for i, text in enumerate(context_texts) if text.strip()]
            
            if non_empty_contexts:
                context_indices, context_strings = zip(*non_empty_contexts)
                context_vecs = np.array([self.get_embedding(text) for text in tqdm(context_strings, desc="上下文嵌入")])
                for idx, vec in zip(context_indices, context_vecs):
                    context_embeddings[idx] = vec
        
        # 3. 创建实体类型和检验项目的one-hot编码
        logger.info("  创建类型和检验项目特征...")
        types = [[entity['type']] for entity in entities]
        test_items = [[entity['test_item']] for entity in entities]
        
        type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        type_features = type_encoder.fit_transform(types)
        
        test_item_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        test_item_features = test_item_encoder.fit_transform(test_items)
        
        # 4. 特征融合 - 加权组合
        logger.info("  融合特征...")
        # 调整one-hot特征维度以匹配嵌入维度
        def expand_features(features, target_dim):
            n_samples, n_features = features.shape
            if n_features >= target_dim:
                return features[:, :target_dim]
            else:
                expanded = np.zeros((n_samples, target_dim))
                expanded[:, :n_features] = features
                return expanded
        
        text_dim = text_embeddings.shape[1]
        expanded_type_features = expand_features(type_features, text_dim)
        expanded_test_item_features = expand_features(test_item_features, text_dim)
        expanded_context_features = context_embeddings  # 已经是正确维度
        
        # 加权融合
        combined_features = (
            0.55 * text_embeddings + 
            0.15 * expanded_context_features +
            0.15 * expanded_type_features +
            0.15 * expanded_test_item_features
        )
        
        # 5. 归一化
        logger.info("  归一化特征...")
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(combined_features)
        
        # 6. 降低维度（可选）
        if normalized_features.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            reduced_features = pca.fit_transform(normalized_features)
            logger.info(f"  PCA降维: {normalized_features.shape[1]} -> {reduced_features.shape[1]} 维, 解释方差: {pca.explained_variance_ratio_.sum():.2%}")
            final_features = reduced_features
        else:
            final_features = normalized_features
        
        feature_info = {
            'text_weight': 0.55,
            'context_weight': 0.15,
            'type_weight': 0.15,
            'test_item_weight': 0.15,
            'type_encoder': type_encoder,
            'test_item_encoder': test_item_encoder,
            'feature_dim': final_features.shape[1],
            'original_dim': text_dim
        }
        
        logger.info(f"特征创建完成。最终特征维度: {final_features.shape[1]}")
        return final_features, feature_info
    
    def determine_optimal_clusters(self, features: np.ndarray, entities: List[Dict], max_k: int = 15) -> Tuple[int, Dict]:
        """
        基于多指标和领域知识确定最佳聚类数量
        
        参数:
            features: 特征矩阵
            entities: 实体列表
            max_k: 最大聚类数量
            
        返回:
            最佳聚类数量和评估指标
        """
        logger.info("确定最佳聚类数量...")
        n_samples = len(entities)
        max_k = min(max_k, max(3, n_samples // 4))  # 动态调整最大k
        
        # 1. 基础指标计算
        k_range = range(2, max_k + 1)
        inertia_values = []
        silhouette_scores = []
        
        logger.info("  计算聚类指标...")
        for k in tqdm(k_range, desc="评估不同k值"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            inertia_values.append(kmeans.inertia_)
            
            if k > 1 and n_samples > k * 2:
                silhouette_scores.append(silhouette_score(features, labels))
            else:
                silhouette_scores.append(0)
        
        # 2. 基于实体类型分布的推荐
        logger.info("  分析实体类型分布...")
        type_counts = Counter([entity['type'] for entity in entities])
        test_item_counts = Counter([entity['test_item'] for entity in entities])
        
        # 基础聚类数量
        recommended_k = 3
        
        # 为高频类型分配更多聚类
        if '试剂/溶液' in type_counts:
            reagent_ratio = type_counts['试剂/溶液'] / n_samples
            recommended_k += int(reagent_ratio * 6)  # 为试剂分配更多聚类
        
        if '具体测定操作' in type_counts:
            operation_ratio = type_counts['具体测定操作'] / n_samples
            recommended_k += int(operation_ratio * 3)
        
        # 为复杂的检验项目分配更多聚类
        complex_items = ['检查', '含量测定']
        for item in complex_items:
            if item in test_item_counts:
                item_ratio = test_item_counts[item] / n_samples
                recommended_k += int(item_ratio * 2)
        
        # 限制在合理范围内
        recommended_k = max(3, min(recommended_k, max_k))
        logger.info(f"  基于实体分布推荐k={recommended_k}")
        
        # 3. 基于轮廓系数的最佳k
        best_k_by_score = 2
        if len(silhouette_scores) > 0:
            max_score = max(silhouette_scores)
            if max_score > 0.1:  # 有意义的聚类
                best_k_by_score = k_range[np.argmax(silhouette_scores)]
                logger.info(f"  基于轮廓系数最佳k={best_k_by_score}, 得分={max_score:.4f}")
            else:
                logger.info(f"  轮廓系数较低 (max={max_score:.4f})，使用基于分布的推荐")
        
        # 4. 综合推荐
        final_k = int((best_k_by_score * 0.4 + recommended_k * 0.6))
        final_k = max(2, min(final_k, max_k))
        
        logger.info(f"  最终推荐聚类数量: k={final_k}")
        
        metrics = {
            'k_range': list(k_range),
            'inertia': inertia_values,
            'silhouette': silhouette_scores,
            'type_distribution': dict(type_counts),
            'test_item_distribution': dict(test_item_counts),
            'recommended_k': recommended_k,
            'best_k_by_score': best_k_by_score,
            'final_k': final_k
        }
        
        # 5. 可视化（可选）
        self._plot_clustering_metrics(metrics, n_samples)
        
        return final_k, metrics
    
    def _plot_clustering_metrics(self, metrics: Dict, n_samples: int):
        """绘制聚类评估指标图"""
        try:
            plt.figure(figsize=(12, 5))
            
            # 惯性图
            plt.subplot(1, 2, 1)
            plt.plot(metrics['k_range'], metrics['inertia'], 'bo-')
            plt.xlabel('聚类数量 (k)')
            plt.ylabel('惯性 (Inertia)')
            plt.title(f'肘部法则 - 样本数: {n_samples}')
            plt.grid(True)
            
            # 轮廓系数图
            plt.subplot(1, 2, 2)
            plt.plot(metrics['k_range'][:len(metrics['silhouette'])], metrics['silhouette'], 'go-')
            plt.xlabel('聚类数量 (k)')
            plt.ylabel('平均轮廓系数')
            plt.title('轮廓系数评估')
            plt.grid(True)
            
            # 标记推荐的k值
            plt.axvline(x=metrics['final_k'], color='r', linestyle='--', label=f'推荐 k={metrics["final_k"]}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'clustering_metrics_{time.strftime("%Y%m%d_%H%M%S")}.png')
            logger.info("聚类评估指标图已保存")
            plt.close()
        except Exception as e:
            logger.warning(f"绘图时出错: {e}")
    
    def perform_clustering(self, features: np.ndarray, n_clusters: int, entities: List[Dict]) -> Tuple[np.ndarray, KMeans]:
        """
        执行改进的聚类，考虑领域知识约束
        
        参数:
            features: 特征矩阵
            n_clusters: 聚类数量
            entities: 实体列表
            
        返回:
            聚类标签和KMeans模型
        """
        logger.info(f"执行聚类 (k={n_clusters})...")
        
        # 1. 初始化：尝试基于检验项目的种子点
        test_items = [entity['test_item'] for entity in entities]
        unique_test_items = list(set(test_items))
        
        # 初始化中心点
        init_method = 'k-means++'
        init_centroids = None
        
        # 如果聚类数小于等于检验项目数，尝试基于检验项目初始化
        if n_clusters <= len(unique_test_items) and n_clusters > 1:
            logger.info("  使用基于检验项目的初始化...")
            centroids = []
            used_items = set()
            
            # 优先选择样本多的检验项目
            item_counts = Counter(test_items)
            sorted_items = [item for item, _ in item_counts.most_common()]
            
            for item in sorted_items:
                if len(centroids) >= n_clusters:
                    break
                    
                indices = [i for i, t in enumerate(test_items) if t == item]
                if indices and item not in used_items:
                    # 使用该检验项目的样本中心
                    centroid = np.mean(features[indices], axis=0)
                    centroids.append(centroid)
                    used_items.add(item)
            
            # 如果数量不足，补充随机中心
            while len(centroids) < n_clusters:
                random_idx = np.random.randint(0, features.shape[0])
                centroids.append(features[random_idx].copy())
            
            init_centroids = np.array(centroids)
            init_method = init_centroids
        
        # 2. 执行聚类
        kmeans = KMeans(
            n_clusters=n_clusters, 
            init=init_method,
            random_state=42, 
            n_init=1 if init_centroids is not None else 10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(features)
        logger.info(f"  聚类完成。聚类分布: {Counter(labels)}")
        
        return labels, kmeans
    
    def optimize_clusters_by_purity(self, labels: np.ndarray, entities: List[Dict], n_clusters: int) -> np.ndarray:
        """
        优化聚类结果，提高检验项目纯度
        
        参数:
            labels: 原始聚类标签
            entities: 实体列表
            n_clusters: 聚类数量
            
        返回:
            优化后的聚类标签
        """
        logger.info("优化聚类结果 (提高检验项目纯度)...")
        from collections import Counter
        
        # 1. 计算每个聚类的检验项目分布
        cluster_items = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(labels):
            if label < n_clusters:  # 确保标签有效
                cluster_items[label].append(entities[i]['test_item'])
        
        # 2. 识别混杂聚类（项目纯度<70%且大小>5）
        mixed_clusters = []
        for cluster_id, items in cluster_items.items():
            if not items or len(items) < 5:
                continue
                
            counter = Counter(items)
            dominant_item = counter.most_common(1)[0]
            purity = dominant_item[1] / len(items)
            
            if purity < 0.7:
                mixed_clusters.append((cluster_id, dominant_item[0], purity, len(items)))
        
        logger.info(f"  识别到 {len(mixed_clusters)} 个混杂聚类")
        
        # 3. 重新分配混杂聚类中的实体
        new_labels = labels.copy()
        next_cluster_id = n_clusters
        
        for cluster_id, dominant_item, purity, size in mixed_clusters:
            logger.info(f"  优化混杂聚类 {cluster_id} (纯度={purity:.2%}, 大小={size})")
            
            # 获取该聚类中的实体索引
            indices = [i for i, label in enumerate(labels) if label == cluster_id]
            items = [entities[i]['test_item'] for i in indices]
            
            # 按检验项目分组
            item_groups = defaultdict(list)
            for idx, item in zip(indices, items):
                item_groups[item].append(idx)
            
            # 为每个主要项目创建新聚类 (至少3个实体)
            major_items = {item: idxs for item, idxs in item_groups.items() if len(idxs) >= 3}
            
            logger.info(f"    将拆分为 {len(major_items)} 个新聚类")
            
            for item, idx_list in major_items.items():
                for idx in idx_list:
                    new_labels[idx] = next_cluster_id
                logger.info(f"      新聚类 {next_cluster_id}: {item} ({len(idx_list)} 个实体)")
                next_cluster_id += 1
        
        logger.info(f"  优化后聚类数量: {next_cluster_id}")
        return new_labels
    
    def extract_cluster_keywords(self, cluster_entities: List[Dict]) -> List[str]:
        """
        从聚类实体中提取关键词
        
        参数:
            cluster_entities: 聚类中的实体列表
            
        返回:
            关键词列表
        """
        # 收集所有文本
        all_texts = [entity['text'] for entity in cluster_entities]
        combined_text = " ".join(all_texts)
        
        # 简单的关键词提取
        keywords = []
        
        # 专业术语模式
        patterns = [
            (r'(紫外|红外|可见|分光光度|光谱)', '光谱'),
            (r'(薄层|高效液相|气相|色谱|HPLC|TLC|GC)', '色谱'),
            (r'(滴定|滴定液)', '滴定'),
            (r'(重金属|砷|铁盐|氯化物|硫酸盐)', '杂质'),
            (r'(滤膜|滤器|针筒)', '过滤'),
            (r'(盐酸|硫酸|硝酸|醋酸|冰醋酸|高氯酸)', '酸'),
            (r'(氢氧化钠|氢氧化钾|碱)', '碱'),
            (r'(甲醇|乙醇|丙酮|乙醚|三氯甲烷|四氯化碳)', '有机溶剂'),
            (r'(碘化钾|溴化钾|氯化钠)', '盐类'),
            (r'(标准品|对照品)', '标准物质')
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, combined_text):
                keywords.append(keyword)
        
        # 如果没有匹配到专业术语，使用高频词
        if not keywords:
            # 分词
            words = jieba.cut(combined_text)
            word_counts = Counter()
            
            for word in words:
                word = word.strip()
                if len(word) >= 2 and word not in self.stopwords and not re.match(r'^\d+$', word):
                    word_counts[word] += 1
            
            # 取前3个高频词
            keywords = [word for word, _ in word_counts.most_common(3)]
        
        return keywords[:3]
    
    def generate_cluster_names(self, clusters: Dict[int, List[Dict]], cluster_id: int) -> str:
        """
        生成描述性聚类名称，避免使用不具信息量的代表实体
        
        参数:
            clusters: 聚类字典 {cluster_id: [entities]}
            cluster_id: 当前聚类ID
            
        返回:
            聚类名称
        """
        cluster_entities = clusters[cluster_id]
        if not cluster_entities:
            return f"空聚类_{cluster_id}"
        
        # 1. 分析聚类内容
        test_items = [e['test_item'] for e in cluster_entities]
        types = [e['type'] for e in cluster_entities]
        texts = [e['text'] for e in cluster_entities]
        
        from collections import Counter
        item_counter = Counter(test_items)
        type_counter = Counter(types)
        
        dominant_item = item_counter.most_common(1)[0][0] if item_counter else "未知"
        dominant_type = type_counter.most_common(1)[0][0] if type_counter else "未知"
        
        # 2. 基于检验项目和类型生成名称
        # 鉴别项目
        if dominant_item == "鉴别":
            if any("光谱" in t or "分光光度" in t or "红外" in t or "紫外" in t for t in texts):
                return "光谱鉴别方法"
            elif any("色谱" in t or "薄层" in t or "HPLC" in t or "TLC" in t for t in texts):
                return "色谱鉴别方法"
            elif any("试剂" in t or "试液" in t or "试纸" in t for t in texts):
                return "化学鉴别试剂"
            elif any("熔点" in t for t in texts):
                return "熔点鉴别"
        
        # 检查项目
        elif dominant_item == "检查":
            if any("杂质" in t or "有关物质" in t or "有机杂质" in t for t in texts):
                return "有机杂质检查"
            elif any("残留" in t or "溶剂" in t for t in texts):
                return "残留溶剂检查"
            elif any("重金属" in t or "砷" in t or "铁盐" in t or "氯化物" in t for t in texts):
                return "无机杂质检查"
            elif any("干燥失重" in t or "水分" in t for t in texts):
                return "干燥失重与水分检查"
        
        # 含量测定
        elif dominant_item == "含量测定":
            if any("滴定" in t or "滴定液" in t for t in texts):
                return "滴定分析方法"
            elif any("色谱" in t or "HPLC" in t for t in texts):
                return "色谱定量方法"
            elif any("紫外" in t or "分光光度" in t for t in texts):
                return "光谱定量方法"
        
        # 3. 基于类型生成名称
        if dominant_type == "试剂/溶液":
            acid_count = sum(1 for t in texts if "盐酸" in t or "硫酸" in t or "醋酸" in t or "酸" in t)
            solvent_count = sum(1 for t in texts if "甲醇" in t or "乙醇" in t or "丙酮" in t or "溶剂" in t)
            
            if acid_count > len(texts) * 0.3:
                return "酸性试剂"
            elif solvent_count > len(texts) * 0.3:
                return "有机溶剂"
            else:
                return "实验试剂与溶液"
        
        elif dominant_type == "具体测定操作":
            return f"{dominant_item}操作步骤"
        
        elif dominant_type == "限度判定指标":
            return f"{dominant_item}限度标准"
        
        elif dominant_type == "仪器设备":
            return f"实验仪器设备"
        
        # 4. 基于高频关键词
        keywords = self.extract_cluster_keywords(cluster_entities)
        if keywords:
            return "、".join(keywords[:2]) + "相关实体"
        
        # 5. 最后手段：使用代表性实体
        text_counts = Counter([e['text'] for e in cluster_entities])
        most_common = text_counts.most_common(3)
        if most_common:
            representative = most_common[0][0]
            if len(representative) > 10:
                representative = representative[:10] + "..."
            return representative
        
        return f"聚类_{cluster_id}"
    
    def evaluate_cluster_quality(self, clusters: Dict[int, List[Dict]], entities: List[Dict]) -> Dict:
        """
        全面评估聚类质量
        
        参数:
            clusters: 聚类结果 {cluster_id: [entities]}
            entities: 所有实体
            
        返回:
            质量评估指标
        """
        logger.info("评估聚类质量...")
        results = {
            'overall_purity': 0,
            'cluster_purities': {},
            'semantic_coherence': {},
            'type_consistency': {},
            'cluster_sizes': {},
            'quality_summary': ''
        }
        
        # 1. 检验项目纯度
        total_entities = 0
        weighted_purity = 0
        
        for cluster_id, cluster_entities in clusters.items():
            results['cluster_sizes'][cluster_id] = len(cluster_entities)
            
            if not cluster_entities:
                continue
                
            test_items = [e['test_item'] for e in cluster_entities]
            from collections import Counter
            counter = Counter(test_items)
            
            dominant_count = counter.most_common(1)[0][1]
            purity = dominant_count / len(cluster_entities)
            
            results['cluster_purities'][cluster_id] = {
                'purity': purity,
                'dominant_item': counter.most_common(1)[0][0],
                'item_distribution': dict(counter),
                'size': len(cluster_entities)
            }
            
            weighted_purity += purity * len(cluster_entities)
            total_entities += len(cluster_entities)
        
        if total_entities > 0:
            results['overall_purity'] = weighted_purity / total_entities
            logger.info(f"  整体检验项目纯度: {results['overall_purity']:.2%}")
        
        # 2. 类型一致性
        for cluster_id, cluster_entities in clusters.items():
            if not cluster_entities:
                continue
                
            types = [e['type'] for e in cluster_entities]
            counter = Counter(types)
            dominant_count = counter.most_common(1)[0][1]
            consistency = dominant_count / len(cluster_entities)
            
            results['type_consistency'][cluster_id] = {
                'consistency': consistency,
                'dominant_type': counter.most_common(1)[0][0],
                'type_distribution': dict(counter)
            }
        
        # 3. 质量摘要
        high_purity_clusters = sum(1 for c in results['cluster_purities'].values() if c['purity'] >= 0.8)
        medium_purity_clusters = sum(1 for c in results['cluster_purities'].values() if 0.6 <= c['purity'] < 0.8)
        low_purity_clusters = sum(1 for c in results['cluster_purities'].values() if c['purity'] < 0.6)
        
        results['quality_summary'] = (
            f"聚类质量摘要:\n"
            f"- 总体检验项目纯度: {results['overall_purity']:.2%}\n"
            f"- 高纯度聚类 (≥80%): {high_purity_clusters} 个\n"
            f"- 中等纯度聚类 (60-80%): {medium_purity_clusters} 个\n"
            f"- 低纯度聚类 (<60%): {low_purity_clusters} 个\n"
            f"- 平均聚类大小: {total_entities/len(clusters):.1f} 个实体"
        )
        
        logger.info("\n" + results['quality_summary'])
        return results
    
    def visualize_clusters(self, features: np.ndarray, labels: np.ndarray, cluster_names: Dict[int, str], 
                          drug_name: str, save_path: str = None):
        """
        可视化聚类结果
        
        参数:
            features: 特征矩阵
            labels: 聚类标签
            cluster_names: 聚类名称映射
            drug_name: 药品名称
            save_path: 保存路径
        """
        logger.info("生成聚类可视化...")
        
        try:
            # 降维到2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            embeddings_2d = tsne.fit_transform(features)
            
            # 创建DataFrame用于绘图
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'cluster': [cluster_names.get(label, f"Cluster {label}") for label in labels],
                'label': [f"{cluster_names.get(label, str(label))[:10]}..." if len(cluster_names.get(label, '')) > 10 else cluster_names.get(label, str(label)) 
                         for label in labels]
            })
            
            # 绘图
            plt.figure(figsize=(14, 10))
            
            # 散点图
            scatter = sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='cluster',
                palette='tab20',
                s=100,
                alpha=0.8,
                edgecolor='w',
                linewidth=0.5
            )
            
            # 添加聚类中心标签
            cluster_centers = {}
            for cluster_id in set(labels):
                mask = np.array(labels) == cluster_id
                if np.any(mask):
                    center_x = np.mean(embeddings_2d[mask, 0])
                    center_y = np.mean(embeddings_2d[mask, 1])
                    cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
                    cluster_centers[cluster_id] = (center_x, center_y)
                    
                    # 添加标签
                    plt.text(center_x, center_y, cluster_name[:15], 
                            horizontalalignment='center',
                            verticalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'),
                            fontsize=9)
            
            plt.title(f'{drug_name} 实体聚类可视化', fontsize=16, fontweight='bold')
            plt.xlabel('t-SNE 1', fontsize=12)
            plt.ylabel('t-SNE 2', fontsize=12)
            plt.legend(title='聚类', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"聚类可视化已保存至: {save_path}")
            else:
                output_path = f"{drug_name.replace(' ', '_')}_clustering_{time.strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"聚类可视化已保存至: {output_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"可视化时出错: {e}")
    
    def generate_analysis_report(self, drug_name: str, entities: List[Dict], clusters: Dict[int, List[Dict]], 
                                cluster_names: Dict[int, str], quality_metrics: Dict) -> str:
        """
        生成分析报告
        
        参数:
            drug_name: 药品名称
            entities: 实体列表
            clusters: 聚类结果
            cluster_names: 聚类名称
            quality_metrics: 质量指标
            
        返回:
            格式化的报告文本
        """
        logger.info("生成分析报告...")
        
        report = []
        report.append("=" * 80)
        report.append(f"药品实体聚类分析报告: {drug_name}")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # 1. 概述
        report.append("1. 分析概述")
        report.append("-" * 40)
        report.append(f"• 总实体数量: {len(entities)}")
        report.append(f"• 聚类数量: {len(clusters)}")
        report.append(f"• 平均聚类大小: {len(entities)/len(clusters):.1f}")
        report.append(f"• 整体检验项目纯度: {quality_metrics['overall_purity']:.2%}")
        report.append("")
        
        # 2. 聚类详情
        report.append("2. 聚类详情")
        report.append("-" * 40)
        
        for cluster_id in sorted(clusters.keys()):
            cluster_entities = clusters[cluster_id]
            if not cluster_entities:
                continue
                
            name = cluster_names.get(cluster_id, f"聚类 {cluster_id}")
            size = len(cluster_entities)
            
            # 检验项目分布
            test_items = [e['test_item'] for e in cluster_entities]
            item_counter = Counter(test_items)
            
            # 类型分布
            types = [e['type'] for e in cluster_entities]
            type_counter = Counter(types)
            
            # 纯度
            purity = quality_metrics['cluster_purities'].get(cluster_id, {}).get('purity', 0)
            
            report.append(f"\n聚类 {cluster_id}: {name}")
            report.append(f"  • 实体数量: {size} ({size/len(entities):.1%} of total)")
            report.append(f"  • 检验项目纯度: {purity:.2%}")
            report.append(f"  • 主要检验项目: {item_counter.most_common(1)[0][0]} ({item_counter.most_common(1)[0][1]}/{size})")
            report.append(f"  • 项目分布: {dict(item_counter)}")
            report.append(f"  • 类型分布: {dict(type_counter)}")
            
            # 显示代表性实体 (最多5个)
            if size > 0:
                report.append("  • 代表性实体:")
                for i, entity in enumerate(cluster_entities[:5]):
                    text = entity['text']
                    if len(text) > 50:
                        text = text[:47] + "..."
                    report.append(f"    {i+1}. [{entity['test_item']}] {text}")
            
            # 聚类改进建议
            if purity < 0.6:
                report.append(f"  • 建议: 该聚类检验项目混杂，建议拆分为更小的聚类或检查预处理")
            elif purity < 0.8:
                report.append(f"  • 建议: 该聚类有一定混杂，可考虑进一步优化")
        
        # 3. 质量评估
        report.append("")
        report.append("3. 质量评估")
        report.append("-" * 40)
        report.append(quality_metrics['quality_summary'])
        
        # 4. 改进建议
        report.append("")
        report.append("4. 改进建议")
        report.append("-" * 40)
        
        # 基于质量指标提供建议
        if quality_metrics['overall_purity'] < 0.7:
            report.append("• 整体聚类质量较低，建议:")
            report.append("  - 增加聚类数量，特别是为'试剂/溶液'类型分配更多聚类")
            report.append("  - 优化特征表示，增强检验项目和实体类型的权重")
            report.append("  - 检查预处理步骤，确保实体标准化和边界实体处理")
        elif quality_metrics['overall_purity'] < 0.85:
            report.append("• 聚类质量中等，建议:")
            report.append("  - 对低纯度聚类进行后处理优化")
            report.append("  - 调整特征权重，增加检验项目和上下文信息的权重")
        else:
            report.append("• 聚类质量良好，可考虑用于更大规模分析。")
            report.append("  - 对于特定药品，可微调参数以获得更好的结果")
            report.append("  - 建议进行小规模人工验证，确认聚类语义合理性")
        
        report.append("")
        report.append("5. 下一步建议")
        report.append("-" * 40)
        report.append("• 对低纯度聚类进行人工审查，确定是否需要调整预处理规则")
        report.append("• 尝试调整聚类数量 (±1~2) 观察质量变化")
        report.append("• 为后续大规模分析建立参数配置模板")
        report.append("")
        report.append("=" * 80)
        
        final_report = "\n".join(report)
        logger.info("分析报告生成完成")
        return final_report
    
    def process_drug_clustering(self, file_path: str, drug_name: str = None) -> Dict:
        """
        处理单个药品的聚类分析
        
        参数:
            file_path: JSON文件路径
            drug_name: 药品名称 (如果为None，从文件名提取)
            
        返回:
            分析结果字典
        """
        start_time = time.time()
        
        # 1. 确定药品名称
        if drug_name is None:
            drug_name = os.path.splitext(os.path.basename(file_path))[0]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"开始处理药品: {drug_name}")
        logger.info(f"文件路径: {file_path}")
        logger.info(f"{'='*80}")
        
        # 2. 加载实体
        logger.info("\n1. 加载实体...")
        entities = self.load_entities_from_json(file_path, drug_name)
        if not entities:
            logger.error("未加载到有效实体，处理终止")
            return {}
        
        # 3. 预处理
        logger.info("\n2. 预处理实体...")
        cleaned_entities = self.preprocess_entities(entities)
        if not cleaned_entities:
            logger.error("预处理后无有效实体，处理终止")
            return {}
        
        # 4. 构建上下文
        logger.info("\n3. 构建实体上下文...")
        context_map = self.build_entity_context(cleaned_entities)
        
        # 5. 创建增强特征
        logger.info("\n4. 创建增强特征...")
        features, feature_info = self.create_enhanced_features(cleaned_entities, context_map)
        
        # 6. 确定最佳聚类数量
        logger.info("\n5. 确定最佳聚类数量...")
        optimal_k, metrics = self.determine_optimal_clusters(features, cleaned_entities)
        
        # 7. 执行聚类
        logger.info("\n6. 执行聚类...")
        cluster_labels, kmeans_model = self.perform_clustering(features, optimal_k, cleaned_entities)
        
        # 8. 后处理优化
        logger.info("\n7. 优化聚类结果...")
        optimized_labels = self.optimize_clusters_by_purity(cluster_labels, cleaned_entities, optimal_k)
        
        # 9. 按聚类分组
        logger.info("\n8. 按聚类分组...")
        clusters = defaultdict(list)
        for i, label in enumerate(optimized_labels):
            clusters[label].append(cleaned_entities[i])
        
        # 10. 生成聚类名称
        logger.info("\n9. 生成聚类名称...")
        cluster_names = {}
        for cluster_id in clusters:
            name = self.generate_cluster_names(clusters, cluster_id)
            cluster_names[cluster_id] = name
            logger.info(f"  聚类 {cluster_id}: {name} ({len(clusters[cluster_id])} 个实体)")
        
        # 11. 评估质量
        logger.info("\n10. 评估聚类质量...")
        quality_metrics = self.evaluate_cluster_quality(clusters, cleaned_entities)
        
        # 12. 生成可视化
        logger.info("\n11. 生成可视化...")
        viz_path = f"{drug_name.replace(' ', '_')}_clustering_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self.visualize_clusters(features, optimized_labels, cluster_names, drug_name, viz_path)
        
        # 13. 生成报告
        logger.info("\n12. 生成分析报告...")
        report = self.generate_analysis_report(
            drug_name, cleaned_entities, clusters, cluster_names, quality_metrics
        )
        
        # 14. 保存结果
        logger.info("\n13. 保存结果...")
        result = {
            'drug_name': drug_name,
            'file_path': file_path,
            'original_entity_count': len(entities),
            'processed_entity_count': len(cleaned_entities),
            'clusters': clusters,
            'cluster_names': cluster_names,
            'cluster_labels': optimized_labels.tolist() if hasattr(optimized_labels, 'tolist') else optimized_labels,
            'quality_metrics': quality_metrics,
            'report': report,
            'feature_info': feature_info,
            'processing_time': time.time() - start_time,
            'visualization_path': viz_path
        }
        
        # 保存JSON结果
        output_file = f"{drug_name.replace(' ', '_')}_clustering_result_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'drug_name': drug_name,
                    'clusters': {
                        str(cluster_id): [e['text'] for e in entities] 
                        for cluster_id, entities in clusters.items()
                    },
                    'cluster_names': cluster_names,
                    'quality_metrics': quality_metrics,
                    'processing_time': result['processing_time']
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存至: {output_file}")
        except Exception as e:
            logger.error(f"保存结果时出错: {e}")
        
        # 保存完整报告
        report_file = f"{drug_name.replace(' ', '_')}_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"分析报告已保存至: {report_file}")
        except Exception as e:
            logger.error(f"保存报告时出错: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"\n处理完成! 总耗时: {total_time:.2f} 秒")
        logger.info(f"平均每实体处理时间: {total_time/len(cleaned_entities):.4f} 秒")
        
        return result
    
    def process_batch_drugs(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, Dict]:
        """
        批量处理多个药品
        
        参数:
            file_paths: JSON文件路径列表
            max_workers: 并行工作进程数
            
        返回:
            {drug_name: result} 字典
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"开始批量处理 {len(file_paths)} 个药品")
        logger.info(f"并行工作进程数: {max_workers}")
        logger.info(f"{'='*80}")
        
        results = {}
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_file = {}
            for file_path in file_paths:
                drug_name = os.path.splitext(os.path.basename(file_path))[0]
                future = executor.submit(self._process_single_drug_wrapper, file_path, drug_name)
                future_to_file[future] = (file_path, drug_name)
            
            # 收集结果
            for future in tqdm(as_completed(future_to_file), total=len(file_paths), desc="处理进度"):
                file_path, drug_name = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results[drug_name] = result
                        logger.info(f"✓ 完成: {drug_name}")
                    else:
                        logger.warning(f"✗ 失败: {drug_name} (无结果返回)")
                except Exception as exc:
                    logger.error(f"✗ 处理失败: {drug_name} - {exc}")
                    log_error(drug_name, exc)
        
        total_time = time.time() - start_time
        logger.info(f"\n批量处理完成! 总耗时: {total_time/60:.2f} 分钟")
        logger.info(f"成功处理: {len(results)}/{len(file_paths)} 个药品")
        logger.info(f"平均每个药品: {total_time/len(file_paths):.2f} 秒")
        
        return results
    
    def _process_single_drug_wrapper(self, file_path: str, drug_name: str) -> Dict:
        """单个药品处理的包装函数，用于并行处理"""
        try:
            # 为了避免子进程中的模型重复加载，每个子进程需要重新初始化
            analyzer = DrugEntityClusterAnalyzer(self.model_name)
            return analyzer.process_drug_clustering(file_path, drug_name)
        except Exception as e:
            logger.error(f"处理 {drug_name} 时出错: {e}")
            return {}


def log_error(drug_name: str, error: Exception):
    """记录错误信息"""
    with open("processing_errors.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {drug_name}: {str(error)}\n")
        f.write(f"Traceback: {error.__traceback__}\n\n")


def main():
    """主函数"""
    analyzer = DrugEntityClusterAnalyzer()
    
    json_files = [
        r"D:\drug_extraction\extracted_results\extracted_entities_阿苯达唑.json",
        r"D:\drug_extraction\extracted_results\extracted_entities_阿苯达唑胶囊.json",
        r"D:\drug_extraction\extracted_results\extracted_entities_阿苯达唑颗粒.json"
    ]

    # 检查是否所有文件都存在
    missing_files = [f for f in json_files if not os.path.exists(f)]

    if missing_files:
        logger.warning(f"以下文件不存在: {missing_files}")
        return

    # 批量处理
    results = analyzer.process_batch_drugs(json_files, max_workers=4)

    # results 是 {drug_name: result_dict, ...}
    if not results:
        logger.warning("批量处理未返回有效结果。")
        return

    print("\n分析报告汇总:")
    for drug_name, res in results.items():
        print("\n" + "="*40)
        print(f"药品: {drug_name}")
        if not res:
            print("  ✗ 该药品处理失败或未返回结果。")
            continue
        # 有的字段可能不存在，先做保险判断
        report = res.get('report')
        if report:
            print(report)
        else:
            # 如果没有完整报告，打印一些摘要信息
            print(f"  • 实体数量: {res.get('processed_entity_count', '未知')}")
            print(f"  • 聚类数量: {len(res.get('clusters', {}))}")
            qm = res.get('quality_metrics')
            if qm:
                print(f"  • 整体检验项目纯度: {qm.get('overall_purity', 0):.2%}")
            print("  （没有'main'报告文本）")


    
    # 批量处理示例 (取消注释以使用)
    '''
    # 获取所有JSON文件
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if json_files:
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        results = analyzer.process_batch_drugs(json_files, max_workers=4)
        
        # 保存汇总报告
        summary_report = f"batch_processing_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_report, 'w', encoding='utf-8') as f:
            f.write("批量处理汇总报告\n")
            f.write("=" * 50 + "\n\n")
            for drug_name, result in results.items():
                f.write(f"药品: {drug_name}\n")
                f.write(f"处理时间: {result['processing_time']:.2f} 秒\n")
                f.write(f"实体数量: {result['processed_entity_count']}\n")
                f.write(f"聚类数量: {len(result['clusters'])}\n")
                f.write(f"整体纯度: {result['quality_metrics']['overall_purity']:.2%}\n")
                f.write(f"可视化: {result['visualization_path']}\n")
                f.write("-" * 50 + "\n\n")
        
        logger.info(f"汇总报告已保存至: {summary_report}")
    '''


if __name__ == "__main__":
    main()