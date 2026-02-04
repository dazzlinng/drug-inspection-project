import json
import re
import os
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import requests

# 配置日志（避免终端输出，仅记录关键信息）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='drug_extraction.log'
)
logger = logging.getLogger('DrugExtractor')

class DrugStandardExtractor:
    def __init__(self, api_key: str, model: str = "qwen-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.entity_categories = {
            "仪器设备": "检验中使用的设备（如HPLC、UV分光光度计等）",
            "试剂/溶液": "检验中使用的试剂、标准品、对照品、缓冲液等",
            "色谱柱/填充剂": "色谱分析中使用的色谱柱、填料（如C18、硅胶等）",
            "其他材料": "检验中使用的辅助材料（如滤膜、离心管等）",
            "前处理过程": "样品前处理步骤（如粉碎、溶解、离心等）",
            "溶液配置过程": "溶液配制步骤（如标准溶液制备、缓冲液配制）",
            "具体测定操作": "仪器操作步骤（如进样、检测、数据分析）",
            "限度判定指标": "检验结果判定标准（如含量限度、杂质限度）",
            "检验项目": "检验类型（鉴别、检查、含量测定、通则）"
        }
        self.processed_count = 0
        self.start_time = time.time()

    def call_qwen_api(self, prompt: str) -> str:
        """调用Qwen API获取实体抽取结果（带重试机制）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 4096  # 适应长文本
            }
        }
        for attempt in range(3):  # 重试3次
            try:
                response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result['output']['text']
            except Exception as e:
                if attempt == 2:  # 最后一次尝试失败
                    logger.error(f"API调用失败: {str(e)}")
                    return ""
                time.sleep(2 ** attempt)  # 指数退避
        return ""

    def extract_entities_from_text(self, text: str, test_name: str) -> Dict[str, Any]:
        """从文本中抽取检验标准实体（优化Prompt设计）"""
        prompt = f"""
请严格根据以下要求从药品检验标准文本中抽取实体：
1. 仅抽取与检验直接相关的实体，忽略背景信息
2. 实体类别必须匹配以下定义：
   - 仪器设备: {self.entity_categories['仪器设备']}
   - 试剂/溶液: {self.entity_categories['试剂/溶液']}
   - 色谱柱/填充剂: {self.entity_categories['色谱柱/填充剂']}
   - 其他材料: {self.entity_categories['其他材料']}
   - 前处理过程: {self.entity_categories['前处理过程']}
   - 溶液配置过程: {self.entity_categories['溶液配置过程']}
   - 具体测定操作: {self.entity_categories['具体测定操作']}
   - 限度判定指标: {self.entity_categories['限度判定指标']}
3. 重要要求：
   - 步骤类实体（前处理/溶液配置/测定操作）必须用数字序号分步描述
   - 限度判定指标需包含具体数值和单位（如：含量不得少于98.0%）
   - 未找到的类别返回"未找到"
   - 保持原意，不添加额外信息

检验项目: {test_name}
文本内容:
{text}

请以严格的JSON格式返回结果，格式如下（注意：必须包含所有字段）：
{{
  "检验项目": "{test_name}",
  "检测资源": {{
    "仪器设备": [],
    "试剂/溶液": [],
    "色谱柱/填充剂": [],
    "其他材料": []
  }},
  "检测步骤": {{
    "前处理过程": "未找到",
    "溶液配置过程": "未找到",
    "具体测定操作": "未找到",
    "限度判定指标": "未找到"
  }}
}}
        """
        api_response = self.call_qwen_api(prompt)
        
        # 增强JSON解析鲁棒性
        try:
            # 提取最内层JSON
            json_match = re.search(r'\{[\s\S]*\}', api_response)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                # 验证结果结构
                self._validate_result(result)
                return result
            else:
                logger.warning(f"API响应无效: {api_response[:500]}...")
                return self._get_empty_result()
        except Exception as e:
            logger.error(f"JSON解析失败: {str(e)}")
            logger.error(f"原始响应: {api_response[:500]}...")
            return self._get_empty_result()

    def _validate_result(self, result: Dict[str, Any]) -> None:
        """验证返回结果结构完整性"""
        required_keys = ["检验项目", "检测资源", "检测步骤"]
        for key in required_keys:
            if key not in result:
                result[key] = {} if key in ["检测资源", "检测步骤"] else ""
        
        # 确保检测资源子字段
        for category in ["仪器设备", "试剂/溶液", "色谱柱/填充剂", "其他材料"]:
            if category not in result["检测资源"]:
                result["检测资源"][category] = []
        
        # 确保检测步骤子字段
        for step in ["前处理过程", "溶液配置过程", "具体测定操作", "限度判定指标"]:
            if step not in result["检测步骤"]:
                result["检测步骤"][step] = "未找到"

    def _get_empty_result(self) -> Dict[str, Any]:
        """返回结构完整的空结果"""
        return {
            "检验项目": "未指定",
            "检测资源": {
                "仪器设备": [],
                "试剂/溶液": [],
                "色谱柱/填充剂": [],
                "其他材料": []
            },
            "检测步骤": {
                "前处理过程": "未找到",
                "溶液配置过程": "未找到",
                "具体测定操作": "未找到",
                "限度判定指标": "未找到"
            }
        }

    def clean_filename(self, name: str) -> str:
        """清理文件名中的非法字符"""
        return re.sub(r'[\\/*?:"<>|]', "_", name)[:50]  # 限制长度

    def process_drug(self, drug: Dict[str, Any], output_dir: str) -> bool:
        """处理单个药品的检验标准"""
        drug_name = drug.get("名称", "Unknown")
        safe_name = self.clean_filename(drug_name)
        output_file = os.path.join(output_dir, f"extracted_entities_{safe_name}.json")
        
        # 仅处理非空药品
        if not drug_name or drug_name == "Unknown":
            logger.warning(f"跳过无效药品: {drug}")
            return False
        
        # 构建药品结果结构
        drug_result = {
            "药品名称": drug_name,
            "检验标准": []
        }
        
        # 处理所有检验项目
        test_sections = {
            "鉴别": drug.get("鉴别", ""),
            "检查": drug.get("检查", ""),
            "含量测定": drug.get("含量测定", ""),
            "通则": drug.get("通则", "")
        }
        
        for section_name, section_text in test_sections.items():
            # 跳过空内容
            if not section_text or section_text.strip() == "未找到":
                continue
                
            # 处理长文本
            if len(section_text) > 10000:
                section_text = section_text[:10000] + " [文本截断]"
            
            # 执行实体抽取
            section_result = self.extract_entities_from_text(section_text, section_name)
            drug_result["检验标准"].append(section_result)
        
        # 添加空结果处理
        if not drug_result["检验标准"]:
            drug_result["检验标准"].append(self._get_empty_result())
        
        # 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(drug_result, f, ensure_ascii=False, indent=2)
            self.processed_count += 1
            return True
        except Exception as e:
            logger.error(f"保存失败 {drug_name}: {str(e)}")
            return False

    def process_large_file(self, input_file: str, output_dir: str = "extracted_results"):
        """处理大型JSON文件（106MB级）"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 读取文件（流式处理，避免内存溢出）
        logger.info(f"开始处理文件: {input_file}")
        logger.info(f"输出目录: {output_dir}")
        
        # 使用流式JSON解析（避免一次性加载大文件）
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取整个文件内容（106MB在内存中可接受，但流式更安全）
            try:
                drug_data = json.load(f)
            except Exception as e:
                logger.error(f"文件加载失败: {str(e)}")
                return
        
        # 2. 处理每个药品（按批次处理，避免内存峰值）
        batch_size = 50  # 每批次处理50个药品
        total_drugs = len(drug_data)
        
        for i in range(0, total_drugs, batch_size):
            batch = drug_data[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(total_drugs + batch_size - 1)//batch_size} "
                        f"({len(batch)}个药品) - {time.strftime('%H:%M:%S')}")
            
            for drug in batch:
                self.process_drug(drug, output_dir)
            
            # 每批次后休息1秒（避免API限流）
            time.sleep(1)
        
        # 3. 生成处理报告
        elapsed = time.time() - self.start_time
        logger.info(f"处理完成! 总药品数: {total_drugs}, 成功处理: {self.processed_count}")
        logger.info(f"平均处理速度: {self.processed_count / elapsed:.2f} 药品/秒")
        logger.info(f"结果已保存到: {output_dir}")

def main():
    # 配置参数（实际使用时替换API_KEY）
    API_KEY = ""  
    INPUT_FILE = "full_drug_standards.json"  # 106MB的输入文件
    OUTPUT_DIR = "extracted_results"
    
    # 初始化提取器
    extractor = DrugStandardExtractor(API_KEY)
    
    # 开始处理
    extractor.process_large_file(INPUT_FILE, OUTPUT_DIR)

if __name__ == "__main__":
    main()