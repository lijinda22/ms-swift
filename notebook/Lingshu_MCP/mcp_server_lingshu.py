#!/usr/bin/env python3
"""
Lingshu FastMCP Server - Simplified implementation using FastMCP framework
灵枢 FastMCP 服务器 - 使用 FastMCP 框架的简化实现

This server wraps the Lingshu model as MCP tools using the FastMCP framework,
providing medical image analysis and report generation services for clients like dHealth Intelligence.
该服务器使用 FastMCP 框架将灵枢模型转换为 MCP 工具，为数智健康等客户端提供医学影像分析和报告生成服务。
"""

import base64
import json
import logging
import asyncio
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import os
import PIL
from fastmcp import FastMCP
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP("lingshu")

DEFAULT_TIMEOUT = 60.0


class LingshuModelClient:
    """
    Lingshu model client | 灵枢模型客户端
    """
    def __init__(self):
        self.llm_client = AsyncOpenAI(
            base_url=os.environ.get("LINGSHU_SERVER_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LINGSHU_SERVER_API", "api_key")  
        )
        self.model = os.environ.get("LINGSHU_MODEL", "Lingshu-7B")
    
    async def generate(self, prompt: str, image_data: Optional[str] = None, 
                      max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """
        调用灵枢模型生成回答
        Call Lingshu model to generate response
        """
        try:
            if image_data:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            else:
                content = prompt
            
            response = await self.llm_client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error calling Lingshu model: {e}")
            raise Exception(f"Lingshu model call failed: {str(e)}")


lingshu_client = LingshuModelClient()


@app.tool(name="analyze_medical_image", description="Analyze medical images using Lingshu")
async def analyze_medical_image( 
    image_path: str,
    analysis_type: str = "radiology",
    patient_context: str = "",
    language: str = "zh"
) -> Dict[str, Any]:
    """
    分析医学影像并提供专业的医学报告
    Analyze medical images and provide professional medical reports
    
    Parameters:
    - image_path: Path to medical image file | 医学影像文件路径
    - analysis_type: Type of analysis | 分析类型 (radiology/pathology/dermatology/ophthalmology)
    - patient_context: Patient clinical background information | 患者临床背景信息
    - language: Report language | 报告语言 (zh/en)
    
    Returns:
    - Dictionary containing detailed medical analysis | 包含详细医学分析的字典
    """
    try:

        if not image_path:
            return {"error": "No image data provided"}
        
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        
        
        valid_types = ["radiology", "pathology", "dermatology", "ophthalmology", "general"]
        if analysis_type not in valid_types:
            analysis_type = "general"
        

        if language == "en":
            prompt = f"""You are a highly trained medical AI assistant specializing in {analysis_type} image analysis.

Analyze the provided medical image and provide a comprehensive assessment including:

1. **Technical Quality Assessment:**
   - Image quality, positioning, and technique
   - Any technical limitations or artifacts

2. **Anatomical Observations:**
   - Detailed description of visible structures
   - Normal anatomical findings
   - Any anatomical variations

3. **Pathological Findings:**
   - Identify abnormal findings with precise descriptions
   - Location, size, morphology, and characteristics
   - Severity assessment where applicable

4. **Clinical Interpretation:**
   - Most likely differential diagnoses
   - Clinical significance and implications
   - Correlation with provided context

5. **Recommendations:**
   - Additional imaging or studies needed
   - Urgent vs routine clinical follow-up
   - Specific management suggestions

Patient Context: {patient_context}

Provide your analysis in a structured, professional medical report format."""
        else:
            prompt = f"""您是一位经验丰富的{analysis_type}影像学专家。请对提供的医学影像进行全面分析：

1. **技术质量评估：**
   - 影像质量、体位和技术参数
   - 技术限制或伪影

2. **解剖学观察：**
   - 可见结构的详细描述  
   - 正常解剖学表现
   - 解剖变异

3. **病理学发现：**
   - 异常发现的精确描述
   - 位置、大小、形态学特征
   - 严重程度评估

4. **临床解读：**
   - 可能的鉴别诊断
   - 临床意义和影响
   - 与临床背景的关联

5. **建议：**
   - 需要的进一步检查
   - 紧急或常规随访建议
   - 具体管理意见

患者背景：{patient_context}

请以结构化的专业医学报告格式提供分析。"""
        
        result = await lingshu_client.generate(
            prompt=prompt,
            image_data=image_base64,
            max_tokens=2048,
            temperature=0.1
        )
        
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "language": language,
            "report": result,
            "timestamp": datetime.now().isoformat(),
            "model": "lingshu"
        }
        
    except Exception as e:
        logger.error(f"Medical image analysis error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.tool(name="generate_medical_report", description="Generate structured medical reports using Lingshu medical model")
async def generate_medical_report(  
    findings: List[str],
    report_type: str = "diagnostic",
    patient_info: Dict[str, Any] = None,
    language: str = "zh",
    template: str = "standard"
) -> Dict[str, Any]:
    """
    根据医学发现生成结构化医学报告
    Generate structured medical reports based on medical findings
    
    Parameters:
    - findings: List of medical findings | 医学发现列表
    - report_type: Type of report | 报告类型 (diagnostic/screening/follow_up/consultation)
    - patient_info: Patient information dictionary | 患者信息字典
    - language: Report language | 报告语言 (zh/en)
    - template: Report template | 报告模板 (standard/detailed/brief)
    
    Returns:
    - Dictionary containing formatted medical report | 包含格式化医学报告的字典
    """
    
    try:
        if not findings:
            return {"error": "No medical findings provided"}
        
        if not patient_info:
            patient_info = {}
        

        findings_text = "\n".join([f"• {finding}" for finding in findings])
        
        patient_text = ""
        if patient_info:
            patient_text = f"Patient Information:\n{json.dumps(patient_info, indent=2)}\n"
        
        if language == "en":
            prompt = f"""You are a medical reporting specialist. Generate a comprehensive {report_type} medical report based on the following findings.

**Clinical Findings:**
{findings_text}

**Patient Information:**
{patient_text if patient_text else "Not provided"}

Generate a structured medical report in the following format:

**MEDICAL REPORT - {report_type.upper()}**
Date: {datetime.now().strftime('%Y-%m-%d')}
Report Type: {report_type.title()}

**CLINICAL HISTORY:**
[Based on provided patient information and context]

**FINDINGS:**
[Detailed analysis of each finding with clinical correlation]

**IMPRESSION:**
[Concise summary of key findings and clinical significance]

**RECOMMENDATIONS:**
[Specific recommendations for patient management, follow-up, or additional studies]

**CLINICAL CORRELATION:**
[How findings correlate with clinical presentation and patient history]

Ensure the report is:
- Medically accurate and professional
- Appropriately detailed for the {template} template
- Clear and actionable for healthcare providers
- Compliant with medical reporting standards"""
        else:
            prompt = f"""您是一位专业的医学报告专家。请基于以下医学发现生成一份全面的{report_type}医学报告。

**临床发现:**
{findings_text}

**患者信息:**
{patient_text if patient_text else "未提供"}

请按以下格式生成结构化医学报告：

**医学报告 - {report_type.upper()}**
日期: {datetime.now().strftime('%Y年%m月%d日')}
报告类型: {report_type}

**临床病史:**
[基于提供的患者信息和背景]

**检查发现:**
[每项发现的详细分析和临床关联]

**诊断印象:**
[关键发现的简洁总结和临床意义]

**建议:**
[患者管理、随访或其他检查的具体建议]

**临床关联:**
[发现与临床表现和患者病史的关联性]

请确保报告：
- 医学准确且专业
- 符合{template}模板的详细程度
- 对医护人员清晰可行
- 符合医学报告标准"""

        result = await lingshu_client.generate(
            prompt=prompt,
            max_tokens=3072,
            temperature=0.1
        )
        
        return {
            "status": "success",
            "report_type": report_type,
            "template": template,
            "language": language,
            "report": result,
            "findings_count": len(findings),
            "timestamp": datetime.now().isoformat(),
            "model": "lingshu"
        }
        
    except Exception as e:
        logger.error(f"Medical report generation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.tool(name="medical_qa", description="Answer medical questions using Lingshu medical model")
async def medical_qa(
    question: str,
    context: str = "",
    specialty: str = "general",
    language: str = "zh"
) -> Dict[str, Any]:
    """
    回答医学相关问题
    Answer medical-related questions
    
    Parameters:
    - question: Medical question | 医学问题
    - context: Relevant background information | 相关背景信息
    - specialty: Medical specialty field | 医学专业领域 (general/radiology/pathology/surgery/etc)
    - language: Answer language | 回答语言 (zh/en)
    
    Returns:
    - Dictionary containing medical Q&A response | 包含医学问答响应的字典
    """
    
    try:
        if not question.strip():
            return {"error": "No question provided"}
        
        if language == "en":
            prompt = f"""You are a knowledgeable medical expert specializing in {specialty}. 
Please provide a comprehensive, accurate, and professional answer to the following medical question.

**Question:** {question}

**Context:** {context if context else "No additional context provided"}

Please provide:
1. A clear, evidence-based answer
2. Relevant medical terminology and explanations
3. Clinical considerations where applicable
4. Any important caveats or limitations
5. Recommendations for further evaluation if needed

Note: This response is for educational purposes and should not replace professional medical consultation."""
        else:
            prompt = f"""您是一位在{specialty}领域知识丰富的医学专家。
请为以下医学问题提供全面、准确、专业的答案。

**问题:** {question}

**背景:** {context if context else "无额外背景信息"}

请提供：
1. 清晰、基于循证医学的答案
2. 相关医学术语和解释
3. 适用的临床考虑因素
4. 重要的注意事项或限制
5. 必要时的进一步评估建议

注意：此回答仅用于教育目的，不应替代专业医学咨询。"""

        result = await lingshu_client.generate(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.2
        )
        
        return {
            "status": "success",
            "question": question,
            "specialty": specialty,
            "language": language,
            "answer": result,
            "timestamp": datetime.now().isoformat(),
            "model": "lingshu"
        }
        
    except Exception as e:
        logger.error(f"Medical QA error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lingshu FastMCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=4200, help="Server port")
    parser.add_argument("--path", default="/lingshu", help="Service path prefix")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    print(f"""
🚀 Lingshu FastMCP Server Starting...

Configuration:
- Server Address: {args.host}:{args.port}
- Service Path: {args.path}  
- Log Level: {args.log_level}

Available Tools:
1. analyze_medical_image - Medical image analysis
2. generate_medical_report - Medical report generation
3. medical_qa - Medical Q&A

Integration Config:
{{
  "mcpServers": {{
    "lingshu": {{
      "command": "python",
      "args": ["{os.path.abspath(__file__)}"]
    }}
  }}
}}
    """)
    
    app.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        path=args.path,
        log_level=args.log_level
    )