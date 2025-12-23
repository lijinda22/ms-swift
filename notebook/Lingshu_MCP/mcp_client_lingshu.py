#!/usr/bin/env python3
"""
Lingshu 32B FastMCP Client Test Script | 灵枢 32B FastMCP 客户端测试脚本

This script demonstrates how to interact with the Lingshu FastMCP server,
simulating the usage scenario of dHealth Intelligence.
本脚本演示了如何与灵枢 FastMCP 服务器进行交互，模拟了数智健康的实际使用场景。
"""

import asyncio
import base64
import json
import httpx
from datetime import datetime
from typing import Dict, Any
import os
import argparse
from fastmcp import Client
from openai import AsyncOpenAI

# 初始化 LLM 客户端，默认连接本地 vLLM 服务
llm_client = AsyncOpenAI(
    base_url=os.environ.get("LLM_SERVER_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_SERVER_API", "api_key")  
)
# 设置模型名称
model = os.environ.get("LLM_MODEL", "qwen3-235b-a22b-instruct-2507")

async def query_mcp_tool(tool_name: str, params: dict):
    """
    调用MCP工具的统一入口
    :param tool_name: 工具名称
    :param params: 工具参数
    :return: 工具执行结果
    """
    async with Client("http://127.0.0.1:4200/lingshu") as client:
        return await client.call_tool(tool_name, params)

async def test_image_analysis(mcp_server_url):
    """
    测试医学影像分析功能
    Test medical image analysis
    """
    print("\n🖼️  Testing medical image analysis...")
    
    # 注意：此测试需要真实的医学影像文件
    print("Note: This test requires a real medical image file")
    
    # 测试影像路径
    test_image_path = "./lung.jpeg"
    async with Client(mcp_server_url) as mcp_client:
        if os.path.exists(test_image_path):
            # 调用 analyze_medical_image 工具
            result = await query_mcp_tool(
                "analyze_medical_image",
                params= {
                    "image_path": test_image_path,
                    "analysis_type": "radiology",
                    "patient_context": "55-year-old male patient, 20-year smoking history, abnormality found in lung during physical examination",
                    "language": "en"
                }
            )
            print(result)
        else:
            print("⚠️  Test image file does not exist, skipping image analysis test")

async def main(mcp_server_url: str):
    """
    实现支持工具调用的对话功能
    1. 连接本地 vLLM 服务
    2. 获取可用工具列表并转换为 OpenAI 函数调用格式
    3. 根据用户问题调用适当的工具
    4. 整合工具结果生成最终回答
    
    Implement chat functionality with tool call support
    1. Connect to local vLLM service
    2. Get available tool list and convert to OpenAI function call format
    3. Call appropriate tools based on user questions
    4. Integrate tool results to generate final response
    """
    
    async with Client(mcp_server_url) as mcp_client:
        # 动态获取 MCP 服务提供的工具列表
        # Dynamically get the list of tools provided by the MCP service
        tools = await mcp_client.list_tools()
        
        # 将 MCP 工具 schema 转换为 OpenAI 函数调用格式
        # Convert MCP tool schemas to OpenAI function call format
        tool_schemas = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": tool.inputSchema.get("type", "object"),
                    "properties": {
                        prop_name: prop_def 
                        for prop_name, prop_def in tool.inputSchema["properties"].items()
                    },
                    "required": tool.inputSchema.get("required", [])
                }
            }
        } for tool in tools]
        
        print("--" * 20)
        print("Available tools:")
        for tool in tools:
            print(f"- {tool}\n")

        print("--" * 20)    
        # 用户查询示例
        user_query = "How to evaluate lung nodules in CT images? What imaging features should be noted?"

        # 第一次调用模型，让其决定是否需要调用工具
        # First call to the model, allowing it to decide if tool calls are needed
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_query}],
            tools=tool_schemas,
            tool_choice="auto"  # 让模型自动选择工具
        )
        
        # 处理工具调用请求
        # Handle tool call requests
        message = response.choices[0].message
        print(message.tool_calls)

        if message.tool_calls:
            print("Tool call request detected:")

            # 按顺序执行模型请求的所有工具
            # Execute all tools requested by the model in order
            for call in message.tool_calls:
                print(f"Executing {call.function.name}...")
                # 调用 MCP 工具并获取结果
                # Call MCP tool and get results
                result = await query_mcp_tool(
                    call.function.name,
                    eval(call.function.arguments)  # 将参数字符串转换为字典
                )
                print(f"Tool returned: {result}")
        else:
            # 如果模型认为不需要工具，直接返回模型回复
            # If the model decides no tools are needed, return the model's reply directly
            print("Direct reply:", message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu 32B FastMCP Client Test Script")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:4200/lingshu",
                        help="MCP server URL (default: http://127.0.0.1:4200/lingshu)")
    args = parser.parse_args()
    
    # 运行主流程或影像分析测试
    # asyncio.run(main(args.mcp_url))
    asyncio.run(test_image_analysis(args.mcp_url))
