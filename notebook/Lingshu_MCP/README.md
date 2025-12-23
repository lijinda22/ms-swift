# 灵枢 FastMCP 医学 AI 服务 (Lingshu FastMCP Medical AI Service)

本项目为灵枢（Lingshu）医学 AI 模型实现了 FastMCP 服务器，并提供了一个用于测试和集成的客户端。

## 组件 (Components)

1. `mcp_server_lingshu.py`: 封装了灵枢模型的 FastMCP 服务器
2. `mcp_client_lingshu.py`: 演示如何与灵枢 FastMCP 服务器交互的测试客户端

## 服务器功能 (Server Features)

- 医学影像分析 (Medical image analysis)
- 结构化医学报告生成 (Structured medical report generation)
- 医学问答 (Medical Q&A)

## 前置条件 (Prerequisites)

- FastMCP 框架
- 兼容 OpenAI API 的 LLM 服务器 (例如 vLLM)
- 必要的 Python 包 (通过 `pip install -r requirements.txt` 安装)

## 安装 (Setup)

1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`

## 使用说明 (Usage)

### 1. 使用 vLLM 部署灵枢模型 (Use vLLM to serve the Lingshu Model)

```bash
vllm serve lingshu-medical-mllm/Lingshu-7B  --dtype float16 --api_key api_key --port 8000  --max-model-len 32768
```

### 2. 启动 FastMCP 服务器 (Wrap the server with FastMCP)

```bash
# 设置环境变量以连接到 vLLM 服务器
export LINGSHU_SERVER_URL="http://localhost:8000/v1" 
export LINGSHU_SERVER_API="api_key"
export LINGSHU_MODEL="lingshu-medical-mllm/Lingshu-7B" # 这里的配置取决于你的 vllm 服务器设置

# 运行服务器
python mcp_server_lingshu.py --host 127.0.0.1 --port 4200 --path /lingshu --log-level info
```

### 3. 使用客户端连接 (Try connecting Lingshu with MCP)

```bash
# 设置客户端所需的 LLM 服务器环境变量 (用于调用 MCP 工具的主模型)
export LLM_SERVER_URL="xxx"
export LLM_SERVER_API="xxx"
export LLM_MODEL="xxx" ## 你的主模型名称

# 运行客户端测试
python mcp_client_lingshu.py  --mcp-url http://127.0.0.1:4200/lingshu # mcp-url 应取决于上一步部署的服务器地址
```
