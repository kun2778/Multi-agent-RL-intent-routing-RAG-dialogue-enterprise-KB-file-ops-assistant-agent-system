"""MCP客户端示例 - 在代码中调用AgentRL Audit MCP Server

这个客户端展示了如何在Python代码中作为MCP客户端
调用AgentRL Audit MCP Server提供的审计和量化工具。
"""

import json
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MCPClient:
    """MCP客户端 - 用于调用MCP Server"""
    
    def __init__(self, server_url: str = None, use_stdio: bool = True):
        """
        初始化MCP客户端
        
        Args:
            server_url: SSE模式下的服务器URL
            use_stdio: 是否使用stdio模式（默认True）
        """
        self.server_url = server_url
        self.use_stdio = use_stdio
        self._server_process = None
        self._reader = None
        self._writer = None
    
    async def connect_stdio(self):
        """通过stdio连接到MCP Server"""
        if not self.use_stdio:
            raise ValueError("Not configured for stdio mode")
        
        # 启动服务器进程
        import subprocess
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mcp_server", "server.py"
        )
        
        self._server_process = subprocess.Popen(
            [sys.executable, "-m", "mcp_server.server", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        return True
    
    async def connect_sse(self, url: str = None):
        """通过SSE连接到MCP Server"""
        try:
            import httpx
            
            url = url or self.server_url
            if not url:
                raise ValueError("SSE URL not provided")
            
            # 测试连接
            async with httpx.AsyncClient() as client:
                response = await client.get(url.replace("/sse", "/health"))
                if response.status_code == 200:
                    return True
            
            return False
        except ImportError:
            print("httpx not installed. Run: pip install httpx")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self._server_process:
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None
    
    async def list_tools(self) -> List[Dict]:
        """列出所有可用工具"""
        # 直接导入工具定义
        from mcp_server.tools.audit_tools import AuditTools
        from mcp_server.tools.metrics_tools import MetricsTools
        from mcp_server.tools.rl_insights import RLInsightsTools
        
        all_tools = []
        all_tools.extend(AuditTools().get_tool_definitions())
        all_tools.extend(MetricsTools().get_tool_definitions())
        all_tools.extend(RLInsightsTools().get_tool_definitions())
        
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict:
        """调用指定工具"""
        # 根据工具名称分发到对应的处理器
        from mcp_server.tools.audit_tools import AuditTools
        from mcp_server.tools.metrics_tools import MetricsTools
        from mcp_server.tools.rl_insights import RLInsightsTools
        
        handlers = {
            "audit": AuditTools(),
            "metrics": MetricsTools(),
            "rl": RLInsightsTools()
        }
        
        # 确定工具前缀
        for prefix, handler in handlers.items():
            tool_names = [t["name"] for t in handler.get_tool_definitions()]
            if tool_name in tool_names:
                return await handler.handle_tool(tool_name, arguments)
        
        return {"error": f"Tool not found: {tool_name}"}


class SyncMCPClient:
    """同步MCP客户端 - 方便在非异步代码中使用"""
    
    def __init__(self):
        from mcp_server.tools.audit_tools import AuditTools
        from mcp_server.tools.metrics_tools import MetricsTools
        from mcp_server.tools.rl_insights import RLInsightsTools
        
        self.audit_tools = AuditTools()
        self.metrics_tools = MetricsTools()
        self.rl_insights = RLInsightsTools()
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        all_tools = []
        all_tools.extend(self.audit_tools.get_tool_definitions())
        all_tools.extend(self.metrics_tools.get_tool_definitions())
        all_tools.extend(self.rl_insights.get_tool_definitions())
        return all_tools
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict:
        """调用工具"""
        handlers = [
            self.audit_tools,
            self.metrics_tools,
            self.rl_insights
        ]
        
        for handler in handlers:
            tool_names = [t["name"] for t in handler.get_tool_definitions()]
            if tool_name in tool_names:
                # 同步调用异步方法
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(handler.handle_tool(tool_name, arguments))
                    return result
                finally:
                    loop.close()
        
        return {"error": f"Tool not found: {tool_name}"}


# ============================================================
# 使用示例
# ============================================================

def example_audit_query():
    """示例：查询审计日志"""
    print("\n=== 示例：查询审计日志 ===")
    
    client = SyncMCPClient()
    
    # 查询最近7天的审计日志
    result = client.call_tool("query_audit_logs", {
        "event_type": "intent_routing",
        "limit": 5
    })
    
    print("查询结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_routing_statistics():
    """示例：获取路由统计"""
    print("\n=== 示例：获取路由统计 ===")
    
    client = SyncMCPClient()
    
    result = client.call_tool("get_routing_statistics", {
        "days": 7
    })
    
    print("路由统计：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_rl_insights():
    """示例：获取RL洞察"""
    print("\n=== 示例：获取RL洞察 ===")
    
    client = SyncMCPClient()
    
    # 解释路由决策
    result = client.call_tool("explain_routing_decision", {
        "query": "添加文件 notes.txt 内容为 hello world"
    })
    
    print("路由决策解释：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_model_report():
    """示例：生成模型报告"""
    print("\n=== 示例：生成模型报告 ===")
    
    client = SyncMCPClient()
    
    result = client.call_tool("generate_model_report", {})
    
    print("模型报告：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_benchmark():
    """示例：执行基准测试"""
    print("\n=== 示例：执行基准测试 ===")
    
    client = SyncMCPClient()
    
    result = client.call_tool("benchmark_routing", {
        "iterations": 50
    })
    
    print("基准测试结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_ocr_recognize():
    """示例：OCR识别图像"""
    print("\n=== 示例：OCR识别图像 ===")
    
    client = SyncMCPClient()
    
    # 检查OCR状态
    status_result = client.call_tool("get_ocr_status", {})
    print("OCR服务状态：")
    print(json.dumps(status_result, ensure_ascii=False, indent=2))
    
    # 注意：要使用OCR功能，需要配置OCR API Key
    # 1. 在.env文件中配置BAIDU_OCR_API_KEY和BAIDU_OCR_SECRET_KEY
    # 2. 或者安装tesseract-ocr（免费本地OCR）
    
    # 示例：识别图像（需要提供图像路径）
    # result = client.call_tool("ocr_recognize_image", {
    #     "image_path": "path/to/your/image.png",
    #     "provider": "baidu"  # 或 "tencent", "tesseract"
    # })
    # print("OCR识别结果：")
    # print(json.dumps(result, ensure_ascii=False, indent=2))


def example_batch_ocr():
    """示例：批量OCR识别"""
    print("\n=== 示例：批量OCR识别 ===")
    
    client = SyncMCPClient()
    
    # 示例：批量识别多个图像
    # result = client.call_tool("batch_ocr_recognize", {
    #     "image_paths": [
    #         "path/to/image1.png",
    #         "path/to/image2.jpg",
    #         "path/to/image3.png"
    #     ],
    #     "provider": "tesseract"
    # })
    # print("批量OCR结果：")
    # print(json.dumps(result, ensure_ascii=False, indent=2))


def list_all_tools():
    """列出所有可用工具"""
    print("\n=== 所有可用MCP工具 ===\n")
    
    client = SyncMCPClient()
    tools = client.list_tools()
    
    # 按类别分组
    categories = {
        "审计工具 (Audit)": [],
        "量化工具 (Metrics)": [],
        "RL洞察工具 (RL Insights)": [],
        "OCR工具": []
    }
    
    for tool in tools:
        name = tool["name"]
        if name.startswith("ocr") or name.startswith("batch_ocr") or name.startswith("get_ocr"):
            categories["OCR工具"].append(tool)
        elif name.startswith("query") or name.startswith("get_routing") or \
           name.startswith("detect") or name.startswith("get_security") or \
           name.startswith("export_audit") or name.startswith("get_file") or \
           name.startswith("analyze_routing"):
            categories["审计工具 (Audit)"].append(tool)
        elif name.startswith("get_system") or name.startswith("get_intent") or \
             name.startswith("get_reward") or name.startswith("get_latency") or \
             name.startswith("get_training") or name.startswith("compare") or \
             name.startswith("export_metrics") or name.startswith("benchmark"):
            categories["量化工具 (Metrics)"].append(tool)
        else:
            categories["RL洞察工具 (RL Insights)"].append(tool)
    
    for category, tool_list in categories.items():
        if tool_list:
            print(f"\n{category}:")
            print("-" * 50)
            for tool in tool_list:
                print(f"  - {tool['name']}")
                print(f"    {tool['description'][:60]}...")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("AgentRL MCP客户端示例")
    print("=" * 60)
    
    list_all_tools()
    
    # 运行示例（注意：需要先启动MCP Server）
    print("\n" + "=" * 60)
    print("注意：以下示例需要先启动MCP Server")
    print("运行命令: python -m mcp_server.server --transport stdio")
    print("=" * 60)
    
    try:
        example_audit_query()
        example_routing_statistics()
        example_rl_insights()
        example_model_report()
        example_benchmark()
        example_ocr_recognize()
        example_batch_ocr()
    except Exception as e:
        print(f"\n示例执行出错: {e}")
        print("请确保MCP Server已启动且数据库已初始化")


if __name__ == "__main__":
    main()
