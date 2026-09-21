"""AgentRL Audit MCP Server - 审计与量化MCP服务入口

支持两种传输方式:
1. stdio: 标准输入输出（适合本地集成）
2. sse: Server-Sent Events（适合远程访问）
"""

import json
import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional

# MCP协议实现
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, ImageContent, EmbeddedResource,
    ServerCapabilities, ToolCapability, ResourceCapability
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tools.audit_tools import AuditTools
from mcp_server.tools.metrics_tools import MetricsTools
from mcp_server.tools.rl_insights import RLInsightsTools
from mcp_server.audit_logger import AuditLogger

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentrl-mcp")


class AgentRLMCPServer:
    """AgentRL审计量化MCP Server"""
    
    def __init__(self):
        self.server = Server("agentrl-audit-mcp")
        
        # 初始化工具
        self.audit_tools = AuditTools()
        self.metrics_tools = MetricsTools()
        self.rl_insights = RLInsightsTools()
        
        # 注册处理函数
        self._setup_handlers()
    
    def _setup_handlers(self):
        """设置MCP协议处理器"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """列出所有可用工具"""
            tools = []
            
            # 审计工具
            for tool_def in self.audit_tools.get_tool_definitions():
                tools.append(Tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["inputSchema"]
                ))
            
            # 量化工具
            for tool_def in self.metrics_tools.get_tool_definitions():
                tools.append(Tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["inputSchema"]
                ))
            
            # RL洞察工具
            for tool_def in self.rl_insights.get_tool_definitions():
                tools.append(Tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["inputSchema"]
                ))
            
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
            """调用指定工具"""
            logger.info(f"Calling tool: {name} with args: {arguments}")
            
            # 尝试所有工具处理器
            handlers = [
                self.audit_tools,
                self.metrics_tools,
                self.rl_insights
            ]
            
            for handler in handlers:
                tool_names = [t["name"] for t in handler.get_tool_definitions()]
                if name in tool_names:
                    result = await handler.handle_tool(name, arguments)
                    
                    if "error" in result:
                        return [TextContent(
                            type="text",
                            text=json.dumps({"error": result["error"]}, ensure_ascii=False)
                        )]
                    
                    return result.get("content", [TextContent(type="text", text="No content")])
            
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Tool not found: {name}"}, ensure_ascii=False)
            )]
        
        @self.server.list_resources()
        async def list_resources():
            """列出可用资源"""
            return []
    
    async def run_stdio(self):
        """以stdio模式运行"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
    
    async def run_sse(self, host: str = "0.0.0.0", port: int = 3000):
        """以SSE模式运行"""
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route, Mount
            import uvicorn
            
            sse = SseServerTransport("/messages/")
            
            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope, request.receive, request.send
                ) as streams:
                    await self.server.run(
                        streams[0], streams[1],
                        self.server.create_initialization_options()
                    )
            
            app = Starlette(
                routes=[
                    Route("/sse", endpoint=handle_sse),
                    Mount("/messages/", app=sse.handle_post_message),
                ]
            )
            
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            logger.error("SSE transport requires additional dependencies. Install with: pip install mcp[sse]")
            raise
    
    def get_capabilities(self) -> ServerCapabilities:
        """返回服务器能力"""
        return ServerCapabilities(
            tools=ToolCapability(),
            resources=ResourceCapability()
        )


async def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentRL Audit MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport type (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE mode")
    parser.add_argument("--port", type=int, default=3000, help="Port for SSE mode")
    
    args = parser.parse_args()
    
    server = AgentRLMCPServer()
    
    logger.info(f"Starting AgentRL MCP Server (transport: {args.transport})")
    
    if args.transport == "stdio":
        await server.run_stdio()
    else:
        await server.run_sse(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
