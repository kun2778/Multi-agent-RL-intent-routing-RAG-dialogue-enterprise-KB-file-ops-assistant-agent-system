"""审计工具 - MCP Tool实现，用于查询和分析审计日志"""

import json
import os
import sys
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.audit_logger import AuditLogger, EventType, Severity


class AuditTools:
    """审计工具集合 - 提供MCP Tool接口用于审计数据查询"""
    
    def __init__(self, audit_logger: AuditLogger = None):
        self.logger = audit_logger or AuditLogger()
    
    def get_tool_definitions(self) -> List[Dict]:
        """获取所有审计工具的MCP定义"""
        return [
            {
                "name": "query_audit_logs",
                "description": "查询审计日志，支持按类型、用户、时间范围筛选",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "事件类型: intent_routing, file_operation, auth_event, security_event, rl_training",
                            "enum": ["intent_routing", "file_operation", "auth_event", "security_event", "system_event", "rl_training", "rl_inference"]
                        },
                        "user_id": {
                            "type": "string",
                            "description": "用户ID"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "开始时间 (ISO格式: 2024-01-01T00:00:00)"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "结束时间 (ISO格式)"
                        },
                        "severity": {
                            "type": "string",
                            "description": "严重级别",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量限制",
                            "default": 100
                        }
                    }
                }
            },
            {
                "name": "get_routing_statistics",
                "description": "获取意图路由统计信息（准确率、置信度、延迟等）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "统计最近N天",
                            "default": 7
                        }
                    }
                }
            },
            {
                "name": "get_routing_accuracy_by_intent",
                "description": "获取每种意图类型的路由准确率",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "default": 7
                        }
                    }
                }
            },
            {
                "name": "detect_anomalies",
                "description": "检测异常使用模式（高频操作、错误激增等）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "检测最近N小时",
                            "default": 24
                        },
                        "threshold": {
                            "type": "number",
                            "description": "异常阈值（标准差倍数）",
                            "default": 2.0
                        }
                    }
                }
            },
            {
                "name": "get_security_events",
                "description": "获取安全事件日志",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "default": 24
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["WARNING", "ERROR", "CRITICAL"]
                        }
                    }
                }
            },
            {
                "name": "export_audit_trail",
                "description": "导出完整审计轨迹（支持JSON/CSV格式）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "开始日期 (YYYY-MM-DD)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "结束日期 (YYYY-MM-DD)"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "csv"],
                            "default": "json"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_file_operation_summary",
                "description": "获取文件操作审计摘要",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "default": 7
                        },
                        "user_id": {
                            "type": "string"
                        }
                    }
                }
            },
            {
                "name": "analyze_routing_errors",
                "description": "分析路由错误模式，找出常见错误类型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "default": 7
                        }
                    }
                }
            },
            # OCR相关工具
            {
                "name": "ocr_recognize_image",
                "description": "OCR识别图像中的文字（支持本地文件路径或base64图像数据）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "图像文件路径"
                        },
                        "image_base64": {
                            "type": "string",
                            "description": "Base64编码的图像数据"
                        },
                        "provider": {
                            "type": "string",
                            "description": "OCR提供商: baidu, tencent, tesseract",
                            "enum": ["baidu", "tencent", "tesseract"]
                        }
                    }
                }
            },
            {
                "name": "get_ocr_status",
                "description": "获取OCR服务状态和可用的提供商",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "batch_ocr_recognize",
                "description": "批量OCR识别多个图像文件",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "图像文件路径列表"
                        },
                        "provider": {
                            "type": "string",
                            "description": "OCR提供商"
                        }
                    },
                    "required": ["image_paths"]
                }
            }
        ]
    
    async def handle_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP Tool调用"""
        handlers = {
            "query_audit_logs": self._query_audit_logs,
            "get_routing_statistics": self._get_routing_statistics,
            "get_routing_accuracy_by_intent": self._get_routing_accuracy_by_intent,
            "detect_anomalies": self._detect_anomalies,
            "get_security_events": self._get_security_events,
            "export_audit_trail": self._export_audit_trail,
            "get_file_operation_summary": self._get_file_operation_summary,
            "analyze_routing_errors": self._analyze_routing_errors,
            # OCR工具
            "ocr_recognize_image": self._ocr_recognize_image,
            "get_ocr_status": self._get_ocr_status,
            "batch_ocr_recognize": self._batch_ocr_recognize,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = handler(params)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]}
        except Exception as e:
            return {"error": str(e)}
    
    def _query_audit_logs(self, params: Dict) -> List[Dict]:
        """查询审计日志"""
        return self.logger.query_events(
            event_type=params.get("event_type"),
            user_id=params.get("user_id"),
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            severity=params.get("severity"),
            limit=params.get("limit", 100)
        )
    
    def _get_routing_statistics(self, params: Dict) -> Dict:
        """获取路由统计"""
        return self.logger.get_routing_statistics(params.get("days", 7))
    
    def _get_routing_accuracy_by_intent(self, params: Dict) -> Dict:
        """获取每种意图的准确率"""
        import sqlite3
        days = params.get("days", 7)
        
        conn = sqlite3.connect(self.logger.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        
        # 按意图类型统计准确率
        cursor = conn.execute(
            "SELECT predicted_intent, "
            "COUNT(*) as total, "
            "SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct, "
            "AVG(confidence) as avg_confidence, "
            "AVG(latency_ms) as avg_latency "
            "FROM intent_routing_audit "
            "WHERE timestamp >= ? AND is_correct IS NOT NULL "
            "GROUP BY predicted_intent",
            (start_time,)
        )
        
        results = {}
        for row in cursor.fetchall():
            intent = row['predicted_intent']
            total = row['total']
            correct = row['correct'] or 0
            results[intent] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else None,
                "avg_confidence": row['avg_confidence'],
                "avg_latency_ms": row['avg_latency']
            }
        
        conn.close()
        return {"period_days": days, "by_intent": results}
    
    def _detect_anomalies(self, params: Dict) -> List[Dict]:
        """检测异常"""
        return self.logger.get_anomalies(
            hours=params.get("hours", 24),
            threshold=params.get("threshold", 2.0)
        )
    
    def _get_security_events(self, params: Dict) -> List[Dict]:
        """获取安全事件"""
        return self.logger.query_events(
            event_type=EventType.SECURITY_EVENT,
            severity=params.get("severity", Severity.WARNING),
            limit=params.get("limit", 100)
        )
    
    def _export_audit_trail(self, params: Dict) -> str:
        """导出审计轨迹"""
        return self.logger.export_audit_trail(
            start_date=params["start_date"],
            end_date=params["end_date"],
            format=params.get("format", "json")
        )
    
    def _get_file_operation_summary(self, params: Dict) -> Dict:
        """获取文件操作摘要"""
        import sqlite3
        
        days = params.get("days", 7)
        user_id = params.get("user_id")
        
        conn = sqlite3.connect(self.logger.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        
        conditions = ["timestamp >= ?"]
        params_list = [start_time]
        
        if user_id:
            conditions.append("user_id = ?")
            params_list.append(user_id)
        
        where_clause = " AND ".join(conditions)
        
        # 按操作类型统计
        cursor = conn.execute(
            f"SELECT operation, COUNT(*) as count, "
            f"SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count "
            f"FROM file_operation_audit "
            f"WHERE {where_clause} "
            f"GROUP BY operation",
            params_list
        )
        
        by_operation = {}
        for row in cursor.fetchall():
            op = row['operation']
            total = row['count']
            success = row['success_count'] or 0
            by_operation[op] = {
                "total": total,
                "success": success,
                "success_rate": success / total if total > 0 else 0
            }
        
        # 总体统计
        cursor = conn.execute(
            f"SELECT COUNT(*) as total, "
            f"SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count "
            f"FROM file_operation_audit WHERE {where_clause}",
            params_list
        )
        
        overall = dict(cursor.fetchone())
        conn.close()
        
        return {
            "period_days": days,
            "total_operations": overall['total'],
            "overall_success_rate": (overall['success_count'] or 0) / overall['total'] if overall['total'] > 0 else None,
            "by_operation": by_operation
        }
    
    def _analyze_routing_errors(self, params: Dict) -> Dict:
        """分析路由错误"""
        import sqlite3
        
        days = params.get("days", 7)
        
        conn = sqlite3.connect(self.logger.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        
        # 获取所有错误路由
        cursor = conn.execute(
            "SELECT query, predicted_intent, expected_intent, confidence "
            "FROM intent_routing_audit "
            "WHERE timestamp >= ? AND is_correct = 0",
            (start_time,)
        )
        
        errors = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # 分析错误模式
        error_patterns = {}
        for error in errors:
            key = f"{error['expected_intent']} -> {error['predicted_intent']}"
            if key not in error_patterns:
                error_patterns[key] = {
                    "count": 0,
                    "examples": [],
                    "avg_confidence": []
                }
            error_patterns[key]['count'] += 1
            error_patterns[key]['examples'].append(error['query'][:50])
            error_patterns[key]['avg_confidence'].append(error['confidence'])
        
        # 计算平均置信度
        for pattern in error_patterns.values():
            confs = pattern['avg_confidence']
            pattern['avg_confidence'] = sum(confs) / len(confs) if confs else 0
        
        # 按频率排序
        sorted_patterns = sorted(error_patterns.items(), 
                                 key=lambda x: x[1]['count'], 
                                 reverse=True)
        
        return {
            "period_days": days,
            "total_errors": len(errors),
            "error_patterns": dict(sorted_patterns[:10])  # 返回前10个模式
        }
    
    def _ocr_recognize_image(self, params: Dict) -> Dict:
        """OCR识别图像"""
        import base64
        from mcp_server.ocr_module import get_ocr_manager, OCRResult
        
        image_path = params.get("image_path")
        image_base64 = params.get("image_base64")
        provider = params.get("provider")
        
        if not image_path and not image_base64:
            return {"error": "Either image_path or image_base64 is required"}
        
        manager = get_ocr_manager()
        
        # 如果提供了base64，解码为bytes
        image_bytes = None
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
        
        try:
            result = manager.recognize(
                image_path=image_path,
                image_bytes=image_bytes,
                provider=provider
            )
            
            if result is None:
                return {
                    "error": "OCR recognition failed",
                    "available_providers": manager.get_available_providers()
                }
            
            return {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "processing_time_ms": result.processing_time_ms,
                "provider_used": provider or manager.config.get("provider", "auto")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_ocr_status(self, params: Dict) -> Dict:
        """获取OCR服务状态"""
        from mcp_server.ocr_module import get_ocr_manager
        
        manager = get_ocr_manager()
        
        providers_status = {}
        for name, provider in manager.providers.items():
            providers_status[name] = {
                "configured": provider.is_configured(),
                "available": provider.is_configured()
            }
        
        return {
            "configured_provider": manager.config.get("provider", "tesseract"),
            "providers": providers_status,
            "available_providers": manager.get_available_providers(),
            "supported_formats": manager.config.get("supported_formats", [])
        }
    
    def _batch_ocr_recognize(self, params: Dict) -> Dict:
        """批量OCR识别"""
        from mcp_server.ocr_module import get_ocr_manager
        
        image_paths = params.get("image_paths", [])
        provider = params.get("provider")
        
        if not image_paths:
            return {"error": "image_paths is required"}
        
        manager = get_ocr_manager()
        results = []
        
        for path in image_paths:
            try:
                result = manager.recognize(image_path=path, provider=provider)
                results.append({
                    "file": path,
                    "success": True,
                    "text": result.text if result else "",
                    "confidence": result.confidence if result else 0
                })
            except Exception as e:
                results.append({
                    "file": path,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total_files": len(image_paths),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
