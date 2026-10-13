"""量化工具 - MCP Tool实现，用于获取系统性能指标和准确率"""

import json
import os
import sys
import time
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from runtime_paths import DB_DIR, RUNTIME_ROOT

# NSP 指标的滚动历史：探针本身是即时快照，趋势要靠这份追加写
NSP_HISTORY_PATH = RUNTIME_ROOT / "nsp_history.jsonl"
NSP_HISTORY_LIMIT = 500


class MetricsTools:
    """量化工具集合 - 提供MCP Tool接口用于性能指标查询"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DB_DIR / "metrics.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化指标数据库"""
        conn = sqlite3.connect(self.db_path)
        
        # 性能指标表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                tags TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # 准确率记录表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                intent_type TEXT,
                accuracy REAL NOT NULL,
                sample_count INTEGER,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # 系统资源监控表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage REAL,
                active_connections INTEGER,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_name ON performance_metrics(metric_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_acc_test ON accuracy_records(test_name)")
        
        conn.commit()
        conn.close()
    
    def record_metric(self, name: str, value: float, tags: Dict = None):
        """记录性能指标"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO performance_metrics (metric_name, metric_value, tags, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (name, value, json.dumps(tags) if tags else None, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    
    def record_accuracy(self, test_name: str, accuracy: float, 
                        intent_type: str = None, sample_count: int = None,
                        details: Dict = None):
        """记录准确率"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO accuracy_records (test_name, intent_type, accuracy, sample_count, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (test_name, intent_type, accuracy, sample_count,
             json.dumps(details, ensure_ascii=False) if details else None,
             datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    
    def get_tool_definitions(self) -> List[Dict]:
        """获取所有量化工具的MCP定义"""
        return [
            {
                "name": "get_system_metrics",
                "description": "获取系统性能指标（CPU、内存、响应时间等）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "查询最近N小时",
                            "default": 24
                        }
                    }
                }
            },
            {
                "name": "get_intent_accuracy",
                "description": "获取意图识别准确率（按类别）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "test_name": {
                            "type": "string",
                            "description": "测试名称"
                        },
                        "days": {
                            "type": "integer",
                            "default": 7
                        }
                    }
                }
            },
            {
                "name": "get_reward_statistics",
                "description": "获取RL奖励信号统计（均值、方差、趋势）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "window": {
                            "type": "integer",
                            "description": "滑动窗口大小",
                            "default": 100
                        }
                    }
                }
            },
            {
                "name": "get_latency_distribution",
                "description": "获取响应延迟分布",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "default": 24
                        },
                        "percentiles": {
                            "type": "array",
                            "items": {"type": "number"},
                            "default": [50, 90, 95, 99]
                        }
                    }
                }
            },
            {
                "name": "get_training_progress",
                "description": "获取RL训练进度（epsilon衰减、奖励收敛等）",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "compare_model_versions",
                "description": "比较不同版本模型的性能",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "version_a": {"type": "string"},
                        "version_b": {"type": "string"}
                    }
                }
            },
            {
                "name": "export_metrics_report",
                "description": "导出性能报告",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "csv", "markdown"],
                            "default": "json"
                        },
                        "days": {
                            "type": "integer",
                            "default": 7
                        }
                    }
                }
            },
            {
                "name": "get_nsp_metrics",
                "description": "获取 NSP 潜空间量化指标：确定性探针"
                               "（NLL/KL/InfoNCE/H 步 rollout 误差/EMA 漂移）"
                               "+ 训练期滚动均值 + 历史趋势",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": ["single", "marl", "both"],
                            "description": "被测架构：single=AgentRL，marl=三专家",
                            "default": "both"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回最近 N 条历史",
                            "default": 50
                        },
                        "no_history": {
                            "type": "boolean",
                            "description": "只读不写，不追加历史记录",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "benchmark_routing",
                "description": "执行路由性能基准测试",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "iterations": {
                            "type": "integer",
                            "description": "测试迭代次数",
                            "default": 100
                        },
                        "concurrent": {
                            "type": "integer",
                            "description": "并发数",
                            "default": 1
                        }
                    }
                }
            }
        ]
    
    async def handle_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP Tool调用"""
        handlers = {
            "get_system_metrics": self._get_system_metrics,
            "get_intent_accuracy": self._get_intent_accuracy,
            "get_reward_statistics": self._get_reward_statistics,
            "get_latency_distribution": self._get_latency_distribution,
            "get_training_progress": self._get_training_progress,
            "get_nsp_metrics": self._get_nsp_metrics,
            "compare_model_versions": self._compare_model_versions,
            "export_metrics_report": self._export_metrics_report,
            "benchmark_routing": self._benchmark_routing,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = handler(params)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_system_metrics(self, params: Dict) -> Dict:
        """获取系统指标"""
        hours = params.get("hours", 24)
        
        # 获取本项目的状态
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        try:
            from agent_rl import AgentRL
            agent = AgentRL()
            
            metrics = {
                "rl_model": {
                    "epsilon": agent.epsilon,
                    "training_steps": agent.training_step,
                    "replay_buffer_size": len(agent.replay_buffer),
                    "is_trained": agent.trained
                }
            }
        except Exception as e:
            metrics = {"rl_model": {"error": str(e)}}
        
        # 获取历史指标
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        cursor = conn.execute(
            "SELECT metric_name, AVG(metric_value) as avg_value, "
            "MIN(metric_value) as min_value, MAX(metric_value) as max_value "
            "FROM performance_metrics "
            "WHERE timestamp >= ? "
            "GROUP BY metric_name",
            (start_time.isoformat(),)
        )
        
        historical = {}
        for row in cursor.fetchall():
            historical[row['metric_name']] = {
                "avg": row['avg_value'],
                "min": row['min_value'],
                "max": row['max_value']
            }
        
        conn.close()
        
        metrics["historical"] = historical
        metrics["period_hours"] = hours
        
        return metrics
    
    def _get_intent_accuracy(self, params: Dict) -> Dict:
        """获取意图准确率"""
        test_name = params.get("test_name")
        days = params.get("days", 7)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        if test_name:
            cursor = conn.execute(
                "SELECT * FROM accuracy_records "
                "WHERE test_name = ? AND timestamp >= ? "
                "ORDER BY timestamp DESC",
                (test_name, start_time.isoformat())
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM accuracy_records "
                "WHERE timestamp >= ? "
                "ORDER BY timestamp DESC",
                (start_time.isoformat(),)
            )
        
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # 按意图类型聚合
        by_intent = {}
        for record in records:
            intent = record.get('intent_type', 'overall')
            if intent not in by_intent:
                by_intent[intent] = {
                    "accuracies": [],
                    "sample_counts": []
                }
            by_intent[intent]["accuracies"].append(record['accuracy'])
            if record['sample_count']:
                by_intent[intent]["sample_counts"].append(record['sample_count'])
        
        # 计算统计值
        result = {}
        for intent, data in by_intent.items():
            accs = data['accuracies']
            result[intent] = {
                "mean_accuracy": np.mean(accs) if accs else None,
                "std_accuracy": np.std(accs) if accs else None,
                "min_accuracy": np.min(accs) if accs else None,
                "max_accuracy": np.max(accs) if accs else None,
                "total_samples": sum(data['sample_counts']) if data['sample_counts'] else None,
                "measurements": len(accs)
            }
        
        return {
            "period_days": days,
            "by_intent": result
        }
    
    def _get_reward_statistics(self, params: Dict) -> Dict:
        """获取奖励统计"""
        window = params.get("window", 100)
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        try:
            from agent_rl import AgentRL
            agent = AgentRL()
            
            rewards = agent.reward_history[-window:] if agent.reward_history else []
            
            if not rewards:
                return {"message": "No reward history available", "window": window}
            
            rewards_array = np.array(rewards)
            
            return {
                "window": window,
                "total_records": len(rewards),
                "statistics": {
                    "mean": float(np.mean(rewards_array)),
                    "std": float(np.std(rewards_array)),
                    "min": float(np.min(rewards_array)),
                    "max": float(np.max(rewards_array)),
                    "median": float(np.median(rewards_array)),
                    "last": float(rewards[-1])
                },
                "trend": {
                    "first_half_mean": float(np.mean(rewards_array[:len(rewards_array)//2])),
                    "second_half_mean": float(np.mean(rewards_array[len(rewards_array)//2:])),
                    "improvement": float(np.mean(rewards_array[len(rewards_array)//2:]) - 
                                        np.mean(rewards_array[:len(rewards_array)//2]))
                },
                "recent_rewards": rewards[-10:] if len(rewards) >= 10 else rewards
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_latency_distribution(self, params: Dict) -> Dict:
        """获取延迟分布"""
        hours = params.get("hours", 24)
        percentiles = params.get("percentiles", [50, 90, 95, 99])
        
        # 从审计日志获取延迟数据
        audit_db = os.path.join(os.path.dirname(self.db_path), "audit.db")
        
        if not os.path.exists(audit_db):
            return {"message": "Audit database not found"}
        
        conn = sqlite3.connect(audit_db)
        cursor = conn.execute(
            "SELECT latency_ms FROM intent_routing_audit "
            "WHERE timestamp >= datetime('now', ?) AND latency_ms IS NOT NULL",
            (f'-{hours} hours',)
        )
        
        latencies = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not latencies:
            return {"message": "No latency data available", "hours": hours}
        
        latencies_array = np.array(latencies)
        
        result = {
            "hours": hours,
            "sample_count": len(latencies),
            "statistics": {
                "mean_ms": float(np.mean(latencies_array)),
                "std_ms": float(np.std(latencies_array)),
                "min_ms": float(np.min(latencies_array)),
                "max_ms": float(np.max(latencies_array))
            },
            "percentiles": {}
        }
        
        for p in percentiles:
            result["percentiles"][f"p{p}"] = float(np.percentile(latencies_array, p))
        
        return result
    
    def _get_training_progress(self, params: Dict) -> Dict:
        """获取训练进度"""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        try:
            from agent_rl import AgentRL
            agent = AgentRL()
            
            # 从审计日志获取训练历史
            audit_db = os.path.join(os.path.dirname(self.db_path), "audit.db")
            
            training_history = []
            if os.path.exists(audit_db):
                conn = sqlite3.connect(audit_db)
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM rl_training_audit ORDER BY timestamp DESC LIMIT 100"
                )
                training_history = [dict(row) for row in cursor.fetchall()]
                conn.close()
            
            return {
                "current_state": {
                    "epsilon": agent.epsilon,
                    "epsilon_min": agent.epsilon_min,
                    "epsilon_decay": agent.epsilon_decay,
                    "training_steps": agent.training_step,
                    "replay_buffer_size": len(agent.replay_buffer),
                    "is_trained": agent.trained
                },
                "training_history": training_history[:10],  # 返回最近10条
                "total_episodes": len(training_history)
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ---------------------------------------------------------- NSP 潜空间
    def _nsp_history(self, limit: int) -> List[Dict]:
        if not NSP_HISTORY_PATH.exists():
            return []
        rows: List[Dict] = []
        try:
            with open(NSP_HISTORY_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return rows[-max(int(limit), 1):]

    def _append_nsp_history(self, records: List[Dict]) -> None:
        """追加一条探针快照；超上限时整体截断成最近 NSP_HISTORY_LIMIT 条。"""
        if not records:
            return
        try:
            os.makedirs(os.path.dirname(str(NSP_HISTORY_PATH)) or ".", exist_ok=True)
            with open(NSP_HISTORY_PATH, "a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total = len(self._nsp_history(NSP_HISTORY_LIMIT + len(records) + 1))
            if total > NSP_HISTORY_LIMIT:
                keep = self._nsp_history(NSP_HISTORY_LIMIT)
                tmp = str(NSP_HISTORY_PATH) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    for rec in keep:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                os.replace(tmp, NSP_HISTORY_PATH)
        except OSError:
            pass

    def _get_nsp_metrics(self, params: Dict) -> Dict:
        """NSP 潜空间量化指标。

        ``probe`` 是固定输入 → 固定输出的确定性快照（可跨机器复现、可写进
        基线）；``training`` 是训练期的滚动均值，只用于观测趋势。
        """
        target = str(params.get("agent", "both"))
        limit = int(params.get("limit", 50))
        no_history = bool(params.get("no_history", False))
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

        probe: Dict[str, Any] = {}
        training: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        if target in ("single", "both"):
            try:
                from agent_rl import AgentRL
                agent = AgentRL(marl_vote=False)
                probe["single"] = agent.nsp_probe()
                training["single"] = agent.nsp_metrics_snapshot()
            except Exception as e:
                errors["single"] = str(e)
        if target in ("marl", "both"):
            try:
                from marl_agent import create_marl_agent
                meta = create_marl_agent()
                probe["marl"] = meta.nsp_probe()
                training["marl"] = meta.nsp_metrics_snapshot()
            except Exception as e:
                errors["marl"] = str(e)

        if not no_history:
            now = datetime.utcnow().isoformat(timespec="seconds")
            self._append_nsp_history(
                [{"ts": now, "arch": arch, "probe": data}
                 for arch, data in probe.items()])

        result: Dict[str, Any] = {
            "probe": probe,
            "training": training,
            "history": self._nsp_history(limit),
        }
        if errors:
            result["errors"] = errors
        return result

    def _compare_model_versions(self, params: Dict) -> Dict:
        """比较模型版本"""
        version_a = params.get("version_a")
        version_b = params.get("version_b")
        
        # 简化实现：比较两次评估的结果
        return {
            "comparison": {
                "version_a": version_a,
                "version_b": version_b
            },
            "message": "Model version comparison requires separate evaluation runs"
        }
    
    def _export_metrics_report(self, params: Dict) -> str:
        """导出指标报告"""
        format_type = params.get("format", "json")
        days = params.get("days", 7)
        
        # 收集所有指标
        metrics = {
            "system": self._get_system_metrics({"hours": days * 24}),
            "accuracy": self._get_intent_accuracy({"days": days}),
            "rewards": self._get_reward_statistics({"window": 500}),
            "training": self._get_training_progress({})
        }
        
        if format_type == "json":
            return json.dumps(metrics, ensure_ascii=False, indent=2)
        elif format_type == "csv":
            lines = ["metric_name,metric_value,description"]
            lines.append(f"epsilon,{metrics['rl_model']['epsilon']},Current epsilon")
            lines.append(f"training_steps,{metrics['rl_model']['training_steps']},Total training steps")
            # ... 更多指标
            return "\n".join(lines)
        elif format_type == "markdown":
            md = "# AgentRL Metrics Report\n\n"
            md += f"## System\n- Epsilon: {metrics['current_state']['epsilon']}\n"
            md += f"- Training Steps: {metrics['current_state']['training_steps']}\n"
            return md
        
        return json.dumps(metrics, ensure_ascii=False, indent=2)
    
    def _benchmark_routing(self, params: Dict) -> Dict:
        """路由性能基准测试"""
        iterations = params.get("iterations", 100)
        concurrent = params.get("concurrent", 1)
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        try:
            from agent_rl import AgentRL
            from agent_state import AgentState
            import random
            
            agent = AgentRL()
            
            test_queries = [
                "添加文件 test.txt 内容为 hello",
                "删除文件 old.py",
                "修改文件 config.yaml",
                "整理文件夹",
                "搜索强化学习",
                "对比分析 a.txt 和 b.txt",
                "你好"
            ]
            
            latencies = []
            correct = 0
            total = 0
            
            for _ in range(iterations):
                query = random.choice(test_queries)
                
                state = AgentState()
                state.increment_round()
                
                start_time = time.time()
                intent, path, confidence, clarifications = agent.process(query, state)
                latency = (time.time() - start_time) * 1000  # ms
                
                latencies.append(latency)
                total += 1
            
            latencies_array = np.array(latencies)
            
            # 记录基准测试结果
            self.record_metric("benchmark_routing_latency_mean", float(np.mean(latencies_array)))
            self.record_metric("benchmark_routing_latency_p99", float(np.percentile(latencies_array, 99)))
            
            return {
                "iterations": iterations,
                "results": {
                    "mean_latency_ms": float(np.mean(latencies_array)),
                    "p50_latency_ms": float(np.percentile(latencies_array, 50)),
                    "p99_latency_ms": float(np.percentile(latencies_array, 99)),
                    "min_latency_ms": float(np.min(latencies_array)),
                    "max_latency_ms": float(np.max(latencies_array))
                },
                "throughput": {
                    "queries_per_second": 1000 / float(np.mean(latencies_array))
                }
            }
        except Exception as e:
            return {"error": str(e)}
