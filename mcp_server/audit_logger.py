"""增强审计日志模块 - 记录系统关键事件用于审计和分析"""

import sqlite3
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime_paths import DB_DIR


class EventType(str, Enum):
    """审计事件类型"""
    INTENT_ROUTING = "intent_routing"
    FILE_OPERATION = "file_operation"
    AUTH_EVENT = "auth_event"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    RL_TRAINING = "rl_training"
    RL_INFERENCE = "rl_inference"


class Severity(str, Enum):
    """事件严重级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEvent:
    """审计事件数据结构"""
    event_type: str
    severity: str
    user_id: Optional[str]
    details: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class AuditLogger:
    """增强审计日志器 - 支持多种事件类型的记录和查询"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DB_DIR / "audit.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化审计数据库"""
        conn = sqlite3.connect(self.db_path)
        
        # 审计事件表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                user_id TEXT,
                details TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 意图路由审计表（专用表，便于查询）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_routing_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                predicted_intent TEXT NOT NULL,
                expected_intent TEXT,
                confidence REAL NOT NULL,
                decision_path TEXT,
                is_correct INTEGER,
                latency_ms INTEGER,
                user_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # 文件操作审计表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_operation_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                target_path TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                user_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # RL训练审计表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_training_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                epsilon REAL NOT NULL,
                training_steps INTEGER NOT NULL,
                loss REAL,
                duration_ms INTEGER,
                timestamp TEXT NOT NULL
            )
        """)
        
        # 创建索引
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON audit_events(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON audit_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON audit_events(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_timestamp ON intent_routing_audit(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_intent ON intent_routing_audit(predicted_intent)")
        
        conn.commit()
        conn.close()
    
    def log(self, event: AuditEvent) -> int:
        """记录审计事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO audit_events (event_type, severity, user_id, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (event.event_type, event.severity, event.user_id, 
             json.dumps(event.details, ensure_ascii=False), event.timestamp)
        )
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return event_id
    
    def log_intent_routing(self, query: str, predicted_intent: str, 
                           confidence: float, decision_path: List[str],
                           expected_intent: str = None, latency_ms: int = None,
                           user_id: str = None) -> int:
        """记录意图路由决策"""
        timestamp = datetime.utcnow().isoformat()
        is_correct = 1 if (expected_intent and predicted_intent == expected_intent) else None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO intent_routing_audit "
            "(query, predicted_intent, expected_intent, confidence, decision_path, "
            "is_correct, latency_ms, user_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query, predicted_intent, expected_intent, confidence,
             json.dumps(decision_path, ensure_ascii=False),
             is_correct, latency_ms, user_id, timestamp)
        )
        routing_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 同时记录到通用事件表
        event = AuditEvent(
            event_type=EventType.INTENT_ROUTING,
            severity=Severity.INFO,
            user_id=user_id,
            details={
                "query": query,
                "predicted_intent": predicted_intent,
                "expected_intent": expected_intent,
                "confidence": confidence,
                "decision_path": decision_path,
                "is_correct": is_correct,
                "latency_ms": latency_ms
            },
            timestamp=timestamp
        )
        self.log(event)
        
        return routing_id
    
    def log_file_operation(self, operation: str, target_path: str, 
                           success: bool, error_message: str = None,
                           user_id: str = None) -> int:
        """记录文件操作"""
        timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO file_operation_audit "
            "(operation, target_path, success, error_message, user_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (operation, target_path, 1 if success else 0, error_message, user_id, timestamp)
        )
        op_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 通用事件
        severity = Severity.INFO if success else Severity.WARNING
        event = AuditEvent(
            event_type=EventType.FILE_OPERATION,
            severity=severity,
            user_id=user_id,
            details={
                "operation": operation,
                "target_path": target_path,
                "success": success,
                "error_message": error_message
            },
            timestamp=timestamp
        )
        self.log(event)
        
        return op_id
    
    def log_rl_training(self, episode: int, total_reward: float, 
                        epsilon: float, training_steps: int,
                        loss: float = None, duration_ms: int = None) -> int:
        """记录RL训练过程"""
        timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO rl_training_audit "
            "(episode, total_reward, epsilon, training_steps, loss, duration_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (episode, total_reward, epsilon, training_steps, loss, duration_ms, timestamp)
        )
        train_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        event = AuditEvent(
            event_type=EventType.RL_TRAINING,
            severity=Severity.INFO,
            user_id=None,
            details={
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": epsilon,
                "training_steps": training_steps,
                "loss": loss,
                "duration_ms": duration_ms
            },
            timestamp=timestamp
        )
        self.log(event)
        
        return train_id
    
    def log_security_event(self, event_type: str, details: Dict[str, Any],
                           severity: Severity = Severity.WARNING,
                           user_id: str = None) -> int:
        """记录安全事件"""
        event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            severity=severity,
            user_id=user_id,
            details={"event_type": event_type, **details}
        )
        return self.log(event)
    
    def query_events(self, event_type: str = None, user_id: str = None,
                     start_time: str = None, end_time: str = None,
                     severity: str = None, limit: int = 100) -> List[Dict]:
        """查询审计事件"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        conditions = []
        params = []
        
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = conn.execute(
            f"SELECT * FROM audit_events WHERE {where_clause} "
            f"ORDER BY timestamp DESC LIMIT ?",
            params + [limit]
        )
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_routing_statistics(self, days: int = 7) -> Dict[str, Any]:
        """获取意图路由统计"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        start_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        
        # 总体统计
        cursor = conn.execute(
            "SELECT COUNT(*) as total, "
            "AVG(confidence) as avg_confidence, "
            "AVG(latency_ms) as avg_latency, "
            "SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count "
            "FROM intent_routing_audit WHERE timestamp >= ?",
            (start_time,)
        )
        overall = dict(cursor.fetchone())
        
        # 按意图类型统计
        cursor = conn.execute(
            "SELECT predicted_intent, COUNT(*) as count, "
            "AVG(confidence) as avg_confidence, "
            "AVG(latency_ms) as avg_latency "
            "FROM intent_routing_audit WHERE timestamp >= ? "
            "GROUP BY predicted_intent",
            (start_time,)
        )
        by_intent = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        accuracy = None
        if overall['total'] > 0 and overall['correct_count'] is not None:
            accuracy = overall['correct_count'] / overall['total']
        
        return {
            "period_days": days,
            "total_routing": overall['total'],
            "accuracy": accuracy,
            "avg_confidence": overall['avg_confidence'],
            "avg_latency_ms": overall['avg_latency'],
            "by_intent": by_intent
        }
    
    def get_anomalies(self, hours: int = 24, threshold: float = 2.0) -> List[Dict]:
        """检测异常使用模式"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT * FROM audit_events "
            "WHERE timestamp >= datetime('now', ?) "
            "ORDER BY timestamp DESC",
            (f'-{hours} hours',)
        )
        
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        anomalies = []
        
        # 检测高频操作
        user_ops = {}
        for event in events:
            user_id = event.get('user_id', 'unknown')
            if user_id not in user_ops:
                user_ops[user_id] = []
            user_ops[user_id].append(event)
        
        for user_id, ops in user_ops.items():
            if len(ops) > 50:  # 阈值
                anomalies.append({
                    "type": "high_frequency",
                    "user_id": user_id,
                    "count": len(ops),
                    "message": f"用户 {user_id} 在 {hours} 小时内执行了 {len(ops)} 次操作"
                })
        
        # 检测错误事件
        error_events = [e for e in events if e.get('severity') in ['ERROR', 'CRITICAL']]
        if len(error_events) > 10:
            anomalies.append({
                "type": "high_error_rate",
                "count": len(error_events),
                "message": f"{hours} 小时内发生 {len(error_events)} 次错误事件"
            })
        
        return anomalies
    
    def export_audit_trail(self, start_date: str, end_date: str, 
                           format: str = "json") -> str:
        """导出审计轨迹"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT * FROM audit_events "
            "WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC",
            (start_date, end_date)
        )
        
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if format == "json":
            return json.dumps(events, ensure_ascii=False, indent=2)
        elif format == "csv":
            if not events:
                return ""
            headers = events[0].keys()
            lines = [",".join(headers)]
            for event in events:
                values = [str(event.get(h, "")) for h in headers]
                lines.append(",".join(values))
            return "\n".join(lines)
        
        return json.dumps(events, ensure_ascii=False, indent=2)
