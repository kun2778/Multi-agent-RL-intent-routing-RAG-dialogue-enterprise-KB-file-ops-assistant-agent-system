"""RL洞察工具 - MCP Tool实现，用于分析RL模型决策过程和训练状态"""

import json
import os
import sys
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RLInsightsTools:
    """RL洞察工具集合 - 提供MCP Tool接口用于RL模型分析"""
    
    def __init__(self):
        pass
    
    def get_tool_definitions(self) -> List[Dict]:
        """获取所有RL洞察工具的MCP定义"""
        return [
            {
                "name": "analyze_convergence",
                "description": "分析RL训练收敛状态（是否过拟合/欠拟合/正常收敛）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "window_size": {
                            "type": "integer",
                            "description": "分析窗口大小",
                            "default": 50
                        }
                    }
                }
            },
            {
                "name": "explain_routing_decision",
                "description": "解释某条输入的路由决策过程（特征提取->分类->仲裁）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要分析的输入文本"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "visualize_q_values",
                "description": "可视化Q值分布（输出JSON格式数据）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sample_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "示例查询列表（可选）"
                        }
                    }
                }
            },
            {
                "name": "get_feature_importance",
                "description": "分析特征重要性（哪些特征影响决策最大）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "num_samples": {
                            "type": "integer",
                            "default": 100
                        }
                    }
                }
            },
            {
                "name": "analyze_exploration_exploitation",
                "description": "分析探索-利用平衡状态",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_decision_boundary",
                "description": "分析决策边界（不同意图之间的区分度）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "intent_pair": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要分析的意图对，如 ['add_file', 'delete_file']"
                        }
                    }
                }
            },
            {
                "name": "analyze_error_patterns",
                "description": "分析错误模式（哪些输入容易被误分类）",
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
                "name": "generate_model_report",
                "description": "生成完整的RL模型分析报告",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP Tool调用"""
        handlers = {
            "analyze_convergence": self._analyze_convergence,
            "explain_routing_decision": self._explain_routing_decision,
            "visualize_q_values": self._visualize_q_values,
            "get_feature_importance": self._get_feature_importance,
            "analyze_exploration_exploitation": self._analyze_exploration_exploitation,
            "get_decision_boundary": self._get_decision_boundary,
            "analyze_error_patterns": self._analyze_error_patterns,
            "generate_model_report": self._generate_model_report,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = handler(params)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]}
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_convergence(self, params: Dict) -> Dict:
        """分析训练收敛状态"""
        window_size = params.get("window_size", 50)
        
        try:
            from agent_rl import AgentRL
            agent = AgentRL()
            
            if not agent.reward_history or len(agent.reward_history) < window_size:
                return {
                    "status": "insufficient_data",
                    "message": f"需要至少 {window_size} 条奖励记录",
                    "current_records": len(agent.reward_history) if agent.reward_history else 0
                }
            
            rewards = np.array(agent.reward_history)
            
            # 分段分析
            n_segments = 4
            segment_size = len(rewards) // n_segments
            segments = []
            
            for i in range(n_segments):
                start = i * segment_size
                end = start + segment_size
                segment = rewards[start:end]
                segments.append({
                    "mean": float(np.mean(segment)),
                    "std": float(np.std(segment)),
                    "trend": "increasing" if i > 0 and np.mean(segment) > np.mean(segments[-1]) else "stable"
                })
            
            # 计算滑动平均
            window = min(window_size, len(rewards) // 4)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            # 收敛判断
            recent_std = float(np.std(rewards[-window:]))
            overall_std = float(np.std(rewards))
            trend = segments[-1]["mean"] - segments[0]["mean"]
            
            # 判断状态
            if recent_std < 0.1 * abs(segments[-1]["mean"]):
                status = "converged"
                description = "模型已收敛，奖励波动很小"
            elif trend > 0 and recent_std < overall_std:
                status = "converging"
                description = "模型正在收敛，奖励呈上升趋势"
            elif agent.epsilon > agent.epsilon_min * 2:
                status = "exploring"
                description = f"模型仍在探索阶段 (ε={agent.epsilon:.4f})"
            else:
                status = "diverging"
                description = "模型可能发散，需要检查超参数"
            
            return {
                "status": status,
                "description": description,
                "metrics": {
                    "current_epsilon": agent.epsilon,
                    "epsilon_min": agent.epsilon_min,
                    "training_steps": agent.training_step,
                    "total_rewards": len(rewards),
                    "recent_mean": segments[-1]["mean"],
                    "overall_mean": float(np.mean(rewards)),
                    "trend": trend
                },
                "segments": segments,
                "moving_average": moving_avg[-10:].tolist() if len(moving_avg) >= 10 else moving_avg.tolist()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _explain_routing_decision(self, params: Dict) -> Dict:
        """解释路由决策过程"""
        query = params.get("query", "")
        
        if not query:
            return {"error": "query is required"}
        
        try:
            from agent_rl import AgentRL
            from agent_state import AgentState
            import torch
            import torch.nn.functional as F
            
            agent = AgentRL()
            
            # 创建状态
            state = AgentState()
            state.corpus_size = 10
            state.increment_round()
            
            # 1. 特征提取
            features = agent._extract_features(query, state)
            semantic_features = agent.feature_extractor.extract(query)
            state_vector = state.get_state_vector()
            
            # 2. 分类头预测
            cls_probs = agent._classify_probs(features)
            cls_action = int(cls_probs.argmax())
            cls_confidence = float(cls_probs[cls_action])
            
            # 3. Q网络预测
            with torch.no_grad():
                q_values = agent.q_network(torch.FloatTensor(features).unsqueeze(0))
                q_probs = F.softmax(q_values, dim=-1).squeeze(0).numpy()
                q_action = int(q_values.argmax())
                q_confidence = float(q_probs[q_action])
            
            # 4. 规则匹配
            top_op = agent.query_rewriter.top_operation(query)
            rule_intent = None
            try:
                from agent_state import IntentType
                rule_intent = IntentType(top_op) if top_op != "vague" else None
            except:
                pass
            
            # 5. 置信度仲裁
            intent, confidence = agent.classify_intent(query, state)
            
            # 构建决策路径
            intent_labels = {
                0: "add_file", 1: "delete_file", 2: "modify_file",
                3: "refactor", 4: "query", 5: "deep_search", 6: "vague"
            }
            
            decision_path = [
                {"step": "semantic_extraction", "features_dim": len(semantic_features)},
                {"step": "classification_head", "predicted": intent_labels[cls_action], "confidence": cls_confidence},
                {"step": "q_network", "predicted": intent_labels[q_action], "confidence": q_confidence},
                {"step": "rule_matching", "matched": top_op, "intent": rule_intent.value if rule_intent else None},
                {"step": "final_arbitration", "intent": intent.value, "confidence": confidence}
            ]
            
            # Q值详情
            q_values_detail = {}
            for i, label in intent_labels.items():
                q_values_detail[label] = {
                    "q_value": float(q_values[0][i]),
                    "probability": float(q_probs[i])
                }
            
            return {
                "query": query,
                "final_decision": {
                    "intent": intent.value,
                    "confidence": confidence
                },
                "decision_path": decision_path,
                "q_values": q_values_detail,
                "classification_probabilities": {
                    intent_labels[i]: float(cls_probs[i]) for i in range(len(cls_probs))
                },
                "semantic_features_norm": float(np.linalg.norm(semantic_features))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _visualize_q_values(self, params: Dict) -> Dict:
        """可视化Q值分布"""
        sample_queries = params.get("sample_queries", [
            "添加文件 test.txt",
            "删除文件 old.py",
            "修改配置文件",
            "整理文件夹",
            "搜索相关信息",
            "分析文档差异",
            "你好"
        ])
        
        try:
            from agent_rl import AgentRL
            from agent_state import AgentState
            import torch
            
            agent = AgentRL()
            intent_labels = ["add_file", "delete_file", "modify_file", 
                           "refactor", "query", "deep_search", "vague"]
            
            results = []
            q_matrix = []
            
            for query in sample_queries:
                state = AgentState()
                state.corpus_size = 10
                state.increment_round()
                
                features = agent._extract_features(query, state)
                
                with torch.no_grad():
                    q_values = agent.q_network(torch.FloatTensor(features).unsqueeze(0))
                
                q_vals = q_values.squeeze().numpy()
                q_matrix.append(q_vals)
                
                results.append({
                    "query": query[:30],
                    "q_values": {intent_labels[i]: float(q_vals[i]) for i in range(len(q_vals))},
                    "best_action": intent_labels[int(q_vals.argmax())]
                })
            
            # 转置矩阵用于热力图
            q_matrix_array = np.array(q_matrix)
            
            return {
                "queries": results,
                "heatmap_data": {
                    "rows": [q[:20] for q in sample_queries],
                    "columns": intent_labels,
                    "values": q_matrix_array.tolist()
                },
                "summary": {
                    "mean_q_per_intent": {intent_labels[i]: float(q_matrix_array[:, i].mean()) 
                                         for i in range(len(intent_labels))},
                    "std_q_per_intent": {intent_labels[i]: float(q_matrix_array[:, i].std()) 
                                         for i in range(len(intent_labels))}
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_feature_importance(self, params: Dict) -> Dict:
        """分析特征重要性"""
        num_samples = params.get("num_samples", 100)
        
        try:
            from agent_rl import AgentRL
            from agent_state import AgentState
            import torch
            
            agent = AgentRL()
            intent_labels = ["add_file", "delete_file", "modify_file", 
                           "refactor", "query", "deep_search", "vague"]
            
            # 采样不同意图的输入
            test_queries = []
            for intent_name, templates in agent.TASK_TEMPLATES.items():
                samples = templates[:min(5, len(templates))]
                for tmpl in samples:
                    f = "test.txt"
                    query = tmpl.format(f=f, g="config.txt", c="示例内容")
                    test_queries.append((query, intent_name.value))
            
            # 提取特征
            all_features = []
            all_labels = []
            
            for query, label in test_queries[:num_samples]:
                state = AgentState()
                state.corpus_size = 10
                state.increment_round()
                
                features = agent._extract_features(query, state)
                all_features.append(features)
                all_labels.append(label)
            
            features_array = np.array(all_features)
            
            # 简单的特征重要性分析（基于方差）
            feature_variance = np.var(features_array, axis=0)
            top_features = np.argsort(feature_variance)[-20:]  # Top 20
            
            return {
                "total_features": features_array.shape[1],
                "samples_analyzed": len(all_features),
                "top_features_by_variance": {
                    "indices": top_features.tolist(),
                    "variances": feature_variance[top_features].tolist()
                },
                "feature_groups": {
                    "semantic_features": {
                        "range": [0, 256],
                        "mean_variance": float(np.mean(feature_variance[:256]))
                    },
                    "state_features": {
                        "range": [256, 268],
                        "mean_variance": float(np.mean(feature_variance[256:]))
                    }
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_exploration_exploitation(self, params: Dict) -> Dict:
        """分析探索-利用平衡"""
        try:
            from agent_rl import AgentRL
            
            agent = AgentRL()
            
            # 计算探索率
            exploration_rate = agent.epsilon
            exploitation_rate = 1 - agent.epsilon
            
            # 基于训练进度估算
            total_steps = agent.training_step
            expected_steps_for_convergence = 10000
            
            progress = min(1.0, total_steps / expected_steps_for_convergence)
            
            # 建议
            if exploration_rate > 0.5:
                status = "high_exploration"
                suggestion = "模型仍在大量探索，建议继续训练"
            elif exploration_rate > 0.1:
                status = "balanced"
                suggestion = "探索-利用平衡良好"
            else:
                status = "high_exploitation"
                suggestion = "模型主要利用已学知识，可能需要增加探索"
            
            return {
                "current_state": {
                    "epsilon": exploration_rate,
                    "exploitation_rate": exploitation_rate,
                    "training_steps": total_steps
                },
                "status": status,
                "suggestion": suggestion,
                "progress": {
                    "to_convergence": progress,
                    "remaining_steps": max(0, expected_steps_for_convergence - total_steps)
                },
                "history_analysis": {
                    "epsilon_decay_rate": agent.epsilon_decay,
                    "epsilon_min": agent.epsilon_min,
                    "steps_to_min": self._estimate_steps_to_min(
                        exploration_rate, agent.epsilon_min, agent.epsilon_decay
                    )
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_steps_to_min(self, current: float, target: float, decay: float) -> int:
        """估算到达最小epsilon需要的步数"""
        if current <= target:
            return 0
        steps = 0
        while current > target and steps < 100000:
            current *= decay
            steps += 1
        return steps
    
    def _get_decision_boundary(self, params: Dict) -> Dict:
        """分析决策边界"""
        intent_pair = params.get("intent_pair", ["add_file", "delete_file"])
        
        try:
            from agent_rl import AgentRL
            from agent_state import AgentState
            import torch
            import torch.nn.functional as F
            
            agent = AgentRL()
            
            # 获取两个意图的模板
            intent_a = intent_pair[0]
            intent_b = intent_pair[1]
            
            templates_a = agent.TASK_TEMPLATES.get(intent_a, [])
            templates_b = agent.TASK_TEMPLATES.get(intent_b, [])
            
            # 提取特征
            features_a = []
            features_b = []
            
            for tmpl in templates_a[:10]:
                query = tmpl.format(f="test.txt", g="config.txt", c="示例内容")
                state = AgentState()
                state.corpus_size = 10
                state.increment_round()
                features_a.append(agent._extract_features(query, state))
            
            for tmpl in templates_b[:10]:
                query = tmpl.format(f="test.txt", g="config.txt", c="示例内容")
                state = AgentState()
                state.corpus_size = 10
                state.increment_round()
                features_b.append(agent._extract_features(query, state))
            
            features_a = np.array(features_a)
            features_b = np.array(features_b)
            
            # 计算类间距离
            mean_a = np.mean(features_a, axis=0)
            mean_b = np.mean(features_b, axis=0)
            
            # 中心距离
            center_distance = float(np.linalg.norm(mean_a - mean_b))
            
            # 计算方差
            var_a = np.mean(np.var(features_a, axis=0))
            var_b = np.mean(np.var(features_b, axis=0))
            
            # 判断可分性
            separability = center_distance / (np.sqrt(var_a) + np.sqrt(var_b) + 1e-8)
            
            return {
                "intent_pair": intent_pair,
                "separability_score": separability,
                "center_distance": center_distance,
                "interpretation": {
                    "very_separable": separability > 2.0,
                    "separable": 1.0 < separability <= 2.0,
                    "overlapping": separability <= 1.0
                },
                "feature_statistics": {
                    intent_a: {
                        "mean_norm": float(np.linalg.norm(mean_a)),
                        "variance": float(var_a)
                    },
                    intent_b: {
                        "mean_norm": float(np.linalg.norm(mean_b)),
                        "variance": float(var_b)
                    }
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_error_patterns(self, params: Dict) -> Dict:
        """分析错误模式"""
        days = params.get("days", 7)
        
        try:
            audit_db = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "audit.db"
            )
            # 兼容新路径
            if not os.path.exists(audit_db):
                from runtime_paths import DB_DIR
                audit_db = str(DB_DIR / "audit.db")
            
            if not os.path.exists(audit_db):
                return {"message": "Audit database not found"}
            
            conn = sqlite3.connect(audit_db)
            conn.row_factory = sqlite3.Row
            
            start_time = datetime.utcnow() - timedelta(days=days)
            
            # 获取错误路由
            cursor = conn.execute(
                "SELECT query, predicted_intent, expected_intent, confidence "
                "FROM intent_routing_audit "
                "WHERE timestamp >= ? AND is_correct = 0",
                (start_time.isoformat(),)
            )
            
            errors = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            if not errors:
                return {
                    "message": "No errors found in the specified period",
                    "days": days
                }
            
            # 分析错误模式
            error_by_type = defaultdict(list)
            for error in errors:
                key = (error['expected_intent'], error['predicted_intent'])
                error_by_type[key].append(error)
            
            # 统计每种错误类型
            patterns = []
            for (expected, predicted), examples in sorted(error_by_type.items(), 
                                                          key=lambda x: len(x[1]), 
                                                          reverse=True):
                avg_confidence = np.mean([e['confidence'] for e in examples])
                patterns.append({
                    "expected": expected,
                    "predicted": predicted,
                    "count": len(examples),
                    "avg_confidence": float(avg_confidence),
                    "example_queries": [e['query'][:50] for e in examples[:3]]
                })
            
            # 分析常见错误原因
            common_causes = []
            for pattern in patterns:
                if pattern['count'] > 3:
                    common_causes.append({
                        "type": f"{pattern['expected']} -> {pattern['predicted']}",
                        "frequency": pattern['count'],
                        "likely_cause": self._suggest_cause(pattern['expected'], pattern['predicted'])
                    })
            
            return {
                "period_days": days,
                "total_errors": len(errors),
                "error_patterns": patterns[:10],
                "common_causes": common_causes,
                "recommendations": self._generate_recommendations(patterns)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _suggest_cause(self, expected: str, predicted: str) -> str:
        """根据错误类型推测原因"""
        causes = {
            ("add_file", "query"): "输入可能包含查询关键词，触发了查询意图",
            ("delete_file", "query"): "输入可能包含查找类关键词",
            ("query", "add_file"): "输入可能包含创建类动词",
            ("vague", "query"): "模糊输入被降级为查询",
        }
        return causes.get((expected, predicted), "需要进一步分析")
    
    def _generate_recommendations(self, patterns: List[Dict]) -> List[str]:
        """根据错误模式生成建议"""
        recommendations = []
        
        for pattern in patterns[:3]:
            expected = pattern['expected']
            predicted = pattern['predicted']
            
            if expected == "add_file" and predicted == "query":
                recommendations.append("增加添加文件的训练样本，特别是包含'新建'、'创建'等关键词的样本")
            elif expected == "delete_file" and predicted == "query":
                recommendations.append("加强删除意图的特征权重，注意区分'删除'和'查找'关键词")
            elif pattern['avg_confidence'] > 0.7:
                recommendations.append(f"模型对 {predicted} 意图过于自信，建议降低相关特征权重")
        
        if not recommendations:
            recommendations.append("继续收集更多训练数据以改进模型")
        
        return recommendations
    
    def _generate_model_report(self, params: Dict) -> Dict:
        """生成完整模型报告"""
        try:
            from agent_rl import AgentRL
            
            agent = AgentRL()
            
            # 收集所有分析
            convergence = self._analyze_convergence({"window_size": 50})
            exploration = self._analyze_exploration_exploitation({})
            
            # 评估
            eval_results = {}
            try:
                eval_results = agent.evaluate()
            except:
                pass
            
            return {
                "report_generated": datetime.utcnow().isoformat(),
                "model_summary": {
                    "epsilon": agent.epsilon,
                    "training_steps": agent.training_step,
                    "replay_buffer_size": len(agent.replay_buffer),
                    "is_trained": agent.trained,
                    "reward_history_length": len(agent.reward_history) if agent.reward_history else 0
                },
                "convergence_analysis": convergence,
                "exploration_analysis": exploration,
                "evaluation_results": eval_results,
                "recommendations": [
                    "继续在线学习以提高准确率",
                    "收集更多边界情况的训练数据",
                    "监控收敛状态避免过拟合"
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
