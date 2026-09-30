"""AgentRL 对话框 Web 服务端：在 5000 端口提供聊天 API，并托管 chat.html。

端点:
  GET  /chat.html                       -> 网页界面
  POST /api/chat                        -> 发送一条消息，返回回复 + 思考过程
  GET  /api/conversations               -> 历史会话列表
  GET  /api/conversations/{id}/messages -> 单个会话的全部消息（含思考过程）
  POST /api/conversations               -> 新建会话（真正的 0 上下文）
  DELETE /api/conversations/{id}        -> 删除会话
  GET  /health                          -> 健康检查

历史会话持久化在项目的 conversations/ 目录（JSON 文件），重启不丢。
"""

import json
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from agent_state import AgentState, SHORT_LABELS, IntentType
from atomic_io import atomic_write_json
from rwlock import RWLock
from runtime_paths import CONVERSATIONS_DIR
from continuous_learning import get_learning_engine, EvalSplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = str(CONVERSATIONS_DIR)
HTML_PATH = os.path.join(BASE_DIR, "chat.html")

OP_LABELS = {
    "add": "添加", "delete": "删除", "modify": "修改",
    "refactor": "整理", "query": "查询", "deep_search": "深度检索",
}

_engine_lock = threading.Lock()   # 串行化会改动共享 RL 状态的整条请求
_engine_rw = RWLock()             # 引擎实例表读多写少：查表走共享读锁，首次构造走写锁
_system = None


def get_system():
    """惰性创建共享的 AgentRL 系统（首次启动自动加载持久化模型）。

    用读写锁而不是普通互斥锁：
      - 已构造 → 只持共享读锁，/health、build_trace 等读取者互不阻塞
      - 未构造 → 持独占写锁构造，避免两个请求同时建出两套系统
    """
    global _system
    if _system is not None:
        with _engine_rw.read():
            return _system
    with _engine_rw.write():
        if _system is None:
            try:
                from main import AgentRLSystem
                _system = AgentRLSystem()
            except Exception as e:
                print(f"[ERROR] 系统初始化失败: {e}")
                traceback.print_exc()
                raise
        return _system


def build_trace(state: AgentState, agent_rl) -> list:
    """把一次路由的完整"思考过程"整理成步骤列表。"""
    steps = []
    if state.current_intent is not None:
        label = SHORT_LABELS.get(state.current_intent, state.current_intent.value)
        path = state.routing_path.value if state.routing_path else "N/A"
        conf = None
        if hasattr(agent_rl, 'intent_history') and agent_rl.intent_history:
            conf = agent_rl.intent_history[-1][1]
        extra = f"路由路径 {path}" + (f"，置信度 {conf:.2%}" if conf is not None else "")
        steps.append({
            "title": "意图识别",
            "detail": f"用户请求被分为【{label}】类 ({state.current_intent.value})",
            "extra": extra,
        })

    if state.current_intent == IntentType.DEEP_SEARCH:
        try:
            system = get_system()
            react_steps = system.router.deep_search_handler.get_react_trace()
            for rs in react_steps:
                step_num = rs.get("step", "?")
                thought = rs.get("thought", "")
                action = rs.get("action", "")
                detail = rs.get("detail", "")
                action_label = "思考+搜索" if action == "search" else "思考+生成回答"
                steps.append({
                    "title": f"ReAct 第{step_num}步 [{action_label}]",
                    "detail": thought[:200],
                    "extra": detail[:200] if detail else "",
                })
        except Exception:
            pass

    for ev in state.operation_history:
        label = OP_LABELS.get(ev.kind, ev.kind)
        mark = "成功" if ev.success else "失败"
        steps.append({
            "title": "文件操作",
            "detail": f"将要执行的操作: {label} 文件「{ev.target or '（未指定）'}」",
            "extra": f"结果[{mark}]: {ev.message}",
        })

    for r in state.evidence:
        files = [d.get("path") for d in r.documents]
        steps.append({
            "title": "文档检索",
            "detail": f"第 {r.round_num} 轮向量检索命中 {len(files)} 个文件",
            "extra": "、".join(files) if files else "无命中文档",
        })

    if state.clarifications:
        steps.append({
            "title": "主动澄清",
            "detail": "指令存在歧义，需要用户补充信息后才能继续",
            "extra": "；".join(state.clarifications),
        })

    if not steps:
        steps.append({"title": "已接收", "detail": "请求已处理完成", "extra": ""})
    return steps


def run_message(message: str, conversation_id: str = ""):
    """处理一条消息，返回 (reply, thinking, intent, path, clarifications, trace_id)。上锁防止并发竞争。"""
    start_time = __import__('time').time()
    with _engine_lock:
        system = get_system()
        router, generator, agent_rl = system.router, system.generator, system.agent_rl

        state = AgentState()
        state.corpus_size = len(system.corpus.registry)
        state.add_dialog(message, "")
        state = router.route(state)

        reply = generator.generate(state)
        thinking = build_trace(state, agent_rl)
        intent = state.current_intent.value if state.current_intent else ""
        path = state.routing_path.value if state.routing_path else ""
        clarifications = list(state.clarifications or [])

    latency_ms = (__import__('time').time() - start_time) * 1000

    # 记录 Trace
    engine = get_learning_engine()
    retrieval_docs = []
    retrieval_scores = []
    for ev in state.evidence:
        for d in ev.documents:
            retrieval_docs.append(d.get("path", ""))
        if ev.score:
            retrieval_scores.append(ev.score)

    trace_id = engine.record_trace(
        user_query=message,
        answer=reply,
        conversation_id=conversation_id,
        intent=intent,
        route=path,
        thinking=thinking,
        clarifications=clarifications,
        retrieval_docs=retrieval_docs,
        retrieval_scores=retrieval_scores,
        latency_ms=round(latency_ms, 2),
        agent=getattr(system, "agent_type", ""),
    )
    engine.increment_trace_count()

    return reply, thinking, intent, path, clarifications, trace_id


class ChatStore:
    """会话存储：每个会话一个 JSON 文件。"""

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, conv_id: str) -> str:
        return os.path.join(self.dir, f"{conv_id}.json")

    def list(self):
        convs = []
        for name in os.listdir(self.dir):
            if not name.endswith(".json"):
                continue
            try:
                with open(self._path(name[:-5]), "r", encoding="utf-8") as f:
                    data = json.load(f)
                convs.append({
                    "id": data["id"],
                    "title": data.get("title", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except (OSError, json.JSONDecodeError):
                continue
        convs.sort(key=lambda c: c["updated_at"], reverse=True)
        return convs

    def get(self, conv_id: str):
        path = self._path(conv_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create(self):
        now = datetime.now().isoformat(timespec="seconds")
        conv = {
            "id": uuid.uuid4().hex[:12],
            "title": "",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.save(conv)
        return conv

    def save(self, conv: dict):
        conv["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(self._path(conv["id"]), conv)

    def delete(self, conv_id: str) -> bool:
        path = self._path(conv_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True


app = FastAPI(title="AgentRL Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ChatStore(CONV_DIR)

# ----------------------------------------------------------- 静态页面
@app.get("/", response_class=HTMLResponse)
@app.get("/chat.html", response_class=HTMLResponse)
def index():
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "chat.html 不存在", 404


# ----------------------------------------------------------- 聊天 API
class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


class NewConversation(BaseModel):
    pass


@app.post("/api/chat")
def chat(req: ChatIn):
    message = req.message.strip()
    if not message:
        return JSONResponse({"error": "请输入内容"}, status_code=400)

    conv = store.get(req.conversation_id) if req.conversation_id else None
    if req.conversation_id and conv is None:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    if conv is None:
        conv = store.create()

    if not conv["title"]:
        conv["title"] = message.replace("\n", " ")[:24]
    conv["messages"].append({"role": "user", "content": message,
                             "thinking": [], "timestamp": datetime.now().isoformat(timespec="seconds")})

    try:
        reply, thinking, intent, path, clarifications, trace_id = run_message(message, conv["id"])
    except Exception as e:
        traceback.print_exc()
        reply = f"处理请求时发生错误: {e}"
        thinking = [{"title": "错误", "detail": str(e), "extra": ""}]
        intent = ""
        path = ""
        clarifications = []
        trace_id = ""

    conv["messages"].append({"role": "assistant", "content": reply,
                             "thinking": thinking, "timestamp": datetime.now().isoformat(timespec="seconds"),
                             "trace_id": trace_id})
    store.save(conv)

    return {
        "conversation_id": conv["id"],
        "title": conv["title"],
        "reply": reply,
        "thinking": thinking,
        "intent": intent,
        "path": path,
        "clarifications": clarifications,
        "trace_id": trace_id,
    }


@app.get("/api/conversations")
def list_conversations():
    return {"conversations": store.list()}


@app.post("/api/conversations")
def create_conversation():
    conv = store.create()
    return {"conversation_id": conv["id"], "title": conv["title"]}


@app.get("/api/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    conv = store.get(conv_id)
    if not conv:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    return {"conversation_id": conv_id, "title": conv["title"], "messages": conv["messages"]}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    ok = store.delete(conv_id)
    if not ok:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    return {"deleted": conv_id}


@app.get("/health")
def health():
    info = {"status": "healthy"}
    try:
        system = get_system()
        info["system"] = "ready"
        info["corpus_size"] = len(system.corpus.registry)
        info["rl_epsilon"] = round(system.agent_rl.epsilon, 4)
        info["rl_buffer"] = len(system.agent_rl.replay_buffer)
    except Exception as e:
        info["system"] = "error"
        info["error"] = str(e)
    return info


# ----------------------------------------------------------- 持续学习 API
class FeedbackIn(BaseModel):
    trace_id: str
    feedback: str  # "positive" / "negative"


class LabelIn(BaseModel):
    trace_id: str
    expected_intent: str = ""
    expected_route: str = ""
    failure_type: str = ""
    notes: str = ""


class EvalSetIn(BaseModel):
    trace_id: str
    split: str  # "train" / "validation" / "test" / "challenge"
    label: str = ""


class VersionIn(BaseModel):
    version_name: str
    description: str = ""
    config: dict = None


@app.post("/api/feedback")
def submit_feedback(req: FeedbackIn):
    engine = get_learning_engine()
    ok = engine.update_feedback(req.trace_id, req.feedback)
    if not ok:
        return JSONResponse({"error": "trace 不存在"}, status_code=404)
    return {"status": "ok", "trace_id": req.trace_id, "feedback": req.feedback}


@app.post("/api/learning/label")
def submit_label(req: LabelIn):
    """人工标注：为 trace 添加期望意图/路由/失败类型。"""
    engine = get_learning_engine()
    trace = engine.get_trace(req.trace_id)
    if not trace:
        return JSONResponse({"error": "trace 不存在"}, status_code=404)

    trace_path = str(engine.TRACES_DIR if hasattr(engine, 'TRACES_DIR') else '') + f"/{req.trace_id}.json"
    from continuous_learning import TRACES_DIR
    trace_path = TRACES_DIR / f"{req.trace_id}.json"
    if not trace_path.exists():
        return JSONResponse({"error": "trace 不存在"}, status_code=404)

    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["human_label"] = {
        "expected_intent": req.expected_intent,
        "expected_route": req.expected_route,
        "failure_type": req.failure_type,
        "notes": req.notes,
    }
    atomic_write_json(trace_path, data)

    return {"status": "ok"}


@app.post("/api/learning/eval-set")
def add_to_eval_set(req: EvalSetIn):
    """将 trace 加入评估集。"""
    engine = get_learning_engine()
    try:
        split = EvalSplit(req.split)
    except ValueError:
        return JSONResponse({"error": f"无效的 split: {req.split}，可选: train/validation/test/challenge"}, status_code=400)
    engine.add_to_eval_set(req.trace_id, split, req.label)
    return {"status": "ok", "split": split.value}


@app.get("/api/learning/stats")
def get_learning_stats():
    """获取持续学习统计数据。"""
    engine = get_learning_engine()
    return engine.get_stats()


@app.get("/api/learning/failures")
def get_failures(limit: int = 50):
    """获取失败案例列表。"""
    engine = get_learning_engine()
    return {"failures": engine.get_failure_cases(limit=limit)}


@app.get("/api/learning/traces")
def get_traces(limit: int = 100):
    """获取 trace 列表。"""
    engine = get_learning_engine()
    return {"traces": engine.get_traces(limit=limit)}


@app.get("/api/learning/traces/{trace_id}")
def get_trace(trace_id: str):
    """获取单条 trace 详情。"""
    engine = get_learning_engine()
    trace = engine.get_trace(trace_id)
    if not trace:
        return JSONResponse({"error": "trace 不存在"}, status_code=404)
    return trace


@app.get("/api/learning/clusters")
def get_failure_clusters():
    """获取失败案例聚类。"""
    engine = get_learning_engine()
    clusters = engine.cluster_failures()
    result = {}
    for k, v in clusters.items():
        result[k] = {"count": len(v), "query": v[0].get("user_query", ""), "trace_ids": [m["trace_id"] for m in v]}
    return {"clusters": result}


@app.post("/api/learning/dedup")
def dedup_failures():
    """去重失败案例。"""
    engine = get_learning_engine()
    removed = engine.deduplicate()
    return {"removed": removed}


@app.post("/api/learning/version")
def create_version(req: VersionIn):
    """创建优化版本。"""
    engine = get_learning_engine()
    vid = engine.create_version(req.version_name, req.description, req.config)
    return {"version_id": vid}


@app.get("/api/learning/versions")
def list_versions():
    """列出所有版本。"""
    engine = get_learning_engine()
    return {"versions": engine.list_versions()}


@app.get("/api/learning/export")
def export_failures():
    """导出失败案例。"""
    engine = get_learning_engine()
    path = engine.export_failures()
    return {"export_path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)