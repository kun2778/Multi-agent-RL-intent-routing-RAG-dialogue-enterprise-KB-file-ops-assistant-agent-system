import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from agent_state import AgentState, IntentType
from file_corpus import FileCorpus
from safety import SafetyManager, OperationType, OperationStatus, OperationRecord


@dataclass
class OperationOutcome:
    """一次真实文件操作的结果（终端反馈 + RL 奖励键 + 状态转移）。"""
    success: bool
    message: str
    reward_key: str
    target: str = ""
    data: Dict = field(default_factory=dict)


class FileOperationHandler:
    """
    文件操作执行器：把 RL 意图转成真实文件系统操作，作用于共享语料库。

    add/delete/modify/refactor 直接调 FileCorpus 完成文件增删改与整理，
    返回 OperationOutcome（含 reward_key，供奖励模块计算即时奖励）。
    """

    INTENT_TO_REWARD = {
        IntentType.ADD_FILE: "add",
        IntentType.DELETE_FILE: "delete",
        IntentType.MODIFY_FILE: "modify",
        IntentType.REFACTOR: "refactor",
    }

    def __init__(self, corpus: FileCorpus, workspace_root: str = None):
        self.corpus = corpus
        self.safety_manager = SafetyManager(workspace_root) if workspace_root else None

    def execute(self, intent: IntentType, query: str, state: AgentState) -> OperationOutcome:
        if intent == IntentType.ADD_FILE:
            return self.add(query, state)
        if intent == IntentType.DELETE_FILE:
            return self.delete(query, state)
        if intent == IntentType.MODIFY_FILE:
            return self.modify(query, state)
        if intent == IntentType.REFACTOR:
            return self.refactor(query, state)
        raise ValueError(f"不支持的操作意图: {intent}")

    def _split_add(self, query: str) -> Tuple[str, str, List[str]]:
        """从指令中拆出 文件名 和 内容。"""
        target = None
        content = ""
        q = query.strip()
        for sep in ("内容为", "内容:", "内容是", "内容如下", "写入", "保存"):
            idx = q.find(sep)
            if idx != -1:
                content = q[idx + len(sep):].strip()
                q = q[:idx]
                break

        action_words = ["添加", "新增", "创建", "新建", "写入", "保存", "删除", "移除", 
                        "修改", "更新", "编辑", "add", "create", "new", "write", "save",
                        "delete", "remove", "modify", "edit", "update"]
        for word in action_words:
            if q.lower().startswith(word.lower()):
                q = q[len(word):].strip()
                break

        q = q.lstrip("我的").strip()
        q = re.sub(r'^[一个]+', '', q).strip()
        q = q.rstrip("文件").strip()

        path_match = re.search(r'[A-Za-z]:\\[^\s]+\.[A-Za-z0-9]+', q)
        if path_match:
            target = path_match.group(0)
        elif "/" in q or "\\" in q:
            parts = q.replace("\\", "/").split("/")
            for i, part in enumerate(parts):
                if "." in part:
                    target = "/".join(parts[:i + 1])
                    break
        elif " " in q:
            words = q.split()
            for w in words:
                if "." in w:
                    target = w
                    break
        else:
            target = q

        tags = []
        return target, content, tags

    def add(self, query: str, state: AgentState) -> OperationOutcome:
        target, content, tags = self._split_add(query)
        if not target:
            state.record_operation("add", "", False, "缺少文件名")
            return OperationOutcome(False, "请提供要添加的文件名。", "missing_slot")
        ok, msg, doc = self.corpus.add_document(target, content, tags)
        state.corpus_size = len(self.corpus.registry)
        state.record_operation("add", doc.get("path", target), ok, msg)
        return OperationOutcome(ok, msg, "add_success" if ok else "add_failure", target, {"document": doc})

    def delete(self, query: str, state: AgentState) -> OperationOutcome:
        target = self._extract_target_from_query(query)
        if not target:
            state.record_operation("delete", "", False, "缺少文件名")
            return OperationOutcome(False, "请提供要删除的文件名。", "missing_slot")

        # 作用域早检：系统目录/非 C 盘在进入 confirm 流程前就拒绝，
        # 避免用户确认后才失败（corpus 内部仍有同样的兜底校验）
        scope_err = self.corpus.check_write_scope(target)
        if scope_err:
            state.record_operation("delete", target, False, scope_err)
            return OperationOutcome(False, scope_err, "delete_failure", target)

        if self.safety_manager and self.safety_manager.needs_confirmation(OperationType.DELETE, 1, query):
            record = self.safety_manager.create_operation_record(
                OperationType.DELETE, query, [target]
            )
            confirmation_msg = self.safety_manager.format_confirmation_message(record)
            state.record_operation("delete", target, False, confirmation_msg)
            return OperationOutcome(False, confirmation_msg, "delete_pending", target, {"operation_id": record.operation_id})

        ok, msg, doc = self.corpus.delete_document(target)
        state.corpus_size = len(self.corpus.registry)
        state.record_operation("delete", target, ok, msg)
        return OperationOutcome(ok, msg, "delete_success" if ok else "delete_failure", target)

    def modify(self, query: str, state: AgentState) -> OperationOutcome:
        target, content, _ = self._split_add(query)
        if not target:
            state.record_operation("modify", "", False, "缺少文件名")
            return OperationOutcome(False, "请提供要修改的目标文件。", "missing_slot")
        if not content:
            matched = self.corpus._resolve_target(target)
            existing = self.corpus.read_text(target) if matched else None
            if existing is not None:
                state.record_operation("modify", target, True, f"文件 {target} 已存在，当前内容可查看 view 命令")
                return OperationOutcome(True, f"文件 {target} 内容未改变（未提供新内容）。可使用 view 查看。", "modify_failure")
            state.record_operation("modify", target, False, "缺少新内容")
            return OperationOutcome(False, "请说明要修改成什么内容。", "missing_slot")
        ok, msg, doc = self.corpus.modify_document(target, new_content=content)
        state.corpus_size = len(self.corpus.registry)
        state.record_operation("modify", doc.get("path", target), ok, msg)
        return OperationOutcome(ok, msg, "modify_success" if ok else "modify_failure", target, {"document": doc})

    def refactor(self, query: str, state: AgentState) -> OperationOutcome:
        targets = []
        for tok in query.split():
            if "." in tok or "/" in tok or "\\" in tok:
                targets.append(tok)

        all_files = list(self.corpus.registry.keys())
        target_list = targets if targets else all_files

        if self.safety_manager and self.safety_manager.needs_confirmation(OperationType.REFACTOR, len(target_list), query):
            record = self.safety_manager.create_operation_record(
                OperationType.REFACTOR, query, target_list[:50]
            )
            confirmation_msg = self.safety_manager.format_confirmation_message(record)
            state.record_operation("refactor", ", ".join(targets) or "全库", False, confirmation_msg)
            return OperationOutcome(False, confirmation_msg, "refactor_pending", "", {"operation_id": record.operation_id})

        ok, msg, moved = self.corpus.refactor(targets=targets or None)
        state.record_operation("refactor", ", ".join(targets) or "全库", ok, msg)
        return OperationOutcome(ok, msg, "refactor_success" if ok else "refactor_failure", "", {"moved": moved})

    def _extract_target_from_query(self, query: str) -> Optional[str]:
        from query_rewriter import QueryRewriter
        target = QueryRewriter().extract_target(query)
        if target:
            return target

        q = query.strip()
        
        path_match = re.search(r'[A-Za-z]:\\.+', q)
        if path_match:
            return path_match.group(0)

        if "/" in q or "\\" in q:
            parts = q.replace("\\", "/").split("/")
            for i, part in enumerate(parts):
                if "." in part:
                    return "/".join(parts[:i + 1])

        action_words = ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete", "rm"]
        for word in action_words:
            idx = q.lower().find(word.lower())
            if idx != -1:
                q = q[idx + len(word):].strip()
                break

        q = q.lstrip("我的").strip()
        q = q.rstrip("文件").strip()
        q = re.sub(r'^[一个]+', '', q).strip()

        file_match = re.search(r'[\w\u4e00-\u9fff]+\.[A-Za-z0-9]{1,6}', q)
        if file_match:
            return file_match.group(0)

        return q if q else None

    def confirm_operation(self, operation_id: str) -> Tuple[bool, str]:
        if not self.safety_manager:
            return False, "Safety manager not available"
        record = None
        for op in self.safety_manager.operations:
            if op.operation_id == operation_id:
                record = op
                break
        if not record:
            return False, f"Operation {operation_id} not found"
        self.safety_manager.confirm_operation(operation_id)
        backup_paths = self.safety_manager.backup_before_operation(record.targets, operation_id)
        if record.operation_type == OperationType.DELETE:
            ok, msg, doc = self.corpus.delete_document(record.targets[0])
            if ok:
                self.safety_manager.execute_operation(operation_id, backup_paths)
            return ok, msg
        elif record.operation_type == OperationType.REFACTOR:
            ok, msg, moved = self.corpus.refactor(targets=None)
            if ok:
                self.safety_manager.execute_operation(operation_id, backup_paths)
            return ok, msg
        return False, f"Unsupported operation type: {record.operation_type}"

    def rollback_operation(self, operation_id: str) -> Tuple[bool, str]:
        if not self.safety_manager:
            return False, "Safety manager not available"
        return self.safety_manager.rollback_operation(operation_id)

    def get_pending_operations(self) -> List[OperationRecord]:
        if not self.safety_manager:
            return []
        return self.safety_manager.get_pending_operations()

    def get_rollbackable_operations(self) -> List[OperationRecord]:
        if not self.safety_manager:
            return []
        return self.safety_manager.get_rollbackable_operations()