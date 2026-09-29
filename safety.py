import os
import json
import time
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from runtime_paths import RUNTIME_ROOT, BACKUP_DIR, LOG_DIR
from atomic_io import atomic_write_json
from write_queue import get_default_write_queue


class OperationType(Enum):
    DELETE = "delete"
    REFACTOR = "refactor"
    MODIFY = "modify"
    ADD = "add"


class OperationStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class OperationRecord:
    operation_id: str
    operation_type: OperationType
    query: str
    targets: List[str]
    status: OperationStatus = OperationStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    backup_paths: Dict[str, str] = field(default_factory=dict)


class SafetyManager:
    DANGEROUS_KEYWORDS = {
        OperationType.DELETE: ["delete", "rm"],
        OperationType.REFACTOR: ["refactor", "rename", "move", "organize"],
    }

    AUTO_CONFIRM_THRESHOLD = 5

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.operation_log_path = str(LOG_DIR / "operation_log.json")
        self.operations: List[OperationRecord] = []
        self._load_operation_log()

    def _load_operation_log(self):
        if os.path.exists(self.operation_log_path):
            try:
                with open(self.operation_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.operations = [
                        OperationRecord(
                            operation_id=op["operation_id"],
                            operation_type=OperationType(op["operation_type"]),
                            query=op["query"],
                            targets=op["targets"],
                            status=OperationStatus(op["status"]),
                            timestamp=op["timestamp"],
                            executed_at=op.get("executed_at"),
                            rolled_back_at=op.get("rolled_back_at"),
                            details=op.get("details", {}),
                            backup_paths=op.get("backup_paths", {}),
                        )
                        for op in data
                    ]
            except (json.JSONDecodeError, KeyError):
                self.operations = []

    def _save_operation_log(self):
        data = []
        for op in self.operations:
            data.append({
                "operation_id": op.operation_id,
                "operation_type": op.operation_type.value,
                "query": op.query,
                "targets": op.targets,
                "status": op.status.value,
                "timestamp": op.timestamp,
                "executed_at": op.executed_at,
                "rolled_back_at": op.rolled_back_at,
                "details": op.details,
                "backup_paths": op.backup_paths,
            })
        # 多个线程都会改操作日志：交给单写者队列串行落盘
        try:
            get_default_write_queue().submit_json(self.operation_log_path, data)
        except RuntimeError:
            atomic_write_json(self.operation_log_path, data)

    def needs_confirmation(self, operation_type: OperationType, target_count: int = 0, query: str = "") -> bool:
        if operation_type in (OperationType.DELETE, OperationType.REFACTOR):
            return True
        if target_count > self.AUTO_CONFIRM_THRESHOLD:
            return True
        return False

    def create_operation_record(self, operation_type: OperationType, query: str,
                                 targets: List[str], details: Dict = None) -> OperationRecord:
        op_id = f"{operation_type.value}_{int(time.time() * 1000)}"
        record = OperationRecord(
            operation_id=op_id,
            operation_type=operation_type,
            query=query,
            targets=targets,
            status=OperationStatus.PENDING,
            details=details or {},
        )
        self.operations.append(record)
        self._save_operation_log()
        return record

    def confirm_operation(self, operation_id: str) -> bool:
        for op in self.operations:
            if op.operation_id == operation_id:
                op.status = OperationStatus.CONFIRMED
                self._save_operation_log()
                return True
        return False

    def execute_operation(self, operation_id: str, backup_paths: Dict[str, str] = None):
        for op in self.operations:
            if op.operation_id == operation_id:
                op.status = OperationStatus.EXECUTED
                op.executed_at = time.time()
                if backup_paths:
                    op.backup_paths = backup_paths
                self._save_operation_log()
                return True
        return False

    def backup_before_operation(self, targets: List[str], operation_id: str) -> Dict[str, str]:
        backup_dir = str(BACKUP_DIR / operation_id)
        os.makedirs(backup_dir, exist_ok=True)
        backup_paths = {}
        for target in targets:
            abs_path = os.path.join(self.workspace_root, target)
            if os.path.exists(abs_path):
                backup_path = os.path.join(backup_dir, os.path.basename(target))
                try:
                    if os.path.isfile(abs_path):
                        shutil.copy2(abs_path, backup_path)
                    elif os.path.isdir(abs_path):
                        shutil.copytree(abs_path, backup_path)
                    backup_paths[target] = backup_path
                except (OSError, PermissionError) as e:
                    pass
        return backup_paths

    def rollback_operation(self, operation_id: str) -> Tuple[bool, str]:
        for op in self.operations:
            if op.operation_id == operation_id:
                if op.status != OperationStatus.EXECUTED:
                    return False, f"Operation {operation_id} status is {op.status.value}, cannot rollback"
                if not op.backup_paths:
                    return False, f"Operation {operation_id} has no backup info"
                restored = []
                failed = []
                for original_path, backup_path in op.backup_paths.items():
                    abs_original = os.path.join(self.workspace_root, original_path)
                    if os.path.exists(backup_path):
                        try:
                            if os.path.isfile(backup_path):
                                os.makedirs(os.path.dirname(abs_original), exist_ok=True)
                                shutil.copy2(backup_path, abs_original)
                                restored.append(original_path)
                            elif os.path.isdir(backup_path):
                                if os.path.exists(abs_original):
                                    shutil.rmtree(abs_original)
                                shutil.copytree(backup_path, abs_original)
                                restored.append(original_path)
                        except (OSError, PermissionError) as e:
                            failed.append(f"{original_path}: {e}")
                    else:
                        failed.append(f"{original_path}: backup not found")
                if restored:
                    op.status = OperationStatus.ROLLED_BACK
                    op.rolled_back_at = time.time()
                    self._save_operation_log()
                    msg = f"Rolled back {len(restored)} files"
                    if failed:
                        msg += f", {len(failed)} failed"
                    return True, msg
                else:
                    return False, f"Rollback failed: {failed}"
        return False, f"Operation {operation_id} not found"

    def get_pending_operations(self) -> List[OperationRecord]:
        return [op for op in self.operations if op.status == OperationStatus.PENDING]

    def get_rollbackable_operations(self) -> List[OperationRecord]:
        return [op for op in self.operations if op.status == OperationStatus.EXECUTED]

    def format_confirmation_message(self, record: OperationRecord) -> str:
        type_names = {
            OperationType.DELETE: "DELETE",
            OperationType.REFACTOR: "REFACTOR",
            OperationType.MODIFY: "MODIFY",
            OperationType.ADD: "ADD",
        }
        type_name = type_names.get(record.operation_type, record.operation_type.value)
        lines = [
            "",
            "=" * 60,
            f"  [SAFETY CONFIRM] {type_name} operation",
            "=" * 60,
            f"  Operation ID: {record.operation_id}",
            f"  Query: {record.query}",
            f"  Target files: {len(record.targets)}",
        ]
        for i, target in enumerate(record.targets[:10], 1):
            lines.append(f"    [{i}] {target}")
        if len(record.targets) > 10:
            lines.append(f"    ... and {len(record.targets) - 10} more files")
        lines.extend([
            "=" * 60,
            f"  Enter 'confirm {record.operation_id}' to confirm",
            f"  Enter 'cancel' to abort",
            "=" * 60,
        ])
        return "\n".join(lines)
