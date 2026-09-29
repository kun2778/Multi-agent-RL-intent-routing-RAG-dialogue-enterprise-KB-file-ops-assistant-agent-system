"""新基础设施单元测试：原子写 / 读写锁 / 单写者队列 / 脱敏 / 回归 gate / PER。

覆盖路线图 二、并发与文件 与 一、增量学习 的关键不变量：
  - atomic_io：写入不产生半成品文件、append_line 有序、mmap 按阈值分级
  - rwlock   ：读共享、写独占、写者不被饿死、超时抛 TimeoutError
  - write_queue：单线程顺序执行、错误被捕获不拖垮线程、close 后拒收
  - redaction：邮箱/密钥/手机号/长 hash/本机用户名全部脱敏
  - badcase_replay：sha1 去重、脱敏后才进 buffer
  - regression_gate：回归集确定性、阈值判定、篡改检测
  - agent_rl：PER 优先级清洗（NaN/Inf + 下限）、τ 退火、Engram 失败即弃权

用法::

    python test_infra.py
    pytest test_infra.py        # 函数名也是 test_*，两种方式都能跑
"""

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

TMP_ROOT = tempfile.mkdtemp(prefix="agentrl_infra_")


def tmp_path(name: str) -> str:
    return os.path.join(TMP_ROOT, name)


# ================================================================ atomic_io
def test_atomic_write_json_roundtrip():
    from atomic_io import atomic_write_json, read_json

    p = tmp_path("a_json.json")
    atomic_write_json(p, {"b": 1, "a": [1, 2, 3]})
    assert read_json(p) == {"b": 1, "a": [1, 2, 3]}
    # 原子写不得留下 .tmp 残骸
    leftovers = [f for f in os.listdir(TMP_ROOT) if f.startswith("a_json")]
    assert leftovers == ["a_json.json"], leftovers


def test_atomic_write_overwrites_existing():
    from atomic_io import atomic_write_text, read_bytes

    p = tmp_path("a_over.txt")
    atomic_write_text(p, "x" * 4096)
    atomic_write_text(p, "short")
    assert read_bytes(p) == b"short"


def test_atomic_write_fsync_off_still_ok():
    from atomic_io import atomic_write_text, read_bytes

    p = tmp_path("a_nofsync.txt")
    atomic_write_text(p, "hello", fsync=False)
    assert read_bytes(p) == b"hello"


def test_append_line_ordered():
    from atomic_io import atomic_append_line, read_bytes

    p = tmp_path("a_append.log")
    for i in range(5):
        atomic_append_line(p, f"line{i}")
    lines = read_bytes(p).decode("utf-8").splitlines()
    assert lines == [f"line{i}" for i in range(5)], lines


def test_append_line_concurrent_no_interleave():
    from atomic_io import atomic_append_line, read_bytes

    p = tmp_path("a_append_mt.log")
    errors = []

    def worker(tag):
        try:
            for i in range(30):
                atomic_append_line(p, f"{tag}-{i:02d}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    lines = read_bytes(p).decode("utf-8").splitlines()
    assert len(lines) == 120, len(lines)
    for tag in "abcd":
        own = [l for l in lines if l.startswith(f"{tag}-")]
        assert own == [f"{tag}-{i:02d}" for i in range(30)], own


def test_open_read_threshold_tiers():
    from atomic_io import open_read, decode_bytes

    small = tmp_path("r_small.bin")
    with open(small, "wb") as fh:
        fh.write("中文 utf-8".encode("utf-8"))
    with open_read(small) as fh:
        data = fh.read() if hasattr(fh, "read") else bytes(fh)
    assert data == "中文 utf-8".encode("utf-8")

    big = tmp_path("r_big.bin")
    payload = ("abcdefgh" * 4096).encode("ascii")   # 32KB
    with open(big, "wb") as fh:
        fh.write(payload)
    with open_read(big, threshold=1024) as fh:
        assert decode_bytes(fh) == payload.decode("ascii")
        # 低于阈值才复制成 bytes，超阈值走 mmap（只读、零拷贝）
        assert not isinstance(fh, (bytes, bytearray)), type(fh)


def test_decode_bytes_fallback():
    from atomic_io import decode_bytes

    assert decode_bytes("hi".encode("utf-8")) == "hi"
    assert decode_bytes("中文".encode("gbk")) == "中文"
    # 全部候选编码都失败时用 latin-1 忽略错误兜底，不抛异常
    assert isinstance(decode_bytes(b"\xff\xfe\x00"), str)


def test_open_blob_size_tiering():
    from atomic_io import open_blob

    small = b"tiny" * 10
    with open_blob(small, threshold=1024) as fh:
        assert type(fh).__name__ == "BytesIO"
        assert fh.read() == small

    big = b"x" * 4096
    with open_blob(big, threshold=1024) as fh:
        assert type(fh).__name__ != "BytesIO", "超阈值应落临时文件而非 BytesIO"
        tmp_name = fh.name
        assert fh.read() == big
    assert not os.path.exists(tmp_name), "退出 with 后临时文件必须被删除"


def test_atomic_savez_roundtrip():
    from atomic_io import atomic_savez

    p = tmp_path("a_npz.npz")
    atomic_savez(p, vec=np.arange(5, dtype=np.float32), mat=np.ones((2, 2)))
    with np.load(p) as data:
        assert data["vec"].tolist() == [0, 1, 2, 3, 4]
        assert data["mat"].shape == (2, 2)


def test_atomic_torch_save_roundtrip():
    import torch
    from atomic_io import atomic_torch_save

    p = tmp_path("a_torch.pt")
    atomic_torch_save(p, {"w": torch.zeros(3)})
    state = torch.load(p, map_location="cpu", weights_only=True)
    assert state["w"].shape == (3,)


def test_fsync_dir_tolerated():
    from atomic_io import fsync_dir

    fsync_dir(TMP_ROOT)   # Windows 上允许退化为 no-op，但不得抛异常


# ================================================================ rwlock
def test_rwlock_readers_share_and_writer_exclusive():
    from rwlock import RWLock

    lock = RWLock()
    active_readers = 0
    peak_readers = 0
    release_readers = threading.Event()
    errors = []

    def reader():
        nonlocal active_readers, peak_readers
        try:
            with lock.read():
                active_readers += 1
                peak_readers = max(peak_readers, active_readers)
                # 停在临界区内，直到主线程验证完写者被挡在门外
                release_readers.wait(5)
                active_readers -= 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()

    deadline = time.time() + 5
    while peak_readers < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert peak_readers >= 3, f"读者未并发，peak={peak_readers}"

    # 三个读者都持有时：写者必须拿不到锁
    assert lock.acquire_write(timeout=0.05) is False, "读者持有时写者必须拿不到锁"

    release_readers.set()
    for t in threads:
        t.join()

    assert not errors, errors
    assert lock.reader_count == 0 and not lock.locked_for_write

    # 读者全部释放后，写者可以拿到锁
    with lock.write(timeout=5):
        assert lock.locked_for_write is True



def test_rwlock_writer_excludes_readers():
    from rwlock import RWLock

    lock = RWLock()
    reader_inside = threading.Event()
    release_reader = threading.Event()
    errors = []

    def reader():
        try:
            with lock.read():
                reader_inside.set()
                release_reader.wait(5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t_r = threading.Thread(target=reader)
    t_r.start()
    assert reader_inside.wait(5), "读者未进入临界区"

    # 读者持有时：写者必须拿不到锁
    assert lock.acquire_write(timeout=0.05) is False, "写者不应在读者持有时拿到锁"
    assert lock.locked_for_write is False

    release_reader.set()
    t_r.join()
    assert not errors, errors

    # 读者释放后写者可以拿到
    assert lock.acquire_write(timeout=5) is True
    assert lock.locked_for_write is True
    lock.release_write()
    assert lock.locked_for_write is False


def test_rwlock_reader_blocks_while_writer_holds():
    from rwlock import RWLock

    lock = RWLock()
    assert lock.acquire_write(timeout=1) is True
    assert lock.acquire_read(timeout=0.05) is False, "写者持有时读者必须排队"
    lock.release_write()
    assert lock.acquire_read(timeout=1) is True
    lock.release_read()


def test_rwlock_mismatched_release_raises():
    from rwlock import RWLock

    lock = RWLock()
    try:
        lock.release_read()
    except RuntimeError:
        pass
    else:
        raise AssertionError("无配对的 release_read() 必须报错")

    try:
        lock.release_write()
    except RuntimeError:
        pass
    else:
        raise AssertionError("无配对的 release_write() 必须报错")


# ================================================================ write_queue
def test_write_queue_fifo_and_flush():
    from write_queue import WriteQueue

    q = WriteQueue(name="test-fifo")
    p = tmp_path("w_fifo.log")
    for i in range(50):
        q.submit_append_line(p, f"{i}")
    q.flush(timeout=10)
    assert q.is_idle is True
    q.close(timeout=10)

    with open(p, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]
    assert lines == [str(i) for i in range(50)], lines


def test_write_queue_survives_task_error():
    from write_queue import WriteQueue

    q = WriteQueue(name="test-errors")
    p = tmp_path("w_err.json")

    def boom():
        raise ValueError("expected failure")

    q.submit(boom)
    q.submit_json(p, {"ok": 1})
    q.flush(timeout=10)
    q.close(timeout=10)

    assert len(q.errors) >= 1, "失败任务必须被记录"
    with open(p, "r", encoding="utf-8") as fh:
        assert json.load(fh) == {"ok": 1}, "一个任务失败不得影响后续任务"


def test_write_queue_rejects_after_close():
    from write_queue import WriteQueue

    q = WriteQueue(name="test-closed")
    q.close(timeout=10)
    try:
        q.submit(lambda: None)
    except Exception:
        return
    raise AssertionError("close() 之后 submit 必须抛异常")


def test_write_queue_no_lost_updates():
    from atomic_io import read_json
    from write_queue import WriteQueue

    q = WriteQueue(name="test-rmw")
    p = tmp_path("w_rmw.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"n": 0}, fh)

    def bump():
        with open(p, "r", encoding="utf-8") as fh:
            cur = json.load(fh)
        cur["n"] += 1
        from atomic_io import atomic_write_json
        atomic_write_json(p, cur)

    def worker():
        for _ in range(50):
            q.submit(bump)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    q.flush(timeout=30)
    q.close(timeout=30)

    assert read_json(p)["n"] == 200, read_json(p)
    assert not q.errors, q.errors


# ================================================================ redaction
def test_redact_sensitive_values():
    from redaction import redact, PLACEHOLDER

    text = ("联系 alice@example.com，token=abcdef0123456789 "
            "password=secret123")
    out = redact(text)
    assert "alice@example.com" not in out, out
    assert PLACEHOLDER in out, out
    assert "secret123" not in out, out
    assert "abcdef0123456789" not in out, out

    assert PLACEHOLDER in redact("手机号 13800138000")
    assert PLACEHOLDER in redact("hash " + "ab" * 32)
    assert PLACEHOLDER in redact("Bearer " + "x" * 32)
    assert PLACEHOLDER in redact("sk-" + "k" * 20)
    assert PLACEHOLDER in redact("访问 192.168.1.10:8080")


def test_redact_path_username():
    from redaction import redact_path

    out = redact_path(r"C:\Users\alice\secret\note.txt")
    assert "alice" not in out, out
    assert r"C:\Users\<USER>" in out, out

    out2 = redact_path("/home/bob/x.log")
    assert "bob" not in out2, out2


def test_redact_truncates_and_type_guards():
    from redaction import redact, MAX_QUERY_LEN

    assert redact(None) == ""
    assert redact(12345) == ""
    # 用 CJK 文本，避免撞上 base64/hex 规则被整段替换成占位符
    assert redact("好" * 5000) == "好" * MAX_QUERY_LEN
    assert len(redact("好" * 400)) == 400


# ================================================================ badcase_replay
def test_badcase_sample_key_stable_and_unique():
    from badcase_replay import _sample_key

    k1 = _sample_key("删除文件 a.txt", "delete_file", "1.json")
    k2 = _sample_key("删除文件 a.txt", "delete_file", "1.json")
    k3 = _sample_key("删除文件 a.txt", "delete_file", "2.json")
    assert k1 == k2, "同一 case 的去重键必须稳定"
    assert k1 != k3, "不同来源必须产生不同去重键"
    assert len(k1) == 40 and set(k1) <= set("0123456789abcdef"), k1


def test_badcase_load_returns_redacted_records():
    from badcase_replay import load_bad_cases

    cases = load_bad_cases(limit=50)
    assert isinstance(cases, list)
    for case in cases:
        # load_bad_cases 落盘前已脱敏：不得再带真实邮箱/手机号
        assert "@example.com" not in case.get("user_query", ""), case
        assert case.get("_source"), "每条 case 需记录来源文件名"


def test_badcase_transitions_from_record():
    from agent_rl import AgentRL, PrioritizedReplayBuffer
    from badcase_replay import transitions_for_case

    agent = AgentRL(marl_vote=False)
    agent.replay_buffer = PrioritizedReplayBuffer(capacity=64)

    # 缺字段 / 未知意图 → 不产生 transition
    assert transitions_for_case(agent, {}) == []
    assert transitions_for_case(agent, {
        "user_query": "删除文件 a.txt", "final_intent": "not_a_real_intent"}) == []

    # 判错 → 至少一条负奖励样本
    wrong = {"user_query": "删除文件 a.txt", "final_intent": "delete_file"}
    neg = transitions_for_case(agent, wrong)
    assert neg, "失败案例必须生成 transition"
    assert all(len(t) == 5 for t in neg), "transition 需是 (s, a, r, s', done)"
    assert all(float(t[2]) < 0 for t in neg), "判错样本必须是负奖励"

    # 有 ground truth → 追加一条正奖励样本
    labeled = dict(wrong, expected_intent="query")
    both = transitions_for_case(agent, labeled)
    assert len(both) == len(neg) + 1, "应额外补一条正确意图的正样本"
    assert float(both[-1][2]) > 0, "正确意图样本必须是正奖励"


# ================================================================ regression_gate
def test_regression_set_deterministic():
    from regression_gate import build_regression_set

    a = build_regression_set()
    b = build_regression_set()
    dump = lambda s: json.dumps(s, ensure_ascii=False, sort_keys=True)
    assert [dump(x) for x in a] == [dump(x) for x in b]
    assert len(a) >= 50, f"回归集过小: {len(a)}"
    labels = {s["label"] for s in a}
    assert len(labels) >= 8, f"意图覆盖过少: {sorted(labels)}"


def test_regression_ensure_set_stable_hash():
    from regression_gate import ensure_regression_set

    s1, h1 = ensure_regression_set()
    s2, h2 = ensure_regression_set()
    assert h1 == h2, "同一磁盘集合两次读取哈希必须一致"
    assert len(s1) == len(s2) and h1


def test_sha256_ignores_line_endings():
    """同一份回归集在 CRLF checkout 下必须算出同一指纹。

    core.autocrlf 会在 Windows 把文件改成 CRLF；指纹若对原始字节哈希，
    同一集合在两台机器上会得到不同 sha256，gate 把机器差异当成篡改。
    """
    from regression_gate import _sha256_file

    lf = tmp_path("set_lf.jsonl")
    crlf = tmp_path("set_crlf.jsonl")
    with open(lf, "wb") as fh:
        fh.write('{"text":"a","label":"q"}\n{"text":"b","label":"q"}\n'.encode("utf-8"))
    with open(crlf, "wb") as fh:
        fh.write('{"text":"a","label":"q"}\r\n{"text":"b","label":"q"}\r\n'.encode("utf-8"))
    assert _sha256_file(Path(lf)) == _sha256_file(Path(crlf))

    # 内容真的不同 → 指纹必须不同（不能因为归一化而失去区分度）
    diff = tmp_path("set_diff.jsonl")
    with open(diff, "wb") as fh:
        fh.write('{"text":"c","label":"q"}\n'.encode("utf-8"))
    assert _sha256_file(Path(lf)) != _sha256_file(Path(diff))


def test_regression_check_flags_regression():
    from regression_gate import GateMetrics, GateThresholds, check

    baseline = GateMetrics(
        overall_accuracy=0.80, macro_accuracy=0.75, sample_count=120,
        set_hash="abc", per_intent={"delete_file": {"accuracy": 0.90, "total": 10}},
    )
    thresholds = GateThresholds()
    worse = GateMetrics(
        overall_accuracy=0.70, macro_accuracy=0.70, sample_count=120,
        set_hash="abc", per_intent={"delete_file": {"accuracy": 0.70, "total": 10}},
    )
    res = check(worse, baseline, thresholds, "abc")
    assert not res.passed, res.reasons
    assert any("overall" in r for r in res.reasons), res.reasons
    assert any("delete_file" in r for r in res.reasons), res.reasons

    # 集合内容被改 → 必须拦下
    tampered = GateMetrics(
        overall_accuracy=0.99, macro_accuracy=0.99, sample_count=120,
        set_hash="abc", per_intent={"delete_file": {"accuracy": 0.99, "total": 10}},
    )
    res2 = check(tampered, baseline, thresholds, "CHANGED")
    assert not res2.passed and any("sha256" in r for r in res2.reasons), res2.reasons

    # 与基线持平 → 通过
    same = GateMetrics(
        overall_accuracy=0.80, macro_accuracy=0.75, sample_count=120,
        set_hash="abc", per_intent={"delete_file": {"accuracy": 0.90, "total": 10}},
    )
    res3 = check(same, baseline, thresholds, "abc")
    assert res3.passed, res3.reasons


def test_regression_thresholds_roundtrip():
    from regression_gate import GateThresholds, GateMetrics

    t = GateThresholds(tolerance=0.05, min_samples=10)
    assert GateThresholds.from_dict(t.to_dict()).tolerance == 0.05
    m = GateMetrics(overall_accuracy=0.5, sample_count=10, set_hash="h")
    assert GateMetrics.from_dict(m.to_dict()).set_hash == "h"


def test_set_hash_matches_baseline():
    """本地回归集哈希必须与已提交基线一致，否则 CI 一定失败。"""
    from regression_gate import ensure_regression_set, load_baseline

    _, sha = ensure_regression_set()
    baseline, _ = load_baseline()
    if baseline is None or not baseline.set_hash:
        return   # 基线缺失时不在此测试内报错
    assert sha == baseline.set_hash, (
        f"回归集哈希 {sha[:12]} != 基线 {baseline.set_hash[:12]}，"
        "集合被改动过？请 python regression_gate.py --update-baseline"
    )


# ================================================================ PER / τ / Engram
def test_per_priority_cleans_nonfinite():
    from agent_rl import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(capacity=32)
    for i in range(3):
        buf.push(np.zeros(4), i, -1.0, np.zeros(4), True)
    batch, indices, weights = buf.sample(3)
    assert len(batch) == 3
    assert np.all(np.isfinite(weights))

    buf.update_priorities(indices, np.array([np.nan, np.inf, -np.inf]))
    assert math.isfinite(buf.tree.total), "优先级树被 NaN/Inf 破坏"
    assert buf.tree.total > 0

    buf.update_priorities(indices, np.array([1e9, 1e9, 1e9]))
    assert math.isfinite(buf.tree.total), "超大 TD 未被 td_clip 裁剪"


def test_per_sample_empty_buffer_safe():
    from agent_rl import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(capacity=8)
    batch, indices, weights = buf.sample(4)
    assert batch == [] and len(indices) == 0


def test_tau_anneals_to_base():
    from agent_rl import AgentRL

    a = AgentRL.__new__(AgentRL)
    a.tau_init = 0.05
    a.tau_base = 0.005
    a.tau_anneal_steps = 1000
    a.training_step = 0
    assert abs(a._current_tau() - 0.05) < 1e-9
    a.training_step = 500
    mid = a._current_tau()
    assert 0.005 < mid < 0.05, mid
    a.training_step = 10_000
    assert abs(a._current_tau() - 0.005) < 1e-9, "退火到 tau_base 后应保持不变"


def test_engram_absent_abstains():
    from agent_rl import AgentRL

    a = AgentRL.__new__(AgentRL)
    a.engram = None
    vote = a._engram_vote(np.zeros(8, dtype=np.float32), 0.95)
    assert vote == (None, 0.0), "无记忆时必须弃权，绝不参与投票"


# ================================================================ harness
def _cleanup():
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


# 直接 python test_infra.py 时 run_all 会清理；被 pytest 收集时靠 atexit 兜底
import atexit as _atexit
_atexit.register(_cleanup)


def run_all() -> bool:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]

    failed = []
    started = time.time()
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print("[PASS]")
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    _cleanup()

    print("\n" + "=" * 70)
    if failed:
        print(f"  失败 {len(failed)}/{len(tests)}:")
        for name, err in failed:
            print(f"    - {name}: {err}")
        print("=" * 70)
        return False
    print(f"  全部通过 {len(tests)}/{len(tests)}  ({time.time() - started:.1f}s)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
