#!/usr/bin/env python3
"""
File reverse tool with confirmation and rollback.
Usage:
    python reverse_moves.py              # Interactive (default)
    python reverse_moves.py --dry-run    # Preview only
    python reverse_moves.py --yes        # Skip confirmation
    python reverse_moves.py --rollback   # Undo last reverse
"""

import os
import sys
import json
import re
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime_paths import LOG_DIR
from atomic_io import atomic_write_json

DESKTOP = Path(os.path.expanduser("~")) / "Desktop"
LOG_FILE = LOG_DIR / "reverse_moves_log.json"


def find_source_file():
    """Find the source txt file on Desktop (handles CJK filename)."""
    for f in DESKTOP.glob("*.txt"):
        if "文本文档" in f.name:
            return f
    return None


def parse_operations(source_file):
    """Parse the move log and return operation list."""
    operations = []
    with open(source_file, "r", encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\s+(.+?)\s+->\s+(.+?)\s+\(", line)
            if m:
                dest = m.group(1).strip()
                src = m.group(2).strip()
                src_exists = os.path.isfile(src)
                dest_exists = os.path.isfile(dest)

                if not src_exists:
                    status = "skip_no_src"
                elif dest_exists:
                    status = "skip_has_dst"
                else:
                    status = "pending"

                operations.append({
                    "source": src,
                    "destination": dest,
                    "src_exists": src_exists,
                    "dest_exists": dest_exists,
                    "status": status,
                })
    return operations


def do_rollback():
    """Undo a previous reverse operation."""
    if not LOG_FILE.exists():
        print("[ERROR] Rollback log not found:", LOG_FILE)
        sys.exit(1)

    with open(LOG_FILE, "r", encoding="utf-8") as fh:
        log = json.load(fh)

    ops = log.get("operations", [])
    print("=" * 50)
    print(" Rollback: moving files back to docs/")
    print(f" Records: {len(ops)}")
    print("=" * 50)

    ok = fail = 0
    for op in ops:
        src = op["destination"]  # current location
        dst = op["source"]       # original location

        if not os.path.isfile(src):
            fail += 1
            print(f"  [SKIP] {os.path.basename(src)}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.move(src, dst)
            ok += 1
            print(f"  [OK] {os.path.basename(src)}")
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {os.path.basename(src)}: {e}")

    print()
    color = "green" if fail == 0 else "yellow"
    print(f"Rollback done: success {ok}, failed {fail}")
    return ok, fail


def do_reverse(dry_run=False, auto_confirm=False):
    """Reverse the file moves."""
    source_file = find_source_file()
    if not source_file:
        print("[ERROR] Source file not found on Desktop")
        sys.exit(1)

    operations = parse_operations(source_file)

    total = len(operations)
    pending = sum(1 for o in operations if o["status"] == "pending")
    skip_no_src = sum(1 for o in operations if o["status"] == "skip_no_src")
    skip_has_dst = sum(1 for o in operations if o["status"] == "skip_has_dst")

    print()
    print("=" * 50)
    print(" Reverse Preview")
    print("=" * 50)
    print(f" Total records:        {total}")
    print(f" Pending:              {pending}")
    print(f" Skip (no source):     {skip_no_src}")
    print(f" Skip (has dest):      {skip_has_dst}")
    print("=" * 50)

    pending_ops = [o for o in operations if o["status"] == "pending"]

    if pending_ops:
        show = min(20, len(pending_ops))
        print(f"\nPending files (first {show}):")
        for i, op in enumerate(pending_ops[:20]):
            leaf = os.path.basename(op["source"])
            to_dir = os.path.dirname(op["destination"])
            print(f"  {i+1}. {leaf}")
            print(f"     -> {to_dir}")
        if len(pending_ops) > 20:
            print(f"  ... and {len(pending_ops) - 20} more")

    if dry_run:
        print("\n[DryRun] Preview only, no changes made.")
        return

    if pending == 0:
        print("\nNothing to reverse.")
        return

    if not auto_confirm:
        print()
        ans = input(f"Reverse {pending} files? (type YES to proceed): ")
        if ans != "YES":
            print("Cancelled.")
            return

    print("\nReversing...")
    ok = fail = 0
    log_ops = []

    for op in pending_ops:
        src = op["source"]
        dst = op["destination"]

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.move(src, dst)
            ok += 1
            log_ops.append(op)
            print(f"  [{ok}/{pending}] OK: {os.path.basename(src)}")
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {os.path.basename(src)}: {e}")

    # Save rollback log
    atomic_write_json(LOG_FILE, {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "operations": log_ops,
    })

    print()
    print("=" * 50)
    print(" Done")
    print("=" * 50)
    print(f" Success: {ok}")
    if fail:
        print(f" Failed:  {fail}")
    print(f" Log:     {LOG_FILE}")
    print()
    print("To undo, run: python reverse_moves.py --rollback")


def main():
    parser = argparse.ArgumentParser(description="Reverse file moves with rollback")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--rollback", action="store_true", help="Undo last reverse")
    args = parser.parse_args()

    if args.rollback:
        do_rollback()
    else:
        do_reverse(dry_run=args.dry_run, auto_confirm=args.yes)


if __name__ == "__main__":
    main()
