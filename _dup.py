import io
import re
import itertools

FILES = ("intent_shared.py", "agent_rl.py", "marl_agent.py", "router.py")
STR = re.compile(r'"([^"]{1,30})"')


def tables(path):
    text = io.open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r'\[[^\[\]]*\]|\([^\(\)]*\)', text):
        blob = m.group(0)
        items = STR.findall(blob)
        if len(items) >= 3:
            lineno = text.count("\n", 0, m.start()) + 1
            out.append((lineno, frozenset(items), " ".join(blob.split())[:90]))
    return out


data = {f: tables(f) for f in FILES}

print("== cross-file duplicate tables (jaccard >= 0.4) ==")
for a, b in itertools.combinations(FILES, 2):
    for la, sa, ta in data[a]:
        for lb, sb, tb in data[b]:
            j = len(sa & sb) / len(sa | sb)
            if j >= 0.4:
                print(f"  {a}:{la}  <->  {b}:{lb}   j={j:.2f}")
                print(f"      A: {ta}")
                print(f"      B: {tb}")

print("== same-file overlap (jaccard >= 0.6) ==")
for f in FILES:
    for (la, sa, ta), (lb, sb, tb) in itertools.combinations(data[f], 2):
        j = len(sa & sb) / len(sa | sb)
        if j >= 0.6:
            print(f"  {f}:{la} <-> {f}:{lb}  j={j:.2f}")
            print(f"      A: {ta}")
            print(f"      B: {tb}")
