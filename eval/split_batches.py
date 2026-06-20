"""
split_batches.py — dzieli eval/results.jsonl na paczki do oceny przez sędziów-LLM.
Wyjście: eval/batches/batch_NN.jsonl  (domyślnie 20 przypadków na paczkę)
Drukuje listę bezwzględnych ścieżek paczek (do przekazania workflowowi jako args).
"""
import os, json, sys, glob

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 20

INFILE = sys.argv[2] if len(sys.argv) > 2 else "eval/results.jsonl"
rows = [l for l in open(INFILE, encoding="utf-8") if l.strip()]
os.makedirs("eval/batches", exist_ok=True)
for f in glob.glob("eval/batches/batch_*.jsonl"):
    os.remove(f)

paths = []
for i in range(0, len(rows), SIZE):
    n = i // SIZE
    path = f"eval/batches/batch_{n:02d}.jsonl"
    open(path, "w", encoding="utf-8").writelines(rows[i:i+SIZE])
    paths.append(os.path.join(ROOT, path).replace("\\", "/"))

print(f"{len(rows)} wyników -> {len(paths)} paczek po max {SIZE}")
print(json.dumps(paths, ensure_ascii=False))
