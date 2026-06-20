"""
build_pilot.py — buduje stratyfikowany podzbiór ~40 przypadków do szybkiej bramki
przed pełnym przebiegiem. Łączy: nowe przypadki taryfikatora, przeniesione
false-negatives i fabrykacje z poprzedniego runu, sprawy czyste/pułapki (kontrola FP)
oraz najsłabsze tematy.
Wyjście: eval/pilot.jsonl
"""
import json, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

tc = {json.loads(l)["id"]: json.loads(l)
      for l in open("eval/testcases.jsonl", encoding="utf-8") if l.strip()}
try:
    J = {j["id"]: j for j in json.load(open("eval/judgments.json", encoding="utf-8"))}
except Exception:
    J = {}

picked = {}
def add(ids, limit):
    c = 0
    for i in ids:
        if c >= limit:
            break
        if i in tc and i not in picked:
            picked[i] = tc[i]
            c += 1

# 1. Nowe przypadki taryfikatora (grunt penalty grounding) — 8
add([i for i in tc if tc[i]["topic"] == "taryfikator"], 8)
# 2. Przeniesione false-negatives (realne naruszenie przeoczone w poprzednim runie) — 10
add(sorted(i for i, j in J.items()
           if j.get("model_verdict") == "brak_naruszenia"
           and tc.get(i, {}).get("expected_verdict") == "naruszenie"), 10)
# 3. Przeniesione fabrykacje kary — 6
add(sorted(i for i, j in J.items() if j.get("penalty_handling") == "fabricated"), 6)
# 4. Sprawy czyste / pułapki (kontrola FP / bias) — 8
add(sorted(i for i in tc if tc[i]["expected_verdict"] != "naruszenie"), 8)
# 5. Najsłabsze tematy — po 2
for topic in ["kontrola_drogowa", "znaki_sygnaly", "swiatla", "wlaczanie", "strony_wlasciwosc"]:
    add(sorted(i for i in tc if tc[i]["topic"] == topic), 2)

rows = list(picked.values())
with open("eval/pilot.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Pilot: {len(rows)} przypadków → eval/pilot.jsonl")
print("Werdykt:", dict(Counter(r["expected_verdict"] for r in rows)))
print("Wymiar:", dict(Counter(r["dimension"] for r in rows)))
print("Tematy:", dict(Counter(r["topic"] for r in rows)))
