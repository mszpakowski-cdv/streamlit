"""
report.py — łączy oceny sędziów (eval/judgments.json) z gold labelami i sygnałami
retrievalu (eval/results.jsonl) i generuje raport: eval/report.md.

Pokazuje: trafność werdyktów, BIAS (false-positive na czystych sprawach),
under-detection, halucynacje, fabrykowanie kar, kategorie błędów E1-E8,
rozbicie wg tematu/wymiaru/trudności oraz retrieval-vs-reasoning.
"""
import json, sys, re, argparse
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="eval/results.jsonl")
ap.add_argument("--judgments", default="eval/judgments.json")
ap.add_argument("--out", default="eval/report.md")
args = ap.parse_args()


def _model_name():
    try:
        txt = open("eval/run_eval.py", encoding="utf-8").read()
        m = re.search(r'^MODEL\s*=\s*"([^"]+)"', txt, re.M)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


MODEL_NAME = _model_name()

results = {json.loads(l)["id"]: json.loads(l)
           for l in open(args.results, encoding="utf-8") if l.strip()}
judg = json.load(open(args.judgments, encoding="utf-8"))
J = {j["id"]: j for j in judg}

ids = [i for i in results if i in J]
n = len(ids)


def gold(i): return results[i]
def jd(i): return J[i]


def rate(pred):
    hit = [i for i in ids if pred(i)]
    return len(hit), (100*len(hit)/n if n else 0), hit


out = []
def w(s=""): out.append(s)

w(f"# Raport ewaluacji — Agent Kodeksu Drogowego (Bielik 4.5B)")
w(f"\nPrzypadków ocenionych: **{n}**  ·  model: `{MODEL_NAME}`\n")

# --- Werdykt ---
vc, vcp, _ = rate(lambda i: jd(i).get("verdict_correct"))
w(f"## Trafność werdyktu (naruszenie / brak naruszenia)\n")
w(f"- **Poprawny werdykt: {vc}/{n} ({vcp:.0f}%)**")

# Bias: gold = brak_naruszenia, model orzekł naruszenie
clean = [i for i in ids if gold(i)["expected_verdict"] != "naruszenie"]
fp = [i for i in clean if jd(i).get("model_verdict") == "naruszenie"]
viol = [i for i in ids if gold(i)["expected_verdict"] == "naruszenie"]
fn = [i for i in viol if jd(i).get("model_verdict") in ("brak_naruszenia", "niejasny")]
w(f"- **BIAS / false-positive** (czysta sprawa uznana za naruszenie): "
  f"{len(fp)}/{len(clean)} ({100*len(fp)/max(len(clean),1):.0f}%)")
w(f"- **Under-detection / false-negative** (naruszenie przeoczone): "
  f"{len(fn)}/{len(viol)} ({100*len(fn)/max(len(viol),1):.0f}%)\n")

# --- Poprawność prawna ---
w("## Poprawność prawna (całościowa ocena sędziego)\n")
lc = Counter(jd(i).get("legal_correctness") for i in ids)
for k in ("correct", "partially_correct", "incorrect"):
    w(f"- {k}: {lc.get(k,0)} ({100*lc.get(k,0)/n:.0f}%)")
w("")

# --- Halucynacje / kary ---
hl, hlp, _ = rate(lambda i: jd(i).get("hallucinated_law"))
ph = Counter(jd(i).get("penalty_handling") for i in ids)
w("## Kontrola halucynacji i kar\n")
w(f"- **Halucynacja prawa** (zmyślony/źle przypisany artykuł): {hl} ({hlp:.0f}%)")
w(f"- Obsługa kar: correct_from_context={ph.get('correct_from_context',0)}, "
  f"fabricated={ph.get('fabricated',0)}, correct_absent_flagged={ph.get('correct_absent_flagged',0)}, "
  f"omitted={ph.get('omitted',0)}, n/a={ph.get('na',0)}\n")

# --- Kategorie błędów ---
w("## Kategorie błędów (E1-E8) — częstość\n")
ec = Counter()
for i in ids:
    for e in jd(i).get("error_categories", []) or []:
        ec[e] += 1
NAMES = {
 "E1_halluc_article":"E1 zmyślony artykuł", "E2_misread_condition":"E2 błędny warunek/kierunek",
 "E3_guilt_bias":"E3 stronniczość ku winie", "E4_fabricated_penalty":"E4 zmyślona kara",
 "E5_source_confusion":"E5 mieszanie źródeł", "E6_under_detection":"E6 przeoczone naruszenie",
 "E7_procedural_error":"E7 błąd proceduralny", "E8_over_citation":"E8 nadmiar cytowań",
}
w("| Kategoria | Liczba | % |")
w("|---|---:|---:|")
for e, c in ec.most_common():
    w(f"| {NAMES.get(e,e)} | {c} | {100*c/n:.0f}% |")
w("")

# --- Rozbicia ---
def breakdown(keyfn, title):
    w(f"## Trafność wg: {title}\n")
    w("| grupa | n | werdykt OK | prawnie correct |")
    w("|---|---:|---:|---:|")
    g = defaultdict(list)
    for i in ids:
        g[keyfn(i)].append(i)
    for k in sorted(g, key=lambda k: -len(g[k])):
        sub = g[k]
        vok = sum(1 for i in sub if jd(i).get("verdict_correct"))
        cok = sum(1 for i in sub if jd(i).get("legal_correctness") == "correct")
        w(f"| {k} | {len(sub)} | {100*vok/len(sub):.0f}% | {100*cok/len(sub):.0f}% |")
    w("")

breakdown(lambda i: gold(i).get("topic","?"), "temat")
breakdown(lambda i: gold(i).get("dimension","?"), "wymiar")
breakdown(lambda i: gold(i).get("difficulty","?"), "trudność")

# --- Retrieval vs reasoning ---
w("## Retrieval vs. rozumowanie\n")
hit = [i for i in ids if gold(i).get("signals",{}).get("retrieval_hit") is True]
miss = [i for i in ids if gold(i).get("signals",{}).get("retrieval_hit") is False]
reason_fail = [i for i in hit if jd(i).get("legal_correctness") != "correct"]
w(f"- Trafny artykuł BYŁ w kontekście: {len(hit)}; z nich błędna odpowiedź "
  f"(**błąd rozumowania**): {len(reason_fail)} ({100*len(reason_fail)/max(len(hit),1):.0f}%)")
w(f"- Trafnego artykułu NIE było w kontekście (**luka retrievalu**): {len(miss)}\n")

# --- Najgorsze tematy ---
w("## Najsłabsze tematy (najniższa poprawność prawna)\n")
g = defaultdict(list)
for i in ids: g[gold(i)["topic"]].append(i)
ranked = sorted(g, key=lambda k: sum(1 for i in g[k] if jd(i).get("legal_correctness")=="correct")/len(g[k]))
for k in ranked[:8]:
    sub=g[k]; cok=sum(1 for i in sub if jd(i).get("legal_correctness")=="correct")
    w(f"- **{k}**: {100*cok/len(sub):.0f}% correct ({cok}/{len(sub)})")
w("")

# --- Przykładowe porażki ---
w("## Przykładowe porażki (major)\n")
majors = [i for i in ids if jd(i).get("severity")=="major"][:12]
for i in majors:
    g_=gold(i); j_=jd(i)
    w(f"- `{i}` [{g_['topic']}/{g_['dimension']}] oczek.: **{g_['expected_verdict']}**, "
      f"model: **{j_.get('model_verdict')}** — {j_.get('rationale','')[:200]}")
w("")

open(args.out, "w", encoding="utf-8").write("\n".join(out))
print(f"Raport zapisany: {args.out} ({n} przypadków)")
print(f"Werdykt OK: {vcp:.0f}% | BIAS FP: {100*len(fp)/max(len(clean),1):.0f}% | "
      f"halucynacje: {hlp:.0f}% | prawnie correct: {100*lc.get('correct',0)/n:.0f}%")
