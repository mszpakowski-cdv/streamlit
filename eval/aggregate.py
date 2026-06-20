"""
aggregate.py — składa zweryfikowane przypadki z workflowu w finalny zbiór.

Wejście:  eval/raw_cases.json  (lista obiektów case z polem 'topic')
Wyjście:  eval/testcases.jsonl  + raport rozkładu/pokrycia na stdout

Nadaje id, usuwa duplikaty (po znormalizowanym scenario), raportuje rozkład
werdyktów/wymiarów/trudności i pokrycie artykułów względem eval/articles.json.
"""
import json, re, sys, os
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

RAW = "eval/raw_cases.json"
OUT = "eval/testcases.jsonl"
ARTICLES = "eval/articles.json"


def norm_scen(s):
    return re.sub(r'\W+', ' ', s.lower()).strip()


def norm_art(label):
    m = re.search(r'(\d+)\s*([a-z]?)', str(label))
    return f"{int(m.group(1))}{m.group(2)}" if m else str(label)


def main():
    cases = json.load(open(RAW, encoding="utf-8"))
    seen, final = set(), []
    counter = Counter()
    for c in cases:
        key = norm_scen(c["scenario"])
        if key in seen:
            continue
        seen.add(key)
        counter[c["topic"]] += 1
        c["id"] = f"{c['topic']}-{counter[c['topic']]:03d}"
        final.append(c)

    with open(OUT, "w", encoding="utf-8") as f:
        for c in final:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- Raport ---
    print(f"Finalny zbiór: {len(final)} przypadków (z {len(cases)} surowych; "
          f"{len(cases)-len(final)} duplikatów usunięto)\n")

    def pct(counter_):
        tot = sum(counter_.values()) or 1
        return ", ".join(f"{k}: {v} ({100*v/tot:.0f}%)" for k, v in counter_.most_common())

    print("WERDYKT:   ", pct(Counter(c["expected_verdict"] for c in final)))
    print("WYMIAR:    ", pct(Counter(c["dimension"] for c in final)))
    print("TRUDNOŚĆ:  ", pct(Counter(c["difficulty"] for c in final)))
    print("PENALTY:   ", pct(Counter(c["penalty_grounding"] for c in final)))

    print("\nPER TEMAT:")
    for topic, n in Counter(c["topic"] for c in final).most_common():
        nv = sum(1 for c in final if c["topic"] == topic and c["expected_verdict"] != "naruszenie")
        print(f"  {topic:32s} {n:3d}  (brak/pułapka: {nv})")

    # Pokrycie artykułów
    art = json.load(open(ARTICLES, encoding="utf-8"))
    corpus = {norm_art(k) for d in art.values() for k in d if k != "brak numeru"}
    used = {norm_art(a["art"]) for c in final for a in c.get("expected_articles", [])}
    print(f"\nPOKRYCIE ARTYKUŁÓW: {len(used)} unikalnych artykułów użytych w testach "
          f"(korpus: {len(corpus)})")
    print(f"  Artykuły testowane spoza korpusu (potencjalny błąd labela): "
          f"{sorted(used - corpus) or 'brak'}")


if __name__ == "__main__":
    main()
