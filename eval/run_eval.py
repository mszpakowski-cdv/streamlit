"""
run_eval.py — uruchamia pełny pipeline (retrieval + Bielik) na zbiorze testowym
i zapisuje surowe wyniki + tanie metryki obiektywne do oceny.

Pipeline odwzorowuje app.py (TOP_K=5, ten sam SYSTEM_PROMPT i format promptu).
Bielik liczony na GPU (embedder/reranker są na CPU w rag.py, więc nie zabierają
VRAM); ~5-25 s/przypadek zamiast ~90 s na CPU.

Zapis przyrostowy do eval/results.jsonl (wznawialny: pomija już policzone id).

Użycie:
    PYTHONUTF8=1 python eval/run_eval.py            # cały zbiór
    PYTHONUTF8=1 python eval/run_eval.py --limit 50 # pilotaż
"""
import os, re, sys, json, time, argparse, subprocess

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from rag import retrieve, format_context, free_resources

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "hf.co/gaianet/Bielik-4.5B-v3.0-Instruct-GGUF:Q6_K"
TOP_K = 5

TESTCASES = "eval/testcases.jsonl"
RESULTS = "eval/results.jsonl"
METADATA = "vectorstore/metadata.json"  # żywy korpus (zawiera też art. k.w. z taryfikatora)

DOC2SRC = {"PoRD": "prawo_o_ruchu_drogowym", "KPW": "kpw", "TARYF": "taryfikator"}

# Ten sam system prompt co w app.py (krytyczne ramowanie).
SYSTEM_PROMPT = """Jesteś asystentem prawnym dla polskich policjantów drogówki. Oceniasz, czy opisana
sytuacja drogowa stanowi naruszenie przepisów, na podstawie DOSTARCZONYCH przepisów.

STANDARD OCENY:
Stosujesz standard administracyjny (jak w sprawach o wykroczenia drogowe), NIE karny.
NIE wymagaj pewności „ponad wszelką wątpliwość". Naruszenie stwierdzasz, gdy z faktów
i treści przepisu wynika ono w sposób przeważający (bardziej prawdopodobne niż nie).
Przeoczenie realnego naruszenia jest BŁĘDEM TAK SAMO POWAŻNYM jak uznanie czystej
sytuacji za naruszenie. Nie faworyzuj żadnej z odpowiedzi z góry — oceniaj fakty.

SPOSÓB PRACY:
1. Opieraj się WYŁĄCZNIE na przepisach z sekcji „PRZEPISY (KONTEKST)". Cytuj dokładny
   numer artykułu i ustępu z tego kontekstu.
2. Dla każdego istotnego przepisu wykonaj ślad faktów:
   a) Przytocz, co przepis NAKAZUJE lub ZABRANIA (kto, komu, co, w którą stronę, pod
      jakim warunkiem, z jakim wyjątkiem). Czytaj dosłownie i nie odwracaj kierunku
      reguły (np. „polecenia osoby kierującej ruchem MAJĄ pierwszeństwo przed sygnałami
      świetlnymi" — nie na odwrót).
   b) Wypisz fakty z opisu — co kierujący FAKTYCZNIE zrobił lub czego zaniechał. Nie
      dopisuj zachowań, których w opisie nie ma (np. nie zakładaj, że „kierujący
      ustąpił", jeśli opis tego nie mówi).
   c) Porównaj: czy fakty spełniają obowiązek, czy go naruszają. Zaniechanie obowiązku
      (np. niezachowanie szczególnej ostrożności) jest naruszeniem tak samo jak
      działanie zakazane.
3. Traktuj obowiązek jako całość — nie rozbijaj jednego obowiązku na osobne reguły, by
   każdą z osobna oddalić.
4. Zanim orzekniesz naruszenie, sprawdź WYJĄTKI — czy treść przepisu nie przewiduje
   okoliczności wyłączającej naruszenie (np. pojazd uprzywilejowany z włączonymi
   sygnałami; zachowanie wprost dopuszczone przez przepis jako wariant; pierwszeństwo po
   stronie kierującego). Jeśli przepis przewiduje taki wyjątek i tu on zachodzi — to
   „Brak naruszenia".
5. Werdykt: jeśli fakty naruszają przepis i nie zachodzi wyjątek — napisz, że doszło do
   naruszenia, i wskaż artykuł; jeśli naruszenie nie wynika — napisz wprost „Brak
   naruszenia" i wyjaśnij, którego warunku nie spełniono; jeśli przepisy nie wystarczają
   — powiedz to otwarcie.

KARA, MANDAT, KWALIFIKACJA:
Jeśli w sekcji „PRZEPISY (KONTEKST)" jest pozycja taryfikatora mandatów odpowiadająca
czynowi — podaj kwalifikację Kodeksu wykroczeń (np. „art. 92a § 1 k.w.") oraz kwotę
grzywny dokładnie tak, jak w taryfikatorze, i zaznacz, że pochodzi z taryfikatora.
Jeśli odpowiedniej pozycji NIE ma w kontekście — NIE wymyślaj artykułu k.w., kwoty ani
punktów. Napisz, że nie wynika to z dostarczonych przepisów, albo umieść pod flagą
„⚠️ Uwaga: poniższe nie wynika z dostarczonych przepisów, podaję z wiedzy ogólnej:".

STRUKTURA ODPOWIEDZI:
1. Analiza — ślad faktów (obowiązek → fakty → wniosek)
2. Ocena — naruszone przepisy ruchu (z numerami) ALBO wyraźne „Brak naruszenia"
3. Kara / mandat — kwalifikacja k.w. i kwota grzywny z taryfikatora w kontekście;
   jeśli brak pozycji, zaznacz to (nie zgaduj kwoty)
4. Tryb postępowania — tylko jeśli wynika z przepisów KPW w kontekście

Odpowiadaj zwięźle, konkretnie i po polsku."""


def norm_art(label: str) -> str:
    m = re.search(r'(\d+)\s*([a-z]?)', str(label))
    return f"{int(m.group(1))}{m.group(2)}" if m else str(label).strip().lower()


def corpus_article_nums():
    """Numery artykułów obecne w zaindeksowanym korpusie (per źródło).
    Czytane z żywych metadanych — dzięki temu artykuły Kodeksu wykroczeń
    wniesione przez taryfikator liczą się jako 'w korpusie'."""
    meta = json.load(open(METADATA, encoding="utf-8"))
    nums = {}
    for c in meta:
        if c["article"] != "brak numeru":
            nums.setdefault(c["source"], set()).add(norm_art(c["article"]))
    return nums


CORPUS = corpus_article_nums()
ALL_CORPUS_NUMS = set().union(*CORPUS.values())


def ask_bielik(user_prompt: str) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        # num_ctx=2048: nasze prompty mieszczą się (~1600 tok wejścia + ~450 wyjścia),
        # a mniejszy bufor KV pozwala zmieścić wagi w VRAM (model na GPU zamiast CPU).
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 2048},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def objective_signals(case, output, retrieved):
    """Tanie, deterministyczne sygnały (autorytatywną ocenę robi sędzia-LLM)."""
    # Artykuły obecne w kontekście (retrieval)
    retr = {(c["source"], norm_art(c["article"])) for c in retrieved}
    expected = {(DOC2SRC.get(a["doc"], a["doc"]), norm_art(a["art"]))
                for a in case.get("expected_articles", [])}
    retrieval_hit = bool(expected & retr) if expected else None

    # Numery artykułów zacytowane w odpowiedzi modelu
    cited = {norm_art(m) for m in re.findall(r'[Aa]rt\.?\s*\d+[a-z]?', output)}
    expected_nums = {norm_art(a["art"]) for a in case.get("expected_articles", [])}
    hit_nums = cited & expected_nums
    hallucinated = sorted(c for c in cited if c not in ALL_CORPUS_NUMS)

    low = output.lower()
    return {
        "retrieved_articles": sorted(f"{s}:{n}" for s, n in retr),
        "retrieval_hit": retrieval_hit,
        "cited_article_nums": sorted(cited),
        "expected_article_nums": sorted(expected_nums),
        "cited_expected_overlap": sorted(hit_nums),
        "hallucinated_article_nums": hallucinated,
        "has_warning_flag": ("wiedzy ogólnej" in low) or ("⚠" in output),
        "says_no_violation": ("brak naruszenia" in low) or ("miał pierwszeństwo" in low),
        "output_chars": len(output),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--infile", default=TESTCASES)
    ap.add_argument("--outfile", default=RESULTS)
    ap.add_argument("--fast", action="store_true",
                    help="Dwufazowo: cały retrieval na GPU (Bielik wyładowany), potem inferencja "
                         "Bielika na GPU. Znika przestój CPU-retrievalu między przypadkami.")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.infile, encoding="utf-8") if l.strip()]
    if args.limit:
        cases = cases[:args.limit]

    done = set()
    if os.path.exists(args.outfile):
        for l in open(args.outfile, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l).get("id"))
    todo = [c for c in cases if c["id"] not in done]
    # Postęp drukujemy na STDERR — harness buforuje stdout zadań w tle, a stderr leci na żywo.
    print(f"Zbiór: {len(cases)} | już policzone: {len(done)} | do zrobienia: {len(todo)}",
          file=sys.stderr, flush=True)

    def make_prompt(case, context):
        return (
            f"PRZEPISY (KONTEKST):\n{context}\n\n"
            f"SYTUACJA ZGŁOSZONA PRZEZ POLICJANTA:\n{case['scenario']}\n\n"
            "Oceń tę sytuację: wykonaj ślad faktów dla każdego istotnego przepisu "
            "(co przepis nakazuje/zabrania → co kierujący faktycznie zrobił → wniosek), "
            "podaj werdykt („naruszenie\" + artykuł, albo „Brak naruszenia\"), a "
            "kwalifikację k.w. i kwotę mandatu podaj wyłącznie z pozycji taryfikatora "
            "obecnej w kontekście — nie zgaduj."
        )

    # ── Faza A (tylko --fast): cały retrieval na GPU, Bielik wyładowany z VRAM ──
    precomp = {}  # id -> (context, chunks)
    _cuda = False
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        _cuda = False
    if args.fast and not _cuda:
        print("⚠️ --fast pominięte: torch bez CUDA (embedder zostaje na CPU).",
              file=sys.stderr, flush=True)
    if args.fast and _cuda and todo:
        print("⏳ Faza A: retrieval na GPU (zwalniam VRAM po Bieliku)...", file=sys.stderr, flush=True)
        subprocess.run(["ollama", "stop", MODEL], capture_output=True)
        os.environ["RAG_DEVICE"] = "cuda"
        tA = time.time()
        for i, case in enumerate(todo, 1):
            chunks = retrieve(case["scenario"], k=TOP_K)
            precomp[case["id"]] = (format_context(chunks), chunks)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - tA
                print(f"  retrieval [{i}/{len(todo)}] ETA {el/i*(len(todo)-i)/60:.0f} min",
                      file=sys.stderr, flush=True)
        free_resources()  # zwolnij embedder+reranker z GPU pod Bielika
        print(f"✅ Faza A gotowa w {(time.time()-tA)/60:.1f} min; ładuję Bielika...",
              file=sys.stderr, flush=True)

    out = open(args.outfile, "a", encoding="utf-8")
    t0 = time.time()
    for i, case in enumerate(todo, 1):
        try:
            if case["id"] in precomp:
                context, chunks = precomp[case["id"]]
            else:
                chunks = retrieve(case["scenario"], k=TOP_K)
                context = format_context(chunks)
            ts = time.time()
            output = ask_bielik(make_prompt(case, context))
            dur = round(time.time() - ts, 1)
            rec = {**case, "bielik_output": output, "latency_s": dur,
                   "signals": objective_signals(case, output, chunks)}
        except Exception as e:
            rec = {**case, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        overall = len(done) + i
        print(f"[{overall}/{len(cases)}] (tura {i}/{len(todo)}) {case['id']} "
              f"({rec.get('latency_s','ERR')}s) ETA {eta/60:.0f} min",
              file=sys.stderr, flush=True)
    out.close()
    print(f"\nGotowe. Wyniki: {args.outfile}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
