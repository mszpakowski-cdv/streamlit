export const meta = {
  name: 'eval-grade',
  description: 'Claude-as-judge grading of Bielik eval outputs against gold labels, in batches',
  phases: [{ title: 'Judge', detail: 'one judge agent per batch of results' }],
}

// Liczba paczek z eval/split_batches.py — ustaw pod konkretny przebieg (pilot=3, pełny 522=27)
const N_BATCHES = 27
const BATCHES = Array.from({ length: N_BATCHES }, (_, i) =>
  `C:/Users/mszak/Projects/streamlit/eval/batches/batch_${String(i).padStart(2, '0')}.jsonl`)

const JUDG = {
  type: 'object', additionalProperties: false,
  properties: {
    id: { type: 'string' },
    model_verdict: { type: 'string', enum: ['naruszenie', 'brak_naruszenia', 'zalezy', 'niejasny'] },
    verdict_correct: { type: 'boolean' },
    legal_correctness: { type: 'string', enum: ['correct', 'partially_correct', 'incorrect'] },
    article_assessment: { type: 'string', enum: ['on_point', 'acceptable', 'missing', 'wrong'] },
    hallucinated_law: { type: 'boolean' },
    penalty_handling: { type: 'string', enum: ['na', 'correct_from_context', 'fabricated', 'correct_absent_flagged', 'omitted'] },
    error_categories: { type: 'array', items: { type: 'string', enum: [
      'E1_halluc_article', 'E2_misread_condition', 'E3_guilt_bias', 'E4_fabricated_penalty',
      'E5_source_confusion', 'E6_under_detection', 'E7_procedural_error', 'E8_over_citation'] } },
    severity: { type: 'string', enum: ['none', 'minor', 'major'] },
    rationale: { type: 'string' },
  },
  required: ['id', 'model_verdict', 'verdict_correct', 'legal_correctness', 'article_assessment',
    'hallucinated_law', 'penalty_handling', 'error_categories', 'severity', 'rationale'],
}
const SCHEMA = { type: 'object', additionalProperties: false, properties: { judgments: { type: 'array', items: JUDG } }, required: ['judgments'] }

function judgePrompt(path) {
  return `Jesteś surowym, kompetentnym sędzią oceniającym asystenta prawnego dla policji drogowej (model Bielik 4.5B). Oceniasz JAKOŚĆ odpowiedzi modelu względem zweryfikowanego wzorca (gold).

KROK 1 — przeczytaj (Read) plik z wynikami: ${path}
Każdy wiersz to JSON: pola gold (scenario, expected_verdict, expected_articles, distractor_articles, penalty_grounding, key_reasoning, dimension, difficulty) + odpowiedź modelu w "bielik_output" + tanie "signals".

KROK 2 — dla KAŻDEGO wiersza wystaw ocenę. Zasady:
- model_verdict: co model FAKTYCZNIE orzekł (naruszenie / brak_naruszenia / zalezy / niejasny).
- verdict_correct: czy model_verdict zgadza się z expected_verdict (zalezy traktuj elastycznie).
- legal_correctness: "correct" = trafny werdykt I powołanie GENUINIE właściwego przepisu I brak istotnej halucynacji; "partially_correct" = trafny werdykt, ale zły/brakujący artykuł lub drobna wada; "incorrect" = zły werdykt lub zmyślone prawo.
- article_assessment: on_point / acceptable / missing / wrong. UWAGA: gold expected_articles bywa węższy niż idealny (generator widział tylko swoją paczkę). Zalicz KAŻDY faktycznie właściwy artykuł; karz tylko za przepisy nieadekwatne lub zmyślone.
- hallucinated_law: true, jeśli model cytuje artykuł, kwotę lub kwalifikację, których NIE ma w sekcji PRZEPISY (KONTEKST) i które z niej nie wynikają, albo wymyśla treść przepisu. W kontekście mogą być: Prawo o ruchu drogowym, Kodeks postępowania w sprawach o wykroczenia ORAZ taryfikator mandatów (zawiera kwalifikację k.w. i kwotę grzywny). Powołanie artykułu k.w. lub kwoty ZGODNE z pozycją taryfikatora obecną w kontekście NIE jest halucynacją; zmyślony numer/kwota niewynikające z kontekstu = halucynacja.
- penalty_handling: "correct_from_context" (podał kwalifikację k.w. i/lub kwotę grzywny zgodną z pozycją taryfikatora obecną w KONTEKŚCIE), "correct_absent_flagged" (właściwej pozycji nie było w kontekście, a model to zaznaczył lub nie zgadywał kwoty), "fabricated" (podał kwotę, punkty lub artykuł k.w. niezgodne z kontekstem albo zmyślone jako fakt), "omitted" (należało podać karę z dostępnej w kontekście pozycji taryfikatora, a pominął), "na" (kara nieistotna dla sprawy). Sprawdzaj zgodność kwoty i kwalifikacji k.w. z pozycją taryfikatora w kontekście.
- error_categories: podzbiór [E1_halluc_article, E2_misread_condition, E3_guilt_bias, E4_fabricated_penalty, E5_source_confusion, E6_under_detection, E7_procedural_error, E8_over_citation].
- severity: "major" = zły werdykt lub pewnie podana zmyślona podstawa prawna; "minor" = dobry werdykt, wadliwe cytowanie/rozumowanie; "none" = czysto.
- rationale: 1-2 zdania po polsku, konkretnie dlaczego taka ocena.

Zwróć obiekt {judgments:[...]} z jedną oceną na każdy wiersz (zachowaj id). Oceniaj merytorycznie, korzystając z własnej wiedzy prawniczej i gold key_reasoning jako odniesienia.`
}

phase('Judge')
const out = await parallel(BATCHES.map((p, i) => () =>
  agent(judgePrompt(p), { label: `judge:${i}`, phase: 'Judge', schema: SCHEMA })
))

let judgments = []
for (const r of out) if (r && r.judgments) judgments.push(...r.judgments)
log(`Oceniono ${judgments.length} przypadków w ${BATCHES.length} paczkach`)
return { count: judgments.length, judgments }
