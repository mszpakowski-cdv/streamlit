# Raport ewaluacji — Agent Kodeksu Drogowego (Bielik 4.5B)

Przypadków ocenionych: **42**  ·  model: `hf.co/gaianet/Bielik-4.5B-v3.0-Instruct-GGUF:Q6_K`

## Trafność werdyktu (naruszenie / brak naruszenia)

- **Poprawny werdykt: 31/42 (74%)**
- **BIAS / false-positive** (czysta sprawa uznana za naruszenie): 5/12 (42%)
- **Under-detection / false-negative** (naruszenie przeoczone): 5/30 (17%)

## Poprawność prawna (całościowa ocena sędziego)

- correct: 10 (24%)
- partially_correct: 13 (31%)
- incorrect: 19 (45%)

## Kontrola halucynacji i kar

- **Halucynacja prawa** (zmyślony/źle przypisany artykuł): 18 (43%)
- Obsługa kar: correct_from_context=5, fabricated=21, correct_absent_flagged=2, omitted=2, n/a=12

## Kategorie błędów (E1-E8) — częstość

| Kategoria | Liczba | % |
|---|---:|---:|
| E4 zmyślona kara | 21 | 50% |
| E5 mieszanie źródeł | 17 | 40% |
| E8 nadmiar cytowań | 15 | 36% |
| E2 błędny warunek/kierunek | 11 | 26% |
| E6 przeoczone naruszenie | 6 | 14% |
| E3 stronniczość ku winie | 6 | 14% |
| E7 błąd proceduralny | 4 | 10% |
| E1 zmyślony artykuł | 3 | 7% |

## Trafność wg: temat

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| cross_synthesis | 14 | 71% | 21% |
| taryfikator | 8 | 88% | 50% |
| def_zasady | 8 | 62% | 0% |
| dokumenty_stan_techniczny | 2 | 50% | 50% |
| kontrola_drogowa | 2 | 100% | 0% |
| znaki_sygnaly | 2 | 100% | 0% |
| swiatla | 2 | 50% | 0% |
| wlaczanie | 2 | 50% | 0% |
| strony_wlasciwosc | 2 | 100% | 100% |

## Trafność wg: wymiar

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| single | 13 | 69% | 15% |
| penalty | 8 | 88% | 50% |
| multi | 8 | 100% | 0% |
| trap | 5 | 80% | 80% |
| exception | 3 | 0% | 0% |
| out_of_scope_penalty | 2 | 50% | 0% |
| no_violation | 2 | 100% | 0% |
| ambiguous | 1 | 0% | 0% |

## Trafność wg: trudność

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| easy | 20 | 75% | 20% |
| medium | 16 | 81% | 31% |
| hard | 6 | 50% | 17% |

## Retrieval vs. rozumowanie

- Trafny artykuł BYŁ w kontekście: 32; z nich błędna odpowiedź (**błąd rozumowania**): 22 (69%)
- Trafnego artykułu NIE było w kontekście (**luka retrievalu**): 10

## Najsłabsze tematy (najniższa poprawność prawna)

- **def_zasady**: 0% correct (0/8)
- **kontrola_drogowa**: 0% correct (0/2)
- **znaki_sygnaly**: 0% correct (0/2)
- **swiatla**: 0% correct (0/2)
- **wlaczanie**: 0% correct (0/2)
- **cross_synthesis**: 21% correct (3/14)
- **taryfikator**: 50% correct (4/8)
- **dokumenty_stan_techniczny**: 50% correct (1/2)

## Przykładowe porażki (major)

- `taryfikator-001` [taryfikator/penalty] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie i art. 20 trafne, ale kara całkowicie błędna: +25 km/h to przedział 21-25 (art. 92a §1, lp.75, 300 zł), a model pewnie podał art. 92a §2 i 1500 zł z lp.80. Dodatkowo zbędny art. 14 
- `taryfikator-002` [taryfikator/penalty] oczek.: **naruszenie**, model: **naruszenie** — Werdykt i art. 20 poprawne, lecz +35 km/h to przedział 31-40 (art. 92a §2, 800 zł, lp.77); model błędnie zakwalifikował jako §1 i podał 400 zł z lp.76, myląc przedział prędkości.
- `taryfikator-004` [taryfikator/penalty] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie trafny, ale +6 km/h to przedział do 10 km/h (art. 92a §1, lp.72, 50 zł); model dziko pomylił przedziały (mówił o 51-70 km/h) i podał art. 92a §2, 1500 zł oraz zbędne art. 21/24/63.
- `taryfikator-007` [taryfikator/penalty] oczek.: **naruszenie**, model: **brak_naruszenia** — Błędny werdykt: dziecko <150 cm musi być w foteliku niezależnie od siedzenia i wieku >3 lat; model uznał brak naruszenia, pominął art. 39 ust. 3 PoRD oraz karę 300 zł (lp.58, art. 97 k.w.).
- `cross_synthesis-003` [cross_synthesis/multi] oczek.: **naruszenie**, model: **naruszenie** — Sedno pytania to tryb postępowania; model zmyślił odpowiedź na bazie art. 82 KPW (rzekome osadzenie/nakaz aresztowania), zamiast trybu przyspieszonego (art. 90 KPW) i wniosku o ukaranie po odmowie man
- `cross_synthesis-013` [cross_synthesis/single] oczek.: **naruszenie**, model: **niejasny** — Wjazd z nieruchomości to włączanie się do ruchu (art. 17 PoRD); model dał się zwieść dystraktorowi art. 25 (skrzyżowanie), zboczył na pieszych (art. 26) i wymyślił kary z lp.46/50 art. 86b, nie dając 
- `cross_synthesis-019` [cross_synthesis/single] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie trafny, ale model powołał dystraktor art. 26 ust. 2 (piesi) zamiast właściwego art. 27 ust. 1a PoRD (ustąpienie rowerzyście na wprost), przekwalifikował rowerzystę na pieszego i pew
- `def_zasady-005` [def_zasady/single] oczek.: **naruszenie**, model: **brak_naruszenia** — Pytanie dotyczyło kwalifikacji: 5 min unieruchomienia to postój (art. 2 pkt 30 PoRD, >1 min). Model nie sklasyfikował zdarzenia, oparł się na nietrafnym art. 46 i orzekł brak naruszenia, pomijając def
- `def_zasady-011` [def_zasady/multi] oczek.: **naruszenie**, model: **naruszenie** — Werdykt o naruszeniu trafny, lecz model nie odpowiedział na pytanie definicyjne (art. 2 pkt 25 - mgła to niedostateczna widoczność; art. 3 - ostrożność), skupił się na nieadekwatnym art. 30 (światła) 
- `def_zasady-017` [def_zasady/out_of_scope_penalty] oczek.: **naruszenie**, model: **naruszenie** — Scenariusz dotyczy pierwszeństwa między pojazdami na skrzyżowaniu, a kwota i punkty są poza korpusem (penalty external); model błędnie przekwalifikował na nieustąpienie pieszemu (art. 15a/86a), pewnie
- `def_zasady-018` [def_zasady/multi] oczek.: **naruszenie**, model: **naruszenie** — Werdykt 'naruszenie' trafny, ale model w ogóle nie zauważył sedna (Art. 2 pkt 18 - osoba prowadząca rower jest pieszym); oparł rozumowanie na nieadekwatnych Art. 15a i 33b (przejazd rowerowy/UTO) oraz
- `dokumenty_stan_techniczny-006` [dokumenty_stan_techniczny/single] oczek.: **naruszenie**, model: **brak_naruszenia** — Błędny werdykt - niebieskie światła błyskowe i syrena to oczywiste naruszenie Art. 66 ust. 4 pkt 3, a model orzekł 'brak naruszenia'; dodatkowo zmyślił treść Art. 66 ust. 4 (światła przeciwmgielne) za
