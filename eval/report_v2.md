# Raport ewaluacji — Agent Kodeksu Drogowego (Bielik 4.5B)

Przypadków ocenionych: **511**  ·  model: `hf.co/gaianet/Bielik-4.5B-v3.0-Instruct-GGUF:Q6_K`

## Trafność werdyktu (naruszenie / brak naruszenia)

- **Poprawny werdykt: 377/511 (74%)**
- **BIAS / false-positive** (czysta sprawa uznana za naruszenie): 67/193 (35%)
- **Under-detection / false-negative** (naruszenie przeoczone): 63/318 (20%)

## Poprawność prawna (całościowa ocena sędziego)

- correct: 114 (22%)
- partially_correct: 236 (46%)
- incorrect: 161 (32%)

## Kontrola halucynacji i kar

- **Halucynacja prawa** (zmyślony/źle przypisany artykuł): 195 (38%)
- Obsługa kar: correct_from_context=35, fabricated=206, correct_absent_flagged=44, omitted=33, n/a=193

## Kategorie błędów (E1-E8) — częstość

| Kategoria | Liczba | % |
|---|---:|---:|
| E2 błędny warunek/kierunek | 206 | 40% |
| E4 zmyślona kara | 203 | 40% |
| E5 mieszanie źródeł | 152 | 30% |
| E8 nadmiar cytowań | 128 | 25% |
| E6 przeoczone naruszenie | 71 | 14% |
| E3 stronniczość ku winie | 67 | 13% |
| E1 zmyślony artykuł | 55 | 11% |
| E7 błąd proceduralny | 24 | 5% |

## Trafność wg: temat

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| taryfikator | 30 | 93% | 53% |
| wlaczanie | 20 | 85% | 10% |
| widocznosc_ostrzeganie | 20 | 70% | 30% |
| swiatla | 20 | 70% | 25% |
| przewoz_pasy_foteliki | 20 | 75% | 55% |
| zakazy_kierujacego | 20 | 85% | 20% |
| uprzywilejowane | 20 | 90% | 20% |
| rowery_uto | 20 | 80% | 10% |
| strefa_autostrada | 20 | 60% | 5% |
| dokumenty_stan_techniczny | 20 | 70% | 5% |
| kontrola_drogowa | 20 | 70% | 35% |
| cross_synthesis | 20 | 80% | 10% |
| def_zasady | 19 | 68% | 5% |
| piesi_przejscia | 19 | 79% | 42% |
| predkosc | 19 | 74% | 5% |
| wyprzedzanie | 19 | 79% | 16% |
| holowanie | 19 | 58% | 26% |
| zatrzymanie_postoj | 19 | 84% | 26% |
| znaki_sygnaly | 18 | 72% | 6% |
| zmiana_kierunku_pasa | 18 | 89% | 28% |
| wymijanie_omijanie_cofanie | 18 | 78% | 28% |
| pierwszenstwo | 17 | 94% | 24% |
| mandat_przeslanki | 16 | 6% | 0% |
| mandat_rodzaje | 16 | 62% | 19% |
| odmowa_mandatu | 16 | 69% | 25% |
| tryby_postepowania | 16 | 50% | 19% |
| strony_wlasciwosc | 12 | 75% | 42% |

## Trafność wg: wymiar

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| single | 248 | 78% | 19% |
| multi | 71 | 73% | 15% |
| exception | 71 | 62% | 25% |
| trap | 68 | 68% | 24% |
| penalty | 30 | 93% | 53% |
| no_violation | 12 | 75% | 42% |
| out_of_scope_penalty | 7 | 57% | 0% |
| ambiguous | 3 | 33% | 0% |
| cross_doc | 1 | 0% | 0% |

## Trafność wg: trudność

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| medium | 254 | 79% | 21% |
| hard | 148 | 62% | 20% |
| easy | 109 | 78% | 28% |

## Retrieval vs. rozumowanie

- Trafny artykuł BYŁ w kontekście: 424; z nich błędna odpowiedź (**błąd rozumowania**): 315 (74%)
- Trafnego artykułu NIE było w kontekście (**luka retrievalu**): 87

## Najsłabsze tematy (najniższa poprawność prawna)

- **mandat_przeslanki**: 0% correct (0/16)
- **strefa_autostrada**: 5% correct (1/20)
- **dokumenty_stan_techniczny**: 5% correct (1/20)
- **def_zasady**: 5% correct (1/19)
- **predkosc**: 5% correct (1/19)
- **znaki_sygnaly**: 6% correct (1/18)
- **wlaczanie**: 10% correct (2/20)
- **rowery_uto**: 10% correct (2/20)

## Przykładowe porażki (major)

- `def_zasady-003` [def_zasady/trap] oczek.: **brak_naruszenia**, model: **naruszenie** — Błędny werdykt: pojazd z lewej nie ma pierwszeństwa, więc brak naruszenia, a model orzekł naruszenie Art. 25, myląc strony. Dodatkowo nieistniejący 'sąd grodzki' (błąd proceduralny).
- `def_zasady-004` [def_zasady/trap] oczek.: **brak_naruszenia**, model: **naruszenie** — 40 sekund to zatrzymanie, nie postój (Art. 2 pkt 29/30) - brak naruszenia, a model orzekł naruszenie Art. 46. Zmyślił też konkretną pozycję 'art. 97 lp. 168, 100 zł' niepotwierdzoną w kontekście.
- `def_zasady-006` [def_zasady/out_of_scope_penalty] oczek.: **zalezy**, model: **naruszenie** — Gold: zalezy (limit 20 km/h i sankcja to wiedza zewnętrzna). Art. 20 o 20 km/h trafny, ale model zmyślił kwalifikację - Art. 94 §1a k.w. (dot. braku uprawnień, nie prędkości) oraz nonsensowne 'Art. 3 
- `def_zasady-009` [def_zasady/trap] oczek.: **brak_naruszenia**, model: **niejasny** — Pominął kluczowe wyłączenie z Art. 2 pkt 10 (twarda+gruntowa to nie skrzyżowanie) i pozostał przy warunkowym 'jeśli skrzyżowanie, to naruszenie', nie dochodząc do brak_naruszenia. Dodatkowo zmyślił ma
- `def_zasady-014` [def_zasady/ambiguous] oczek.: **zalezy**, model: **naruszenie** — Gold: zalezy (status strefy ruchu decyduje). Model orzekł naruszenie, karząc kierowcę za samo zadanie pytania - rażący guilt bias. Zmyślił 'Art. 86a k.w.' i karę 'od kilkuset do kilku tysięcy zł'.
- `def_zasady-017` [def_zasady/out_of_scope_penalty] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie trafny, ale podstawa zmyślona: 'Art. 15a ust. 3 UoRW' o pieszych (scenariusz dot. pierwszeństwa pojazdów) i jako fakt podana kwota 200 zł, choć gold wskazuje, że kwota i punkty są w
- `def_zasady-018` [def_zasady/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Błędny werdykt: osoba prowadząca rower JEST pieszym (Art. 2 pkt 18), więc doszło do naruszenia. Model przyjął błędny argument kierowcy i zmyślił, że 'nie była pieszym', błędnie powołując Art. 13 ust. 
- `znaki_sygnaly-001` [znaki_sygnaly/single] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie trafny tylko przypadkiem - model pominął kluczowy Art. 5 (polecenia kierującego ruchem ważniejsze niż sygnały) i wymyślił nieistniejący fakt 'brak miejsca na skrzyżowaniu' (Art. 25 
- `znaki_sygnaly-003` [znaki_sygnaly/single] oczek.: **brak_naruszenia**, model: **niejasny** — Oczekiwano brak_naruszenia (Art. 5 ust. 3 - sygnały świetlne nad znakami). Model zmyślił obecność pieszych, rozważał Art. 26/25/23 i skończył mętnym 'naruszenie ustąpienia pierwszeństwa pieszym (jeśli
- `znaki_sygnaly-006` [znaki_sygnaly/multi] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie zgodny z gold, ale z całkowicie błędnego powodu (Art. 28 ust. 3 pkt 2 - brak miejsca za przejazdem) zamiast Art. 5/6. Pewnie podana zmyślona kara: art. 97a k.w., 2000/4000 zł, lp. 1
- `znaki_sygnaly-007` [znaki_sygnaly/multi] oczek.: **naruszenie**, model: **naruszenie** — Werdykt naruszenie przypadkiem zgodny, ale model wymyślił 'Art. 57a ust. 1' o światłach awaryjnych i orzekł naruszenie przez kierowcę autobusu (nie sprawcę), pomijając Art. 5/6 ust. 1 pkt 7. Zmyślona 
- `znaki_sygnaly-008` [znaki_sygnaly/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Oczekiwano naruszenie (Art. 5/6 ust. 1 pkt 8 - strażnik leśny uprawniony w lesie). Model pogubił się, oceniał czy to strażnik naruszył prawo i orzekł brak dowodów na naruszenie - nie wykrył wykroczeni
