# Raport ewaluacji — Agent Kodeksu Drogowego (Bielik 4.5B)

Przypadków ocenionych: **480**  ·  model: `SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0`

## Trafność werdyktu (naruszenie / brak naruszenia)

- **Poprawny werdykt: 256/480 (53%)**
- **BIAS / false-positive** (czysta sprawa uznana za naruszenie): 8/195 (4%)
- **Under-detection / false-negative** (naruszenie przeoczone): 200/285 (70%)

## Poprawność prawna (całościowa ocena sędziego)

- correct: 97 (20%)
- partially_correct: 151 (31%)
- incorrect: 232 (48%)

## Kontrola halucynacji i kar

- **Halucynacja prawa** (zmyślony/źle przypisany artykuł): 59 (12%)
- Obsługa kar: fabricated=68, correct_flagged=17, omitted=13, n/a=382

## Kategorie błędów (E1-E8) — częstość

| Kategoria | Liczba | % |
|---|---:|---:|
| E2 błędny warunek/kierunek | 266 | 55% |
| E6 przeoczone naruszenie | 200 | 42% |
| E8 nadmiar cytowań | 120 | 25% |
| E4 zmyślona kara | 69 | 14% |
| E5 mieszanie źródeł | 68 | 14% |
| E1 zmyślony artykuł | 53 | 11% |
| E7 błąd proceduralny | 33 | 7% |
| E3 stronniczość ku winie | 10 | 2% |

## Trafność wg: temat

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| wlaczanie | 20 | 45% | 15% |
| widocznosc_ostrzeganie | 20 | 55% | 40% |
| swiatla | 20 | 35% | 20% |
| zakazy_kierujacego | 20 | 55% | 10% |
| uprzywilejowane | 20 | 40% | 10% |
| rowery_uto | 20 | 35% | 20% |
| strefa_autostrada | 20 | 65% | 25% |
| dokumenty_stan_techniczny | 20 | 50% | 15% |
| cross_synthesis | 20 | 75% | 20% |
| def_zasady | 19 | 74% | 16% |
| znaki_sygnaly | 19 | 37% | 21% |
| piesi_przejscia | 19 | 58% | 16% |
| predkosc | 19 | 42% | 26% |
| wymijanie_omijanie_cofanie | 19 | 53% | 11% |
| pierwszenstwo | 19 | 74% | 26% |
| holowanie | 19 | 47% | 32% |
| zatrzymanie_postoj | 19 | 53% | 26% |
| kontrola_drogowa | 19 | 47% | 0% |
| zmiana_kierunku_pasa | 18 | 39% | 22% |
| wyprzedzanie | 18 | 61% | 28% |
| przewoz_pasy_foteliki | 16 | 69% | 25% |
| mandat_przeslanki | 16 | 69% | 19% |
| mandat_rodzaje | 16 | 75% | 38% |
| odmowa_mandatu | 16 | 50% | 19% |
| tryby_postepowania | 16 | 31% | 19% |
| strony_wlasciwosc | 13 | 62% | 8% |

## Trafność wg: wymiar

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| single | 244 | 34% | 9% |
| multi | 74 | 53% | 15% |
| exception | 70 | 84% | 34% |
| trap | 68 | 84% | 46% |
| no_violation | 12 | 92% | 67% |
| out_of_scope_penalty | 8 | 38% | 0% |
| ambiguous | 3 | 67% | 0% |
| cross_doc | 1 | 100% | 0% |

## Trafność wg: trudność

| grupa | n | werdykt OK | prawnie correct |
|---|---:|---:|---:|
| medium | 235 | 55% | 22% |
| hard | 147 | 56% | 18% |
| easy | 98 | 45% | 20% |

## Retrieval vs. rozumowanie

- Trafny artykuł BYŁ w kontekście: 430; z nich błędna odpowiedź (**błąd rozumowania**): 334 (78%)
- Trafnego artykułu NIE było w kontekście (**luka retrievalu**): 50

## Najsłabsze tematy (najniższa poprawność prawna)

- **kontrola_drogowa**: 0% correct (0/19)
- **strony_wlasciwosc**: 8% correct (1/13)
- **zakazy_kierujacego**: 10% correct (2/20)
- **uprzywilejowane**: 10% correct (2/20)
- **wymijanie_omijanie_cofanie**: 11% correct (2/19)
- **wlaczanie**: 15% correct (3/20)
- **dokumenty_stan_techniczny**: 15% correct (3/20)
- **def_zasady**: 16% correct (3/19)

## Przykładowe porażki (major)

- `def_zasady-001` [def_zasady/multi] oczek.: **naruszenie**, model: **naruszenie** — Trafny werdykt naruszenia, a Art. 26 jest realnie adekwatny do przejścia dla pieszych, ale model pewnie podał karę z 'Art. 86a Kodeksu wykroczeń' - akt spoza kontekstu i przepis zmyślony jako fakt.
- `def_zasady-005` [def_zasady/single] oczek.: **naruszenie**, model: **brak_naruszenia** — Zły werdykt: 5 minut to postój wg Art. 2 pkt 30, a model uznał brak naruszenia i pominął kluczową definicję, błędnie analizując tylko Art. 46.
- `def_zasady-006` [def_zasady/out_of_scope_penalty] oczek.: **zalezy**, model: **brak_naruszenia** — Model podał limit 20 km/h jako fakt z korpusu (wiedza zewnętrzna), a potem wewnętrznie sprzecznie uznał, że jadący 40 km/h 'jechał zgodnie z 20 km/h' - błędny werdykt i zmyślona podstawa.
- `def_zasady-007` [def_zasady/multi] oczek.: **naruszenie**, model: **naruszenie** — Trafny werdykt naruszenia (Art. 26 adekwatny do przejścia), ale ponownie pewnie podana zmyślona kara z 'Art. 86a Kodeksu wykroczeń' - akt spoza kontekstu.
- `def_zasady-011` [def_zasady/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Zły werdykt: mgła to niedostateczna widoczność (Art. 2 pkt 25) wymagająca szczególnej ostrożności (Art. 3); model rozproszył się na przepisy o światłach i przejazdach kolejowych i uznał brak naruszeni
- `def_zasady-012` [def_zasady/single] oczek.: **naruszenie**, model: **naruszenie** — Trafny werdykt z prawidłowym Art. 3 ust. 1 (zaniechanie = działanie), ale błędnie powołał Art. 44 (wypadek) i podał karę z 'art. 97 Kodeksu wykroczeń' - akt spoza kontekstu.
- `def_zasady-015` [def_zasady/single] oczek.: **naruszenie**, model: **naruszenie** — Trafny werdykt naruszenia (Art. 25 adekwatny przy znaku ustąpienia pierwszeństwa), ale pewnie podana zmyślona kara z 'Art. 86a Kodeksu wykroczeń' (akt spoza kontekstu); Art. 97 Kpw był dopuszczalny.
- `def_zasady-017` [def_zasady/out_of_scope_penalty] oczek.: **naruszenie**, model: **brak_naruszenia** — Zły werdykt: scenariusz wprost opisuje naruszenie pierwszeństwa zmuszające do hamowania, a model uznał brak naruszenia; plus: poprawnie nie zmyślił kwoty mandatu ani punktów.
- `def_zasady-018` [def_zasady/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Zły werdykt i sprzeczne rozumowanie: model słusznie zauważył, że osoba prowadząca rower to pieszy, ale skonkludował brak naruszenia; dodatkowo zmyślił treść 'Art. 2 ust. 18a' (dotyczy UTO, nie roweru)
- `znaki_sygnaly-001` [znaki_sygnaly/single] oczek.: **naruszenie**, model: **brak_naruszenia** — Zły werdykt i odwrócona treść prawa: model twierdzi, że sygnały świetlne mają pierwszeństwo przed poleceniami kierującego ruchem, co jest dokładnym przeciwieństwem Art. 5 - polecenia policjanta mają p
- `znaki_sygnaly-004` [znaki_sygnaly/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Gold oczekuje 'naruszenie' (strażnik miejski jest w zamkniętym katalogu Art. 6 ust. 1, kierowca miał obowiązek się zastosować). Model orzekł brak naruszenia, nie powołał Art. 6, błędnie uznał uprawnie
- `znaki_sygnaly-006` [znaki_sygnaly/multi] oczek.: **naruszenie**, model: **brak_naruszenia** — Gold: naruszenie — pracownik kolejowy (Art. 6 ust. 1 pkt 4) jest uprawniony do sygnałów, które mają pierwszeństwo niezależnie od rogatek. Model analizował tylko Art. 28 (zapory/przejazd), pominął upra
