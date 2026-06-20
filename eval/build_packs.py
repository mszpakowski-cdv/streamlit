"""
build_packs.py — buduje per-temat paczki z treścią artykułów (do groundingu
agentów generujących testy) oraz rejestr tematów eval/topics.json.

Wejście:  eval/articles.json  (doc -> etykieta artykułu -> tekst)
Wyjście:  eval/packs/<key>.md  + eval/topics.json
"""
import json, re, os

ART = json.load(open("eval/articles.json", encoding="utf-8"))
PORD = "prawo_o_ruchu_drogowym"
KPW = "kodeks_wykroczen"
DOC_LABEL = {
    PORD: "Prawo o ruchu drogowym",
    KPW: "Kodeks postępowania w sprawach o wykroczenia",
}


def _norm(label: str) -> str:
    m = re.search(r'(\d+)\s*([a-z]?)', label)
    return f"{int(m.group(1))}{m.group(2)}" if m else label


def _norm_map(doc: str) -> dict:
    out = {}
    for label, text in ART[doc].items():
        if label == "brak numeru":
            continue
        out.setdefault(_norm(label), []).append(text)
    return {k: "\n".join(v) for k, v in out.items()}


NORM = {PORD: _norm_map(PORD), KPW: _norm_map(KPW)}

# (key, nazwa, doc, [numery artykułów], target, focus/pułapki)
TOPICS = [
    ("def_zasady", "Definicje i zasady ogólne", PORD, ["2","3","4"], 20,
     "szczególna ostrożność, ograniczone zaufanie, definicje (skrzyżowanie, droga, strefa zamieszkania/ruchu, ustąpienie pierwszeństwa)"),
    ("znaki_sygnaly", "Znaki, sygnały i hierarchia poleceń", PORD, ["5","6","7","8"], 20,
     "polecenia kierującego ruchem > sygnały świetlne > znaki; sygnały dawane przez osobę"),
    ("piesi_przejscia", "Ruch pieszych i przejścia dla pieszych", PORD, ["11","12","13","14","26"], 20,
     "pierwszeństwo pieszego na/przed przejściem, wejście a pierwszeństwo, kolumny pieszych"),
    ("wlaczanie", "Włączanie się do ruchu", PORD, ["17"], 20,
     "co JEST a co NIE JEST włączaniem się do ruchu (strefa zamieszkania vs strefa ruchu vs droga publiczna)"),
    ("predkosc", "Prędkość dopuszczalna i odległość", PORD, ["19","20","21","27"], 20,
     "prędkości dopuszczalne wg obszaru/pojazdu, odległość, prędkość bezpieczna; brak naruszenia gdy w limicie"),
    ("zmiana_kierunku_pasa", "Zmiana kierunku jazdy i pasa ruchu", PORD, ["22"], 20,
     "sygnalizowanie, ustąpienie pierwszeństwa przy zmianie pasa, zawracanie i jego zakazy"),
    ("wymijanie_omijanie_cofanie", "Wymijanie, omijanie i cofanie", PORD, ["23"], 20,
     "różnice wymijanie/omijanie/cofanie; cofanie a ustąpienie pierwszeństwa; zakazy cofania"),
    ("wyprzedzanie", "Wyprzedzanie", PORD, ["24"], 20,
     "zakazy wyprzedzania (skrzyżowania, przejścia), wyprzedzanie z prawej na jezdni jednokierunkowej, sygnalizujący skręt w lewo"),
    ("pierwszenstwo", "Pierwszeństwo i przecinanie kierunków ruchu", PORD, ["25"], 20,
     "ZASADA PRAWEJ RĘKI — pojazd z LEWEJ NIE ma pierwszeństwa; skręt w lewo; pojazd szynowy; poza skrzyżowaniem"),
    ("widocznosc_ostrzeganie", "Ograniczona widoczność i ostrzeganie", PORD, ["28","29","30"], 20,
     "sygnał dźwiękowy/świetlny, jazda we mgle, dozwolone/zakazane użycie sygnałów"),
    ("holowanie", "Holowanie", PORD, ["31"], 20,
     "warunki holowania, prędkość, połączenie, zakazy (np. na autostradzie)"),
    ("swiatla", "Używanie świateł zewnętrznych", PORD, ["51","52"], 20,
     "światła mijania w dzień, drogowe a oślepianie, postojowe, przeciwmgłowe — kiedy wolno/nie wolno"),
    ("zatrzymanie_postoj", "Zatrzymanie i postój", PORD, ["46","47","48","49","50","50a"], 20,
     "różnica zatrzymanie vs postój; gdzie zabronione; odległości od przejścia/skrzyżowania; postój a hamulec/silnik"),
    ("przewoz_pasy_foteliki", "Przewóz osób i ładunków, pasy, foteliki", PORD, ["39","40","41","42","43","44","61","62","63"], 20,
     "pasy bezpieczeństwa i wyjątki, foteliki dla dzieci, przewóz osób/ładunku, kask"),
    ("zakazy_kierujacego", "Zakazy kierującego (telefon, stan, manewry)", PORD, ["45"], 20,
     "zakaz korzystania z telefonu wymagającego trzymania, jazda w stanie nietrzeźwości, inne zakazy z art. 45"),
    ("uprzywilejowane", "Pojazdy uprzywilejowane", PORD, ["9","53","53a"], 20,
     "warunki uprzywilejowania (sygnały + ostrożność), obowiązek ustąpienia, kiedy pojazd NIE jest uprzywilejowany"),
    ("rowery_uto", "Rowery, hulajnogi, UTO", PORD, ["15a","33","33a"], 20,
     "korzystanie z drogi dla rowerów, chodnika, przewóz dziecka, pierwszeństwo UTO/pieszych"),
    ("strefa_autostrada", "Strefa zamieszkania/ruchu i autostrady", PORD, ["16","20"], 20,
     "20 km/h i pierwszeństwo pieszego w strefie zamieszkania; zakazy na autostradzie/ekspresowej; ruch prawostronny"),
    ("dokumenty_stan_techniczny", "Dokumenty i stan techniczny pojazdu", PORD, ["38","66","71"], 20,
     "wymagane dokumenty, warunki techniczne i dopuszczenie do ruchu, rejestracja/dowód"),
    ("kontrola_drogowa", "Kontrola ruchu drogowego (uprawnienia Policji)", PORD, ["129","130a","131","132","135"], 20,
     "uprawnienia policjanta, badanie trzeźwości, usunięcie pojazdu, zatrzymanie dowodu rejestracyjnego / prawa jazdy"),
    # --- KPW (procedura) ---
    ("mandat_przeslanki", "Postępowanie mandatowe — przesłanki i granice grzywny", KPW, ["95","96"], 16,
     "kiedy WOLNO nałożyć mandat, górne granice grzywny, sytuacje wykluczające mandat"),
    ("mandat_rodzaje", "Rodzaje mandatów i ich skutki", KPW, ["98","97"], 16,
     "mandat gotówkowy/kredytowany/zaoczny, kiedy staje się prawomocny, pouczenie"),
    ("odmowa_mandatu", "Odmowa przyjęcia mandatu", KPW, ["97","99"], 16,
     "skutek odmowy: skierowanie wniosku o ukaranie do sądu; prawo do odmowy"),
    ("tryby_postepowania", "Tryby postępowania (zwyczajny/przyspieszony/nakazowy)", KPW, ["2","54","89","90","91","92","93","94"], 16,
     "kiedy tryb przyspieszony, kiedy nakazowy, czynności wyjaśniające"),
    ("strony_wlasciwosc", "Strony, właściwość i środki zaskarżenia", KPW, ["9","10","11","17","103","104","105","106","107"], 16,
     "Policja jako oskarżyciel publiczny, sąd właściwy, apelacja/zażalenie, sprzeciw od nakazu"),
]

os.makedirs("eval/packs", exist_ok=True)
registry = []
missing_report = []

for key, name, doc, arts, target, focus in TOPICS:
    lines = [f"# {name}", f"_Źródło: {DOC_LABEL[doc]}_\n"]
    present = []
    for a in arts:
        text = NORM[doc].get(_norm(a))
        if text:
            present.append(a)
            lines.append(f"## Art. {a} ({DOC_LABEL[doc]})\n{text.strip()}\n")
        else:
            missing_report.append((key, a))
            lines.append(f"## Art. {a} — [BRAK W KORPUSIE]\n")
    open(f"eval/packs/{key}.md", "w", encoding="utf-8").write("\n".join(lines))
    registry.append({
        "key": key, "name": name, "doc": doc, "doc_label": DOC_LABEL[doc],
        "articles": present, "target": target, "focus": focus,
        "pack": f"eval/packs/{key}.md",
    })

json.dump(registry, open("eval/topics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"Zbudowano {len(registry)} paczek tematycznych.")
print(f"Suma targetów: {sum(t['target'] for t in registry)} przypadków.")
if missing_report:
    print("\nArtykuły nieobecne w korpusie (pominięte w paczkach):")
    for k, a in missing_report:
        print(f"  {k}: Art. {a}")
else:
    print("Wszystkie zadeklarowane artykuły znaleziono w korpusie.")
