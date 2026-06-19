"""
guardrails.py — filtrowanie zapytań niezwiązanych z ruchem drogowym.
"""

TRAFFIC_KEYWORDS = [
    # wykroczenia i kary
    "mandat", "grzywna", "kara", "wykroczenie", "przepis", "artykuł",
    "postępowanie", "ukaranie", "kodeks",
    # uczestnicy ruchu
    "kierowca", "pieszy", "rowerzysta", "motocyklista", "pojazd",
    "samochód", "ciężarówka", "autobus", "motocykl", "rower", "tir",
    # droga i infrastruktura
    "droga", "skrzyżowanie", "autostrada", "chodnik", "przejście",
    "parking", "jezdnia", "pas", "znak", "sygnalizacja", "światła",
    # naruszenia
    "prędkość", "wyprzedzanie", "pierwszeństwo", "alkohol", "trzeźwość",
    "pasy", "telefon", "rejestracja", "prawo jazdy", "oświetlenie",
    "hamowanie", "cofanie", "zawracanie", "parkowanie", "zatrzymanie",
    # zdarzenia
    "wypadek", "kolizja", "stłuczka", "zdarzenie", "interwencja",
    # policja
    "policjant", "drogówka", "patrol", "kontrola", "zatrzymać",
]

def is_traffic_related(text: str) -> bool:
    """Zwraca True jeśli zapytanie dotyczy ruchu drogowego lub wykroczeń."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in TRAFFIC_KEYWORDS)
