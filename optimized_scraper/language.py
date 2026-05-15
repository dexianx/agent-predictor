import re

PIDGIN_MARKERS = [
    "dey", "e dey", "e no", "na im", "wetin", "abi", "oya", "sha",
    "sef", "wahala", "e shock me", "e don", "chaii", "chai",
    "no be lie", "e sweet", "e pain", "sabi", "wey", "dem",
    "make i", "how e be", "e good", "e bad", "abeg",
]
YORUBA_MARKERS = ["ehn", "abi o", "to wa", "jare", "mo ti", "e joor", "ko bad"]
HAUSA_MARKERS = ["wallahi", "kai", "ban san", "ba shi", "insha"]
IGBO_MARKERS = ["chineke", "chukwu", "nna", "oya na", "biko"]

NIGERIAN_CITIES = [
    "Port Harcourt", "Benin City", "Lagos", "Abuja", "Kano", "Ibadan",
    "Kaduna", "Enugu", "Onitsha", "Aba", "Jos", "Uyo", "Warri",
    "Ilorin", "Abeokuta", "Owerri", "Calabar", "Asaba", "Akure",
]


def _contains_marker(text: str, marker: str) -> bool:
    lower = text.lower()
    if " " in marker:
        return marker in lower
    return re.search(rf"\b{re.escape(marker)}\b", lower) is not None


def detect_language_signal(text: str) -> str:
    for marker in PIDGIN_MARKERS:
        if _contains_marker(text, marker):
            return "en-pidgin"
    for marker in YORUBA_MARKERS:
        if _contains_marker(text, marker):
            return "en-yoruba-mix"
    for marker in HAUSA_MARKERS:
        if _contains_marker(text, marker):
            return "en-hausa-mix"
    for marker in IGBO_MARKERS:
        if _contains_marker(text, marker):
            return "en-igbo-mix"
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha and latin / alpha < 0.5:
        return "other"
    return "en"


def extract_city_signal(text: str) -> str | None:
    lower = text.lower()
    for city in NIGERIAN_CITIES:
        if city.lower() in lower:
            return city
    return None
