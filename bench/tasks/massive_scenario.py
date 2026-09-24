"""MASSIVE scenario classification: 18-way Choice, English and French.

Dataset: AmazonScience/massive, professionally translated and parallel by construction,
so en-US and fr-FR rows with the same id are the same utterance. Classifying the 18
scenarios here; the 60-intent version is a separate task.

Four conditions, because they fail differently:
  EN_V1  english utterance, first-contact english criteria   where criteria writing starts
  EN_EN  english utterance, english criteria                 baseline
  FR_EN  french utterance, english criteria                  the realistic enterprise case
  FR_FR  french utterance, french criteria                   the fully localized case

Watch confidence, not just accuracy. Calibration degrades faster than accuracy under
distribution shift, so the model can stay accurate while becoming overconfident. If that
happens, confidence gating silently stops working in French, which is product-breaking
rather than inconvenient.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

NAME = "massive_scenario"

# Read the official Amazon archive rather than going through `datasets`: the HF repo ships
# a loader script, which datasets 4+ refuses to execute. See README for the fetch command.
DATA = Path(__file__).resolve().parents[2] / "data" / "1.1" / "data"

# V1: criteria written from the scenario names alone, the way anyone would on first
# contact with the label set. Kept so the V2 comparison is honest about the starting point.
SCENARIOS_EN_V1 = {
    "alarm": "Setting, changing or cancelling an alarm",
    "audio": "Adjusting volume or audio output",
    "calendar": "Calendar events, appointments and reminders",
    "cooking": "Recipes and cooking instructions",
    "datetime": "Current date, time or date calculations",
    "email": "Sending, reading or managing email",
    "general": "General chit-chat, greetings or questions about the assistant itself",
    "iot": "Controlling smart home devices such as lights, heating or plugs",
    "lists": "Creating or editing lists such as shopping or to-do lists",
    "music": "Playing, identifying or controlling music",
    "news": "News headlines and current events",
    "play": "Playing media such as radio, podcasts, audiobooks or games",
    "qa": "Factual questions seeking an answer",
    "recommendation": "Asking for recommendations such as places, events or media",
    "social": "Social media posts and interactions",
    "takeaway": "Ordering food for delivery or collection",
    "transport": "Travel, traffic, public transport and ticket queries",
    "weather": "Weather forecasts and conditions",
}

# V2: same 18 labels, rewritten from the dataset's own intent lists. The boundaries the
# taxonomy actually draws, notably play (start playback, including music) against music
# (opinions and settings about music), which V1 got backwards.
SCENARIOS_EN = {
    "alarm": "Set, query or cancel an alarm",
    "audio": "Change the device volume: up, down, mute",
    "calendar": "Query, create or remove a calendar or diary entry, appointment or reminder",
    "cooking": "Ask for a recipe or a cooking question",
    "datetime": "Ask the current date or time, or convert between time zones or date formats",
    "email": "Send or query email, or add and look up a contact",
    "general": "Greet the assistant, ask it for a joke, or make a quirky open-ended remark "
               "to it that fits no other category",
    "iot": "Control a smart home device: lights, plugs, coffee machine, robot vacuum",
    "lists": "Create, query or remove an entry on a shopping list or to-do list",
    "music": "Express liking or disliking of music, identify what is playing, or change "
             "music settings. Not starting playback",
    "news": "Ask for news or current events",
    "play": "Start playing something: music, radio, a podcast, an audiobook or a game",
    "qa": "Ask a factual question with a lookup answer: a definition, a fact, a calculation, "
          "a currency rate or a stock price",
    "recommendation": "Ask for a recommendation of an event, a place or a movie",
    "social": "Post to social media or query social media activity",
    "takeaway": "Order food for delivery or collection, or ask about a takeaway order",
    "transport": "Ask about travel, traffic, a taxi or a transport ticket",
    "weather": "Ask about the weather",
}

SCENARIOS_FR = {
    "alarm": "Programmer, consulter ou annuler une alarme",
    "audio": "Modifier le volume de l'appareil : augmenter, baisser, couper le son",
    "calendar": "Consulter, créer ou supprimer une entrée d'agenda, un rendez-vous ou un rappel",
    "cooking": "Demander une recette ou poser une question de cuisine",
    "datetime": "Demander la date ou l'heure actuelle, ou convertir un fuseau horaire ou un format de date",
    "email": "Envoyer ou consulter un e-mail, ajouter ou rechercher un contact",
    "general": "Saluer l'assistant, lui demander une blague, ou lui adresser une remarque "
               "ouverte et fantaisiste qui ne rentre dans aucune autre catégorie",
    "iot": "Commander un appareil domotique : lumières, prises, machine à café, aspirateur robot",
    "lists": "Créer, consulter ou supprimer un élément d'une liste de courses ou de tâches",
    "music": "Exprimer un goût ou un dégoût musical, identifier ce qui passe, ou modifier "
             "les réglages musicaux. Pas démarrer la lecture",
    "news": "Demander les actualités ou l'information",
    "play": "Lancer la lecture de quelque chose : musique, radio, podcast, livre audio ou jeu",
    "qa": "Poser une question factuelle dont la réponse se cherche : une définition, un fait, "
          "un calcul, un taux de change ou un cours de bourse",
    "recommendation": "Demander une recommandation d'événement, de lieu ou de film",
    "social": "Publier sur les réseaux sociaux ou consulter l'activité des réseaux sociaux",
    "takeaway": "Commander un repas à livrer ou à emporter, ou se renseigner sur une commande",
    "transport": "Se renseigner sur un trajet, le trafic, un taxi ou un titre de transport",
    "weather": "Demander la météo",
}

INSTRUCTIONS_EN = "Which scenario does this virtual assistant request belong to"
INSTRUCTIONS_FR = "À quel scénario appartient cette requête adressée à un assistant vocal"

CONDITIONS = {
    "EN_V1": ("en", INSTRUCTIONS_EN, SCENARIOS_EN_V1),
    "EN_EN": ("en", INSTRUCTIONS_EN, SCENARIOS_EN),
    "FR_EN": ("fr", INSTRUCTIONS_EN, SCENARIOS_EN),
    "FR_FR": ("fr", INSTRUCTIONS_FR, SCENARIOS_FR),
}


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error over the probability assigned to the chosen class.

    This, not `confidence`, is what the vendor's calibration claim is about: an answer
    given 0.8 should be right about 80% of the time.
    """
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        buckets[min(int(row["p_chosen"] * bins), bins - 1)].append(
            (row["p_chosen"], row["correct"])
        )
    return sum(
        len(vals) / len(rows)
        * abs(statistics.mean(p for p, _ in vals) - statistics.mean(c for _, c in vals))
        for vals in buckets.values()
    )


def read_split(locale: str, partition: str = "test") -> dict[str, dict]:
    path = DATA / f"{locale}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}. See README: fetch and extract the MASSIVE archive.")
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {row["id"]: row for row in rows if row["partition"] == partition}


def load_pairs(sample: int, seed: int, split: str = "test", gold: str = "scenario") -> list[dict]:
    english, french = read_split("en-US", split), read_split("fr-FR", split)

    shared = sorted(english.keys() & french.keys(), key=int)
    random.Random(seed).shuffle(shared)

    return [
        {
            "id": key,
            "gold": english[key][gold],
            "en": english[key]["utt"],
            "fr": french[key]["utt"],
        }
        for key in shared[:sample]
    ]


def question(condition: str) -> tuple[str, dict]:
    """The utterance language and the question, keyed as the models receive it: Jev and
    Laya read the key, so renaming it would change what was measured."""
    lang, instructions, criteria = CONDITIONS[condition]
    return lang, {"scenario": {"type": "choice", "instructions": instructions,
                               "criteria": criteria}}
