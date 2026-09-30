"""MASSIVE intent classification: 60-way Choice, English and French.

Same items as massive_scenario (same `load_pairs`, same ids), graded on the intent instead
of the scenario. Published reference for this task: fine-tuned xlm-r-base
(`xlm-r-base-amazon-massive-intent`) = 87.75%, en-US.

The criteria were written from the `dev` partition only: about eight en-US dev utterances
per intent, read beside the intent name. No `test` row was looked at, so the criteria are
not fitted to the rows they are scored on. `audio_volume_other` has no dev row; its
criterion was written from `train`.

No V1 condition: the V1/V2 comparison is massive_scenario's.
"""

from __future__ import annotations

from functools import partial

from bench.tasks.massive_scenario import load_pairs as scenario_pairs

NAME = "massive_intent"

INTENTS_EN = {
    "alarm_query": "Ask which alarms are set or whether a given alarm is set",
    "alarm_remove": "Cancel, turn off or remove one or more alarms",
    "alarm_set": "Set a new alarm or wake-up call",
    "audio_volume_down": "Lower the device volume or ask it to speak more quietly",
    "audio_volume_mute": "Mute the device or its speakers, or ask it to stop making sound",
    "audio_volume_other": "Open or adjust the volume and sound settings without saying up, "
                          "down or mute",
    "audio_volume_up": "Raise the device volume",
    "calendar_query": "Ask what is on the calendar: meetings, events, the agenda for a day",
    "calendar_remove": "Delete or cancel calendar events or meetings",
    "calendar_set": "Add an event, meeting or reminder to the calendar",
    "cooking_query": "Ask a general cooking question that is not a request for a recipe",
    "cooking_recipe": "Ask for a recipe, how to cook a dish, or what to cook with given "
                      "ingredients",
    "datetime_convert": "Convert a time between time zones or ask the time difference between "
                        "places",
    "datetime_query": "Ask the current time or date, the time in a place, or the day a date "
                      "falls on",
    "email_addcontact": "Save a person or an email address as a new contact",
    "email_query": "Check the inbox or ask about received emails",
    "email_querycontact": "Look up a contact's details: email address, phone number, name or "
                          "other information",
    "email_sendemail": "Send, write or reply to an email",
    "general_greet": "Greet the assistant",
    "general_joke": "Ask for a joke",
    "general_quirky": "Chit-chat, an odd remark or an open question to the assistant that fits "
                      "no other intent",
    "iot_cleaning": "Start the robot vacuum or ask for the home to be cleaned",
    "iot_coffee": "Make coffee or start the coffee machine",
    "iot_hue_lightchange": "Change the colour or mood setting of the lights",
    "iot_hue_lightdim": "Dim the lights or make a room darker",
    "iot_hue_lightoff": "Turn the lights off",
    "iot_hue_lighton": "Turn the lights on, or say it is too dark",
    "iot_hue_lightup": "Brighten the lights",
    "iot_wemo_off": "Turn off a smart plug or socket, or the appliance plugged into it",
    "iot_wemo_on": "Turn on a smart plug or socket, or the appliance plugged into it",
    "lists_createoradd": "Create a new list or add an item to a list",
    "lists_query": "Ask what lists exist or what is on a list",
    "lists_remove": "Remove an item from a list, or clear or delete a list",
    "music_dislikeness": "Say that a song, artist or genre is disliked",
    "music_likeness": "Say that a song, artist or genre is liked, or save a song to favourites",
    "music_query": "Ask what song or artist is playing, or for lyrics or details of a song",
    "music_settings": "Change playback settings: shuffle or repeat",
    "news_query": "Ask for news, headlines or updates on a current event",
    "play_audiobook": "Play, pause or resume an audiobook",
    "play_game": "Play a game with the assistant",
    "play_music": "Play music: a song, an artist, an album, a genre or a playlist",
    "play_podcasts": "Play a podcast or move between its episodes",
    "play_radio": "Play the radio or a radio station",
    "qa_currency": "Ask an exchange rate or which currency a country uses",
    "qa_definition": "Ask the meaning, definition or spelling of a word",
    "qa_factoid": "Ask a general factual question about the world",
    "qa_maths": "Ask for a calculation or a maths answer",
    "qa_stock": "Ask a stock price or for stock market updates",
    "recommendation_events": "Ask what events or shows are happening, nearby or on a date",
    "recommendation_locations": "Ask for a nearby place to go: a restaurant, a shop, a bar",
    "recommendation_movies": "Ask which movies are showing or which to watch, or for a cinema",
    "social_post": "Post, share or send something to social media",
    "social_query": "Ask about social media activity: updates, comments, likes, trends",
    "takeaway_order": "Order food for delivery or takeaway",
    "takeaway_query": "Ask about takeaway: the status of an order, delivery options, or "
                      "whether a restaurant delivers",
    "transport_query": "Ask about a journey: train times, schedules, routes or where a station is",
    "transport_taxi": "Book or call a taxi or an Uber",
    "transport_ticket": "Book or buy a train, plane or coach ticket",
    "transport_traffic": "Ask about traffic conditions",
    "weather_query": "Ask about the weather or the forecast",
}

INTENTS_FR = {
    "alarm_query": "Demander quelles alarmes sont programmées ou si une alarme donnée l'est",
    "alarm_remove": "Annuler, désactiver ou supprimer une ou plusieurs alarmes",
    "alarm_set": "Programmer une nouvelle alarme ou un réveil",
    "audio_volume_down": "Baisser le volume de l'appareil ou lui demander de parler moins fort",
    "audio_volume_mute": "Couper le son de l'appareil ou de ses haut-parleurs, ou lui demander "
                         "de se taire",
    "audio_volume_other": "Ouvrir ou régler les paramètres de volume et de son sans dire "
                          "monter, baisser ou couper",
    "audio_volume_up": "Monter le volume de l'appareil",
    "calendar_query": "Demander ce qu'il y a à l'agenda : réunions, événements, programme "
                      "d'une journée",
    "calendar_remove": "Supprimer ou annuler des événements ou des réunions de l'agenda",
    "calendar_set": "Ajouter un événement, une réunion ou un rappel à l'agenda",
    "cooking_query": "Poser une question de cuisine générale qui ne demande pas une recette",
    "cooking_recipe": "Demander une recette, comment cuisiner un plat, ou quoi cuisiner avec "
                      "certains ingrédients",
    "datetime_convert": "Convertir une heure d'un fuseau horaire à un autre ou demander le "
                        "décalage horaire entre deux lieux",
    "datetime_query": "Demander l'heure ou la date actuelle, l'heure dans un lieu, ou le jour "
                      "d'une date",
    "email_addcontact": "Enregistrer une personne ou une adresse e-mail comme nouveau contact",
    "email_query": "Consulter la boîte de réception ou se renseigner sur des e-mails reçus",
    "email_querycontact": "Rechercher les coordonnées d'un contact : adresse e-mail, numéro, "
                          "nom ou autre information",
    "email_sendemail": "Envoyer, rédiger ou répondre à un e-mail",
    "general_greet": "Saluer l'assistant",
    "general_joke": "Demander une blague",
    "general_quirky": "Bavarder, faire une remarque insolite ou poser une question ouverte à "
                      "l'assistant qui ne relève d'aucune autre intention",
    "iot_cleaning": "Lancer l'aspirateur robot ou demander de nettoyer le logement",
    "iot_coffee": "Faire du café ou lancer la machine à café",
    "iot_hue_lightchange": "Changer la couleur ou l'ambiance des lumières",
    "iot_hue_lightdim": "Baisser l'intensité des lumières ou assombrir une pièce",
    "iot_hue_lightoff": "Éteindre les lumières",
    "iot_hue_lighton": "Allumer les lumières, ou dire qu'il fait trop sombre",
    "iot_hue_lightup": "Augmenter la luminosité des lumières",
    "iot_wemo_off": "Éteindre une prise connectée, ou l'appareil qui y est branché",
    "iot_wemo_on": "Allumer une prise connectée, ou l'appareil qui y est branché",
    "lists_createoradd": "Créer une nouvelle liste ou ajouter un élément à une liste",
    "lists_query": "Demander quelles listes existent ou ce que contient une liste",
    "lists_remove": "Retirer un élément d'une liste, ou vider ou supprimer une liste",
    "music_dislikeness": "Dire qu'on n'aime pas une chanson, un artiste ou un genre",
    "music_likeness": "Dire qu'on aime une chanson, un artiste ou un genre, ou enregistrer une "
                      "chanson dans ses favoris",
    "music_query": "Demander quelle chanson ou quel artiste passe, ou les paroles ou les "
                   "détails d'une chanson",
    "music_settings": "Modifier les réglages de lecture : aléatoire ou répétition",
    "news_query": "Demander les actualités, les gros titres ou des nouvelles d'un événement",
    "play_audiobook": "Lancer, mettre en pause ou reprendre un livre audio",
    "play_game": "Jouer à un jeu avec l'assistant",
    "play_music": "Lancer de la musique : une chanson, un artiste, un album, un genre ou une "
                  "playlist",
    "play_podcasts": "Lancer un podcast ou passer d'un épisode à l'autre",
    "play_radio": "Lancer la radio ou une station de radio",
    "qa_currency": "Demander un taux de change ou la monnaie d'un pays",
    "qa_definition": "Demander le sens, la définition ou l'orthographe d'un mot",
    "qa_factoid": "Poser une question factuelle générale sur le monde",
    "qa_maths": "Demander un calcul ou le résultat d'une opération",
    "qa_stock": "Demander un cours de bourse ou des nouvelles de la bourse",
    "recommendation_events": "Demander quels événements ou spectacles ont lieu, à proximité ou "
                             "à une date",
    "recommendation_locations": "Demander un lieu où aller à proximité : restaurant, magasin, bar",
    "recommendation_movies": "Demander quels films passent ou lequel regarder, ou un cinéma",
    "social_post": "Publier, partager ou envoyer quelque chose sur les réseaux sociaux",
    "social_query": "Se renseigner sur l'activité des réseaux sociaux : nouvelles, "
                    "commentaires, mentions j'aime, tendances",
    "takeaway_order": "Commander un repas à livrer ou à emporter",
    "takeaway_query": "Se renseigner sur un repas à emporter : état d'une commande, options de "
                      "livraison, ou si un restaurant livre",
    "transport_query": "Se renseigner sur un trajet : horaires de train, itinéraire ou "
                       "emplacement d'une gare",
    "transport_taxi": "Réserver ou appeler un taxi ou un Uber",
    "transport_ticket": "Réserver ou acheter un billet de train, d'avion ou de car",
    "transport_traffic": "Se renseigner sur l'état de la circulation",
    "weather_query": "Demander la météo ou les prévisions",
}

INSTRUCTIONS_EN = "Which intent does this virtual assistant request express"
INSTRUCTIONS_FR = "Quelle intention exprime cette requête adressée à un assistant vocal"

load_pairs = partial(scenario_pairs, gold="intent")

CONDITIONS = {
    "EN_EN": ("en", INSTRUCTIONS_EN, INTENTS_EN),
    "FR_EN": ("fr", INSTRUCTIONS_EN, INTENTS_EN),
    "FR_FR": ("fr", INSTRUCTIONS_FR, INTENTS_FR),
}


def question(condition: str) -> tuple[str, dict]:
    lang, instructions, criteria = CONDITIONS[condition]
    return lang, {"intent": {"type": "choice", "instructions": instructions,
                             "criteria": criteria}}
