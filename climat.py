"""
Module climat — génération et transformation de séries de température
pour le calcul des degrés-heures (DH).

Aucune dépendance Dash ici : uniquement Numpy/Pandas.
Point d'entrée à remplacer plus tard : generer_serie_horaire_synthetique()
sera substituée par un import de données réelles (CSV MétéoSuisse/IDAWEB).
"""

import numpy as np
import pandas as pd

# --- Paramètres de site (à vérifier/ajuster) ---
ALTITUDE_STATION_PULLY_M = 460      # altitude officielle de la station Pully à vérifier sur MétéoSuisse
ALTITUDE_CHARDONNE_M = 592          # donnée du village de Chardonne
GRADIENT_THERMIQUE_C_PAR_100M = -0.65  # gradient adiabatique standard (~-0.65°C/100m)

# Référence conventionnelle pour les degrés-heures "bruts" (graphs 1 & 2),
# avant introduction de la consigne réelle (graph 3).
# Convention SIA "degrés-jours 20/12" : 20°C intérieur / 12°C = limite de mise
# en route du chauffage (Heizgrenze). Cette limite dépend de la qualité
# d'isolation réelle du bâtiment — à ajuster (voir table OFEN dans la docstring
# de calculer_degres_heures).
BASE_DH_CONVENTIONNELLE_C = 20.0
LIMITE_CHAUFFE_CONVENTIONNELLE_C = 12.0


def generer_serie_horaire_synthetique(
    annee: int = 2024,
    temp_moy_annuelle: float = 10.5,   # climat lémanique tempéré par le lac
    amplitude_annuelle: float = 9.0,   # écart saisonnier hiver/été
    amplitude_journaliere: float = 3.0,  # cycle jour/nuit
    bruit_std: float = 1.2,
    seed: int = 42,
) -> pd.Series:
    """
    PLACEHOLDER — série horaire synthétique pour valider le pipeline.
    À remplacer par le chargement d'un vrai fichier (MétéoSuisse/IDAWEB)
    une fois disponible : la suite du pipeline (correction altitude,
    calcul DH) n'a pas besoin de changer.

    Returns:
        pd.Series indexée par datetime horaire, valeurs en °C
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(f"{annee}-01-01", f"{annee}-12-31 23:00", freq="h")
    heures = np.arange(len(index))

    cycle_annuel = -amplitude_annuelle * np.cos(2 * np.pi * heures / (365.25 * 24))
    cycle_journalier = -amplitude_journaliere * np.cos(2 * np.pi * heures / 24)
    bruit = rng.normal(0, bruit_std, size=len(index))

    temp = temp_moy_annuelle + cycle_annuel + cycle_journalier + bruit
    return pd.Series(temp, index=index, name="temperature_C")


def corriger_altitude(
    temp_serie: pd.Series,
    delta_altitude_m: float,
    gradient_c_par_100m: float = GRADIENT_THERMIQUE_C_PAR_100M,
) -> pd.Series:
    """
    Corrige une série de température pour un écart d'altitude.
    delta_altitude_m > 0 si le lieu cible est plus haut que la station.
    """
    correction = gradient_c_par_100m * (delta_altitude_m / 100)
    return temp_serie + correction


def calculer_degres_heures(
    temp_serie: pd.Series,
    reference_c: float,
    limite_chauffe_c: float = LIMITE_CHAUFFE_CONVENTIONNELLE_C,
) -> pd.Series:
    """
    DH horaire = (reference - T_ext) uniquement si T_ext < limite_chauffe_c, 0 sinon.

    Convention SIA "degrés-jours 20/12" : au-dessus de la limite de chauffage
    (12°C par défaut), les apports internes/solaires sont supposés couvrir les
    pertes, même si reference - T_ext est théoriquement positif.
    La limite de chauffage dépend de la qualité d'isolation du bâtiment
    (cf. valeurs indicatives OFEN : 9-14°C Minergie, 12-15°C bâti 1995-2010,
    14-17°C bâti plus ancien) — à ajuster selon l'état réel de l'enveloppe.

    Sommée par jour, elle donne les degrés-heures/jour habituellement utilisés
    pour caractériser un besoin de chauffe.
    """
    sous_limite = temp_serie < limite_chauffe_c
    dh_horaire = (reference_c - temp_serie).where(sous_limite, 0.0).clip(lower=0)
    return dh_horaire.resample("D").sum()