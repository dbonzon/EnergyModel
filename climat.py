"""
Module climat — génération et transformation de séries de température
pour le calcul des degrés-heures (DH).

Aucune dépendance Dash ici : uniquement Numpy/Pandas.
Point d'entrée à remplacer plus tard : generer_serie_horaire_synthetique()
sera substituée par un import de données réelles (CSV MétéoSuisse/IDAWEB).
"""

import warnings
from io import StringIO

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

# Code sentinelle MétéoSuisse pour une mesure absente : valeur max d'un
# entier signé 16 bits (2^15 - 1), convention héritée des formats de fichiers
# à largeur fixe où NaN/null n'existait pas encore.
VALEUR_MANQUANTE_METEOSUISSE = 32767


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


def charger_temperature_meteosuisse(
    chemin_fichier: str,
    valeur_manquante: int = VALEUR_MANQUANTE_METEOSUISSE,
    fuseau_horaire: str = "Europe/Zurich",
) -> pd.Series:
    """
    Charge un export horaire MétéoSuisse (format "Climap", .dat) — GÉNÉRIQUE,
    fonctionne pour n'importe quel fichier de ce format (station/année/
    chemin passés en paramètre, rien de codé en dur ici).

    Remplace generer_serie_horaire_synthetique() : le reste du pipeline
    (correction altitude, calcul DH, calibration H) n'a rien à changer —
    il consomme une pd.Series horaire, peu importe son origine.

    Format attendu : en-tête de longueur variable, puis une ligne de colonnes
    "STA JAHR MO TG HH MM <code_param>", suivie des données (séparateur
    espaces, largeur libre). Heures du fichier en UTC.

    Args:
        chemin_fichier: chemin vers le .dat MétéoSuisse
        valeur_manquante: code sentinelle signalant une mesure absente
            (voir VALEUR_MANQUANTE_METEOSUISSE)
        fuseau_horaire: fuseau cible pour la conversion UTC -> heure locale

    Returns:
        pd.Series indexée par datetime horaire LOCAL (naïve, sans tz), en °C.
        Les valeurs manquantes sont interpolées linéairement (et un avertissement
        est émis si on en trouve).

    Limite connue : la conversion UTC -> heure locale introduit une heure
    "manquante" au changement d'heure de mars et une heure dupliquée en
    octobre (DST). Impact négligeable sur un total annuel de DH (en dehors
    de la saison de chauffe), mais à garder en tête pour une analyse fine
    de ces deux mois précis.
    """
    with open(chemin_fichier, "rb") as f:
        lignes = f.read().decode("latin-1").splitlines()

    # Repère la ligne d'en-tête des colonnes de façon générique (pas de
    # numéro de ligne codé en dur, au cas où l'en-tête MétéoSuisse varie
    # en longueur d'un export à l'autre).
    idx_entete = next(
        i for i, l in enumerate(lignes) if "STA" in l and "JAHR" in l and "HH" in l
    )
    lignes_donnees = "\n".join(l for l in lignes[idx_entete + 1:] if l.strip())

    df = pd.read_csv(
        StringIO(lignes_donnees),
        sep=r"\s+",
        header=None,
        names=["station", "annee", "mois", "jour", "heure", "minute", "valeur"],
    )

    n_manquantes = int((df["valeur"] == valeur_manquante).sum())
    if n_manquantes:
        warnings.warn(
            f"{n_manquantes} valeur(s) manquante(s) (code {valeur_manquante}) "
            f"dans {chemin_fichier} — interpolées linéairement."
        )
    df["valeur"] = df["valeur"].replace(valeur_manquante, np.nan)

    timestamps_utc = pd.to_datetime(
        dict(year=df["annee"], month=df["mois"], day=df["jour"], hour=df["heure"])
    ).dt.tz_localize("UTC")
    timestamps_locaux = timestamps_utc.dt.tz_convert(fuseau_horaire).dt.tz_localize(None)

    serie = pd.Series(df["valeur"].values, index=timestamps_locaux, name="temperature_C")
    serie = serie.sort_index().interpolate(method="linear").ffill().bfill()
    return serie


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


def calculer_dh_horaire(
    temp_serie: pd.Series,
    reference_c: float,
    limite_chauffe_c: float = LIMITE_CHAUFFE_CONVENTIONNELLE_C,
) -> pd.Series:
    """
    DH horaire (°C), une valeur par heure de l'année — utilisée telle quelle
    pour l'affichage (graphs DH) comme pour la calibration/puissance (kW),
    pas besoin d'agrégation intermédiaire (Plotly gère 8760 points sans souci).

    DH horaire = (reference - T_ext) uniquement si T_ext < limite_chauffe_c, 0 sinon.
    Convention SIA "degrés-jours 20/12" : au-dessus de la limite de chauffage
    (12°C par défaut), les apports internes/solaires sont supposés couvrir les
    pertes, même si reference - T_ext est théoriquement positif.
    """
    sous_limite = temp_serie < limite_chauffe_c
    return (reference_c - temp_serie).where(sous_limite, 0.0).clip(lower=0)


def redistribuer_dh_horaire(
    dh_horaire: pd.Series,
    heure_debut: int,
    heure_fin: int,
) -> pd.Series:
    """
    Concentre le besoin DH quotidien (24h) sur une plage horaire active
    (heure_debut inclus, heure_fin exclu) — simule un chauffage qui ne
    fonctionne pas en continu (ex. plage tarifaire, restriction de bruit).

    HYPOTHÈSE : le bâtiment a assez d'inertie thermique (ou un ballon tampon)
    pour lisser le besoin sur 24h en ne le délivrant que sur la plage active.
    Le total journalier d'énergie est conservé (aucune perte/gain net), mais
    la puissance instantanée pendant la plage active augmente d'autant.

    Args:
        dh_horaire: série DH horaire sur toute la période (pas de Series)
        heure_debut: heure de début de fonctionnement (0-23, incluse)
        heure_fin: heure de fin de fonctionnement (1-24, exclue)

    Returns:
        pd.Series DH horaire redistribué : 0 hors plage, valeur amplifiée
        dans la plage de sorte que le total par jour reste inchangé.
    """
    if not (0 <= heure_debut < heure_fin <= 24):
        raise ValueError("Plage horaire invalide : 0 <= heure_debut < heure_fin <= 24 requis.")

    n_heures_actives = heure_fin - heure_debut
    total_par_jour = dh_horaire.resample("D").sum()
    valeur_horaire_active = total_par_jour / n_heures_actives

    jour = dh_horaire.index.normalize()
    valeur_reportee = valeur_horaire_active.reindex(jour).values

    heure_du_jour = dh_horaire.index.hour
    dans_plage = (heure_du_jour >= heure_debut) & (heure_du_jour < heure_fin)

    return pd.Series(
        np.where(dans_plage, valeur_reportee, 0.0),
        index=dh_horaire.index,
        name="dh_redistribue",
    )