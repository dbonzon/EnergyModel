"""
Module pac — calibration du coefficient de déperdition du bâtiment à partir
de la consommation mazout historique, et dérivation de la courbe de
puissance thermique requise (kW).

Aucune dépendance Dash ici : uniquement Pandas.

VERSION TEST simplifiée : un seul total annuel de mazout (pas de détail
mensuel). Limitation connue et assumée pour l'instant : sans ventilation
mensuelle, impossible de séparer la charge de base (eau chaude sanitaire)
de la charge de chauffage pur — le coefficient H calibré ici inclut donc
l'ECS, ce qui le SURESTIME. Prochaine étape : régression linéaire
conso_mensuelle vs DH_mensuel pour isoler la charge de chauffage.
"""

import numpy as np
import pandas as pd

# Pouvoir calorifique du mazout extra-léger (valeur standard Suisse)
PCI_MAZOUT_KWH_PAR_LITRE = 10.0

# Rendement de combustion/distribution typique d'une chaudière mazout
# existante — à ajuster selon l'âge/état réel de l'installation.
RENDEMENT_CHAUDIERE_DEFAUT = 0.85


def energie_mazout_vers_kwh(litres: float, pci_kwh_par_litre: float = PCI_MAZOUT_KWH_PAR_LITRE) -> float:
    """Convertit un volume de mazout (litres) en énergie brute (kWh)."""
    return litres * pci_kwh_par_litre


def calibrer_coefficient_deperdition(
    dh_annuel_historique_c_h: float,
    energie_mazout_kwh: float,
    rendement_chaudiere: float = RENDEMENT_CHAUDIERE_DEFAUT,
) -> float:
    """
    Calibre le coefficient de déperdition global H (kW/°C) du bâtiment.

    H = énergie utile annuelle / DH annuel total
      - énergie utile = énergie mazout brute × rendement chaudière
      - dh_annuel_historique_c_h : somme des DH de l'année de référence,
        calculée avec la consigne et la limite de chauffage HISTORIQUES
        (celles qui prévalaient pendant la période de consommation mazout) —
        pas forcément la consigne cible choisie pour la future PAC.

    Returns:
        H en kW/°C (un écart de température de 1°C soutenu pendant 1h
        correspond à 1 °C·h de DH, donc H[kW/°C] x DH[°C·h] / 1h = kW).
    """
    if dh_annuel_historique_c_h <= 0:
        raise ValueError("DH annuel historique doit être positif pour calibrer H.")
    energie_utile_kwh = energie_mazout_kwh * rendement_chaudiere
    return energie_utile_kwh / dh_annuel_historique_c_h


def calculer_puissance_horaire_kw(dh_horaire_c: pd.Series, h_kw_par_c: float) -> pd.Series:
    """
    Puissance thermique instantanée requise, heure par heure (kW).
    P(t) = H x DH_horaire(t) — le DH horaire est déjà un écart sur 1h,
    donc H [kW/°C] x DH [°C] = kW directement.
    """
    return dh_horaire_c * h_kw_par_c


def calculer_besoin_tampon_kwh(production_kw: pd.Series, demande_kw: pd.Series) -> pd.Series:
    """
    Capacité utile (kWh) nécessaire au ballon tampon, JOUR PAR JOUR.

    Principe : le ballon absorbe le surplus produit pendant les heures actives
    (production constante, lissée sur la plage horaire) et le restitue
    pendant les heures creuses (où la demande réelle du bâtiment continue,
    mais la production est nulle). La capacité nécessaire pour ce jour-là
    est l'amplitude (max - min) de l'état de charge cumulé (production -
    demande), remis à zéro à chaque début de journée.

    Args:
        production_kw: puissance produite, heure par heure (profil lissé/
            redistribué sur la plage horaire active — ce qui sort de
            calculer_puissance_horaire_kw après redistribuer_dh_horaire)
        demande_kw: puissance réellement nécessaire au bâtiment, heure par
            heure, SANS redistribution (perte de chaleur physique continue,
            24h/24 — H x DH horaire brut)

    Returns:
        pd.Series indexée par jour (minuit), capacité tampon requise ce
        jour-là en kWh. Le maximum sur l'année = dimensionnement du ballon.
    """
    ecart = production_kw - demande_kw
    jours = ecart.index.normalize()
    etat_charge = ecart.groupby(jours).cumsum()
    return etat_charge.groupby(jours).agg(lambda s: s.max() - s.min())


def convertir_kwh_vers_litres_eau(energie_kwh: float, delta_t_c: float = 20.0) -> float:
    """
    Convertit une capacité énergétique (kWh) en volume d'eau (litres) pour
    un ballon tampon, étant donné un écart de température utile (°C) entre
    l'eau "chaude" (départ PAC) et "froide" (retour) du ballon.

    Formule : volume[L] = énergie[kWh] x 3600 / (cp_eau x ΔT)
    avec cp_eau = 4.186 kJ/(kg·°C), densité eau ≈ 1 kg/L.
    """
    cp_eau_kj_par_kg_c = 4.186
    return energie_kwh * 3600 / (cp_eau_kj_par_kg_c * delta_t_c)


# Écart minimal entre l'eau de départ et la consigne intérieure pour que
# l'émetteur (radiateur/plancher chauffant) puisse encore céder suffisamment
# de chaleur par différence de température — valeur indicative à ajuster
# selon le dimensionnement réel de l'émetteur.
MARGE_SECURITE_EMISSION_C = 5.0


def calculer_delta_t_ballon(temp_depart_c: float, consigne_c: float, marge_c: float = MARGE_SECURITE_EMISSION_C) -> float:
    """
    Dérive le ΔT utile du ballon à partir de grandeurs physiquement
    significatives plutôt que d'un écart abstrait :
      - temp_depart_c : température de départ visée par la PAC (°C)
      - consigne_c : température intérieure de consigne (°C)
      - marge_c : écart minimal requis pour que l'émetteur cède encore
        de la chaleur (eau trop proche de la température ambiante = plus
        d'échange utile)

    Returns:
        ΔT utile (°C). Peut être négatif ou nul si temp_depart_c est trop
        basse pour la consigne visée — c'est un signal que le système
        d'émission n'est pas adapté à cette consigne, pas une erreur à masquer.
    """
    return temp_depart_c - (consigne_c + marge_c)


# Fraction typique du COP de Carnot théorique atteinte par une PAC air/eau
# réelle du marché (0.40-0.50 usuel) — exposé comme paramètre ajustable.
FACTEUR_QUALITE_DEFAUT = 0.45
COP_MIN = 1.0
COP_MAX = 6.0


def calculer_cop_carnot(
    temp_source_c: pd.Series,
    temp_depart_c: float,
    facteur_qualite: float = FACTEUR_QUALITE_DEFAUT,
) -> pd.Series:
    """
    COP horaire selon un modèle de Carnot corrigé par un facteur de qualité.

    COP_Carnot = T_départ[K] / (T_départ[K] - T_source[K])
    COP_réel   = COP_Carnot x facteur_qualite, borné à [COP_MIN, COP_MAX]

    LIMITE CONNUE : ce modèle thermodynamique théorique ignore le givrage de
    l'évaporateur (pertes de dégivrage, significatives entre environ -2°C et
    +5°C côté air extérieur) — le COP réel y est généralement plus bas que
    ce que ce modèle lisse prédit. Pour une étude plus fine, remplacer par
    une courbe de performance constructeur (table COP en fonction de T_ext
    et T_départ, propre à chaque modèle de PAC).

    Args:
        temp_source_c: température de la source froide (air extérieur), °C
        temp_depart_c: température de départ visée par la PAC, °C (constante)
        facteur_qualite: fraction du COP de Carnot théorique atteinte en pratique
    """
    temp_depart_k = temp_depart_c + 273.15
    temp_source_k = temp_source_c + 273.15
    ecart_k = (temp_depart_k - temp_source_k).clip(lower=0.5)  # évite division par zéro/négatif hors saison de chauffe
    cop_carnot = temp_depart_k / ecart_k
    cop_reel = cop_carnot * facteur_qualite
    return cop_reel.clip(lower=COP_MIN, upper=COP_MAX)


def calculer_puissance_electrique_kw(puissance_thermique_kw: pd.Series, cop: pd.Series) -> pd.Series:
    """Puissance électrique absorbée par la PAC, heure par heure : P_elec = P_thermique / COP."""
    return puissance_thermique_kw / cop


def calculer_cout_annuel_chf(puissance_electrique_kw: pd.Series, prix_chf_par_kwh: float) -> float:
    """Coût annuel (CHF) = énergie électrique totale (kWh, chaque point = 1h) x prix unitaire."""
    return puissance_electrique_kw.sum() * prix_chf_par_kwh


def interpoler_lineaire_extrapolee(x, xp, fp):
    """
    Comme np.interp, mais EXTRAPOLE linéairement au-delà des bornes de xp
    (np.interp se contente de clipper aux valeurs d'extrémité, ce qui sous-
    estime le COP aux températures douces hors grille constructeur).
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    resultat = np.interp(x, xp, fp)
    sous_bornes = x < xp[0]
    if np.any(sous_bornes):
        pente = (fp[1] - fp[0]) / (xp[1] - xp[0])
        resultat = np.where(sous_bornes, fp[0] + pente * (x - xp[0]), resultat)
    sur_bornes = x > xp[-1]
    if np.any(sur_bornes):
        pente = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        resultat = np.where(sur_bornes, fp[-1] + pente * (x - xp[-1]), resultat)
    return resultat


# Données constructeur réelles (EN14511), PAC air/eau Templari KITA LP-22 :
# https://www.templari.com/wp-content/uploads/2025/11/Product-Data-Sheet-KITA-LP.pdf
# Grille : T_ext (air extérieur) x T_départ (eau), valeurs de COP mesurées.
TABLE_COP_TEMP_EXT_C = np.array([-7.0, 2.0, 7.0])
TABLE_COP_TEMP_DEPART_C = np.array([35.0, 55.0])
TABLE_COP_VALEURS = np.array([
    [3.08, 2.11],  # T_ext = -7°C : COP à W35, W55
    [4.09, 2.75],  # T_ext =  2°C
    [4.52, 3.01],  # T_ext =  7°C
])
SOURCE_TABLE_COP = "Templari KITA LP-22, EN14511 (A-7/A2/A7, W35/W55)"


def calculer_cop_table_reelle(
    temp_source_c: pd.Series,
    temp_depart_c: float,
    table_temp_ext_c: np.ndarray = TABLE_COP_TEMP_EXT_C,
    table_temp_depart_c: np.ndarray = TABLE_COP_TEMP_DEPART_C,
    table_valeurs: np.ndarray = TABLE_COP_VALEURS,
) -> pd.Series:
    """
    COP horaire interpolé (bilinéaire) à partir d'une table de points
    constructeur RÉELS (EN14511) — par défaut SOURCE_TABLE_COP, mais
    entièrement remplaçable par la fiche technique d'un autre modèle
    (mêmes dimensions : 3 T_ext x 2 T_départ) sans toucher à cette fonction.

    Étape 1 : pour chaque T_ext de la grille, interpole le COP à la
    température de départ visée (entre les colonnes de la table).
    Étape 2 : interpole ensuite ces valeurs sur la série T_ext réelle.

    LIMITES À CONNAÎTRE :
    - Le profil réel varie d'un fabricant/modèle à l'autre (technologie du
      compresseur, injection de vapeur, etc.) — remplace table_valeurs par
      la fiche technique du modèle réellement visé pour affiner.
    - Hors des bornes de la table (T_ext ou T_départ) : EXTRAPOLATION
      linéaire (pas de mesure réelle au-delà) — prudence sur les valeurs
      extrêmes.
    """
    cop_par_text_grille = np.array([
        interpoler_lineaire_extrapolee(temp_depart_c, table_temp_depart_c, ligne)
        for ligne in table_valeurs
    ])
    valeurs = interpoler_lineaire_extrapolee(temp_source_c.values, table_temp_ext_c, cop_par_text_grille)
    return pd.Series(valeurs, index=temp_source_c.index, name="cop").clip(lower=COP_MIN, upper=COP_MAX)


def calculer_cout_mazout_chf(energie_mazout_kwh: float, prix_chf_par_litre: float) -> float:
    """
    Coût annuel mazout (CHF) = volume brûlé (L) x prix au litre.
    Le volume se déduit de l'énergie brute (avant rendement chaudière) via
    le pouvoir calorifique — on paie le mazout brûlé, pas l'énergie utile.
    """
    litres = energie_mazout_kwh / PCI_MAZOUT_KWH_PAR_LITRE
    return litres * prix_chf_par_litre


def calculer_masque_heures_creuses(index_horaire: pd.DatetimeIndex, heure_debut_hc: int, duree_hc_h: int) -> np.ndarray:
    """
    Masque booléen (True = heure creuse) pour un index horaire, étant donné
    une heure de début et une durée — gère nativement le passage de minuit
    (ex. débute à 22h, dure 8h -> couvre 22h-23h ET 0h-5h).

    Une durée exprimée plutôt qu'une heure de fin évite toute ambiguïté
    UI sur le sens du slider quand la plage traverse minuit.
    """
    heure = index_horaire.hour.values
    heures_creuses = {(heure_debut_hc + i) % 24 for i in range(duree_hc_h)}
    return np.isin(heure, list(heures_creuses))


def calculer_cout_annuel_bitarif_chf(
    puissance_elec_kw: pd.Series,
    heure_debut_hc: int,
    duree_hc_h: int,
    prix_hc_chf_kwh: float,
    prix_hp_chf_kwh: float,
) -> dict:
    """
    Coût annuel électrique avec tarification heures creuses (HC) / heures
    pleines (HP) — répartit l'énergie horaire selon le masque HC, applique
    le prix correspondant à chaque groupe, puis additionne.

    Returns:
        dict avec kwh_hc, kwh_hp, cout_hc_chf, cout_hp_chf, cout_total_chf
    """
    masque_hc = calculer_masque_heures_creuses(puissance_elec_kw.index, heure_debut_hc, duree_hc_h)
    kwh_hc = puissance_elec_kw[masque_hc].sum()
    kwh_hp = puissance_elec_kw[~masque_hc].sum()
    cout_hc = kwh_hc * prix_hc_chf_kwh
    cout_hp = kwh_hp * prix_hp_chf_kwh
    return {
        "kwh_hc": kwh_hc, "kwh_hp": kwh_hp,
        "cout_hc_chf": cout_hc, "cout_hp_chf": cout_hp,
        "cout_total_chf": cout_hc + cout_hp,
    }


def calculer_cout_annuel_bitarif_masque_chf(
    puissance_elec_kw: pd.Series,
    masque_heures_creuses_24: list,
    prix_hc_chf_kwh: float,
    prix_hp_chf_kwh: float,
) -> dict:
    """
    Comme calculer_cout_annuel_bitarif_chf, mais accepte un masque EXPLICITE
    de 24 booléens (un par heure 0-23) plutôt qu'un bloc contigu — permet de
    représenter n'importe quel motif tarifaire (nuit + mi-journée pour capter
    le solaire, week-end différent du motif actuel une fois étendu, etc.),
    au lieu d'un seul créneau continu.

    Args:
        masque_heures_creuses_24: liste de 24 bool, True = heure creuse, index 0 = minuit
    """
    masque_array = np.asarray(masque_heures_creuses_24, dtype=bool)
    heure = puissance_elec_kw.index.hour.values
    masque_hc = masque_array[heure]
    kwh_hc = puissance_elec_kw[masque_hc].sum()
    kwh_hp = puissance_elec_kw[~masque_hc].sum()
    cout_hc = kwh_hc * prix_hc_chf_kwh
    cout_hp = kwh_hp * prix_hp_chf_kwh
    return {
        "kwh_hc": kwh_hc, "kwh_hp": kwh_hp,
        "cout_hc_chf": cout_hc, "cout_hp_chf": cout_hp,
        "cout_total_chf": cout_hc + cout_hp,
    }