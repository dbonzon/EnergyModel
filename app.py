from dash import Dash, dash_table, dcc, html, Input, Output, State, ALL, ctx, no_update
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import climat
import pac

app = Dash(__name__)
server = app.server

# --- Calculs indépendants des sliders : faits une seule fois au démarrage ---
# Placer le fichier .dat MétéoSuisse à côté de app.py, ou ajuster le chemin ci-dessous.
temp_pully = climat.charger_temperature_meteosuisse("Data/T2m_Pully_20220101-20221231.dat")

# Repères OFEN pour la limite de chauffage (référencés à une consigne de 20°C)
MARQUES_LIMITE_CHAUFFE = {
    9: "9 (passif)", 12: "12 (SIA std)", 14: "14 (1995-2010)",
    16: "16 (1977-95)", 17: "17 (avant 1977)",
}

# --- Explications pédagogiques par étape, affichées en infobulle au survol du ⓘ ---
EXPLICATIONS = {
    "temperature_brute": (
        "Donnée brute telle qu'importée du fichier MétéoSuisse : température horaire à 2m du "
        "sol, station de Pully, sans aucune transformation.\n\n"
        "La ligne pointillée marque la limite de chauffage (slider) : en dessous, l'heure "
        "contribue au DH (Graph 1) ; au-dessus, DH = 0 pour cette heure.\n\n"
        "C'est cette courbe qui est transformée, heure par heure, pour produire le Graph 1."
    ),
    "pully": (
        "DH horaire = (référence − T_ext), uniquement si T_ext < limite de chauffage, sinon 0.\n\n"
        "Référence = 20°C (convention SIA \"degrés-jours 20/12\" : température intérieure "
        "conventionnelle en Suisse).\n\n"
        "Limite de chauffage = 12°C par défaut (réglable via le slider) : au-dessus, on suppose "
        "que les apports internes/solaires couvrent les pertes, même si l'écart à 20°C est positif.\n\n"
        "Série de température : export horaire réel MétéoSuisse (format Climap) pour la "
        "station de Pully, converti d'UTC en heure locale Europe/Zurich."
    ),
    "construction": (
        "Même formule DH que le graph 1, mais appliquée à la température corrigée pour l'altitude "
        "de la construction.\n\n"
        "Correction d'altitude : gradient thermique vertical standard ≈ −0.65°C/100m "
        "(air plus froid en altitude).\n\n"
        "T_corrigée = T_Pully + gradient × (Δaltitude / 100)\n\n"
        "⚠️ Ce gradient est une moyenne atmosphérique générique, pas une valeur officielle SIA/OFEN "
        "pour ce lieu précis — à affiner si possible avec les données SIA 2028 de Chardonne."
    ),
    "consigne": (
        "Même formule DH, mais la référence 20°C est remplacée par la consigne intérieure visée "
        "(slider) — c'est la courbe qui reflète vraiment TON objectif de confort, pas la convention "
        "SIA par défaut.\n\n"
        "La limite de chauffage reste liée à la qualité de l'enveloppe du bâtiment, donc "
        "indépendante de ce slider."
    ),
    "puissance": (
        "Calibration : H [kW/°C] = (énergie mazout × rendement chaudière) / DH annuel total\n"
        "— le DH utilisé ici est celui avec la référence 20°C (proxy de la consigne historique "
        "pendant les années de conso mazout), pas la consigne cible.\n\n"
        "Puissance : P(t) = H × DH(t), où DH(t) utilise la consigne CIBLE (slider), redistribué "
        "au préalable sur la plage horaire de fonctionnement choisie (ex. 6h-22h) — le besoin "
        "quotidien total reste le même, mais concentré sur moins d'heures actives, donc la "
        "pointe de puissance augmente d'autant.\n\n"
        "⚠️ Version test : un seul total annuel de mazout → impossible de séparer la charge de "
        "chauffage de l'eau chaude sanitaire (ECS). H est donc probablement surestimé. "
        "À affiner avec la conso mois par mois (régression linéaire).\n\n"
        "Contrôle : le panneau de droite cumule la puissance en énergie (kWh). À consigne = 20°C "
        "(référence historique), ce total doit rejoindre exactement la ligne pointillée bleue "
        "(énergie utile mazout) — c'est la définition même de H. À toute autre consigne, l'écart "
        "est normal : plus/moins chaud visé = plus/moins d'énergie requise."
    ),
    "tampon": (
        "Chiffre l'hypothèse d'inertie/tampon du Graph 4 : le ballon doit stocker le surplus produit "
        "pendant les heures actives et le restituer pendant les heures creuses, où le bâtiment "
        "continue à perdre de la chaleur en continu.\n\n"
        "Capacité d'un jour = amplitude (max − min) de l'écart cumulé (production lissée − demande "
        "réelle continue) sur ce jour — même principe que le dimensionnement d'une batterie contre "
        "une courbe de charge/décharge.\n\n"
        "Le maximum sur l'année (jour le plus froid) donne la capacité à installer.\n\n"
        "ΔT dérivé de grandeurs physiques plutôt que fixé arbitrairement : "
        "ΔT = T_départ − (consigne + marge de sécurité). La marge garantit que l'eau reste "
        "assez chaude par rapport à l'air ambiant pour que l'émetteur (radiateur/plancher) "
        "cède encore de la chaleur — sinon le ballon \"semble\" plein mais n'est plus utile."
    ),
    "cop": (
        "COP interpolé (bilinéaire) sur des données constructeur RÉELLES (norme EN14511), "
        f"pas un modèle théorique : {pac.SOURCE_TABLE_COP}.\n\n"
        "Pour chaque T_ext de la grille (-7, 2, 7°C), interpolation du COP entre les points "
        "W35/W55 à la température de départ visée (slider) ; puis interpolation sur la série "
        "T_ext réelle (corrigée altitude).\n\n"
        "⚠️ LIMITES : grille basée sur UN modèle précis — un autre fabricant/modèle aura un "
        "profil différent (technologie compresseur, injection vapeur...). Hors de la grille "
        "(T_ext < -7°C ou > 7°C, T_départ hors 35-55°C), extrapolation linéaire — prudence sur "
        "les valeurs extrêmes.\n\n"
        "Courbe affichée uniquement pendant les heures de besoin réel (puissance > 0)."
    ),
    "electrique": (
        "Puissance électrique absorbée = puissance thermique (Graph 4) ÷ COP horaire (Graph 6, "
        "données réelles).\n\n"
        "Tarification bi-horaire (HC/HP) : chaque heure est classée creuse ou pleine via la "
        "grille cliquable (n'importe quel motif — nuit seule, nuit + mi-journée solaire, etc.), "
        "puis facturée à son propre prix. Coût PAC annuel = somme des deux.\n\n"
        "Note : la plage horaire de fonctionnement de la PAC (6h-22h par défaut, en haut) et la "
        "grille tarifaire HC sont indépendantes et peuvent se chevaucher ou non — ajuste les "
        "deux pour simuler ton vrai contrat électrique.\n"
        "Coût mazout équivalent = (énergie mazout brute saisie plus haut ÷ pouvoir calorifique) "
        "× prix mazout (slider) — c'est le coût mazout historique réel, pas recalculé depuis le DH.\n\n"
        "Économie affichée = mazout − PAC, pour les prix actuellement sélectionnés sur les deux "
        "sliders.\n\n"
        "⚠️ Le mazout est un marché volatil (le prix a par exemple bondi de +32% en un mois au "
        "printemps 2026 lors de tensions géopolitiques) — le défaut du slider suit le niveau "
        "courant, mais ajuste-le à ta facture la plus récente pour un calcul fiable."
    ),
}


# Masque HC par défaut : nuit 22h-6h (même motif qu'avant, mais exprimé
# comme 24 booléens explicites — modifiable librement par clic dans l'UI)
MASQUE_HC_DEFAUT = [1 if (h >= 22 or h < 6) else 0 for h in range(24)]


def style_bouton_heure(actif):
    return {
        "width": "30px", "height": "38px", "fontSize": "11px", "padding": "0",
        "border": "1px solid #ccc", "borderRadius": "4px", "cursor": "pointer",
        "backgroundColor": "#1f77b4" if actif else "#eee",
        "color": "white" if actif else "#333",
    }


def generer_grille_heures(masque_initial):
    boutons = [
        html.Button(
            str(h), id={"type": "bouton-heure", "index": h}, n_clicks=0,
            style=style_bouton_heure(masque_initial[h]),
        )
        for h in range(24)
    ]
    return html.Div(boutons, style={"display": "flex", "gap": "3px", "flexWrap": "wrap"})


ICONE_INFO_SVG = (
    "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20"
    "viewBox%3D%220%200%2024%2024%22%20width%3D%2216%22%20height%3D%2216%22%3E%0A"
    "%3Ccircle%20cx%3D%2212%22%20cy%3D%2212%22%20r%3D%2210%22%20fill%3D%22none%22%20"
    "stroke%3D%22%23888%22%20stroke-width%3D%221.6%22/%3E%0A"
    "%3Ccircle%20cx%3D%2212%22%20cy%3D%227.5%22%20r%3D%221.1%22%20fill%3D%22%23888%22/%3E%0A"
    "%3Crect%20x%3D%2211%22%20y%3D%2210.5%22%20width%3D%222%22%20height%3D%227%22%20rx%3D%221%22%20fill%3D%22%23888%22/%3E%0A"
    "%3C/svg%3E"
)


def titre_avec_info(texte, cle_explication):
    """Titre de graph + icône info (SVG) avec infobulle CSS (survol, classes dans assets/style.css)."""
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
        children=[
            html.H4(texte, style={"margin": 0}),
            html.Span(
                className="info-icone",
                children=[
                    html.Img(src=ICONE_INFO_SVG, style={"width": "16px", "height": "16px", "display": "block"}),
                    html.Div(EXPLICATIONS[cle_explication], className="info-bulle"),
                ],
            ),
        ],
    )


def figure_dh_avec_integrale(serie_dh, titre, couleur):
    """Courbe DH/jour à gauche, DH cumulé (intégrale sur l'année) à droite."""
    cumul = serie_dh.cumsum()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("DH / jour", f"DH cumulé — total {cumul.iloc[-1]:,.0f} °C·h".replace(",", "'")),
    )
    fig.add_trace(
        go.Scatter(x=serie_dh.index, y=serie_dh.values, mode="lines", line=dict(color=couleur), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cumul.index, y=cumul.values, mode="lines", line=dict(color=couleur, dash="dot"), showlegend=False),
        row=1, col=2,
    )
    fig.update_layout(
        title=titre,
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
    )
    return fig


def figure_temperature_brute(temp_serie, limite_chauffe, titre):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=temp_serie.index, y=temp_serie.values, mode="lines",
                   line=dict(color="#7f7f7f", width=0.8), name="T° à 2m")
    )
    fig.add_hline(
        y=limite_chauffe, line_dash="dot", line_color="#d62728",
        annotation_text=f"Limite de chauffage ({limite_chauffe}°C)",
        annotation_position="top left",
    )
    fig.update_layout(
        title=titre,
        xaxis_title="Date",
        yaxis_title="Température (°C)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        showlegend=False,
    )
    return fig


def figure_besoin_tampon(serie_kwh_jour, titre):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_kwh_jour.index, y=serie_kwh_jour.values, mode="lines", line=dict(color="#9467bd")))
    fig.update_layout(
        title=f"{titre} — pire jour {serie_kwh_jour.max():.1f} kWh",
        xaxis_title="Date",
        yaxis_title="Capacité tampon requise (kWh)",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def figure_cop(serie_cop_masque, titre):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_cop_masque.index, y=serie_cop_masque.values, mode="lines", line=dict(color="#17becf")))
    moyenne = serie_cop_masque.mean()
    fig.update_layout(
        title=f"{titre} — moyenne (heures de chauffe) {moyenne:.2f}",
        xaxis_title="Date",
        yaxis_title="COP",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def figure_puissance_electrique(serie_kw_elec, titre):
    cumul_kwh = serie_kw_elec.cumsum()
    total_kwh = cumul_kwh.iloc[-1]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Puissance électrique (kW)",
            f"Énergie cumulée — {total_kwh:,.0f} kWh".replace(",", "'"),
        ),
    )
    fig.add_trace(
        go.Scatter(x=serie_kw_elec.index, y=serie_kw_elec.values, mode="lines", line=dict(color="#e377c2"), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cumul_kwh.index, y=cumul_kwh.values, mode="lines", line=dict(color="#e377c2", dash="dot"), showlegend=False),
        row=1, col=2,
    )
    fig.update_layout(
        title=titre,
        margin=dict(l=40, r=20, t=60, b=40),
        height=340,
    )
    return fig


def figure_puissance_avec_integrale(serie_kw, titre, energie_utile_reference_kwh=None):
    """
    Puissance instantanée (kW) à gauche, énergie cumulée (kWh) à droite —
    chaque point horaire de kW équivaut à 1 kWh, donc la somme cumulée (cumsum)
    donne directement l'énergie en kWh, sans facteur de conversion supplémentaire.

    Si energie_utile_reference_kwh est fourni, une ligne pointillée horizontale
    sert de repère de contrôle : à consigne = référence historique (20°C), le
    total cumulé doit exactement la rejoindre (c'est la définition même de H) ;
    à toute autre consigne, l'écart entre les deux est normal et attendu.
    """
    cumul_kwh = serie_kw.cumsum()
    total_kwh = cumul_kwh.iloc[-1]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Puissance (kW)",
            f"Énergie cumulée — total {total_kwh:,.0f} kWh".replace(",", "'"),
        ),
    )
    fig.add_trace(
        go.Scatter(x=serie_kw.index, y=serie_kw.values, mode="lines", line=dict(color="#d62728"), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cumul_kwh.index, y=cumul_kwh.values, mode="lines", line=dict(color="#d62728", dash="dot"), showlegend=False),
        row=1, col=2,
    )
    if energie_utile_reference_kwh is not None:
        fig.add_hline(
            y=energie_utile_reference_kwh, line_dash="dash", line_color="#1f77b4",
            annotation_text=f"Énergie utile mazout ({energie_utile_reference_kwh:,.0f} kWh)".replace(",", "'"),
            annotation_position="bottom right",
            row=1, col=2,
        )
    fig.update_layout(
        title=f"{titre} — pointe {serie_kw.max():.1f} kW",
        margin=dict(l=40, r=20, t=60, b=40),
        height=340,
    )
    return fig


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Dimensionnement PAC — pipeline degrés-heures", style={"textAlign": "center"}),

        # --- Paramètres climat/bâtiment ---
        html.Div(
            style={"padding": "16px 24px", "border": "1px solid #ddd", "borderRadius": "8px", "marginBottom": "24px"},
            children=[
                html.Label("Limite de chauffage — température extérieure sous laquelle on chauffe (échelle OFEN, °C)"),
                dcc.Slider(
                    id="slider-limite-chauffe",
                    min=9, max=17, step=0.5, value=climat.LIMITE_CHAUFFE_CONVENTIONNELLE_C,
                    marks=MARQUES_LIMITE_CHAUFFE,
                ),
                html.Br(),

                html.Label("Température de consigne intérieure (°C)"),
                dcc.Slider(
                    id="slider-consigne",
                    min=15, max=23, step=0.5, value=20,
                    marks={t: str(t) for t in range(15, 24, 1)},
                ),
                html.Br(),

                html.Label(f"Altitude de la construction (m) — station de référence Pully à {climat.ALTITUDE_STATION_PULLY_M} m"),
                dcc.Slider(
                    id="slider-altitude",
                    min=300, max=2000, step=10, value=climat.ALTITUDE_CHARDONNE_M,
                    marks={a: str(a) for a in range(400, 2001, 200)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),

        # --- Paramètres calibration mazout (version test : total annuel) ---
        html.Div(
            style={"padding": "16px 24px", "border": "1px solid #ddd", "borderRadius": "8px", "marginBottom": "24px"},
            children=[
                html.H4("Calibration — consommation mazout historique (test : total annuel)"),
                html.P(
                    "Sans détail mensuel, l'énergie de chauffage ne peut pas être séparée de l'ECS : "
                    "le coefficient H calibré ci-dessous est donc surestimé. À affiner avec la conso mois par mois.",
                    style={"fontSize": "13px", "color": "#888"},
                ),
                html.Div(
                    style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.Label("Énergie mazout totale annuelle (kWh)"),
                            dcc.Input(
                                id="input-mazout-kwh", type="number",
                                value=18000, min=0, step=100,
                                style={"width": "160px", "marginLeft": "8px"},
                            ),
                        ]),
                        html.Div([
                            html.Label("Rendement chaudière existante"),
                            dcc.Slider(
                                id="slider-rendement",
                                min=0.70, max=0.95, step=0.01, value=pac.RENDEMENT_CHAUDIERE_DEFAUT,
                                marks={r: f"{r:.0%}" for r in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]},
                            ),
                        ], style={"flex": "1", "minWidth": "300px"}),
                    ],
                ),
            ],
        ),

        # --- Plage horaire de fonctionnement (affecte uniquement le Graph 4) ---
        html.Div(
            style={"padding": "16px 24px", "border": "1px solid #ddd", "borderRadius": "8px", "marginBottom": "24px"},
            children=[
                html.H4("Plage horaire de fonctionnement de la PAC (affecte le Graph 4 uniquement)"),
                html.P(
                    "Hypothèse : le bâtiment a assez d'inertie thermique pour lisser le besoin sur 24h "
                    "en ne le délivrant que sur cette plage — le total journalier d'énergie reste "
                    "inchangé, mais la puissance instantanée pendant la plage active augmente.",
                    style={"fontSize": "13px", "color": "#888"},
                ),
                dcc.RangeSlider(
                    id="slider-horaire-chauffe",
                    min=0, max=24, step=1, value=[6, 22],
                    marks={h: f"{h}h" for h in range(0, 25, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                html.Label("Température de départ visée par la PAC (°C) — plancher chauffant ~35°C, radiateurs ~45-55°C"),
                dcc.Slider(
                    id="slider-temp-depart",
                    min=25, max=60, step=1, value=35,
                    marks={t: f"{t}°C" for t in range(25, 61, 5)},
                ),
            ],
        ),

        # --- Rendement PAC (COP table réelle) et comparaison de coûts ---
        html.Div(
            style={"padding": "16px 24px", "border": "1px solid #ddd", "borderRadius": "8px", "marginBottom": "24px"},
            children=[
                html.H4("Rendement PAC (données constructeur réelles) et comparaison de coûts"),
                html.P(
                    f"COP par défaut : {pac.SOURCE_TABLE_COP}. Remplace les valeurs ci-dessous "
                    "par celles de la fiche technique EN14511 de ton modèle visé (points standards "
                    "A-7/A2/A7 à W35/W55) pour un calcul fidèle à TA future PAC.",
                    style={"fontSize": "13px", "color": "#888"},
                ),
                dash_table.DataTable(
                    id="table-cop-constructeur",
                    columns=[
                        {"name": "T° extérieure", "id": "t_ext", "editable": False},
                        {"name": "COP @ départ 35°C", "id": "cop_w35", "type": "numeric", "editable": True},
                        {"name": "COP @ départ 55°C", "id": "cop_w55", "type": "numeric", "editable": True},
                    ],
                    data=[
                        {"t_ext": f"{t:g}°C", "cop_w35": round(float(w35), 2), "cop_w55": round(float(w55), 2)}
                        for t, (w35, w55) in zip(pac.TABLE_COP_TEMP_EXT_C, pac.TABLE_COP_VALEURS)
                    ],
                    style_cell={"textAlign": "center", "padding": "6px", "fontFamily": "sans-serif", "fontSize": "13px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5"},
                    style_table={"width": "480px", "marginBottom": "16px"},
                ),
                html.Div(
                    style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.Label("Prix électricité — heures creuses (CHF/kWh)"),
                            dcc.Slider(
                                id="slider-prix-hc",
                                min=0.10, max=0.40, step=0.01, value=0.22,
                                marks={p: f"{p:.2f}" for p in [0.10, 0.18, 0.26, 0.34, 0.40]},
                            ),
                        ], style={"flex": "1", "minWidth": "280px"}),
                        html.Div([
                            html.Label("Prix électricité — heures pleines (CHF/kWh)"),
                            dcc.Slider(
                                id="slider-prix-hp",
                                min=0.15, max=0.50, step=0.01, value=0.32,
                                marks={p: f"{p:.2f}" for p in [0.15, 0.25, 0.35, 0.45, 0.50]},
                            ),
                        ], style={"flex": "1", "minWidth": "280px"}),
                        html.Div([
                            html.Label("Prix mazout (CHF/L) — marché volatil, ajuste à ta facture récente"),
                            dcc.Slider(
                                id="slider-prix-mazout",
                                min=0.60, max=2.20, step=0.01, value=1.65,
                                marks={p: f"{p:.2f}" for p in [0.60, 0.90, 1.20, 1.50, 1.80, 2.10]},
                            ),
                        ], style={"flex": "1", "minWidth": "280px"}),
                    ],
                ),
                html.Br(),
                html.Label("Heures creuses — clique chaque case pour basculer HC (bleu) / HP (gris)"),
                dcc.Store(id="store-heures-creuses", data=MASQUE_HC_DEFAUT),
                generer_grille_heures(MASQUE_HC_DEFAUT),
            ],
        ),

        titre_avec_info("Graph 0 — Température brute à 2m, station de Pully", "temperature_brute"),
        dcc.Graph(id="graph-temperature-brute"),

        titre_avec_info("Graph 1 — DH à la station de Pully (base 20°C)", "pully"),
        dcc.Graph(id="graph-dh-pully"),

        titre_avec_info("Graph 2 — DH corrigé pour l'altitude de la construction", "construction"),
        dcc.Graph(id="graph-dh-construction"),

        titre_avec_info("Graph 3 — DH à la construction, différentiel à la consigne intérieure", "consigne"),
        dcc.Graph(id="graph-dh-consigne"),

        titre_avec_info("Graph 4 — Puissance thermique requise (kW), calibrée sur le mazout", "puissance"),
        dcc.Graph(id="graph-puissance"),

        titre_avec_info("Graph 5 — Besoin de stockage tampon (dimensionnement ballon)", "tampon"),
        dcc.Graph(id="graph-tampon"),

        titre_avec_info("Graph 6 — COP horaire (modèle Carnot corrigé)", "cop"),
        dcc.Graph(id="graph-cop"),

        titre_avec_info("Graph 7 — Puissance électrique consommée (kW) et coût annuel", "electrique"),
        dcc.Graph(id="graph-electrique"),
    ],
)


@app.callback(
    Output("store-heures-creuses", "data"),
    Input({"type": "bouton-heure", "index": ALL}, "n_clicks"),
    State("store-heures-creuses", "data"),
    prevent_initial_call=True,
)
def basculer_heure(_n_clicks_liste, masque_actuel):
    """Inverse l'état HC/HP de l'heure sur laquelle on vient de cliquer."""
    declencheur = ctx.triggered_id
    if declencheur is None:
        return no_update
    index = declencheur["index"]
    nouveau_masque = list(masque_actuel)
    nouveau_masque[index] = 0 if nouveau_masque[index] else 1
    return nouveau_masque


@app.callback(
    Output({"type": "bouton-heure", "index": ALL}, "style"),
    Input("store-heures-creuses", "data"),
)
def recolorer_grille(masque):
    """Recolore les 24 boutons selon l'état courant du masque."""
    return [style_bouton_heure(actif) for actif in masque]


@app.callback(
    Output("graph-temperature-brute", "figure"),
    Output("graph-dh-pully", "figure"),
    Output("graph-dh-construction", "figure"),
    Output("graph-dh-consigne", "figure"),
    Output("graph-puissance", "figure"),
    Output("graph-tampon", "figure"),
    Output("graph-cop", "figure"),
    Output("graph-electrique", "figure"),
    Input("slider-limite-chauffe", "value"),
    Input("slider-consigne", "value"),
    Input("slider-altitude", "value"),
    Input("input-mazout-kwh", "value"),
    Input("slider-rendement", "value"),
    Input("slider-horaire-chauffe", "value"),
    Input("slider-temp-depart", "value"),
    Input("slider-prix-hc", "value"),
    Input("slider-prix-hp", "value"),
    Input("store-heures-creuses", "data"),
    Input("slider-prix-mazout", "value"),
    Input("table-cop-constructeur", "data"),
)
def mettre_a_jour_graphes(
    limite_chauffe, consigne, altitude, mazout_kwh, rendement, plage_horaire,
    temp_depart, prix_hc, prix_hp, masque_heures_creuses, prix_mazout,
    table_cop_data,
):
    delta_altitude = altitude - climat.ALTITUDE_STATION_PULLY_M
    temp_construction = climat.corriger_altitude(temp_pully, delta_altitude)

    fig0 = figure_temperature_brute(
        temp_pully, limite_chauffe, "Température brute — Pully (2m, non corrigée altitude)"
    )

    # DH horaire (°C), utilisé tel quel pour l'affichage ET la calibration —
    # pas besoin d'agrégation intermédiaire.
    dh_pully = climat.calculer_dh_horaire(
        temp_pully, climat.BASE_DH_CONVENTIONNELLE_C, limite_chauffe
    )
    # Référence 20°C = proxy de la consigne historique (années de conso mazout),
    # réutilisé à la fois pour le graph 2 et pour la calibration de H ci-dessous.
    dh_construction_historique = climat.calculer_dh_horaire(
        temp_construction, climat.BASE_DH_CONVENTIONNELLE_C, limite_chauffe
    )
    # Consigne CIBLE (slider) = ce qu'on vise pour la future PAC.
    dh_construction_cible = climat.calculer_dh_horaire(
        temp_construction, consigne, limite_chauffe
    )

    fig1 = figure_dh_avec_integrale(dh_pully, "DH — Pully", "#1f77b4")
    fig2 = figure_dh_avec_integrale(
        dh_construction_historique, f"DH — construction à {altitude} m (Δ={delta_altitude:+d} m)", "#ff7f0e"
    )
    fig3 = figure_dh_avec_integrale(
        dh_construction_cible, f"DH — construction à {altitude} m, différentiel à {consigne} °C", "#2ca02c"
    )

    # --- Graph 4 : calibration H puis courbe de puissance (kW) ---
    mazout_kwh = mazout_kwh or 0
    try:
        H = pac.calibrer_coefficient_deperdition(
            dh_construction_historique.sum(), mazout_kwh, rendement
        )
        energie_utile_kwh = mazout_kwh * rendement
        # Demande réelle du bâtiment : continue, SANS redistribution (perte physique 24h/24)
        demande_kw = pac.calculer_puissance_horaire_kw(dh_construction_cible, H)

        # Production : redistribuée sur la plage horaire de fonctionnement, PUIS H appliqué
        heure_debut, heure_fin = plage_horaire
        dh_cible_redistribue = climat.redistribuer_dh_horaire(dh_construction_cible, heure_debut, heure_fin)
        puissance_kw = pac.calculer_puissance_horaire_kw(dh_cible_redistribue, H)
        fig4 = figure_puissance_avec_integrale(
            puissance_kw, f"Puissance requise — H={H:.3f} kW/°C", energie_utile_kwh
        )

        # Graph 5 : capacité tampon nécessaire pour combler l'écart production/demande
        besoin_tampon_kwh = pac.calculer_besoin_tampon_kwh(puissance_kw, demande_kw)
        delta_t_ballon = pac.calculer_delta_t_ballon(temp_depart, consigne)
        if delta_t_ballon <= 0:
            fig5 = go.Figure()
            fig5.update_layout(
                title=(
                    f"Température de départ ({temp_depart}°C) trop basse pour la consigne "
                    f"{consigne}°C + marge {pac.MARGE_SECURITE_EMISSION_C}°C — augmente le départ ou baisse la consigne"
                ),
                height=320,
            )
        else:
            volume_l = pac.convertir_kwh_vers_litres_eau(besoin_tampon_kwh.max(), delta_t_ballon)
            fig5 = figure_besoin_tampon(
                besoin_tampon_kwh,
                f"Capacité tampon requise (départ {temp_depart}°C, ΔT={delta_t_ballon:.0f}°C → {volume_l:,.0f} L)".replace(",", "'"),
            )

        # Graph 6 : COP horaire (table constructeur réelle EN14511), masqué hors heures de besoin réel
        try:
            table_valeurs_custom = np.array([
                [float(ligne["cop_w35"]), float(ligne["cop_w55"])] for ligne in table_cop_data
            ])
        except (TypeError, ValueError, KeyError):
            # Cellule vide ou non numérique pendant la frappe : on garde la table par défaut
            # plutôt que de planter — l'utilisateur voit le dernier calcul valide le temps de finir.
            table_valeurs_custom = pac.TABLE_COP_VALEURS
        cop_horaire = pac.calculer_cop_table_reelle(temp_construction, temp_depart, table_valeurs=table_valeurs_custom)
        cop_masque = cop_horaire.where(puissance_kw > 0)
        fig6 = figure_cop(cop_masque, "COP horaire (données réelles)")

        # Graph 7 : puissance électrique, coût annuel PAC (bi-tarif HC/HP), et comparaison au mazout
        prix_mazout = prix_mazout or 0
        puissance_elec_kw = pac.calculer_puissance_electrique_kw(puissance_kw, cop_horaire)
        detail_cout = pac.calculer_cout_annuel_bitarif_masque_chf(
            puissance_elec_kw, masque_heures_creuses, prix_hc, prix_hp
        )
        cout_annuel_pac_chf = detail_cout["cout_total_chf"]
        cout_annuel_mazout_chf = pac.calculer_cout_mazout_chf(mazout_kwh, prix_mazout)
        economie_chf = cout_annuel_mazout_chf - cout_annuel_pac_chf
        fig7 = figure_puissance_electrique(
            puissance_elec_kw,
            (
                f"Puissance électrique — PAC {cout_annuel_pac_chf:,.0f} CHF/an "
                f"(HC {detail_cout['kwh_hc']:,.0f} kWh / HP {detail_cout['kwh_hp']:,.0f} kWh) vs "
                f"mazout {cout_annuel_mazout_chf:,.0f} CHF/an → économie {economie_chf:,.0f} CHF/an"
            ).replace(",", "'"),
        )
    except ValueError:
        fig4 = go.Figure()
        fig4.update_layout(title="Entrez une énergie mazout > 0 pour calibrer H", height=340)
        fig5 = go.Figure()
        fig5.update_layout(title="Entrez une énergie mazout > 0 pour calibrer H", height=320)
        fig6 = go.Figure()
        fig6.update_layout(title="Entrez une énergie mazout > 0 pour calibrer H", height=320)
        fig7 = go.Figure()
        fig7.update_layout(title="Entrez une énergie mazout > 0 pour calibrer H", height=340)

    return fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7


if __name__ == "__main__":
    app.run(debug=True)