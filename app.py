from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import climat

app = Dash(__name__)
server = app.server

# --- Calculs indépendants des sliders : faits une seule fois au démarrage ---
temp_pully = climat.generer_serie_horaire_synthetique()

delta_altitude = climat.ALTITUDE_CHARDONNE_M - climat.ALTITUDE_STATION_PULLY_M
temp_chardonne = climat.corriger_altitude(temp_pully, delta_altitude)

# Repères OFEN pour la limite de chauffage (référencés à une consigne de 20°C)
MARQUES_LIMITE_CHAUFFE = {
    9: "9 (passif)", 12: "12 (SIA std)", 14: "14 (1995-2010)",
    16: "16 (1977-95)", 17: "17 (avant 1977)",
}


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


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Dimensionnement PAC — pipeline degrés-heures", style={"textAlign": "center"}),

        html.Label("Limite de chauffage — température extérieure sous laquelle on chauffe (échelle OFEN, °C)"),
        dcc.Slider(
            id="slider-limite-chauffe",
            min=9, max=17, step=0.5, value=climat.LIMITE_CHAUFFE_CONVENTIONNELLE_C,
            marks=MARQUES_LIMITE_CHAUFFE,
        ),
        html.Br(),

        html.H4("Graph 1 — DH à la station de Pully (série synthétique, base 20°C)"),
        dcc.Graph(id="graph-dh-pully"),

        html.H4(f"Graph 2 — DH corrigé pour l'altitude de Chardonne ({climat.ALTITUDE_CHARDONNE_M} m, Δ={delta_altitude:+d} m)"),
        dcc.Graph(id="graph-dh-chardonne"),

        html.H4("Graph 3 — DH à Chardonne, différentiel à la consigne intérieure"),
        html.Label("Température de consigne intérieure (°C)"),
        dcc.Slider(
            id="slider-consigne",
            min=15, max=23, step=0.5, value=20,
            marks={t: str(t) for t in range(15, 24, 1)},
        ),
        dcc.Graph(id="graph-dh-consigne"),
    ],
)


@app.callback(
    Output("graph-dh-pully", "figure"),
    Output("graph-dh-chardonne", "figure"),
    Output("graph-dh-consigne", "figure"),
    Input("slider-limite-chauffe", "value"),
    Input("slider-consigne", "value"),
)
def mettre_a_jour_graphes(limite_chauffe, consigne):
    dh_pully = climat.calculer_degres_heures(
        temp_pully, climat.BASE_DH_CONVENTIONNELLE_C, limite_chauffe
    )
    dh_chardonne = climat.calculer_degres_heures(
        temp_chardonne, climat.BASE_DH_CONVENTIONNELLE_C, limite_chauffe
    )
    dh_consigne = climat.calculer_degres_heures(
        temp_chardonne, consigne, limite_chauffe
    )

    fig1 = figure_dh_avec_integrale(dh_pully, "DH — Pully", "#1f77b4")
    fig2 = figure_dh_avec_integrale(dh_chardonne, "DH — Chardonne (corrigé altitude)", "#ff7f0e")
    fig3 = figure_dh_avec_integrale(dh_consigne, f"DH — Chardonne, différentiel à {consigne} °C", "#2ca02c")
    return fig1, fig2, fig3


if __name__ == "__main__":
    app.run(debug=True)