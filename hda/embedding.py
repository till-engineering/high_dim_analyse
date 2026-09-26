"""Gemeinsame Darstellung fuer nichtlineare Einbettungen (UMAP, t-SNE).

Bei UMAP und t-SNE haben die Achsen keine physikalische Bedeutung: nur die
Nachbarschaften zaehlen, nicht Richtung, Massstab oder Abstand zwischen weit
entfernten Gruppen. Deshalb werden die Achsen hier bewusst ohne Zahlen
gezeichnet - eine Achsenskala wuerde eine Genauigkeit vortaeuschen, die die
Methode nicht hat.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotstyle import kopf, rahmen, scatter_nach_gruppe


def leere_achsen(ax, t: dict, x_label: str, y_label: str) -> None:
    """Achsen ohne Zahlen - Position und Abstand tragen hier keine Einheit."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(x_label, fontsize=9.5)
    ax.set_ylabel(y_label, fontsize=9.5)
    ax.set_facecolor(t["surface"])
    # Rahmen ja, Teilstriche nein: eine Skala gibt es hier nicht.
    rahmen(ax, t, ticks_innen=False)


def einbettung_plot(Y, labels, t, titel: str, unter: str, label_name=None,
                    quelle: str = "", achsen_praefix: str = "Dimension"):
    """Streudiagramm einer 2D-Einbettung."""
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    gruppen = scatter_nach_gruppe(ax, Y[:, 0], Y[:, 1], labels, t,
                                  groesse=46, alpha=0.85)
    leere_achsen(ax, t, achsen_praefix + " 1", achsen_praefix + " 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    kopf(ax, titel, unter, t)

    if gruppen:
        ax.legend(title=label_name, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  title_fontsize=9, labelspacing=0.7, handletextpad=0.5)
    if quelle:
        fig.text(0.01, -0.01, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()
    return fig


def raster_plot(einbettungen, labels, t, titel: str, unter: str,
                label_name=None, quelle: str = "", spalten: int = 3):
    """Kleine Vielfache: dieselbe Datenmenge unter verschiedenen Parametern.

    ``einbettungen`` ist eine Liste von ``(beschriftung, Y)``.
    """
    n = len(einbettungen)
    spalten = 2 if n == 4 else min(spalten, n)  # 2x2 statt 3+1
    zeilen = int(np.ceil(n / spalten))
    fig, achsen = plt.subplots(zeilen, spalten,
                               figsize=(4.1 * spalten, 4.0 * zeilen),
                               squeeze=False)
    gruppen = []
    for idx, ax in enumerate(achsen.flat):
        if idx >= n:
            ax.axis("off")
            continue
        beschriftung, Y = einbettungen[idx]
        g = scatter_nach_gruppe(ax, Y[:, 0], Y[:, 1], labels, t,
                                groesse=17, alpha=0.8)
        gruppen = g or gruppen
        leere_achsen(ax, t, "", "")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(beschriftung, loc="left", fontsize=10,
                     color=t["text"], pad=8)

    fig.suptitle(titel, x=0.012, ha="left", fontsize=14, fontweight="semibold",
                 color=t["text"], y=0.995)
    fig.text(0.012, 0.955, unter, ha="left", fontsize=9.5, color=t["text2"])
    if gruppen:
        griffe, namen = achsen.flat[0].get_legend_handles_labels()
        fig.legend(griffe, namen, title=label_name, loc="lower center",
                   ncol=min(len(namen), 6), frameon=False, fontsize=9,
                   title_fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if quelle:
        fig.text(0.012, -0.055, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


def tsne(X, perplexity: float, seed: int, iterationen: int = 1000,
         metrik: str = "euclidean"):
    """Ein t-SNE-Lauf. Start ueber PCA, damit das Ergebnis reproduzierbar bleibt."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", max_iter=iterationen, metric=metrik,
                random_state=seed, verbose=0).fit_transform(X)


def speichere_koordinaten(Y, ds, pfad: Path, spaltennamen=("Dim_1", "Dim_2")) -> Path:
    """Schreibt die 2D-Koordinaten mit Beschriftung als Excel weg."""
    df = pd.DataFrame(Y, columns=list(spaltennamen))
    if ds.labels is not None:
        df.insert(0, ds.label_name, ds.labels)
    if not ds.meta.empty:
        for spalte in reversed(ds.meta.columns):
            df.insert(0, spalte, ds.meta[spalte])
    pfad.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(pfad, index=False)
    return pfad


def guete(X, Y, n_nachbarn: int = 12) -> float:
    """Trustworthiness: Anteil erhaltener Nachbarschaften (1.0 = perfekt).

    Grob: ab etwa 0.9 sind die sichtbaren Gruppen belastbar, darunter sollte
    man den Plot vorsichtig lesen.
    """
    from sklearn.manifold import trustworthiness
    n_nachbarn = min(n_nachbarn, (len(X) - 1) // 2)
    return float(trustworthiness(X, Y, n_neighbors=max(2, n_nachbarn)))
