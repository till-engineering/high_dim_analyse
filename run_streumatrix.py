"""Streudiagramm-Matrix: jede Messgroesse gegen jede andere.

Gezeichnet wird das untere Dreieck samt Diagonale - die obere Haelfte waere
nur gespiegelt. Feld (Zeile i, Spalte j) zeigt Groesse j auf der x-Achse gegen
Groesse i auf der y-Achse, in Originaleinheiten. Auf der Diagonalen steht jede Groesse gegen
sich selbst (gestrichelter Rahmen) - die Punkte liegen dort immer auf einer Geraden; ihre Verteilung
entlang der Geraden zeigt, wo die Werte der Gruppen liegen. Die Zahl oben
links in jedem Feld ist die Korrelation r (Pearson, gezeichnete Punkte);
der Rahmen jedes Feldes zeigt |r| in fuenf Stufen - Farbe (grau -> dunkelrot)
und Strichstaerke steigen gemeinsam. Der Betrag, weil ein stark fallender
Zusammenhang genauso eng ist wie ein stark steigender; das Vorzeichen steht in
der Zahl.

Eingefaerbt wird nach der Label-Spalte oder - mit --cluster - nach den Gruppen,
die run_pca.py auf den Hauptkomponenten findet.

Erzeugt im Ausgabeordner:
  streumatrix.png

Aufruf:  python run_streumatrix.py [--cluster kmeans] [--spalten A B C ...]
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA

from hda import cluster as hc
from hda.cli import ausgabeordner, basis_parser, blatt_wert, lies_argumente, starte
from hda.data_io import lade_daten, skaliere
from hda.plotstyle import (rahmen, scatter_nach_gruppe, set_style, speichere,
                           zeige_oder_schliesse)


# Rahmen nach |r|: fuenf Stufen statt eines stufenlosen Verlaufs - duenne
# Linien in fast gleichem Rot kann niemand auseinanderhalten, fuenf klar
# getrennte Stufen schon. Eine Farbe (rot), deren Helligkeit gleichmaessig
# mit |r| wandert, dazu waechst die Strichstaerke: die Stufe ist so auch
# ohne Farbsehen und im Schwarzweissdruck lesbar. Schwache Zusammenhaenge
# sind grau und treten zurueck, starke springen ins Auge.
R_GRENZEN = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
R_STUFEN = {
    # hell: helles Grau -> dunkles Rot; dunkel: gespiegelt, stark = hell
    "#fcfcfb": ["#cfcec9", "#eb8a89", "#e0504f", "#b52a2a", "#6e1414"],
    "#1a1a19": ["#454440", "#9c2a2a", "#d03b3b", "#ec7372", "#ffb8b7"],
}
R_STRICH = [0.6, 1.0, 1.4, 1.9, 2.5]
NAME_KLEIN = 4.6          # Achsennamen in jedem Feld, pt


def r_stufe(wert: float) -> int:
    """|r| -> Stufe 0..4."""
    if np.isnan(wert):
        return 0
    return int(np.clip(np.searchsorted(R_GRENZEN, abs(wert), side="right") - 1, 0, 4))


def streumatrix(X, labels, label_name, t, quelle="", neutral=None):
    """Unteres Dreieck samt Diagonale; die obere Haelfte bleibt leer.

    Feld (i, j) und (j, i) zeigen dasselbe Paar nur gespiegelt - eine Haelfte
    genuegt. Zahlen nur am linken und unteren Rand, Namen in jedem Feld.
    """
    merkmale = list(X.columns)
    n = len(merkmale)
    feld = 1.45 if n <= 10 else 1.2
    fig, achsen = plt.subplots(n, n, figsize=(feld * n + 1.5, feld * n + 2.0),
                               squeeze=False)
    werte = X.to_numpy(dtype=float)
    r = np.atleast_2d(np.corrcoef(werte, rowvar=False))
    groesse = 7 if len(X) <= 1000 else 3
    alle = None if labels is None else sorted(set(map(str, labels)))
    stufen_farben = R_STUFEN.get(t["surface"], R_STUFEN["#fcfcfb"])
    klein = dict(fontsize=NAME_KLEIN, color=t["text2"], zorder=6,
                 bbox=dict(boxstyle="round,pad=0.1", linewidth=0,
                           facecolor=t["surface"], alpha=0.75))

    for i in range(n):          # Zeile -> y-Achse
        for j in range(n):      # Spalte -> x-Achse
            ax = achsen[i, j]
            if j > i:
                ax.axis("off")
                continue
            scatter_nach_gruppe(ax, werte[:, j], werte[:, i], labels, t,
                                groesse=groesse, alpha=0.7, neutral=neutral,
                                alle=alle)
            rahmen(ax, t, ticks_innen=False)
            stufe = r_stufe(r[i, j])
            for seite in ax.spines.values():
                seite.set_linewidth(R_STRICH[stufe])
                seite.set_color(stufen_farben[stufe])
                if i == j:
                    seite.set_linestyle((0, (4, 2)))   # Diagonale gestrichelt
            ax.tick_params(labelsize=6, length=2, pad=1)
            if i == j:
                ax.set_facecolor(t["grid"] + "40")   # Diagonale leicht absetzen
            elif not np.isnan(r[i, j]):
                ax.text(0.04, 0.95, "r=%+.2f" % r[i, j], transform=ax.transAxes,
                        ha="left", va="top", fontsize=6.5,
                        color=t["text"] if abs(r[i, j]) >= 0.5 else t["muted"],
                        fontweight="semibold" if abs(r[i, j]) >= 0.8 else "normal",
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15", linewidth=0,
                                            facecolor=t["surface"], alpha=0.8))

            # Achsennamen in jedem Feld - innen am Rand, damit das Layout
            # nicht auseinanderzieht.
            ax.text(0.5, 0.015, merkmale[j], transform=ax.transAxes,
                    ha="center", va="bottom", **klein)
            ax.text(0.015, 0.5, merkmale[i], transform=ax.transAxes, rotation=90,
                    ha="left", va="center", **klein)

            # Zahlen und grosse Namen nur an den Randfeldern.
            if i == n - 1:
                ax.set_xlabel(merkmale[j], fontsize=7.5, rotation=35, ha="right",
                              rotation_mode="anchor", labelpad=2)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(merkmale[i], fontsize=7.5, rotation=0, ha="right",
                              va="center", labelpad=4)
            else:
                ax.set_yticklabels([])
            ax.locator_params(nbins=3)

    # Kopfzeile in festen Zoll-Abstaenden statt Bildanteilen - sonst rutschen
    # Titel, Farbleiste und Legende bei kleinen Matrizen ineinander.
    breite, hoehe = fig.get_size_inches()
    oben = lambda zoll: 1.0 - zoll / hoehe            # Zoll von oben -> Anteil
    links = lambda zoll: zoll / breite

    fig.text(links(0.12), oben(0.10), "Jede Messgroesse gegen jede", ha="left",
             va="top", fontsize=15, fontweight="semibold", color=t["text"])
    fig.text(links(0.12), oben(0.42), "Zeile = y-Achse, Spalte = x-Achse, "
             "Originaleinheiten · Diagonale: Groesse gegen sich selbst · "
             "r = Korrelation (fett ab |r| ≥ 0.8)", ha="left", va="top",
             fontsize=10, color=t["text2"])

    # Farbleiste links unter dem Untertitel, Legende rechts daneben
    leiste_breite = 2.6
    leiste_ax = fig.add_axes([links(0.12), oben(0.98), links(leiste_breite),
                              0.09 / hoehe])
    leiste = fig.colorbar(ScalarMappable(BoundaryNorm(R_GRENZEN, 5),
                                         ListedColormap(stufen_farben)),
                          cax=leiste_ax, orientation="horizontal",
                          ticks=R_GRENZEN, spacing="uniform")
    leiste.outline.set_visible(False)
    # 2px-Fuge zwischen den Stufen, damit sie als getrennte Klassen lesen
    for grenze in R_GRENZEN[1:-1]:
        leiste.ax.axvline(grenze, color=t["surface"], linewidth=2)
    leiste.ax.tick_params(labelsize=7.5, length=2, color=t["text2"],
                          labelcolor=t["text2"])
    leiste.ax.set_title("Rahmen = |r|  (Farbe und Strichstaerke)", fontsize=8.5,
                        color=t["text2"], pad=3, loc="left")

    if labels is not None:
        griffe, namen = [], []
        for ax in achsen.flat:
            for g, name in zip(*ax.get_legend_handles_labels()):
                if name not in namen:
                    griffe.append(g)
                    namen.append(name)
        fig.legend(griffe, namen, title=label_name, loc="upper left",
                   bbox_to_anchor=(links(0.12 + leiste_breite + 0.6), oben(0.66)),
                   ncol=min(len(namen), 6), frameon=False, fontsize=9,
                   title_fontsize=9, markerscale=2.2, alignment="left")
    if quelle:
        fig.text(0.01, 0.0, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout(rect=(0, 0.01, 1, oben(1.45)), h_pad=0.25, w_pad=0.25)
    return fig


def main() -> None:
    p = basis_parser("Streudiagramm-Matrix: jede Messgroesse gegen jede andere.")
    p.add_argument("--spalten", nargs="+", default=None,
                   help="nur diese Messgroessen zeigen (Standard: alle)")
    p.add_argument("--cluster", default="aus",
                   choices=["aus", "hdbscan", "kmeans", "gmm"],
                   help="nach gefundenen Gruppen einfaerben statt nach der Label-Spalte")
    p.add_argument("--gruppen", type=int, default=None,
                   help="feste Gruppenzahl fuer kmeans/gmm (Standard: automatisch)")
    p.add_argument("--min-gruppe", type=int, default=None,
                   help="hdbscan: kleinste Gruppe (Standard: max(5, n/50))")
    p.add_argument("--cluster-varianz", type=float, default=0.9,
                   help="auf so vielen PCs clustern, wie fuer diesen "
                        "Varianzanteil noetig sind")
    p.add_argument("--max-punkte", type=int, default=3000,
                   help="bei mehr Zeilen eine Zufallsauswahl zeichnen (Lesbarkeit, Tempo)")
    args = lies_argumente(p, "streumatrix")

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    X = ds.X
    if args.spalten:
        fehlt = [s for s in args.spalten if s not in X.columns]
        if fehlt:
            sys.exit("Unbekannte Spalten: %s\nVorhanden: %s"
                     % (", ".join(fehlt), ", ".join(X.columns)))
        X = X[args.spalten]

    labels, label_name, neutral = ds.labels, ds.label_name, None
    if args.cluster != "aus":
        # Gruppen wie in run_pca.py: auf den Hauptkomponenten ALLER Messgroessen,
        # auch wenn nur ein Teil davon gezeichnet wird.
        Z, _ = skaliere(ds.X, args.skalierung)
        pca = PCA(random_state=args.seed)
        scores = pca.fit_transform(Z)
        cl = hc.finde_gruppen(scores, pca.explained_variance_ratio_, args.cluster,
                              args.gruppen, args.min_gruppe, args.cluster_varianz,
                              args.seed)
        labels, label_name, neutral = cl.namen, "Gruppe", hc.RAUSCHEN
        print("\nGruppen: %s, %d Gruppen" % (args.cluster.upper(), len(cl.gruppen)))

    if labels is not None:
        labels = np.asarray(labels)
    if len(X) > args.max_punkte:
        auswahl = np.random.default_rng(args.seed).choice(len(X), args.max_punkte,
                                                          replace=False)
        print("  Hinweis : %d von %d Zeilen gezeichnet (--max-punkte)"
              % (args.max_punkte, len(X)))
        X = X.iloc[np.sort(auswahl)]
        labels = None if labels is None else labels[np.sort(auswahl)]

    n = X.shape[1]
    print("\nStreumatrix: %d Messgroessen -> %d Felder" % (n, n * n))
    if n > 20:
        print("  Hinweis : bei %d Groessen wird es eng - mit --spalten auswaehlen" % n)

    fig = streumatrix(X, labels, label_name, t, quelle, neutral)
    speichere(fig, out / "streumatrix.png")
    zeige_oder_schliesse(fig, args.zeigen)


if __name__ == "__main__":
    starte(main)
