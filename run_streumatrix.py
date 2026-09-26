"""Streudiagramm-Matrix: jede Messgroesse gegen jede andere.

Feld (Zeile i, Spalte j) zeigt Groesse j auf der x-Achse gegen Groesse i auf
der y-Achse, in Originaleinheiten. Auf der Diagonalen steht jede Groesse gegen
sich selbst - die Punkte liegen dort immer auf einer Geraden; ihre Verteilung
entlang der Geraden zeigt, wo die Werte der Gruppen liegen. Die Zahl oben
links in jedem Feld ist die Korrelation r (Pearson, gezeichnete Punkte).

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
from sklearn.decomposition import PCA

from hda import cluster as hc
from hda.cli import ausgabeordner, basis_parser, blatt_wert
from hda.data_io import lade_daten, skaliere
from hda.plotstyle import (rahmen, scatter_nach_gruppe, set_style, speichere,
                           zeige_oder_schliesse)


def streumatrix(X, labels, label_name, t, quelle="", neutral=None):
    """n x n Felder; Achsenbeschriftung nur am linken und unteren Rand."""
    merkmale = list(X.columns)
    n = len(merkmale)
    feld = 1.45 if n <= 10 else 1.2
    fig, achsen = plt.subplots(n, n, figsize=(feld * n + 1.5, feld * n + 1.2),
                               squeeze=False)
    werte = X.to_numpy(dtype=float)
    r = np.corrcoef(werte, rowvar=False)
    groesse = 7 if len(X) <= 1000 else 3
    alle = None if labels is None else sorted(set(map(str, labels)))

    for i in range(n):          # Zeile -> y-Achse
        for j in range(n):      # Spalte -> x-Achse
            ax = achsen[i, j]
            scatter_nach_gruppe(ax, werte[:, j], werte[:, i], labels, t,
                                groesse=groesse, alpha=0.7, neutral=neutral,
                                alle=alle)
            rahmen(ax, t, ticks_innen=False)
            for seite in ax.spines.values():
                seite.set_linewidth(0.6)
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

            # Nur die Randfelder tragen Zahlen und Namen, sonst verschwinden
            # die Punkte hinter der Beschriftung.
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

    fig.suptitle("Jede Messgroesse gegen jede", x=0.01, ha="left", fontsize=15,
                 fontweight="semibold", color=t["text"], y=1.0)
    fig.text(0.01, 0.985, "Zeile = y-Achse, Spalte = x-Achse, Originaleinheiten · "
             "Diagonale: Groesse gegen sich selbst · r = Korrelation "
             "(fett ab |r| ≥ 0.8)", ha="left", va="top", fontsize=10,
             color=t["text2"])
    if labels is not None:
        griffe, namen = [], []
        for ax in achsen.flat:
            for g, name in zip(*ax.get_legend_handles_labels()):
                if name not in namen:
                    griffe.append(g)
                    namen.append(name)
        fig.legend(griffe, namen, title=label_name, loc="upper right",
                   bbox_to_anchor=(0.995, 0.995), ncol=min(len(namen), 5),
                   frameon=False, fontsize=9, title_fontsize=9, markerscale=2.2)
    if quelle:
        fig.text(0.01, 0.0, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout(rect=(0, 0.01, 1, 0.965), h_pad=0.25, w_pad=0.25)
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
    p.add_argument("--max-punkte", type=int, default=3000,
                   help="bei mehr Zeilen eine Zufallsauswahl zeichnen (Lesbarkeit, Tempo)")
    args = p.parse_args()

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
                              args.gruppen, seed=args.seed)
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
    main()
