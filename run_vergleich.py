"""Finden PCA, UMAP und t-SNE dieselben Gruppen?

Jedes Verfahren sieht die Daten anders: PCA linear, UMAP und t-SNE ueber
Nachbarschaften. In jeder der drei Darstellungen werden unabhaengig Gruppen
gesucht und danach verglichen. Gruppen, die alle drei finden, stecken in den
Daten; Gruppen, die nur ein Verfahren zeigt, sind vermutlich ein Artefakt
genau dieses Verfahrens.

Erzeugt im Ausgabeordner:
  vergleich_ueberschneidung.png  Kreuztabellen je Verfahrenspaar: wer landet wo
  vergleich_karten.png           3x3-Raster: jede Zeile ein Verfahren, das die
                                 Gruppen vorgibt (links), rechts dieselben
                                 Farben in den Karten der anderen Verfahren
  vergleich_gruppen.xlsx         Zuordnung jeder Zeile unter jedem Verfahren

Aufruf:  python run_vergleich.py [--cluster kmeans] [--gruppen 3] ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from hda import cluster as hc
from hda.cli import ausgabeordner, basis_parser, blatt_wert
from hda.data_io import lade_daten, skaliere
from hda.embedding import leere_achsen, tsne
from hda.plotstyle import (rahmen, scatter_nach_gruppe, set_style, speichere,
                           zeige_oder_schliesse)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def ueberschneidung_plot(namen, nummern, paare, t, quelle=""):
    """Je Verfahrenspaar eine Kreuztabelle: Zeilen = Gruppen von A, Spalten = B.

    Nach dem Angleichen der Nummern steht Uebereinstimmung auf der Diagonalen.
    Farbe = Anteil der Zeile (wie viel von Gruppe X bei B in Gruppe Y landet),
    Zahl = Anzahl Punkte.
    """
    cmap = LinearSegmentedColormap.from_list("hda_seq", [t["surface"], t["series"][0]])
    fig, achsen = plt.subplots(1, len(paare), figsize=(4.9 * len(paare), 5.0),
                               squeeze=False)
    for ax, (a, b) in zip(achsen.flat, paare):
        tab = pd.crosstab(pd.Series(namen[a], name=a),
                          pd.Series(namen[b], name=b))
        anteil = tab.div(tab.sum(axis=1), axis=0)
        ax.imshow(anteil.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                n = tab.iat[i, j]
                if n == 0:
                    continue
                ax.text(j, i, "%d" % n, ha="center", va="center", fontsize=9,
                        color="#ffffff" if anteil.iat[i, j] > 0.6 else t["text"])
        ax.set_xticks(range(tab.shape[1]), [s.replace("Gruppe ", "G") for s in tab.columns])
        ax.set_yticks(range(tab.shape[0]), [s.replace("Gruppe ", "G") for s in tab.index])
        ax.set_xlabel("Gruppen bei " + b)
        ax.set_ylabel("Gruppen bei " + a)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        rahmen(ax, t, ticks_innen=False)
        ax.set_xticks(np.arange(-0.5, tab.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, tab.shape[0], 1), minor=True)
        ax.grid(which="minor", color=t["surface"], linewidth=2)
        ax.tick_params(which="minor", length=0)
        ari = hc.uebereinstimmung(nummern[a], nummern[b])
        ax.set_title("%s ↔ %s · ARI %.2f" % (a, b, ari), loc="left", fontsize=11,
                     color=t["text"], pad=34)

    fig.suptitle("Finden die Verfahren dieselben Gruppen?", x=0.01, ha="left",
                 fontsize=14, fontweight="semibold", color=t["text"], y=1.04)
    fig.text(0.01, 0.975, "Uebereinstimmung steht auf der Diagonalen · Farbe = Anteil "
             "der Zeilengruppe · ARI 1 = identische Aufteilung, 0 = Zufall",
             ha="left", fontsize=9.5, color=t["text2"])
    if quelle:
        fig.text(0.01, -0.03, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


ANDERS = "dort anders zugeordnet"


def karten_plot(darstellungen, namen, nummern, t, quelle=""):
    """3x3-Raster: Zeile = Verfahren, das die Gruppen vorgibt.

    Links steht die Karte des vorgebenden Verfahrens mit seinen eigenen
    Gruppen. Rechts daneben dieselben Punkte in den Karten der anderen
    Verfahren - eingefaerbt weiter nach den Gruppen der Zeile. Bleiben die
    Farben dort beisammen, sieht das andere Verfahren dieselbe Struktur.
    Ein Ring markiert Punkte, die das gezeigte Verfahren selbst einer
    anderen Gruppe zuordnen wuerde.
    """
    verfahren = list(darstellungen)
    n = len(verfahren)
    # Feste Farbplaetze ueber alle Felder: Gruppe 3 ist ueberall gleich gefaerbt.
    alle = sorted(set(np.concatenate([namen[v] for v in verfahren]).tolist()))
    fig, achsen = plt.subplots(n, n, figsize=(4.3 * n, 4.1 * n), squeeze=False)

    griffe = {}
    for zeile, vorgabe in enumerate(verfahren):
        reihe = [vorgabe] + [v for v in verfahren if v != vorgabe]
        for spalte, karte in enumerate(reihe):
            ax = achsen[zeile, spalte]
            _, Y, achsen_namen, _ = darstellungen[karte]
            scatter_nach_gruppe(ax, Y[:, 0], Y[:, 1], namen[vorgabe], t,
                                groesse=14, alpha=0.85, neutral=hc.RAUSCHEN,
                                alle=alle)
            if spalte == 0:
                titel = "%s gibt die Gruppen vor" % vorgabe
            else:
                a, b = nummern[vorgabe], nummern[karte]
                beide = (a >= 0) & (b >= 0)
                abweichend = beide & (a != b)
                gleich = 100.0 * (beide & (a == b)).sum() / max(beide.sum(), 1)
                if abweichend.any():
                    ax.scatter(Y[abweichend, 0], Y[abweichend, 1], s=42,
                               facecolors="none", edgecolors=t["text"],
                               linewidths=0.8, zorder=4,
                               label=ANDERS)
                titel = "%s-Karte · %.0f %% gleich zugeordnet" % (karte, gleich)
            leere_achsen(ax, t, *achsen_namen)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(titel, loc="left", fontsize=10.5, pad=7,
                         color=t["text"], fontweight="semibold" if spalte == 0
                         else "normal")
            for g, name in zip(*ax.get_legend_handles_labels()):
                griffe.setdefault(name, g)
        # Kennzeichnung der Zeile links neben dem ersten Feld
        achsen[zeile, 0].annotate(
            "Gruppen aus %s" % vorgabe, xy=(0, 0.5), xytext=(-30, 0),
            xycoords="axes fraction", textcoords="offset points", rotation=90,
            ha="right", va="center", fontsize=11.5, fontweight="semibold",
            color=t["text"])

    fig.suptitle("Finden die anderen Verfahren dieselben Gruppen?", x=0.01,
                 ha="left", fontsize=15, fontweight="semibold", color=t["text"],
                 y=1.0)
    fig.text(0.01, 0.967, "Jede Zeile: links gibt ein Verfahren die Gruppen vor, "
             "rechts dieselben Farben in den Karten der anderen · Ring = das "
             "gezeigte Verfahren ordnet den Punkt einer anderen Gruppe zu",
             ha="left", fontsize=10, color=t["text2"])
    reihenfolge = [g for g in alle if g in griffe and g != hc.RAUSCHEN]
    reihenfolge += [g for g in griffe if g not in reihenfolge]
    fig.legend([griffe[g] for g in reihenfolge],
               reihenfolge,
               loc="lower center", ncol=min(len(reihenfolge), 7), frameon=False,
               fontsize=10, markerscale=1.8, bbox_to_anchor=(0.5, -0.02))
    if quelle:
        fig.text(0.01, -0.03, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left")
    fig.tight_layout(rect=(0.02, 0.015, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
def main() -> None:
    p = basis_parser("Gruppen unter PCA, UMAP und t-SNE finden und vergleichen.")
    p.add_argument("--cluster", default="kmeans", choices=["hdbscan", "kmeans", "gmm"],
                   help="Verfahren zur Gruppensuche, fuer alle Darstellungen gleich")
    p.add_argument("--gruppen", type=int, default=None,
                   help="feste Gruppenzahl fuer kmeans/gmm; ohne Angabe waehlt "
                        "jedes Verfahren selbst - der strengere Test")
    p.add_argument("--min-gruppe", type=int, default=None,
                   help="hdbscan: kleinste Gruppe (Standard: max(5, n/50))")
    p.add_argument("--cluster-varianz", type=float, default=0.9,
                   help="PCA: auf so vielen PCs clustern, wie fuer diesen "
                        "Varianzanteil noetig sind")
    p.add_argument("--nachbarn", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--min-dist", type=float, default=0.0,
                   help="UMAP min_dist; 0.0 packt Gruppen dicht - gut fuers Clustern")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE Perplexity")
    args = p.parse_args()

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    X, _ = skaliere(ds.X, args.skalierung)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    # ---- Drei Darstellungen derselben Daten ------------------------------ #
    pca = PCA(random_state=args.seed)
    scores = pca.fit_transform(X)
    varianz = pca.explained_variance_ratio_
    d = hc.dimensionen_fuer(varianz, args.cluster_varianz)
    darstellungen = {"PCA": (scores[:, :d], scores[:, :2], ("PC1", "PC2"),
                             "PCA · PC1-PC%d (%.0f %% Varianz)"
                             % (d, varianz[:d].sum() * 100))}
    try:
        import umap
        nachbarn = max(2, min(args.nachbarn, len(X) - 1))
        Y = umap.UMAP(n_components=2, n_neighbors=nachbarn, min_dist=args.min_dist,
                      random_state=args.seed).fit_transform(X)
        darstellungen["UMAP"] = (Y, Y, ("UMAP 1", "UMAP 2"),
                                 "UMAP · n_neighbors=%d" % nachbarn)
    except ImportError:
        print("  Hinweis : umap-learn fehlt - Vergleich nur PCA gegen t-SNE")
    perplexity = min(args.perplexity, max(5.0, (len(X) - 1) / 3.0))
    Y = tsne(X, perplexity, args.seed)
    darstellungen["t-SNE"] = (Y, Y, ("t-SNE 1", "t-SNE 2"),
                              "t-SNE · Perplexity %.0f" % perplexity)

    # ---- In jeder Darstellung unabhaengig Gruppen suchen ----------------- #
    print("\nGruppen: %s in jeder Darstellung unabhaengig" % args.cluster.upper())
    nummern = {}
    for name, (Z, *_rest) in darstellungen.items():
        cl = hc.finde_gruppen_in(Z, args.cluster, args.gruppen, args.min_gruppe,
                                 args.seed)
        nummern[name] = cl.nummern
        ohne = (cl.nummern < 0).sum()
        print("  %-6s auf %d Dim.: %d Gruppen%s"
              % (name, Z.shape[1], len(cl.gruppen),
                 ", %d Punkte ohne Gruppe" % ohne if ohne else ""))

    # Gleiche Nummer = gleiche Gruppe: alle an PCA ausrichten.
    ref = "PCA"
    for name in nummern:
        if name != ref:
            nummern[name] = hc.angleichen(nummern[ref], nummern[name])
    breite = 2 if max(max(n.max(), 0) for n in nummern.values()) >= 10 else 1
    namen = {name: hc.benenne(n, breite) for name, n in nummern.items()}

    verfahren = list(nummern)
    paare = [(a, b) for i, a in enumerate(verfahren) for b in verfahren[i + 1:]]
    ari = pd.DataFrame(np.eye(len(verfahren)), index=verfahren, columns=verfahren)
    print("\n  Uebereinstimmung (ARI, 1 = identisch, 0 = Zufall):")
    for a, b in paare:
        ari.loc[a, b] = ari.loc[b, a] = hc.uebereinstimmung(nummern[a], nummern[b])
        print("    %-6s <-> %-6s  %.2f" % (a, b, ari.loc[a, b]))

    alle = np.column_stack([nummern[v] for v in verfahren])
    einig = (alle[:, :1] == alle).all(axis=1) & (alle[:, 0] >= 0)
    print("\n  Alle Verfahren einig bei %d von %d Punkten (%.0f %%)"
          % (einig.sum(), len(einig), einig.mean() * 100))
    for g in sorted(set(nummern[ref].tolist()) - {-1}):
        m = nummern[ref] == g
        print("    %s (n=%d): %.0f %% einig"
              % (hc.benenne([g], breite)[0], m.sum(), einig[m].mean() * 100))
    if ari.where(~np.eye(len(verfahren), dtype=bool)).min().min() < 0.5:
        print("\n  Achtung: mindestens ein Paar liegt unter ARI 0.5 - die Verfahren "
              "sehen verschiedene Strukturen. Nur Gruppen trauen, die ueberall "
              "auftauchen.")

    # ---- Plots ----------------------------------------------------------- #
    print("\nPlots:")
    fig = ueberschneidung_plot(namen, nummern, paare, t, quelle)
    speichere(fig, out / "vergleich_ueberschneidung.png")
    zeige_oder_schliesse(fig, args.zeigen)

    fig = karten_plot(darstellungen, namen, nummern, t, quelle)
    speichere(fig, out / "vergleich_karten.png")
    zeige_oder_schliesse(fig, args.zeigen)

    # ---- Excel ----------------------------------------------------------- #
    zuordnung = pd.concat([ds.zeilen, ds.meta.reset_index(drop=True)], axis=1)
    if ds.labels is not None:
        zuordnung[ds.label_name] = ds.labels
    for v in verfahren:
        zuordnung["Gruppe_" + v] = namen[v]
    zuordnung["alle_einig"] = np.where(einig, "ja", "nein")
    zuordnung = pd.concat([zuordnung, ds.roh[ds.merkmale].reset_index(drop=True)],
                          axis=1)

    ziel = out / "vergleich_gruppen.xlsx"
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        ari.to_excel(writer, sheet_name="Uebereinstimmung_ARI")
        zuordnung.to_excel(writer, sheet_name="Zuordnung", index=False)
        zuordnung[~einig].to_excel(writer, sheet_name="Uneinig", index=False)
        zeile = 0
        for a, b in paare:
            tab = pd.crosstab(pd.Series(namen[a], name=a), pd.Series(namen[b], name=b))
            pd.DataFrame([["%s gegen %s  (ARI = %.2f)" % (a, b, ari.loc[a, b])]]).to_excel(
                writer, sheet_name="Kreuztabellen", startrow=zeile,
                index=False, header=False)
            tab.to_excel(writer, sheet_name="Kreuztabellen", startrow=zeile + 1)
            zeile += len(tab) + 4
    print("  gespeichert: %s" % ziel)


if __name__ == "__main__":
    main()
