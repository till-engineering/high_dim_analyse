"""t-SNE-Einbettung hochdimensionaler Daten in zwei Dimensionen.

t-SNE bildet vor allem die lokale Nachbarschaft ab. Beim Lesen der Plots gilt:

* Die Groesse der Zwischenraeume zwischen Clustern bedeutet nichts.
* Die Clustergroesse bedeutet nichts - t-SNE blaeht duenne Bereiche auf.
* Nur "diese Punkte liegen beieinander" ist eine belastbare Aussage.
* Das Ergebnis haengt stark von der Perplexity ab, deshalb lohnt --sweep.

Erzeugt im Ausgabeordner:
  tsne_einbettung.png    die Einbettung
  tsne_perplexity.png    (mit --sweep) dieselben Daten unter mehreren Perplexities
  tsne_koordinaten.xlsx  die 2D-Koordinaten je Datenpunkt
  mit --cluster zusaetzlich: tsne_gruppen.png, tsne_gruppenprofil.png,
                             tsne_gruppen.xlsx (Gruppen auf der Karte gesucht)

Aufruf:  python run_tsne.py [--perplexity 30] [--sweep]
"""

from __future__ import annotations

import pandas as pd

from hda import cluster as hc
from hda import gruppen as hg
from hda.cli import ausgabeordner, basis_parser, blatt_wert, starte
from hda.data_io import lade_daten, skaliere
from hda.embedding import (TSNE_STARTS, einbettung_plot, guete, raster_plot,
                           speichere_koordinaten, tsne)
from hda.plotstyle import set_style, speichere, zeige_oder_schliesse


def main() -> None:
    p = basis_parser("t-SNE: nichtlineare Projektion auf zwei Dimensionen.")
    p.add_argument("-p", "--perplexity", type=float, default=30.0,
                   help="Perplexity - effektive Anzahl beruecksichtigter Nachbarn")
    p.add_argument("--iterationen", type=int, default=1000,
                   help="Optimierungsschritte (mehr = stabiler, langsamer)")
    p.add_argument("--tsne-start", default="random", choices=TSNE_STARTS,
                   help="Startpositionen: random = unabhaengig von der PCA, "
                        "pca = PC1/PC2 als Start")
    p.add_argument("--metrik", default="euclidean",
                   help="Abstandsmass (euclidean, manhattan, cosine, ...)")
    p.add_argument("--sweep", action="store_true",
                   help="zusaetzlich ein Raster mehrerer Perplexity-Werte")
    hg.cluster_optionen(p)
    args = p.parse_args()

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    X, _ = skaliere(ds.X, args.skalierung)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    # Harte Bedingung von t-SNE: perplexity < Anzahl Datenpunkte.
    obergrenze = max(5.0, (len(X) - 1) / 3.0)
    perplexity = min(args.perplexity, obergrenze)
    if perplexity != args.perplexity:
        print("  Hinweis : Perplexity auf %.1f begrenzt (nur %d Datenpunkte)"
              % (perplexity, len(X)))

    print("\nt-SNE : perplexity=%.1f, %d Iterationen, Metrik '%s', Start '%s'"
          % (perplexity, args.iterationen, args.metrik, args.tsne_start))
    Y = tsne(X, perplexity, args.seed, args.iterationen, args.metrik, args.tsne_start)

    q = guete(X, Y)
    print("        Trustworthiness: %.3f  (1.0 = Nachbarschaften vollstaendig erhalten)" % q)

    unter = ("Perplexity = %.0f · %d Iterationen · Start %s · Trustworthiness %.2f · "
             "Achsen ohne Einheit, Abstaende zwischen Clustern nicht deutbar"
             % (perplexity, args.iterationen, args.tsne_start, q))

    print("\nPlots:")
    fig = einbettung_plot(Y, ds.labels, t, "t-SNE-Einbettung", unter,
                          ds.label_name, quelle, "t-SNE")
    speichere(fig, out / "tsne_einbettung.png")
    zeige_oder_schliesse(fig, args.zeigen)

    if args.sweep:
        werte = [w for w in (5, 15, 30, 50, 100) if w <= obergrenze]
        werte = sorted(set(werte + [int(perplexity)]))
        laeufe = []
        for w in werte:
            print("        Sweep: perplexity=%d" % w)
            laeufe.append(("Perplexity = %d" % w,
                           tsne(X, float(w), args.seed, args.iterationen, args.metrik,
                                args.tsne_start)))
        fig = raster_plot(laeufe, ds.labels, t, "t-SNE unter verschiedenen Perplexities",
                          "Was bei jeder Einstellung zusammenbleibt, ist echte Struktur",
                          ds.label_name, quelle)
        speichere(fig, out / "tsne_perplexity.png")
        zeige_oder_schliesse(fig, args.zeigen)

    ziel = speichere_koordinaten(Y, ds, out / "tsne_koordinaten.xlsx",
                                 ("tSNE_1", "tSNE_2"))
    print("  gespeichert: %s" % ziel)

    if args.cluster != "aus":
        cl = hc.finde_gruppen_in(Y, args.cluster, args.gruppen, args.min_gruppe,
                                 args.seed)
        print("\nGruppen: %s auf der %s-Karte, %s"
              % (cl.methode.upper(), "t-SNE",
                 ", ".join("%s=%s" % kv for kv in cl.parameter.items())))
        namen = hg.beschreibe(cl, Y, ds, t, out, "tsne", "t-SNE",
                              args.stabilitaet, args.seed, quelle, args.zeigen,
                              zusatz=pd.DataFrame(Y, columns=["tSNE_1", "tSNE_2"]))
        if namen is not None:
            fig = einbettung_plot(Y, namen, t, "t-SNE · gefundene Gruppen", unter,
                                  "Gruppe", quelle, "t-SNE", neutral=hc.RAUSCHEN)
            speichere(fig, out / "tsne_gruppen.png")
            zeige_oder_schliesse(fig, args.zeigen)



if __name__ == "__main__":
    starte(main)
