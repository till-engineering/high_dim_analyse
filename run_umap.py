"""UMAP-Einbettung hochdimensionaler Daten in zwei Dimensionen.

UMAP erhaelt sowohl die lokale Nachbarschaft als auch grob die globale
Struktur. Zwei Stellschrauben bestimmen das Ergebnis:

  --nachbarn  klein (5-15)  -> feine, lokale Struktur, viele kleine Inseln
              gross (50+)   -> grosszuegiger Blick, globale Form

  --min-dist  klein (0.0-0.1) -> dichte, kompakte Cluster
              gross (0.5-0.9) -> gleichmaessiger verteilte Punktwolke

Erzeugt im Ausgabeordner:
  umap_einbettung.png    die Einbettung
  umap_parameter.png     (mit --sweep) dieselben Daten unter mehreren Parametern
  umap_koordinaten.xlsx  die 2D-Koordinaten je Datenpunkt
  mit --cluster zusaetzlich: umap_gruppen.png, umap_gruppenprofil.png,
                             umap_gruppen.xlsx (Gruppen auf der Karte gesucht)

Aufruf:  python run_umap.py [--nachbarn 15] [--min-dist 0.1] [--sweep]
"""

from __future__ import annotations

import sys

import pandas as pd

from hda import cluster as hc
from hda import gruppen as hg
from hda.cli import ausgabeordner, basis_parser, blatt_wert, lies_argumente, starte
from hda.data_io import lade_daten, skaliere
from hda.embedding import (einbettung_plot, guete, raster_plot,
                           speichere_koordinaten)
from hda.plotstyle import set_style, speichere, zeige_oder_schliesse


def main() -> None:
    p = basis_parser("UMAP: nichtlineare Projektion auf zwei Dimensionen.")
    p.add_argument("-n", "--nachbarn", type=int, default=15,
                   help="n_neighbors - Groesse der betrachteten Nachbarschaft")
    p.add_argument("-m", "--min-dist", type=float, default=0.1,
                   help="min_dist - wie dicht Punkte zusammenruecken duerfen")
    p.add_argument("--metrik", default="euclidean",
                   help="Abstandsmass (euclidean, manhattan, cosine, ...)")
    p.add_argument("--sweep", action="store_true",
                   help="zusaetzlich ein Raster mehrerer Parameterkombinationen")
    hg.cluster_optionen(p)
    args = lies_argumente(p, "umap")

    try:
        import umap
    except ImportError:
        sys.exit("umap-learn fehlt. Installation:  pip install umap-learn")

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    X, _ = skaliere(ds.X, args.skalierung)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    nachbarn = max(2, min(args.nachbarn, len(X) - 1))
    if nachbarn != args.nachbarn:
        print("  Hinweis : n_neighbors auf %d begrenzt (nur %d Datenpunkte)"
              % (nachbarn, len(X)))

    print("\nUMAP  : n_neighbors=%d, min_dist=%.2f, Metrik '%s'"
          % (nachbarn, args.min_dist, args.metrik))
    reducer = umap.UMAP(n_components=2, n_neighbors=nachbarn,
                        min_dist=args.min_dist, metric=args.metrik,
                        random_state=args.seed)
    Y = reducer.fit_transform(X)

    q = guete(X, Y, metrik=args.metrik)
    print("        Trustworthiness: %.3f  (1.0 = Nachbarschaften vollstaendig erhalten)" % q)

    unter = ("n_neighbors = %d · min_dist = %.2f · Metrik %s · "
             "Trustworthiness %.2f · Achsen ohne Einheit"
             % (nachbarn, args.min_dist, args.metrik, q))

    print("\nPlots:")
    fig = einbettung_plot(Y, ds.labels, t, "UMAP-Einbettung", unter,
                          ds.label_name, quelle, "UMAP")
    speichere(fig, out / "umap_einbettung.png")
    zeige_oder_schliesse(fig, args.zeigen)

    if args.sweep:
        kombis = [(n, d) for n in (5, nachbarn, 50) for d in (0.0, 0.3)]
        kombis = [(min(n, len(X) - 1), d) for n, d in kombis]
        kombis = list(dict.fromkeys(kombis))
        laeufe = []
        for n, d in kombis:
            print("        Sweep: n_neighbors=%d, min_dist=%.2f" % (n, d))
            r = umap.UMAP(n_components=2, n_neighbors=n, min_dist=d,
                          metric=args.metrik, random_state=args.seed)
            laeufe.append(("n_neighbors=%d · min_dist=%.2f" % (n, d), r.fit_transform(X)))
        fig = raster_plot(laeufe, ds.labels, t, "UMAP unter verschiedenen Parametern",
                          "Stabile Gruppen bleiben ueber alle Einstellungen hinweg zusammen",
                          ds.label_name, quelle)
        speichere(fig, out / "umap_parameter.png")
        zeige_oder_schliesse(fig, args.zeigen)

    ziel = speichere_koordinaten(Y, ds, out / "umap_koordinaten.xlsx",
                                 ("UMAP_1", "UMAP_2"))
    print("  gespeichert: %s" % ziel)

    if args.cluster != "aus":
        cl = hc.finde_gruppen_in(Y, args.cluster, args.gruppen, args.min_gruppe,
                                 args.seed)
        print("\nGruppen: %s auf der %s-Karte, %s"
              % (cl.methode.upper(), "UMAP",
                 ", ".join("%s=%s" % kv for kv in cl.parameter.items())))
        namen = hg.beschreibe(cl, Y, ds, t, out, "umap", "UMAP",
                              args.stabilitaet, args.seed, quelle, args.zeigen,
                              zusatz=pd.DataFrame(Y, columns=["UMAP_1", "UMAP_2"]))
        if namen is not None:
            fig = einbettung_plot(Y, namen, t, "UMAP · gefundene Gruppen", unter,
                                  "Gruppe", quelle, "UMAP", neutral=hc.RAUSCHEN)
            speichere(fig, out / "umap_gruppen.png")
            zeige_oder_schliesse(fig, args.zeigen)



if __name__ == "__main__":
    starte(main)
