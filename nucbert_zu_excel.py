"""Wandelt die NucBERT-Embeddings (data/nucbert_embeddings.npy) in eine Excel im Stil der Testdaten.

Eigenstaendig - braucht nur numpy, pandas und openpyxl, nichts aus hda/.

Die .npy-Datei enthaelt eine Matrix Zeilen x Dimensionen (z. B. 100 x 1024):
jede Zeile ist das Embedding einer Sequenz, ohne Namen oder Beschriftung.

Die Excel sieht aus wie data/beispiel_messdaten.xlsx:
  * Kopfzeile, darunter eine Zeile je Embedding
  * Probe_ID   Textspalte, eindeutig je Zeile (wird nicht als Gruppe gewaehlt)
  * Dim_0001 ... Dim_1024   die Embedding-Dimensionen als Zahlenspalten

Die Analyse-Skripte lesen nur das erste Blatt.

Aufruf:  python nucbert_zu_excel.py
         python nucbert_zu_excel.py --quelle data/andere.npy -o data/andere.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
EXCEL_MAX_SPALTEN = 16384


def main() -> None:
    p = argparse.ArgumentParser(
        description="NucBERT-Embeddings (.npy) -> Excel im Stil der Testdaten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--quelle", default=str(HIER / "data" / "nucbert_embeddings.npy"),
                   help="NumPy-Datei mit einer Matrix Zeilen x Dimensionen")
    p.add_argument("-o", "--ziel", default=str(HIER / "data" / "nucbert_embeddings.xlsx"),
                   help="Ausgabedatei")
    args = p.parse_args()

    quelle = Path(args.quelle)
    if not quelle.is_file():
        sys.exit("Datei nicht gefunden: %s" % quelle)
    emb = np.load(quelle, allow_pickle=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    elif emb.ndim > 2:
        # z. B. (Proben, 1, Dim) - alles ausser der ersten Achse zusammenlegen
        emb = emb.reshape(emb.shape[0], -1)
    n, dim = emb.shape
    if dim + 1 > EXCEL_MAX_SPALTEN:
        sys.exit("%d Dimensionen passen nicht in eine Excel-Tabelle (max. %d Spalten)"
                 % (dim, EXCEL_MAX_SPALTEN - 1))

    stellen = len(str(dim))
    df = pd.DataFrame(emb.astype(np.float64),
                      columns=["Dim_%0*d" % (stellen, i + 1) for i in range(dim)])
    df.insert(0, "Probe_ID", ["Seq_%0*d" % (len(str(n)), i + 1) for i in range(n)])

    ziel = Path(args.ziel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Embeddings", index=False)

    werte = df.iloc[:, 1:]
    konstant = int((werte.nunique() <= 1).sum())
    print("Geschrieben: %s" % ziel)
    print("  %d Embeddings x %d Dimensionen, Wertebereich %.3f bis %.3f"
          % (n, dim, float(np.nanmin(emb)), float(np.nanmax(emb))))
    if np.isnan(emb).any():
        print("  Hinweis: %d fehlende Werte (NaN)" % int(np.isnan(emb).sum()))
    if konstant:
        print("  Konstant (entfernt die Analyse automatisch): %d Dimensionen" % konstant)
    print("\nTesten, z. B.:")
    anzeige = ziel.relative_to(HIER) if ziel.is_relative_to(HIER) else ziel
    print('  python run_all.py --datei "%s" --nur pca umap tsne vergleich' % anzeige)
    print("  (Streumatrix/Konstanz mit %d Spalten nur mit Spaltenauswahl sinnvoll)" % dim)


if __name__ == "__main__":
    main()
