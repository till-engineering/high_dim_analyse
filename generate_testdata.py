"""Erzeugt eine Beispiel-Excel in data/ zum Testen der Plots.

Die Daten simulieren einen Antriebspruefstand mit 15 Messgroessen und drei
Betriebszustaenden. Sie sind bewusst so gebaut, dass die Plots etwas zu
erzaehlen haben:

* Drehmoment, Strom und Leistung haengen physikalisch zusammen -> ihre Pfeile
  im PCA-Biplot zeigen fast in dieselbe Richtung.
* Der Wirkungsgrad laeuft den Temperaturen entgegen -> Pfeil in Gegenrichtung.
* Vibration, Koerperschall und Schalldruck bilden eine eigene Richtung, die den
  Lagerschaden von der Ueberlast trennt -> zwei klar getrennte Cluster in
  UMAP/t-SNE.
* Spannung und Betriebsstunden sind weitgehend unabhaengiges Rauschen
  -> kurze Pfeile, wenig Erklaerungsbeitrag.

Aufruf:  python generate_testdata.py [--zeilen 300] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ZUSTAENDE = {
    # Name            Anteil  Last   Verschleiss  Kuehlung
    "Normalbetrieb":   (0.45,  0.00,   0.00,        0.00),
    "Ueberlast":       (0.30,  2.10,   0.35,       -0.60),
    "Lagerschaden":    (0.25,  0.30,   2.30,       -0.20),
}


def erzeuge(zeilen: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    namen, last, verschleiss, kuehlung = [], [], [], []
    for name, (anteil, mu_l, mu_v, mu_k) in ZUSTAENDE.items():
        n = max(5, int(round(zeilen * anteil)))
        namen += [name] * n
        last += list(rng.normal(mu_l, 0.55, n))
        verschleiss += list(rng.normal(mu_v, 0.50, n))
        kuehlung += list(rng.normal(mu_k, 0.60, n))

    zustand = np.array(namen)
    L = np.array(last)          # Lastfaktor
    V = np.array(verschleiss)   # Verschleissfaktor
    K = np.array(kuehlung)      # Kuehlleistungsfaktor
    n = len(zustand)

    def rauschen(sigma):
        return rng.normal(0.0, sigma, n)

    drehzahl = 1450 + 60 * L - 25 * V + rauschen(18)
    drehmoment = 95 + 34 * L + 4 * V + rauschen(4.5)
    leistung = drehzahl * drehmoment / 9550 * (1 + rauschen(0.012))
    strom = 27 + 8.6 * L + 0.9 * V + rauschen(1.1)
    spannung = 400 + rauschen(3.2)                     # nahezu unabhaengig
    cos_phi = 0.87 + 0.035 * L - 0.02 * V + rauschen(0.012)

    wicklung = 72 + 13.5 * L + 2.0 * V - 6.5 * K + rauschen(2.4)
    lager = 58 + 6.0 * L + 9.5 * V - 3.0 * K + rauschen(2.0)
    oel = 61 + 8.5 * L + 4.0 * V - 5.0 * K + rauschen(2.2)

    vibration = 1.6 + 0.35 * L + 3.10 * V + rauschen(0.28)
    koerperschall = 0.42 + 0.09 * L + 0.78 * V + rauschen(0.07)
    schalldruck = 74 + 1.8 * L + 6.4 * V + rauschen(1.3)

    kuehlluft = 340 + 55 * K - 12 * L + rauschen(16)
    wirkungsgrad = 94.5 - 1.9 * np.abs(L) - 1.4 * V - 0.03 * (wicklung - 72) + rauschen(0.45)
    betriebsstunden = rng.uniform(200, 14000, n)       # nahezu unabhaengig

    df = pd.DataFrame({
        "Pruefling_ID": [f"P-{i + 1:04d}" for i in range(n)],
        "Betriebszustand": zustand,
        "Drehzahl_rpm": drehzahl,
        "Drehmoment_Nm": drehmoment,
        "Leistung_kW": leistung,
        "Strom_A": strom,
        "Spannung_V": spannung,
        "Leistungsfaktor": cos_phi,
        "Wirkungsgrad_pct": wirkungsgrad,
        "Wicklungstemp_C": wicklung,
        "Lagertemp_C": lager,
        "Getriebeoeltemp_C": oel,
        "Vibration_mm_s": vibration,
        "Koerperschall_g": koerperschall,
        "Schalldruck_dB": schalldruck,
        "Kuehlluft_m3_h": kuehlluft,
        "Betriebsstunden_h": betriebsstunden,
    })

    # Zeilen mischen, damit die Gruppen nicht blockweise sortiert sind.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["Pruefling_ID"] = [f"P-{i + 1:04d}" for i in range(len(df))]

    zahl_spalten = df.columns[2:]
    df[zahl_spalten] = df[zahl_spalten].round(3)
    return df


def schreibe(df: pd.DataFrame, pfad: Path) -> Path:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    from openpyxl.styles import Font

    with pd.ExcelWriter(pfad, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Messdaten", index=False)
        ws = writer.sheets["Messdaten"]
        ws.freeze_panes = "A2"
        for i, spalte in enumerate(df.columns, start=1):
            breite = max(len(str(spalte)), 12) + 3
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = breite
            ws.cell(row=1, column=i).font = Font(bold=True)
    return pfad


def main() -> None:
    p = argparse.ArgumentParser(
        description="Erzeugt eine Beispiel-Excel mit 15 Messgroessen in data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--zeilen", type=int, default=300, help="Anzahl Datenpunkte")
    p.add_argument("--seed", type=int, default=42, help="Zufallsstartwert")
    p.add_argument("--out", default="data/beispiel_messdaten.xlsx",
                   help="Zieldatei")
    args = p.parse_args()

    df = erzeuge(args.zeilen, args.seed)
    ziel = schreibe(df, Path(args.out))

    print(f"Geschrieben: {ziel}")
    print(f"  {len(df)} Zeilen, {len(df.columns)} Spalten "
          f"(15 Messgroessen + ID + Betriebszustand)")
    print("  Gruppen:")
    for name, anzahl in df["Betriebszustand"].value_counts().items():
        print(f"    {name:<16} {anzahl}")


if __name__ == "__main__":
    main()
