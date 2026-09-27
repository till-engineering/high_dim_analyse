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

Dazu kommen Messreihen, in denen sich nur EINE Ursache aendert (fuer
run_konstanz.py). Die uebrigen Groessen streuen dort nur um die
Messgenauigkeit, nicht um die volle Streuung des Pruefstands:

* Dauerlauf mit beginnendem Lagerschaden: Betriebsstunden steigen, nur der
  Koerperschall steigt mit - im Fruehstadium zeigt sich ein Lagerschaden im
  hochfrequenten Koerperschall, bevor Schwinggeschwindigkeit, Temperatur und
  Geraeusch reagieren.
* Dauerlauf eines gesunden Motors: nichts aendert sich (Gegenprobe).
* Spannungsreihe 380-420 V (+-5 %) bei gleicher Last: der Strom sinkt mit 1/U, die
  Leistung bleibt. Vereinfacht: cos phi und Temperaturen aendern sich in
  diesem kleinen Bereich nur innerhalb der Messgenauigkeit.

Die Spalte "Messreihe" sagt, zu welcher Reihe eine Zeile gehoert (leer =
freier Betriebspunkt). Die freien Betriebspunkte sind mit und ohne
Messreihen dieselben.

Mit --rauschen entsteht stattdessen eine Vergleichsdatei aus reinem Rauschen:
dieselben 15 Messgroessen mit denselben Mittelwerten und Streuungen, aber
jede Spalte unabhaengig gezogen - keine Zusammenhaenge, keine Gruppen. So
sieht man, wie die Plots aussehen, wenn es nichts zu finden gibt.

Aufruf:  python generate_testdata.py [--zeilen 300] [--seed 42]
         python generate_testdata.py --ohne-messreihen   # nur die freien Betriebspunkte
         python generate_testdata.py --rauschen
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


def erzeuge(zeilen: int = 300, seed: int = 42, mit_messreihen: bool = True) -> pd.DataFrame:
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

    werte = modell(L, V, K, lambda _groesse, sigma: rng.normal(0.0, sigma, n))
    betriebsstunden = rng.uniform(200, 14000, n)       # nahezu unabhaengig

    df = pd.DataFrame({
        "Pruefling_ID": [f"P-{i + 1:04d}" for i in range(n)],
        "Betriebszustand": zustand,
        **werte,
        "Betriebsstunden_h": betriebsstunden,
    })

    # Zeilen mischen, damit die Gruppen nicht blockweise sortiert sind.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["Pruefling_ID"] = [f"P-{i + 1:04d}" for i in range(len(df))]

    if mit_messreihen:
        # Hinten anhaengen statt einmischen: so bleiben die freien
        # Betriebspunkte samt Nummern genau wie ohne Messreihen.
        reihen = messreihen(seed)
        reihen.insert(0, "Pruefling_ID",
                      [f"P-{len(df) + i + 1:04d}" for i in range(len(reihen))])
        df.insert(2, "Messreihe", "")
        df = pd.concat([df, reihen[df.columns]], ignore_index=True)

    zahl_spalten = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[zahl_spalten] = df[zahl_spalten].round(3)
    return df


def modell(L, V, K, rauschen) -> dict:
    """Die 14 abhaengigen Messgroessen aus Last-, Verschleiss- und Kuehlfaktor.

    ``rauschen(groesse, sigma)`` liefert die Streuung je Groesse - fuer die
    freien Betriebspunkte die des Pruefstands, fuer Messreihen die
    Messgenauigkeit. Die Reihenfolge der Aufrufe legt die Zufallszahlen fest:
    nicht umstellen, sonst aendern sich die Beispieldaten.
    """
    drehzahl = 1450 + 60 * L - 25 * V + rauschen("Drehzahl_rpm", 18)
    drehmoment = 95 + 34 * L + 4 * V + rauschen("Drehmoment_Nm", 4.5)
    leistung = drehzahl * drehmoment / 9550 * (1 + rauschen("Leistung_kW", 0.012))
    strom = 27 + 8.6 * L + 0.9 * V + rauschen("Strom_A", 1.1)
    spannung = 400 + rauschen("Spannung_V", 3.2)       # nahezu unabhaengig
    cos_phi = 0.87 + 0.035 * L - 0.02 * V + rauschen("Leistungsfaktor", 0.012)

    wicklung = 72 + 13.5 * L + 2.0 * V - 6.5 * K + rauschen("Wicklungstemp_C", 2.4)
    lager = 58 + 6.0 * L + 9.5 * V - 3.0 * K + rauschen("Lagertemp_C", 2.0)
    oel = 61 + 8.5 * L + 4.0 * V - 5.0 * K + rauschen("Getriebeoeltemp_C", 2.2)

    vibration = 1.6 + 0.35 * L + 3.10 * V + rauschen("Vibration_mm_s", 0.28)
    koerperschall = 0.42 + 0.09 * L + 0.78 * V + rauschen("Koerperschall_g", 0.07)
    schalldruck = 74 + 1.8 * L + 6.4 * V + rauschen("Schalldruck_dB", 1.3)

    kuehlluft = 340 + 55 * K - 12 * L + rauschen("Kuehlluft_m3_h", 16)
    wirkungsgrad = (94.5 - 1.9 * np.abs(L) - 1.4 * V - 0.03 * (wicklung - 72)
                    + rauschen("Wirkungsgrad_pct", 0.45))
    return {
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
    }


# Wiederholgenauigkeit der Messkette (1 sigma): so eng liegen die Punkte
# einer Messreihe beieinander, wenn sich an der Ursache nichts aendert.
MESSGENAUIGKEIT = {
    "Drehzahl_rpm": 0.5, "Drehmoment_Nm": 0.1, "Leistung_kW": 0.01,
    "Strom_A": 0.02, "Spannung_V": 0.05, "Leistungsfaktor": 0.0005,
    "Wirkungsgrad_pct": 0.02, "Wicklungstemp_C": 0.1, "Lagertemp_C": 0.1,
    "Getriebeoeltemp_C": 0.1, "Vibration_mm_s": 0.02, "Koerperschall_g": 0.003,
    "Schalldruck_dB": 0.1, "Kuehlluft_m3_h": 0.8,
}

#  Name, Betriebszustand, Last, Verschleiss, Kuehlung, Betriebsstunden (Start)
DAUERLAEUFE = [
    ("Dauerlauf A: Lager beginnt zu schaedigen", "Normalbetrieb", 0.2, 0.1, 0.1, 3000),
    ("Dauerlauf B: Lager beginnt zu schaedigen", "Normalbetrieb", 0.9, 0.3, -0.2, 6500),
    ("Dauerlauf C: gesund", "Normalbetrieb", 0.5, 0.0, 0.3, 1500),
]
SPANNUNGSREIHEN = [
    ("Spannungsreihe Teillast", "Normalbetrieb", -0.6, 0.1, 0.2, 5200),
    ("Spannungsreihe Nennlast", "Normalbetrieb", 0.0, 0.2, 0.0, 8100),
    ("Spannungsreihe Ueberlast", "Ueberlast", 2.0, 0.4, -0.5, 10400),
]
DAUERLAUF_PUNKTE = 7                                # alle 500 h gemessen
SPANNUNGEN = np.array([380.0, 390.0, 400.0, 410.0, 420.0])   # +-5 %, Bereich A nach IEC 60034-1


def messreihen(seed: int) -> pd.DataFrame:
    """Versuchsreihen, in denen sich genau eine Ursache aendert.

    Eigener Zufallsgenerator - die freien Betriebspunkte bleiben dadurch
    dieselben wie ohne Messreihen.
    """
    rng = np.random.default_rng(seed + 2)

    def betriebspunkt(name, zustand, last, verschleiss, kuehlung, m):
        """m Messungen am selben Betriebspunkt, nur mit Messrauschen."""
        werte = modell(np.full(m, float(last)), np.full(m, float(verschleiss)),
                       np.full(m, float(kuehlung)),
                       lambda groesse, _: rng.normal(0.0, MESSGENAUIGKEIT[groesse], m))
        # Im Modell ist die Leistung relativ gestreut - hier absolut wie ein Messgeraet.
        werte["Leistung_kW"] = (werte["Drehzahl_rpm"] * werte["Drehmoment_Nm"] / 9550
                                + rng.normal(0.0, MESSGENAUIGKEIT["Leistung_kW"], m))
        df = pd.DataFrame(werte)
        df.insert(0, "Messreihe", name)
        df.insert(0, "Betriebszustand", zustand)
        return df

    stuecke = []
    for name, zustand, last, verschleiss, kuehlung, start in DAUERLAEUFE:
        df = betriebspunkt(name, zustand, last, verschleiss, kuehlung, DAUERLAUF_PUNKTE)
        stunden = start + 500.0 * np.arange(DAUERLAUF_PUNKTE)
        df["Betriebsstunden_h"] = stunden
        if "gesund" not in name:
            # Fruehstadium: der Koerperschall waechst beschleunigt, sonst nichts.
            t = (stunden - start) / 1000.0
            df["Koerperschall_g"] += 0.12 * t + 0.04 * t ** 2
        stuecke.append(df)

    for name, zustand, last, verschleiss, kuehlung, stunden in SPANNUNGSREIHEN:
        df = betriebspunkt(name, zustand, last, verschleiss, kuehlung, len(SPANNUNGEN))
        # Gleiche Leistung bei anderer Spannung: I ~ 1/U.
        df["Strom_A"] = df["Strom_A"] * 400.0 / SPANNUNGEN
        df["Spannung_V"] += SPANNUNGEN - 400.0
        df["Betriebsstunden_h"] = float(stunden)
        stuecke.append(df)

    return pd.concat(stuecke, ignore_index=True)


def erzeuge_rauschen(zeilen: int = 300, seed: int = 42) -> pd.DataFrame:
    """Gleiche Spalten, Mittelwerte und Streuungen wie :func:`erzeuge` - ohne Struktur.

    Jede Messgroesse ist unabhaengig normalverteilt. Mittelwert und Streuung
    stammen aus den strukturierten Daten, damit Einheiten und Groessenordnungen
    identisch sind und nur die Zusammenhaenge fehlen.
    """
    vorlage = erzeuge(zeilen, seed, mit_messreihen=False)
    rng = np.random.default_rng(seed + 1)
    messgroessen = vorlage.columns[2:]
    df = pd.DataFrame({
        spalte: rng.normal(vorlage[spalte].mean(), vorlage[spalte].std(), len(vorlage))
        for spalte in messgroessen
    }).round(3)
    df.insert(0, "Pruefling_ID", [f"R-{i + 1:04d}" for i in range(len(df))])
    return df


def schreibe(df: pd.DataFrame, pfad: Path, zusatz: pd.DataFrame | None = None) -> Path:
    """Blatt "Messdaten"; ``zusatz`` als zweites Blatt "Zuordnung" (liest die Analyse nicht)."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    from openpyxl.styles import Font

    blaetter = {"Messdaten": df}
    if zusatz is not None:
        blaetter["Zuordnung"] = zusatz
    with pd.ExcelWriter(pfad, engine="openpyxl") as writer:
        for name, blatt in blaetter.items():
            blatt.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            for i, spalte in enumerate(blatt.columns, start=1):
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
    p.add_argument("--ohne-messreihen", action="store_true",
                   help="nur die freien Betriebspunkte, keine Messreihen")
    p.add_argument("--ohne-labels", action="store_true",
                   help="Betriebszustand und Messreihe nur auf Blatt 'Zuordnung' - "
                        "zum Testen der Gruppensuche ohne Vorwissen")
    p.add_argument("--rauschen", action="store_true",
                   help="reines Rauschen statt strukturierter Daten")
    p.add_argument("--out", default=None,
                   help="Zieldatei (Standard: data/beispiel_messdaten.xlsx bzw. "
                        "data/rauschen_messdaten.xlsx)")
    args = p.parse_args()

    if args.rauschen:
        df = erzeuge_rauschen(args.zeilen, args.seed)
        ziel = schreibe(df, Path(args.out or "data/rauschen_messdaten.xlsx"))
        print(f"Geschrieben: {ziel}")
        print(f"  {len(df)} Zeilen, 15 Messgroessen aus unabhaengigem Rauschen + ID")
        print("  Keine Gruppen, keine Zusammenhaenge - jede gefundene Struktur ist Zufall.")
        return

    df = erzeuge(args.zeilen, args.seed, mit_messreihen=not args.ohne_messreihen)
    zusatz = None
    if args.ohne_labels:
        labels = [c for c in ("Betriebszustand", "Messreihe") if c in df]
        zusatz = df[["Pruefling_ID", *labels]]
        df = df.drop(columns=labels)
    ziel = schreibe(df, Path(args.out or "data/beispiel_messdaten.xlsx"), zusatz)

    print(f"Geschrieben: {ziel}")
    print(f"  {len(df)} Zeilen, {len(df.columns)} Spalten (15 Messgroessen + ID"
          + ("" if zusatz is not None else " + Betriebszustand")
          + (" + Messreihe" if "Messreihe" in df else "") + ")")
    if zusatz is not None:
        df = zusatz            # fuer die Uebersicht unten
        print("  Betriebszustand und Messreihe nur auf Blatt 'Zuordnung'")
    print("  Gruppen:")
    for name, anzahl in df["Betriebszustand"].value_counts().items():
        print(f"    {name:<16} {anzahl}")
    if "Messreihe" in df:
        print("  Messreihen (nur eine Ursache aendert sich):")
        for name, anzahl in df.loc[df["Messreihe"] != "", "Messreihe"].value_counts(
                sort=False).items():
            print(f"    {name:<42} {anzahl} Punkte")


if __name__ == "__main__":
    main()
