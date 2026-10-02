"""Wandelt den Datensatz "Condition monitoring of hydraulic systems" in eine Excel im Stil der Testdaten.

Eigenstaendig - braucht nur numpy, pandas und openpyxl, nichts aus hda/.

Der Datensatz (ZeMA gGmbH, Helwig et al. 2015) stammt von einem hydraulischen
Pruefstand, der 2205 gleiche Lastzyklen von je 60 s faehrt. Dabei wird der
Zustand von vier Komponenten (Kuehler, Ventil, Pumpe, Hydrospeicher) gezielt
verschlechtert. Rohdaten in data/cond_monitoring/:
  * je Sensor eine Datei <Sensor>.txt, tab-getrennt, ohne Kopfzeile:
    Zeile = Zyklus, Spalte = Messzeitpunkt im Zyklus
    (100 Hz -> 6000 Werte, 10 Hz -> 600 Werte, 1 Hz -> 60 Werte)
  * profile.txt: je Zyklus der Zustand der vier Komponenten + Stabil-Flag

Die 43680 Rohwerte je Zyklus passen nicht in eine Excel (max. 16384 Spalten)
und waeren als einzelne Messgroessen auch nicht sinnvoll. Deshalb wird jede
Sensor-Zeitreihe je Zyklus zu Kennwerten verdichtet (Mittelwert,
Standardabweichung, ...) - so wie in den Veroeffentlichungen zum Datensatz.

Die Excel sieht aus wie data/beispiel_messdaten.xlsx:
  * Kopfzeile mit ausgeschriebenen Namen samt Einheit, z. B.
    Druck_1_bar_Mittelwert, Motorleistung_W_Standardabweichung
  * Textspalten zur Kontrolle der Gruppensuche:
      Zyklus_ID            eindeutig je Zeile
      Kuehler_Zustand      100 % volle Effizienz / 20 % reduziert / 3 % kurz vor Totalausfall
      Ventil_Zustand       100 % optimal / 90 % leichte / 80 % starke Verzoegerung / 73 % kurz vor Totalausfall
      Pumpe_Leckage        keine / schwache / starke innere Leckage
      Hydrospeicher_Druck  130 / 115 / 100 / 90 bar
  * Zyklusnummer, Zustandswerte als Zahl und das Stabil-Flag stehen auf dem
    zweiten Blatt - als Spalten im ersten Blatt wuerden die Zahlen als
    Messgroessen zaehlen und das Flag (nur 2 Werte) als Gruppe gewaehlt.

Die Analyse-Skripte lesen nur das erste Blatt.

Aufruf:  python hydraulik_zu_excel.py                       # Mittelwert + Standardabweichung
         python hydraulik_zu_excel.py --kennwerte mittelwert std min max steigung
         python hydraulik_zu_excel.py --nur-stabil
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent

# Laut description.txt / documentation.txt.  (Datei, Name mit Einheit)
SENSOREN = [
    ("PS1", "Druck_1_bar"),
    ("PS2", "Druck_2_bar"),
    ("PS3", "Druck_3_bar"),
    ("PS4", "Druck_4_bar"),
    ("PS5", "Druck_5_bar"),
    ("PS6", "Druck_6_bar"),
    ("EPS1", "Motorleistung_W"),
    ("FS1", "Volumenstrom_1_l_min"),
    ("FS2", "Volumenstrom_2_l_min"),
    ("TS1", "Temperatur_1_C"),
    ("TS2", "Temperatur_2_C"),
    ("TS3", "Temperatur_3_C"),
    ("TS4", "Temperatur_4_C"),
    ("VS1", "Vibration_mm_s"),
    ("CE", "Kuehleffizienz_virtuell_Prozent"),
    ("CP", "Kuehlleistung_virtuell_kW"),
    ("SE", "Wirkungsgrad_Prozent"),
]


def _steigung(werte: np.ndarray) -> np.ndarray:
    """Lineare Steigung je Zyklus, in Einheit pro Sekunde (ein Zyklus = 60 s)."""
    t = np.linspace(0.0, 60.0, werte.shape[1])
    t = t - t.mean()
    return (werte - werte.mean(axis=1, keepdims=True)) @ t / (t @ t)


# Kuerzel auf der Kommandozeile -> (Spaltenendung, Berechnung ueber die Zeitachse)
KENNWERTE = {
    "mittelwert": ("Mittelwert", lambda w: w.mean(axis=1)),
    "std": ("Standardabweichung", lambda w: w.std(axis=1, ddof=1)),
    "min": ("Minimum", lambda w: w.min(axis=1)),
    "max": ("Maximum", lambda w: w.max(axis=1)),
    "median": ("Median", lambda w: np.median(w, axis=1)),
    "steigung": ("Steigung_pro_s", _steigung),
}

# Zustandsspalten aus profile.txt: (Spaltenname, {Zahl: Text})
ZUSTAENDE = [
    ("Kuehler_Zustand", {100: "100 % volle Effizienz",
                         20: "20 % reduzierte Effizienz",
                         3: "3 % kurz vor Totalausfall"}),
    ("Ventil_Zustand", {100: "100 % optimales Schalten",
                        90: "90 % leichte Verzoegerung",
                        80: "80 % starke Verzoegerung",
                        73: "73 % kurz vor Totalausfall"}),
    ("Pumpe_Leckage", {0: "keine Leckage",
                       1: "schwache Leckage",
                       2: "starke Leckage"}),
    ("Hydrospeicher_Druck", {130: "130 bar optimal",
                             115: "115 bar leicht reduziert",
                             100: "100 bar stark reduziert",
                             90: "90 bar kurz vor Totalausfall"}),
]


def lies_matrix(pfad: Path) -> np.ndarray:
    if not pfad.is_file():
        sys.exit("Datei nicht gefunden: %s" % pfad)
    return pd.read_csv(pfad, sep="\t", header=None).to_numpy(dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hydraulik-Zustandsdaten (ZeMA) -> Excel im Stil der Testdaten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ordner", default=str(HIER / "data" / "cond_monitoring"),
                   help="Ordner mit PS1.txt ... VS1.txt und profile.txt")
    p.add_argument("--kennwerte", nargs="+", default=["mittelwert", "std"],
                   choices=list(KENNWERTE),
                   help="welche Kennwerte je Sensor und Zyklus berechnet werden")
    p.add_argument("--nur-stabil", action="store_true",
                   help="nur Zyklen, in denen der Pruefstand eingeschwungen war (Stabil-Flag 0)")
    p.add_argument("-o", "--ziel", default=str(HIER / "data" / "hydraulik_zustand.xlsx"),
                   help="Ausgabedatei")
    args = p.parse_args()

    ordner = Path(args.ordner)
    if not ordner.is_dir():
        sys.exit("Ordner nicht gefunden: %s" % ordner)

    profil = pd.read_csv(ordner / "profile.txt", sep="\t", header=None)
    if profil.shape[1] != 5:
        sys.exit("profile.txt: 5 Spalten erwartet, %d gefunden" % profil.shape[1])
    n = len(profil)

    daten = pd.DataFrame(index=range(n))
    daten["Zyklus_ID"] = ["Zyklus_%04d" % (i + 1) for i in range(n)]
    for i, (name, texte) in enumerate(ZUSTAENDE):
        unbekannt = sorted(set(profil[i]) - set(texte))
        if unbekannt:
            sys.exit("profile.txt, %s: unbekannte Werte %s" % (name, unbekannt))
        daten[name] = profil[i].map(texte)

    merkmale = {}
    for datei, name in SENSOREN:
        werte = lies_matrix(ordner / ("%s.txt" % datei))
        if werte.shape[0] != n:
            sys.exit("%s.txt: %d Zyklen, profile.txt hat %d" % (datei, werte.shape[0], n))
        for kuerzel in args.kennwerte:
            endung, rechne = KENNWERTE[kuerzel]
            merkmale["%s_%s" % (name, endung)] = np.round(rechne(werte), 6)
        print("  gelesen: %-5s %4d Werte je Zyklus -> %s" % (datei, werte.shape[1], name))
    daten = pd.concat([daten, pd.DataFrame(merkmale)], axis=1)

    zyklen = pd.DataFrame({
        "Zyklus_ID": daten["Zyklus_ID"],
        "Zyklus": np.arange(1, n + 1),
        "Kuehler_Effizienz_Prozent": profil[0],
        "Ventil_Schaltverhalten_Prozent": profil[1],
        "Pumpe_Leckage_Stufe": profil[2],
        "Hydrospeicher_Druck_bar": profil[3],
        "Stabil": np.where(profil[4] == 0, "stabil", "evtl. nicht eingeschwungen"),
    })

    if args.nur_stabil:
        stabil = (profil[4] == 0).to_numpy()
        daten = daten[stabil].reset_index(drop=True)
        zyklen = zyklen[stabil].reset_index(drop=True)

    ziel = Path(args.ziel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        daten.to_excel(writer, sheet_name="Messdaten", index=False)
        zyklen.to_excel(writer, sheet_name="Zyklen", index=False)

    werte = daten.iloc[:, 1 + len(ZUSTAENDE):]
    konstant = [c for c in werte.columns if werte[c].nunique() <= 1]
    print("\nGeschrieben: %s" % ziel)
    print("  %d von %d Zyklen, %d Sensoren x %d Kennwerte = %d Messgroessen"
          % (len(daten), n, len(SENSOREN), len(args.kennwerte), werte.shape[1]))
    for name, _ in ZUSTAENDE:
        print("  %-20s %s" % (name, ", ".join(
            "%s %d" % kv for kv in daten[name].value_counts().sort_index().items())))
    if konstant:
        print("  Konstant (entfernt die Analyse automatisch): %s" % ", ".join(konstant))
    print("\nTesten, z. B.:")
    anzeige = ziel.relative_to(HIER) if ziel.is_relative_to(HIER) else ziel
    print('  python run_all.py --datei "%s" --label-spalte Ventil_Zustand' % anzeige)
    print("  (ohne --label-spalte wird nach 'Kuehler_Zustand' eingefaerbt - "
          "die Textspalte mit den wenigsten Werten)")


if __name__ == "__main__":
    main()
