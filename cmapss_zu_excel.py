"""Wandelt die NASA-C-MAPSS-Triebwerksdaten in eine Excel im Stil der Testdaten.

Eigenstaendig - braucht nur numpy, pandas und openpyxl, nichts aus hda/.

C-MAPSS (Saxena et al., PHM08) simuliert eine Flotte gleicher Turbofan-
Triebwerke bis zum Ausfall. Jede Zeile der Rohdaten ist ein Flugzyklus eines
Triebwerks: Nummer, Zyklus, 3 Betriebseinstellungen, 21 Sensoren - ohne
Kopfzeile, durch Leerzeichen getrennt.

Die Excel sieht aus wie data/beispiel_messdaten.xlsx:
  * Kopfzeile mit sprechenden Namen samt Einheit, Temperaturen in °C,
    Druecke in bar (Umrechnung linear - aendert an PCA & Co. nichts)
  * Textspalten zur Kontrolle der Gruppensuche:
      Pruefling_ID   Datensatz-Triebwerk-Zyklus, eindeutig je Zeile
      Triebwerk      welches Triebwerk
      Betriebspunkt  Flugzustand (Hoehe, Machzahl, Schubhebel) - 6 bei FD002/FD004
      Zustand        nach Restlebensdauer: frueh / mittel / kurz vor Ausfall
  * Zyklus und Restlebensdauer (RUL) stehen auf dem zweiten Blatt - als
    Zahlenspalten im ersten Blatt wuerden sie sonst als Messgroessen zaehlen.

Die Analyse-Skripte lesen nur das erste Blatt.

Aufruf:  python cmapss_zu_excel.py                     # FD002, Training, 3000 Zeilen
         python cmapss_zu_excel.py --satz FD001 --alle
         python cmapss_zu_excel.py --satz FD004 --teil beide --stichprobe 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent

# Einheiten laut Saxena et al. (2008), Tabelle 2.
#  (Rohname, neuer Name, Umrechnung)
R_ZU_C = lambda r: (r - 491.67) * 5.0 / 9.0          # Rankine -> Celsius
PSI_ZU_BAR = lambda p: p * 0.0689476
GLEICH = lambda x: x
SENSOREN = [
    ("s1", "T2_Fan_Eintritt_C", R_ZU_C),
    ("s2", "T24_LPC_Austritt_C", R_ZU_C),
    ("s3", "T30_HPC_Austritt_C", R_ZU_C),
    ("s4", "T50_LPT_Austritt_C", R_ZU_C),
    ("s5", "P2_Fan_Eintritt_bar", PSI_ZU_BAR),
    ("s6", "P15_Bypass_bar", PSI_ZU_BAR),
    ("s7", "P30_HPC_Austritt_bar", PSI_ZU_BAR),
    ("s8", "Fan_Drehzahl_rpm", GLEICH),
    ("s9", "Kern_Drehzahl_rpm", GLEICH),
    ("s10", "Druckverhaeltnis_EPR", GLEICH),
    ("s11", "Ps30_statisch_bar", PSI_ZU_BAR),
    ("s12", "Brennstoff_zu_Ps30", GLEICH),
    ("s13", "Fan_Drehzahl_korr_rpm", GLEICH),
    ("s14", "Kern_Drehzahl_korr_rpm", GLEICH),
    ("s15", "Bypassverhaeltnis", GLEICH),
    ("s16", "Brennstoff_Luft_Verh", GLEICH),
    ("s17", "Zapfluft_Enthalpie", GLEICH),
    ("s18", "Fan_Drehzahl_Soll_rpm", GLEICH),
    ("s19", "Fan_Drehzahl_korr_Soll_rpm", GLEICH),
    ("s20", "HPT_Kuehlluft_lbm_s", GLEICH),
    ("s21", "LPT_Kuehlluft_lbm_s", GLEICH),
]
EINSTELLUNGEN = [
    ("e1", "Flughoehe_kft"),
    ("e2", "Machzahl"),
    ("e3", "Schubhebel_TRA_grad"),
]
ROHSPALTEN = ["einheit", "zyklus"] + [e for e, _ in EINSTELLUNGEN] + [s for s, _, _ in SENSOREN]

# Grenzen fuer die Zustandsklassen, in verbleibenden Zyklen bis zum Ausfall.
KURZ_VOR_AUSFALL = 30
MITTEL = 100


def lies_roh(pfad: Path) -> pd.DataFrame:
    df = pd.read_csv(pfad, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(ROHSPALTEN):
        sys.exit("%s: %d Spalten erwartet, %d gefunden"
                 % (pfad.name, len(ROHSPALTEN), df.shape[1]))
    df.columns = ROHSPALTEN
    return df


def restlebensdauer(df: pd.DataFrame, rul_ende: np.ndarray | None) -> pd.Series:
    """RUL je Zeile. Training: bis zum letzten Zyklus. Test: plus die RUL-Datei."""
    letzter = df.groupby("einheit")["zyklus"].transform("max")
    rul = letzter - df["zyklus"]
    if rul_ende is not None:
        rul = rul + df["einheit"].map(dict(enumerate(rul_ende, start=1)))
    return rul.astype(int)


def betriebspunkte(df: pd.DataFrame) -> pd.Series:
    """Die Einstellungen streuen nur um wenige feste Flugzustaende - runden genuegt."""
    schluessel = pd.DataFrame({
        "h": df["e1"].round(0).astype(int).abs(),
        "m": df["e2"].round(2).abs(),
        "t": df["e3"].round(0).astype(int),
    })
    punkte = schluessel.drop_duplicates().sort_values(["h", "m", "t"]).reset_index(drop=True)
    namen = {tuple(z): "BP%d: %d kft, Mach %.2f, TRA %d" % (i + 1, z.h, z.m, z.t)
             for i, z in punkte.iterrows()}
    return pd.Series([namen[tuple(z)] for z in schluessel.itertuples(index=False)],
                     index=df.index)


def zustand(rul: pd.Series) -> pd.Series:
    return pd.Series(np.select([rul <= KURZ_VOR_AUSFALL, rul <= MITTEL],
                               ["kurz vor Ausfall", "mittel"], "frueh"),
                     index=rul.index)


def baue(ordner: Path, satz: str, teil: str) -> pd.DataFrame:
    teile = ["train", "test"] if teil == "beide" else [teil]
    stuecke = []
    for t in teile:
        roh = lies_roh(ordner / f"{t}_{satz}.txt")
        rul_ende = None
        if t == "test":
            rul_ende = pd.read_csv(ordner / f"RUL_{satz}.txt", header=None).iloc[:, 0].to_numpy()
        roh["rul"] = restlebensdauer(roh, rul_ende)
        # Test-Triebwerke sind andere Maschinen als die gleich nummerierten im Training.
        roh["kennung"] = "%s-%s" % (satz, "T" if t == "train" else "P")
        stuecke.append(roh)
    roh = pd.concat(stuecke, ignore_index=True)

    df = pd.DataFrame(index=roh.index)
    df["Pruefling_ID"] = ["%s-E%03d-Z%03d" % (k, e, z)
                          for k, e, z in zip(roh["kennung"], roh["einheit"], roh["zyklus"])]
    df["Triebwerk"] = ["%s-E%03d" % (k, e) for k, e in zip(roh["kennung"], roh["einheit"])]
    df["Betriebspunkt"] = betriebspunkte(roh)
    df["Zustand"] = zustand(roh["rul"])
    for roh_name, name in EINSTELLUNGEN:
        df[name] = roh[roh_name]
    for roh_name, name, umrechnung in SENSOREN:
        df[name] = umrechnung(roh[roh_name]).round(4)
    df["_zyklus"] = roh["zyklus"]
    df["_rul"] = roh["rul"]
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="C-MAPSS-Rohdaten -> Excel im Stil der Testdaten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ordner", default=str(HIER / "data" / "CMAPSSData"),
                   help="Ordner mit train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt")
    p.add_argument("--satz", default="FD002", choices=["FD001", "FD002", "FD003", "FD004"],
                   help="FD002/FD004: 6 Betriebspunkte; FD003/FD004: zwei Fehlerarten")
    p.add_argument("--teil", default="train", choices=["train", "test", "beide"],
                   help="train = alle Triebwerke bis zum Ausfall")
    p.add_argument("--stichprobe", type=int, default=3000,
                   help="zufaellige Zeilen behalten (t-SNE und Stabilitaetspruefung "
                        "werden bei zehntausenden Zeilen sehr langsam)")
    p.add_argument("--alle", action="store_true", help="keine Stichprobe, alle Zeilen")
    p.add_argument("--ohne-einstellungen", action="store_true",
                   help="Flughoehe, Machzahl, Schubhebel weglassen - dann muessen "
                        "die Sensoren allein die Betriebspunkte verraten")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--ziel", default=None,
                   help="Ausgabedatei (Standard: data/cmapss_<satz>_<teil>.xlsx)")
    args = p.parse_args()

    ordner = Path(args.ordner)
    if not ordner.is_dir():
        sys.exit("Ordner nicht gefunden: %s" % ordner)
    df = baue(ordner, args.satz, args.teil)
    gesamt = len(df)

    if not args.alle and len(df) > args.stichprobe:
        auswahl = np.random.default_rng(args.seed).choice(len(df), args.stichprobe,
                                                          replace=False)
        df = df.iloc[np.sort(auswahl)].reset_index(drop=True)
    if args.ohne_einstellungen:
        df = df.drop(columns=[name for _, name in EINSTELLUNGEN])

    zyklen = df[["Pruefling_ID", "Triebwerk", "_zyklus", "_rul"]].rename(
        columns={"_zyklus": "Zyklus", "_rul": "Restlebensdauer_Zyklen"})
    daten = df.drop(columns=["_zyklus", "_rul"])

    ziel = Path(args.ziel) if args.ziel else HIER / "data" / (
        "cmapss_%s_%s.xlsx" % (args.satz, args.teil))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        daten.to_excel(writer, sheet_name="Messdaten", index=False)
        zyklen.to_excel(writer, sheet_name="Zyklen", index=False)

    konstant = [c for c in daten.columns[4:] if daten[c].nunique() <= 1]
    print("Geschrieben: %s" % ziel)
    print("  %s %s: %d von %d Zeilen, %d Triebwerke, %d Messgroessen"
          % (args.satz, args.teil, len(daten), gesamt, daten["Triebwerk"].nunique(),
             daten.shape[1] - 4))
    print("  Betriebspunkte: %d | Zustand: %s"
          % (daten["Betriebspunkt"].nunique(),
             ", ".join("%s %d" % kv for kv in daten["Zustand"].value_counts().items())))
    if konstant:
        print("  Konstant (entfernt die Analyse automatisch): %s" % ", ".join(konstant))
    print("\nTesten, z. B.:")
    print('  python run_all.py --datei "%s" --label-spalte Betriebspunkt'
          % ziel.relative_to(HIER) if ziel.is_relative_to(HIER) else ziel)
    print("  (ohne --label-spalte wird nach 'Zustand' eingefaerbt - die Spalte "
          "mit den wenigsten Werten)")


if __name__ == "__main__":
    main()
