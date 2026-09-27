"""Wandelt die Legierungsdatenbank data/combined_data.csv in eine Excel im Stil der Testdaten.

Eigenstaendig - braucht nur numpy, pandas und openpyxl, nichts aus hda/.

Die CSV sammelt Hochentropie-Legierungen aus der Literatur: je Zeile eine
Summenformel (z. B. Al0.5CoCrFeNi, (CoCrFeMnNi)98C2), die gefundenen Phasen,
die Herstellung und - lueckenhaft - mechanische Kennwerte.

Die Formel gibt Stoffmengenverhaeltnisse an. Daraus wird je Element der
Massenanteil in % gerechnet:  w_i = x_i * M_i / sum(x_j * M_j).

Klammern in Formeln:
  * (ABC)n  - die Gruppe wird auf 1 normiert und mit n gewichtet,
              (CoCrFeMnNi)98C2 = je 19.6 at-% Co, Cr, Fe, Mn, Ni + 2 at-% C
  * (WC)n, (TaC)n - Verbindungen: n Formeleinheiten, also n W und n C

Blaetter der Excel (die Analyse-Skripte lesen nur das erste):
  Legierungen   Textspalten + Massenanteil je Element -> das sind die Messgroessen
      Legierung_ID  eindeutig je Zeile
      Formel        wie in der CSV
      Phasenklasse  aus den Phasen verdichtet: FCC, BCC, FCC+BCC, jeweils
                    "+IM", wenn weitere (intermetallische) Phasen dabei sind.
                    L12 zaehlt als FCC, B2 als BCC (geordnete Varianten).
      Phasen        Originalangabe, Leerzeichen vereinheitlicht
      Gefuege       einphasig / mehrphasig
      Herstellung   Kuerzel wie in der CSV (AC gegossen, A gegluecht,
                    CR/HR kalt-/warmgewalzt, HIP, SPS ...)
      DOI           Quelle
  Atomprozent   dieselben Legierungen in at-% plus Anzahl Elemente
                (mit --je-legierung auch, wie oft sie in der CSV steht)
  Eigenschaften Kennwerte als Zahlen (">50" -> 50, "150-200" -> 175,
                "865 $\\pm$ 39" -> 865), mit Legierung_ID verknuepft

Elemente, die in weniger als --min-zeilen Legierungen vorkommen, werden zu
"Sonstige_Masse%" zusammengefasst - eine Spalte, die fast ueberall 0 ist,
bekommt nach der Standardisierung sonst ein riesiges Gewicht fuer die paar
Zeilen, in denen sie nicht 0 ist.

Aufruf:  python legierungen_zu_excel.py
         python legierungen_zu_excel.py --je-legierung      # gleiche Zusammensetzung nur einmal
         python legierungen_zu_excel.py --min-zeilen 1      # jedes Element eigene Spalte
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent

# Standard-Atommassen in g/mol (IUPAC, gerundet).
ATOMMASSE = {
    "Ag": 107.868, "Al": 26.982, "B": 10.81, "C": 12.011, "Ca": 40.078,
    "Co": 58.933, "Cr": 51.996, "Cu": 63.546, "Fe": 55.845, "Ga": 69.723,
    "Ge": 72.630, "Hf": 178.49, "Ir": 192.217, "Li": 6.94, "Mg": 24.305,
    "Mn": 54.938, "Mo": 95.95, "N": 14.007, "Nb": 92.906, "Nd": 144.24,
    "Ni": 58.693, "O": 15.999, "P": 30.974, "Pd": 106.42, "Pt": 195.084,
    "Re": 186.207, "Ru": 101.07, "Sc": 44.956, "Si": 28.085, "Sn": 118.71,
    "Ta": 180.948, "Ti": 47.867, "V": 50.942, "W": 183.84, "Y": 88.906,
    "Zn": 65.38, "Zr": 91.224,
}
# Klammergruppen, die als Verbindung gemeint sind und nicht normiert werden.
VERBINDUNGEN = {"WC", "TaC"}

# (Spalte in der CSV, Name in der Excel)
EIGENSCHAFTEN = [
    ("PROPERTY: Type of test", "Pruefart"),
    ("PROPERTY: Test temperature ($^\\circ$C)", "Prueftemperatur_C"),
    ("PROPERTY: grain size ($\\mu$m)", "Korngroesse_um"),
    ("PROPERTY: ROM Density (g/cm$^3$)", "Dichte_Mischungsregel_g_cm3"),
    ("PROPERTY: Exp. Density (g/cm$^3$)", "Dichte_gemessen_g_cm3"),
    ("PROPERTY: HV", "Haerte_HV"),
    ("PROPERTY: YS (MPa)", "Streckgrenze_MPa"),
    ("PROPERTY: UTS (MPa)", "Zugfestigkeit_MPa"),
    ("PROPERTY: Elongation (%)", "Dehnung_%"),
    ("PROPERTY: Elongation plastic (%)", "Dehnung_plastisch_%"),
    ("PROPERTY: ROM Young modulus (GPa)", "E_Modul_Mischungsregel_GPa"),
    ("PROPERTY: Exp. Young modulus (GPa)", "E_Modul_gemessen_GPa"),
    ("PROPERTY: O content (wppm)", "O_Gehalt_wppm"),
    ("PROPERTY: N content (wppm)", "N_Gehalt_wppm"),
    ("PROPERTY: C content (wppm)", "C_Gehalt_wppm"),
    ("REFERENCE: comment", "Kommentar"),
]
PRUEFART = {"C": "Druck", "T": "Zug"}
GEFUEGE = {"S": "einphasig", "M": "mehrphasig"}


# --------------------------------------------------------------------------- #
# Formel -> Stoffmengenanteile
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\d+\.?\d*|\.\d+)")


def zerlege(formel: str) -> dict[str, float]:
    """'(CoCrFeMnNi)98C2' -> {'Co': 0.196, ..., 'C': 0.02}, Summe 1."""
    text = formel.replace(" ", "")
    teile = _TOKEN.findall(text)
    if "".join(teile) != text:
        raise ValueError("Formel nicht lesbar: %r" % formel)
    pos = 0

    def zahl(standard: float) -> float:
        nonlocal pos
        if pos < len(teile) and teile[pos][0] in "0123456789.":
            pos += 1
            return float(teile[pos - 1])
        return standard

    def folge() -> dict[str, float]:
        nonlocal pos
        anteile: dict[str, float] = {}
        while pos < len(teile) and teile[pos] != ")":
            t = teile[pos]
            pos += 1
            if t == "(":
                start = pos
                innen = folge()
                if pos >= len(teile):
                    raise ValueError("Klammer nicht geschlossen: %r" % formel)
                wortlaut = "".join(teile[start:pos])
                pos += 1
                n = zahl(1.0)
                if wortlaut not in VERBINDUNGEN:
                    summe = sum(innen.values())
                    innen = {e: x / summe for e, x in innen.items()}
                for e, x in innen.items():
                    anteile[e] = anteile.get(e, 0.0) + x * n
            elif t[0].isupper():
                if t not in ATOMMASSE:
                    raise ValueError("Unbekanntes Element %r in %r" % (t, formel))
                anteile[t] = anteile.get(t, 0.0) + zahl(1.0)
            else:
                raise ValueError("Zahl ohne Element in %r" % formel)
        return anteile

    anteile = folge()
    if pos != len(teile):
        raise ValueError("Klammer zu viel: %r" % formel)
    summe = sum(anteile.values())
    if summe <= 0:
        raise ValueError("Formel ohne Anteile: %r" % formel)
    return {e: x / summe for e, x in anteile.items()}


def masseprozent(atomanteile: dict[str, float]) -> dict[str, float]:
    masse = {e: x * ATOMMASSE[e] for e, x in atomanteile.items()}
    summe = sum(masse.values())
    return {e: 100.0 * m / summe for e, m in masse.items()}


# --------------------------------------------------------------------------- #
# Textspalten aufbereiten
# --------------------------------------------------------------------------- #
def phasenklasse(phasen) -> str:
    """'FCC + BCC + $\\sigma$' -> 'FCC+BCC+IM'."""
    if not isinstance(phasen, str) or not phasen.strip():
        return "(unbekannt)"
    gitter, weitere = set(), False
    for teil in re.split(r"\+", phasen):
        t = teil.strip().lower()
        if not t:
            continue
        if re.fullmatch(r"\d?(fcc\d?|l12)", t):
            gitter.add("FCC")
        elif re.fullmatch(r"\d?(bcc\d?|b2)", t):
            gitter.add("BCC")
        elif re.fullmatch(r"\d?hcp", t):
            gitter.add("HCP")
        else:
            weitere = True
    teile = [g for g in ("FCC", "BCC", "HCP") if g in gitter]
    if weitere:
        teile.append("IM")
    return "+".join(teile) if gitter else "nur IM"


def zahl_aus(wert) -> float:
    """'865 $\\pm$ 39' -> 865, '>50' -> 50, '150-200' -> 175, '' -> NaN."""
    if isinstance(wert, (int, float, np.number)):
        return float(wert)
    if not isinstance(wert, str):
        return np.nan
    t = wert.replace(",", ".")
    bereich = re.fullmatch(r"\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*", t)
    if bereich:
        return (float(bereich.group(1)) + float(bereich.group(2))) / 2
    treffer = re.search(r"-?\d+\.?\d*", t)
    return float(treffer.group()) if treffer else np.nan


# --------------------------------------------------------------------------- #
# Aufbau
# --------------------------------------------------------------------------- #
def baue(roh: pd.DataFrame, min_zeilen: int):
    fehler = []
    atom, masse = [], []
    for formel in roh["FORMULA"].astype(str):
        try:
            x = zerlege(formel)
        except ValueError as f:
            fehler.append(str(f))
            x = {}
        atom.append(x)
        masse.append(masseprozent(x) if x else {})
    if fehler:
        sys.exit("Formeln nicht lesbar:\n  " + "\n  ".join(sorted(set(fehler))))

    atom = pd.DataFrame(atom, index=roh.index).fillna(0.0) * 100.0
    masse = pd.DataFrame(masse, index=roh.index).fillna(0.0)
    # Nach Haeufigkeit, dann alphabetisch - die wichtigen Elemente stehen vorn.
    vorkommen = (atom > 0).sum()
    reihenfolge = sorted(atom.columns, key=lambda e: (-vorkommen[e], e))
    atom, masse = atom[reihenfolge], masse[reihenfolge]

    selten = [e for e in reihenfolge if vorkommen[e] < min_zeilen]
    haeufig = [e for e in reihenfolge if e not in selten]

    text = pd.DataFrame(index=roh.index)
    text["Formel"] = roh["FORMULA"].astype(str).str.strip()
    phasen = roh["PROPERTY: Type of phases"]
    text["Phasenklasse"] = phasen.map(phasenklasse)
    text["Phasen"] = phasen.where(phasen.isna(), phasen.astype(str).str.replace(
        r"\s*\+\s*", " + ", regex=True).str.strip(" +"))
    text["Gefuege"] = roh["PROPERTY: Single/Multiphase"].map(GEFUEGE)
    text["Herstellung"] = roh["PROPERTY: synthesis method"].astype(str).str.replace(
        r"\s*\+\s*", "+", regex=True).str.strip().where(
        roh["PROPERTY: synthesis method"].notna())
    text["DOI"] = roh["REFERENCE: doi"]

    mp = masse[haeufig].rename(columns=lambda e: "%s_Masse%%" % e)
    if selten:
        mp["Sonstige_Masse%"] = masse[selten].sum(axis=1)
    ap = atom.rename(columns=lambda e: "%s_at%%" % e)
    ap["Anzahl_Elemente"] = (atom > 0).sum(axis=1)

    eig = pd.DataFrame(index=roh.index)
    for spalte, name in EIGENSCHAFTEN:
        if spalte not in roh.columns:
            continue
        if name == "Pruefart":
            eig[name] = roh[spalte].map(PRUEFART)
        elif name == "Kommentar":
            eig[name] = roh[spalte]
        else:
            eig[name] = roh[spalte].map(zahl_aus)
    return text, mp.round(4), ap.round(4), eig, selten


def je_legierung(text, mp, ap):
    """Gleiche Zusammensetzung -> eine Zeile. Textspalten: haeufigster Wert."""
    schluessel = ap.drop(columns="Anzahl_Elemente").round(2).apply(tuple, axis=1)
    gruppe = pd.Series(pd.factorize(schluessel)[0], index=text.index)

    def haeufigster(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else np.nan

    ap_neu = ap.groupby(gruppe).first()
    # Nicht ins erste Blatt - als Zahlenspalte liefe sie dort als Messgroesse mit.
    ap_neu["Eintraege_in_CSV"] = gruppe.value_counts().sort_index()
    return text.groupby(gruppe).agg(haeufigster), mp.groupby(gruppe).first(), ap_neu, gruppe


def main() -> None:
    p = argparse.ArgumentParser(
        description="Legierungsdatenbank (CSV) -> Excel mit Massenanteilen je Element",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", default=str(HIER / "data" / "combined_data.csv"))
    p.add_argument("--min-zeilen", type=int, default=5,
                   help="seltenere Elemente in 'Sonstige_Masse%%' zusammenfassen")
    p.add_argument("--je-legierung", action="store_true",
                   help="gleiche Zusammensetzung nur einmal (die CSV fuehrt viele "
                        "Legierungen mehrfach, je Pruefung/Temperatur eine Zeile)")
    p.add_argument("-o", "--ziel", default=None,
                   help="Ausgabedatei (Standard: data/legierungen[_je_legierung].xlsx)")
    args = p.parse_args()

    quelle = Path(args.csv)
    if not quelle.is_file():
        sys.exit("CSV nicht gefunden: %s" % quelle)
    roh = pd.read_csv(quelle, index_col=0).reset_index(drop=True)
    text, mp, ap, eig, selten = baue(roh, args.min_zeilen)

    if args.je_legierung:
        text, mp, ap, gruppe = je_legierung(text, mp, ap)
        ids = pd.Series(["HEA-%04d" % (i + 1) for i in range(len(text))], index=text.index)
        eig.insert(0, "Legierung_ID", gruppe.map(ids).to_numpy())
    else:
        ids = pd.Series(["HEA-%04d" % (i + 1) for i in range(len(text))], index=text.index)
        eig.insert(0, "Legierung_ID", ids.to_numpy())
    eig.insert(1, "Formel", roh["FORMULA"].astype(str).str.strip().to_numpy())
    text.insert(0, "Legierung_ID", ids)
    ap.insert(0, "Legierung_ID", ids)
    ap.insert(1, "Formel", text["Formel"])

    daten = pd.concat([text, mp], axis=1)
    ziel = Path(args.ziel) if args.ziel else HIER / "data" / (
        "legierungen%s.xlsx" % ("_je_legierung" if args.je_legierung else ""))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        daten.to_excel(writer, sheet_name="Legierungen", index=False)
        ap.to_excel(writer, sheet_name="Atomprozent", index=False)
        eig.to_excel(writer, sheet_name="Eigenschaften", index=False)

    print("Geschrieben: %s" % ziel)
    print("  %d Zeilen (CSV: %d), %d Messgroessen (Massenanteile in %%)"
          % (len(daten), len(roh), mp.shape[1]))
    abweichung = (mp.sum(axis=1) - 100).abs().max()
    print("  Summe der Massenanteile je Zeile: 100 %% (max. Abweichung %.3f)" % abweichung)
    if selten:
        print("  In 'Sonstige_Masse%%' zusammengefasst (< %d Zeilen): %s"
              % (args.min_zeilen, ", ".join(selten)))
    print("  Phasenklassen: %s"
          % ", ".join("%s %d" % kv for kv in daten["Phasenklasse"].value_counts().items()))
    print("\nTesten, z. B.:")
    anzeige = ziel.relative_to(HIER) if ziel.is_relative_to(HIER) else ziel
    print('  python run_all.py --datei "%s" --label-spalte Phasenklasse' % anzeige)
    print("  (ohne --label-spalte wird nach 'Gefuege' eingefaerbt - die Spalte "
          "mit den wenigsten Werten)")


if __name__ == "__main__":
    main()
