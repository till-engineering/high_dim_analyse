"""Gemeinsame Kommandozeilen-Optionen aller Analyse-Skripte."""

from __future__ import annotations

import argparse
from pathlib import Path


def basis_parser(beschreibung: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=beschreibung,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-d", "--datei", default=None,
                   help="Excel-Datei; ohne Angabe die neueste in data/")
    p.add_argument("-b", "--blatt", default=0,
                   help="Tabellenblatt (Name oder Index)")
    p.add_argument("-l", "--label-spalte", default=None,
                   help="Spalte zum Einfaerben; ohne Angabe die Textspalte mit den "
                        "wenigsten verschiedenen Werten")
    p.add_argument("--id-spalten", nargs="*", default=None,
                   help="numerische Spalten, die keine Merkmale sind (z. B. IDs)")
    p.add_argument("-s", "--skalierung", default="zscore",
                   choices=["zscore", "minmax", "robust", "keine"],
                   help="Standardisierung der Merkmale")
    p.add_argument("--nan", default="median", choices=["median", "drop"],
                   help="Umgang mit fehlenden Werten")
    p.add_argument("-o", "--out", default="output",
                   help="Ausgabeordner fuer Plots und Tabellen")
    p.add_argument("--theme", default="light", choices=["light", "dark"],
                   help="Farbschema der Plots")
    p.add_argument("--zeigen", action="store_true",
                   help="Plotfenster oeffnen statt nur speichern")
    p.add_argument("--seed", type=int, default=42,
                   help="Zufallsstartwert (Reproduzierbarkeit)")
    return p


def starte(main) -> None:
    """Fuehrt ``main`` aus; Eingabefehler und gesperrte Dateien als Klartext.

    Unter Windows ist die haeufigste Ursache fuer "Permission denied" eine
    Ergebnisdatei, die noch in Excel offen ist.
    """
    import sys

    from .data_io import DatenFehler

    try:
        main()
    except DatenFehler as fehler:
        sys.exit("Fehler: %s" % fehler)
    except PermissionError as fehler:
        sys.exit("Fehler: Datei gesperrt - ist sie noch in Excel geoeffnet?\n  %s"
                 % (fehler.filename or fehler))


def blatt_wert(wert):
    """'0' -> 0, 'Messdaten' -> 'Messdaten'."""
    try:
        return int(wert)
    except (TypeError, ValueError):
        return wert


def ausgabeordner(pfad: str) -> Path:
    p = Path(pfad)
    p.mkdir(parents=True, exist_ok=True)
    return p
