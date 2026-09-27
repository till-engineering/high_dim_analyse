"""Gemeinsame Kommandozeilen-Optionen aller Analyse-Skripte.

Alle Optionen lassen sich auch in einer Konfigurationsdatei setzen
(Standard: config.toml im Projektordner, siehe dort). Reihenfolge, das
Spaetere gewinnt:  Programm-Standard < [allgemein] < [gruppen] < [<skript>]
< Kommandozeile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CONFIG_STANDARD = Path(__file__).resolve().parent.parent / "config.toml"
# Abschnitte, die fuer mehrere Skripte gelten: Schluessel, die ein Skript
# nicht kennt, werden dort still uebergangen (z. B. min_gruppe bei der Streumatrix).
GRUPPEN_SCHLUESSEL = {"cluster", "gruppen", "min_gruppe", "stabilitaet", "cluster_varianz"}
SKRIPT_ABSCHNITTE = ["pca", "umap", "tsne", "vergleich", "streumatrix", "konstanz",
                     "ablauf"]
# Relativ zum Ordner der Konfigurationsdatei, nicht zum Aufrufort.
PFAD_SCHLUESSEL = {"datei", "out"}
# In TOML gibt es kein "leer" - "" oder "auto" heisst: Programm-Standard.
AUTOMATISCH = {"", "auto"}


def basis_parser(beschreibung: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=beschreibung,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-c", "--config", default=None,
                   help="Konfigurationsdatei (TOML); ohne Angabe config.toml im "
                        "Projektordner, falls vorhanden; 'keine' = ohne")
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


def lade_config(pfad: str | None) -> tuple[dict, Path | None]:
    """Liest die TOML-Datei; ({}, None), wenn keine verwendet wird."""
    from .data_io import DatenFehler

    if pfad is None:
        if not CONFIG_STANDARD.is_file():
            return {}, None
        datei = CONFIG_STANDARD
    elif pfad.strip().lower() == "keine":
        return {}, None
    else:
        datei = Path(pfad).expanduser().resolve()
        if not datei.is_file():
            raise DatenFehler(f"--config: Datei nicht gefunden: {datei}")
    try:
        import tomllib
    except ImportError:              # Python 3.10
        import tomli as tomllib
    try:
        with open(datei, "rb") as f:
            return tomllib.load(f), datei
    except tomllib.TOMLDecodeError as fehler:
        raise DatenFehler(f"{datei.name}: {fehler}\n  Windows-Pfade in einfache "
                          "Anfuehrungszeichen setzen: datei = 'C:\\Daten\\x.xlsx'")


def _wandle(aktion: argparse.Action, wert, ordner: Path, ort: str):
    """Einen Wert aus der TOML-Datei so pruefen und umwandeln wie argparse."""
    from .data_io import DatenFehler

    if isinstance(aktion, argparse._StoreTrueAction):
        if not isinstance(wert, bool):
            raise DatenFehler(f"{ort}: true oder false erwartet, nicht {wert!r}")
        return wert
    # "auto" als echte Auswahl (z. B. --anteile auto) bleibt stehen.
    if (isinstance(wert, str) and wert.strip().lower() in AUTOMATISCH
            and not (aktion.choices and wert in aktion.choices)):
        return None

    def eins(w):
        if isinstance(w, (list, dict)):
            raise DatenFehler(f"{ort}: einzelner Wert erwartet, nicht {w!r}")
        if aktion.type is not None:
            try:
                w = aktion.type(w)
            except (TypeError, ValueError):
                raise DatenFehler(f"{ort}: {w!r} passt nicht (erwartet "
                                  f"{aktion.type.__name__})") from None
        if aktion.choices is not None and w not in aktion.choices:
            raise DatenFehler(f"{ort}: {w!r} nicht erlaubt, moeglich: "
                              + ", ".join(map(str, aktion.choices)))
        return w

    if aktion.nargs in ("*", "+"):
        werte = [eins(w) for w in (wert if isinstance(wert, list) else [wert])]
        if aktion.nargs == "+" and not werte:
            raise DatenFehler(f"{ort}: mindestens ein Wert noetig")
        return werte
    if isinstance(aktion.nargs, int) and aktion.nargs > 1:
        if not isinstance(wert, list) or len(wert) != aktion.nargs:
            raise DatenFehler(f"{ort}: Liste mit {aktion.nargs} Werten erwartet, "
                              f"z. B. [\"A\", \"B\"]")
        return [eins(w) for w in wert]
    wert = eins(wert)
    if aktion.dest in PFAD_SCHLUESSEL:
        pfad = Path(str(wert)).expanduser()
        wert = str(pfad if pfad.is_absolute() else ordner / pfad)
    return wert


def lies_argumente(p: argparse.ArgumentParser, abschnitt: str) -> argparse.Namespace:
    """Wie ``p.parse_args()``, vorher Standardwerte aus der Konfigurationsdatei.

    Gelesen werden [allgemein], [gruppen] und [<abschnitt>]. Tippfehler in
    Abschnitts- oder Schluesselnamen fuehren zu einer Fehlermeldung, statt
    still ignoriert zu werden.
    """
    from .data_io import DatenFehler

    vorab = argparse.ArgumentParser(add_help=False)
    vorab.add_argument("-c", "--config", default=None)
    config, datei = lade_config(vorab.parse_known_args()[0].config)

    if datei:
        unbekannt = sorted(set(config) - {"allgemein", "gruppen", *SKRIPT_ABSCHNITTE})
        if unbekannt:
            raise DatenFehler(f"{datei.name}: unbekannte Abschnitte {', '.join(unbekannt)}. "
                              f"Moeglich: allgemein, gruppen, {', '.join(SKRIPT_ABSCHNITTE)}")
        aktionen = {a.dest: a for a in p._actions if a.dest not in ("help", "config")}
        allgemein = {a.dest for a in basis_parser("")._actions} - {"help", "config"}
        werte = {}
        for name in ("allgemein", "gruppen", abschnitt):
            eintraege = config.get(name, {})
            if not isinstance(eintraege, dict):
                raise DatenFehler(f"{datei.name}: [{name}] muss ein Abschnitt sein")
            erlaubt = {"allgemein": allgemein, "gruppen": GRUPPEN_SCHLUESSEL}.get(name)
            for schluessel, wert in eintraege.items():
                dest = schluessel.replace("-", "_")
                ort = f"{datei.name} [{name}] {schluessel}"
                if erlaubt is not None and dest not in erlaubt:
                    raise DatenFehler(f"{ort}: unbekannter Schluessel. Moeglich: "
                                      + ", ".join(sorted(erlaubt)))
                if dest not in aktionen:
                    if erlaubt is not None:
                        continue             # gilt fuer ein anderes Skript
                    raise DatenFehler(f"{ort}: unbekannter Schluessel. Moeglich: "
                                      + ", ".join(sorted(set(aktionen) - allgemein)))
                werte[dest] = _wandle(aktionen[dest], wert, datei.parent, ort)
        p.set_defaults(**werte)

    args = p.parse_args()
    args.config_datei = datei
    args.config_daten = config
    if datei:
        print(f"Konfiguration: {datei}")
    return args


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
