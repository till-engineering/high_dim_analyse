"""Einlesen und Vorbereiten der Excel-Daten.

Erwartetes Format: eine Kopfzeile mit den Spaltennamen, darunter die Daten.
Numerische Spalten werden als Merkmale (Features) verwendet, Text-Spalten
gelten als Beschriftung/Gruppe und werden zum Einfaerben der Plots genutzt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class DatenFehler(Exception):
    """Etwas an der Eingabedatei oder den Spaltenangaben passt nicht.

    Wird von :func:`hda.cli.starte` als kurze Meldung ausgegeben statt als
    Traceback - der Fehler liegt dann in der Eingabe, nicht im Programm.
    """


@dataclass
class Datensatz:
    """Ergebnis von :func:`lade_daten`."""

    X: pd.DataFrame                       # nur numerische Spalten
    labels: pd.Series | None = None       # Gruppenspalte oder None
    label_name: str | None = None
    quelle: Path | None = None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)  # uebrige Textspalten
    zeilen: pd.Series | None = None       # Zeilennummer in der Excel-Datei
    roh: pd.DataFrame = field(default_factory=pd.DataFrame)   # alle Spalten, Luecken ungefuellt

    @property
    def merkmale(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n(self) -> int:
        return len(self.X)

    def __str__(self) -> str:
        z = f"{self.n} Zeilen x {len(self.merkmale)} numerische Spalten"
        if self.label_name:
            z += f" | Gruppenspalte '{self.label_name}' ({self.labels.nunique()} Gruppen)"
        return z


def finde_excel(ordner: str | Path = "data") -> Path:
    """Sucht die (einzige/neueste) Excel-Datei im Datenordner."""
    ordner = Path(ordner)
    treffer = sorted(
        [p for p in ordner.glob("*.xls*") if not p.name.startswith("~$")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not treffer:
        raise DatenFehler(
            f"Keine Excel-Datei in '{ordner}' gefunden. "
            "Mit 'python generate_testdata.py' laesst sich eine Beispieldatei erzeugen."
        )
    return treffer[0]


def _waehle_label(df: pd.DataFrame, text_spalten: list[str]) -> str | None:
    """Waehlt die plausibelste Gruppenspalte: wenige, wiederkehrende Werte.

    Eine ID-Spalte hat so viele Auspraegungen wie Zeilen und faellt damit raus,
    eine Zustands-/Klassenspalte hat eine Handvoll.
    """
    n = len(df)
    kandidaten = [(df[c].nunique(dropna=True), c) for c in text_spalten]
    passend = [(k, c) for k, c in kandidaten if 2 <= k <= max(2, n // 2)]
    if not passend:
        return None
    return min(passend)[1]


def lade_daten(pfad: str | Path | None = None, blatt: str | int = 0,
               label_spalte: str | None = None, id_spalten: list[str] | None = None,
               nan: str = "median", still: bool = False) -> Datensatz:
    """Liest die Excel-Datei ein und trennt Merkmale von Beschriftungen.

    Parameter
    ---------
    pfad : Pfad zur Excel-Datei; ohne Angabe die neueste Datei in ``data/``.
    blatt : Name oder Index des Tabellenblatts.
    label_spalte : Spalte zum Einfaerben. ``None`` = Textspalte mit den wenigsten
        verschiedenen Werten (siehe :func:`_waehle_label`).
    id_spalten : Spalten, die trotz Zahlenwerten keine Merkmale sind (z. B. IDs).
    nan : ``"median"`` (fehlende Werte ersetzen) oder ``"drop"`` (Zeilen verwerfen).
    """
    pfad = Path(pfad) if pfad else finde_excel()
    df = pd.read_excel(pfad, sheet_name=blatt, header=0)
    df.columns = [str(c).strip() for c in df.columns]

    # Spalten, die als Text eingelesen wurden, aber Zahlen enthalten, retten
    # (z. B. "1,23" aus einer deutschen Excel-Einstellung).
    for spalte in df.columns:
        # Nicht auf dtype == object pruefen: neuere pandas-Versionen lesen
        # Textspalten als eigenen 'str'-Typ ein.
        if not pd.api.types.is_numeric_dtype(df[spalte]):
            versuch = pd.to_numeric(
                df[spalte].astype(str).str.replace(",", ".", regex=False).str.strip(),
                errors="coerce")
            if versuch.notna().mean() > 0.9:
                df[spalte] = versuch

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    ausschluss = set(id_spalten or [])
    unbekannt = sorted(ausschluss - set(df.columns))
    if unbekannt:
        # Ein Tippfehler hier wuerde sonst still ignoriert, und die Spalte
        # liefe weiter als Messgroesse mit.
        raise DatenFehler(f"--id-spalten: unbekannte Spalte(n) {', '.join(unbekannt)}. "
                          f"Vorhanden: {', '.join(df.columns)}")
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c]) and c not in ausschluss]
    text = [c for c in df.columns if c not in num]

    if label_spalte is not None:
        if label_spalte not in df.columns:
            raise DatenFehler(f"--label-spalte: Spalte '{label_spalte}' existiert "
                              f"nicht. Vorhanden: {', '.join(df.columns)}")
        num = [c for c in num if c != label_spalte]
        text = [c for c in df.columns if c not in num]
    elif text:
        label_spalte = _waehle_label(df, text)

    X = df[num].copy()
    if nan == "drop":
        behalten = X.notna().all(axis=1)
    else:
        behalten = pd.Series(True, index=X.index)
        X = X.fillna(X.median(numeric_only=True))
    verloren = int((~behalten).sum())
    X, df = X[behalten], df[behalten]

    # Konstante Spalten fliegen raus - sie tragen keine Varianz und
    # wuerden bei der Standardisierung durch Null teilen.
    konstant = [c for c in X.columns if np.isclose(X[c].std(ddof=0), 0.0)]
    if konstant:
        X = X.drop(columns=konstant)
    # Erst nach dem Entfernen pruefen - sonst scheitert es spaeter irgendwo
    # in der PCA mit einem unverstaendlichen Indexfehler.
    if X.shape[1] < 2:
        raise DatenFehler(f"Mindestens 2 veraenderliche Zahlenspalten noetig, "
                          f"gefunden: {list(X.columns) or 'keine'}"
                          + (f" (konstant entfernt: {', '.join(konstant)})" if konstant else ""))
    if len(X) < 3:
        raise DatenFehler(f"Zu wenige Zeilen ({len(X)}) fuer eine Analyse")

    # Leere Zellen der Gruppenspalte als eigene, lesbare Gruppe statt "nan"
    labels = (df[label_spalte].astype(object).where(df[label_spalte].notna(), "(leer)")
              .astype(str) if label_spalte else None)
    # Kopfzeile ist Excel-Zeile 1, Index 0 also Zeile 2 - so bleibt jede Zeile
    # nach dem Verwerfen von Luecken bis in die Originaldatei zurueckverfolgbar.
    zeilen = pd.Series(df.index + 2, name="Excel_Zeile")
    # Auch ohne Textspalten ein DataFrame mit den Zeilen - sonst scheitert
    # spaeter jedes pd.concat mit den Zeilennummern.
    meta = df[[c for c in text if c != label_spalte]]

    ds = Datensatz(X=X.reset_index(drop=True),
                   labels=labels.reset_index(drop=True) if labels is not None else None,
                   label_name=label_spalte, quelle=pfad, meta=meta.reset_index(drop=True),
                   zeilen=zeilen.reset_index(drop=True), roh=df.reset_index(drop=True))

    if not still:
        print(f"Datei     : {pfad}")
        print(f"Datensatz : {ds}")
        if konstant:
            print(f"  Hinweis : konstante Spalten entfernt: {', '.join(konstant)}")
        if verloren:
            print(f"  Hinweis : {verloren} Zeilen mit Luecken verworfen")
    return ds


def skaliere(X: pd.DataFrame, methode: str = "zscore"):
    """Standardisiert die Merkmale.

    Bei PCA/UMAP/t-SNE ist das fast immer noetig: ohne Skalierung dominiert
    schlicht die Spalte mit den groessten Zahlenwerten (z. B. Drehzahl in rpm).
    """
    if methode in (None, "keine", "none"):
        return X.to_numpy(dtype=float), None
    if methode == "zscore":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif methode == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif methode == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unbekannte Skalierung: {methode}")
    return scaler.fit_transform(X.to_numpy(dtype=float)), scaler
