"""Einheitliches Look-and-Feel fuer alle Plots.

Die Farben sind ein validiertes, farbfehlsichtigkeits-sicheres Set. Fuer
Streudiagramme (alle Paare gleichzeitig sichtbar) tragen nur die ersten drei
Slots die volle Pruefung - deshalb wird Farbe hier IMMER mit einer zweiten
Codierung (Markerform) kombiniert, damit die Identitaet nie allein an der
Farbe haengt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Kategoriale Palette, feste Reihenfolge - nie zyklisch weiterdrehen.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#8a8983",
        "grid": "#e6e5e1",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "diverging": ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                      "#f0a3a2", "#d03b3b", "#8c1f1f"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "muted": "#8a8983",
        "grid": "#333330",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "diverging": ["#104281", "#2a78d6", "#86b6ef", "#383835",
                      "#e08a89", "#d03b3b", "#8c1f1f"],
    },
}

# Zweite Codierung neben der Farbe (gleiche feste Reihenfolge).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def set_style(theme: str = "light") -> dict:
    """Setzt die rcParams und liefert das aktive Theme-Dict zurueck."""
    t = THEMES[theme]
    mpl.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": t["text"],
        "axes.labelcolor": t["text2"],
        "axes.titlecolor": t["text"],
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.edgecolor": t["grid"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": t["text2"],
        "ytick.color": t["text2"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
    })
    return t


def scatter_nach_gruppe(ax, x, y, labels=None, theme: dict | None = None,
                        groesse: float = 46, alpha: float = 0.9,
                        neutral: str | None = None, alle=None):
    """Streudiagramm, eingefaerbt nach Gruppe - Farbe plus Markerform.

    ``labels`` darf ``None`` sein (dann eine einzige Serie, keine Legende).
    ``neutral`` benennt eine Gruppe, die keine Farbe verdient (z. B. Punkte
    ohne Clusterzuordnung): sie wird grau und im Hintergrund gezeichnet und
    belegt keinen Farbplatz.
    ``alle`` legt die Farbplaetze fest, wenn mehrere Plots dieselben Gruppen
    zeigen, aber nicht jeder alle enthaelt - sonst verrutschen die Farben.
    Liefert die Liste der gezeichneten Gruppennamen.
    """
    t = theme or THEMES["light"]
    if labels is None:
        ax.scatter(x, y, s=groesse, c=t["series"][0], marker="o",
                   alpha=alpha, linewidths=0.8, edgecolors=t["surface"],
                   zorder=3)
        return []

    # Alphabetisch sortiert: so bekommt eine Gruppe ueber alle Plots und
    # Laeufe hinweg dieselbe Farbe, unabhaengig von der Zeilenreihenfolge.
    labels = [str(v) for v in labels]
    plaetze = sorted(set(labels if alle is None else map(str, alle)) - {neutral})
    if len(plaetze) > len(t["series"]):
        # Ab Slot 9 keine neuen Farben erfinden: Rest wird zu "Sonstige".
        kern = plaetze[: len(t["series"]) - 1]
        labels = [g if g in kern or g == neutral else "Sonstige" for g in labels]
        plaetze = kern + ["Sonstige"]

    vorhanden = set(labels)
    gruppen = [g for g in plaetze if g in vorhanden]
    for g in gruppen:
        i = plaetze.index(g)
        maske = [lab == g for lab in labels]
        ax.scatter([xi for xi, m in zip(x, maske) if m],
                   [yi for yi, m in zip(y, maske) if m],
                   s=groesse, c=t["series"][i % len(t["series"])],
                   marker=MARKERS[i % len(MARKERS)], alpha=alpha,
                   linewidths=0.8, edgecolors=t["surface"], zorder=3,
                   label=str(g))
    if neutral is not None and neutral in labels:
        maske = [lab == neutral for lab in labels]
        ax.scatter([xi for xi, m in zip(x, maske) if m],
                   [yi for yi, m in zip(y, maske) if m],
                   s=groesse * 0.6, c=t["muted"], marker="o", alpha=0.5,
                   linewidths=0, zorder=2, label=neutral)
        gruppen = gruppen + [neutral]
    return gruppen


def gitter(ax, t: dict, achse: str = "both") -> None:
    """Zurueckhaltendes Gitter hinter den Datenpunkten."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=achse, color=t["grid"], linewidth=0.8, zorder=0)


def rahmen(ax, t: dict, ticks_innen: bool = True) -> None:
    """Geschlossener Rahmen ringsum - der klassische Look wissenschaftlicher Plots.

    Im hellen Schema schwarz, im dunklen weiss: ein fest schwarzer Rahmen waere
    auf dunklem Grund unsichtbar.
    """
    for seite in ("top", "right", "bottom", "left"):
        ax.spines[seite].set_visible(True)
        ax.spines[seite].set_color(t["text"])
        ax.spines[seite].set_linewidth(1.1)
    if ticks_innen:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True,
                       color=t["text"], width=0.9)
        ax.tick_params(which="major", length=5.5)
        ax.tick_params(which="minor", length=3.0)


def kopf(ax, titel: str, unter: str, t: dict) -> None:
    """Titel plus Erklaerzeile - die Zeile traegt den Kontext, den die Achsen nicht zeigen."""
    ax.set_title(unter, loc="left", fontsize=9.5, color=t["text2"],
                 fontweight="normal", pad=10)
    ax.annotate(titel, xy=(0, 1), xytext=(0, 28), xycoords="axes fraction",
                textcoords="offset points", fontsize=13, fontweight="semibold",
                color=t["text"], va="bottom", ha="left", annotation_clip=False)


def speichere(fig, pfad: Path, still: bool = False, versuche: int = 3) -> Path:
    """Speichert die Grafik; wiederholt bei kurzzeitig gesperrter Datei.

    Unter Windows greifen Virenscanner, Vorschaufenster oder der Indexdienst
    gern genau dann auf die frisch geschriebene Datei zu - ein zweiter Anlauf
    nach einem Moment genuegt.
    """
    import time

    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    for versuch in range(versuche):
        try:
            fig.savefig(pfad)
            break
        except OSError:
            if versuch == versuche - 1:
                raise
            time.sleep(0.5)
    if not still:
        print(f"  gespeichert: {pfad}")
    return pfad


def zeige_oder_schliesse(fig, zeigen: bool) -> None:
    if zeigen:
        plt.show()
    else:
        plt.close(fig)
