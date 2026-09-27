"""Streumatrix nur mit Punkten, bei denen alle uebrigen Groessen konstant bleiben.

Die normale Streumatrix zeigt jede Groesse gegen jede - aber in jedem Feld
aendern sich zugleich alle anderen Groessen mit, der Zusammenhang ist
vermischt. Hier zeigt Feld (x, y) nur Serien von Zeilen, in denen sich
ausser x und y nichts aendert (im Rahmen von --toleranz). Innerhalb einer
Serie ist die Steigung also der Einfluss von x auf y bei sonst gleichen
Bedingungen - soweit die Daten das hergeben.

Serien finden (je Feld):
  1. Alle Groessen ausser x und y auf ihre Standardabweichung skalieren.
  2. Zeilen, die sich in KEINER dieser Groessen um mehr als --toleranz
     (Anteil der Standardabweichung) unterscheiden, bilden eine Serie -
     jedes Paar innerhalb der Serie, nicht nur Nachbarn (Complete Linkage).
     Sonst ketten sich Zeilen, die langsam driften, zu einer "konstanten"
     Serie zusammen.
  3. Eine Serie braucht mindestens --min-punkte Zeilen, und x oder y muss
     sich tatsaechlich aendern. Bei Anteilen zaehlen nur Zeilen, die y
     enthalten (y > 0).

Anteile (--anteile): Summieren sich die Groessen je Zeile zu 100 (z. B.
Masse-% einer Legierung), kann eine Groesse nicht steigen, ohne dass die
anderen sinken - streng konstant bleibt dann fast nichts. Mit Anteilen gilt
"die uebrigen bleiben konstant" im Verhaeltnis zueinander: sie werden ohne x
und y wieder auf 100 normiert und erst dann verglichen. AlxCoCrFeNi mit
steigendem x ist so eine Serie. Achtung: zwischen zwei Anteilen ist die
Steigung dann weitgehend Rechnung (was x gewinnt, verlieren die anderen).

Gezeichnet wird das untere Dreieck: Zeile = y, Spalte = x. Grau im
Hintergrund alle Zeilen, farbig und verbunden die Serien (nach x sortiert).
Die Zahl im Feld ist die Anzahl der Serien.

Erzeugt im Ausgabeordner:
  konstanz_matrix.png           die Matrix
  konstanz_<x>_<y>.png          mit --paar: ein Feld gross, mit Legende
  konstanz_serien.xlsx          Uebersicht je Feld, jede Serie mit Steigung,
                                jeder Punkt mit Excel-Zeile. Steigung dy/dx nur,
                                wenn sich x ueber die Toleranz hinaus aendert;
                                r nur, wenn sich x und y aendern.

Aufruf:  python run_konstanz.py
         python run_konstanz.py --toleranz 0.05 --spalten A B C D
         python run_konstanz.py --paar Al_Masse% Ni_Masse%
"""

from __future__ import annotations

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from hda.cli import ausgabeordner, basis_parser, blatt_wert, lies_argumente, starte
from hda.data_io import DatenFehler, lade_daten
from hda.plotstyle import kopf, rahmen, set_style, speichere, zeige_oder_schliesse

NAME_KLEIN = 4.6          # Achsennamen in jedem Feld, pt


# --------------------------------------------------------------------------- #
# Serien finden
# --------------------------------------------------------------------------- #
def ist_anteil(W: np.ndarray) -> bool:
    """Summieren sich alle Zeilen zu 100 (oder 1)?"""
    summe = W.sum(axis=1)
    return bool(np.allclose(summe, 100, atol=0.5) or np.allclose(summe, 1, atol=0.005))


def serien(W: np.ndarray, x: int, y: int, toleranz: float, anteile: bool,
           min_punkte: int) -> list[np.ndarray]:
    """Zeilengruppen, in denen sich ausser Spalte x und y nichts aendert.

    Liefert je Serie die Zeilenindizes, nach x sortiert; groesste Serie zuerst.
    """
    rest = [c for c in range(W.shape[1]) if c not in (x, y)]
    Z = W[:, rest]
    if anteile:
        summe = Z.sum(axis=1, keepdims=True)
        Z = np.divide(Z, summe, out=np.zeros_like(Z), where=summe > 0) * 100
    sd = Z.std(axis=0)
    Z = Z / np.where(sd > 0, sd, 1.0)

    n = len(Z)
    paare = cKDTree(Z).query_pairs(toleranz, p=np.inf, output_type="ndarray")
    if len(paare) == 0:
        return []
    netz = coo_matrix((np.ones(len(paare)), (paare[:, 0], paare[:, 1])), shape=(n, n))
    _, komponente = connected_components(netz, directed=False)

    sd_x, sd_y = W[:, x].std(), W[:, y].std()
    gefunden = []
    for k in np.unique(komponente):
        idx = np.flatnonzero(komponente == k)
        if len(idx) < min_punkte:
            continue
        # Nachbarschaft allein kettet; erst Complete Linkage stellt sicher,
        # dass JEDES Paar der Serie innerhalb der Toleranz liegt.
        if np.ptp(Z[idx], axis=0).max() > toleranz:
            teil = fcluster(linkage(Z[idx], "complete", metric="chebyshev"),
                            toleranz, criterion="distance")
            gruppen = [idx[teil == t] for t in np.unique(teil)]
        else:
            gruppen = [idx]
        for g in gruppen:
            # Bei Anteilen ist "y enthalten" und "y fehlt" ein qualitativer
            # Unterschied - sonst springt die Serie zwischen einer Legierung
            # ohne y und einer mit y hin und her. Ohne y gibt es keinen
            # Einfluss auf y zu sehen, dieser Teil faellt weg.
            if anteile:
                g = g[W[g, y] > 0]
            if len(g) < min_punkte:
                continue
            # Mindestens eine der beiden muss sich aendern - welche auf der
            # x-Achse steht, ist nur eine Frage der Spaltenreihenfolge.
            if (np.ptp(W[g, x]) <= toleranz * sd_x
                    and np.ptp(W[g, y]) <= toleranz * sd_y):
                continue
            gefunden.append(g[np.argsort(W[g, x], kind="stable")])
    return sorted(gefunden, key=lambda g: (-len(g), g[0]))


def alle_serien(W, toleranz, anteile, min_punkte) -> dict:
    """{(x, y): [serie, ...]} fuer jedes Paar x < y."""
    return {(i, j): serien(W, i, j, toleranz, anteile, min_punkte)
            for i, j in itertools.combinations(range(W.shape[1]), 2)}


def kennwerte(xw: np.ndarray, yw: np.ndarray, grenze_x: float, grenze_y: float):
    """Was aendert sich in der Serie, Steigung dy/dx und r.

    Aendert sich x nur im Rahmen der Toleranz, ist dy/dx Rauschen geteilt
    durch Rauschen - dann keine Steigung. r braucht Aenderung in beiden.
    """
    dx, dy = np.ptp(xw) > grenze_x, np.ptp(yw) > grenze_y
    aendert = "x und y" if dx and dy else ("x" if dx else "y")
    s = float(np.polyfit(xw, yw, 1)[0]) if dx else np.nan
    r = float(np.corrcoef(xw, yw)[0, 1]) if dx and dy else np.nan
    return aendert, s, r


# --------------------------------------------------------------------------- #
# Zeichnen
# --------------------------------------------------------------------------- #
def zeichne_feld(ax, W, x, y, gefunden, t, groesse_bg=4, groesse=9, linie=0.9):
    """Grauer Hintergrund aus allen Zeilen, darueber die Serien."""
    ax.scatter(W[:, x], W[:, y], s=groesse_bg, c=t["muted"], alpha=0.22,
               linewidths=0, zorder=1)
    farben = t["series"]
    for k, g in enumerate(gefunden):
        f = farben[k % len(farben)]
        ax.plot(W[g, x], W[g, y], "-", color=f, linewidth=linie, alpha=0.85, zorder=3)
        ax.scatter(W[g, x], W[g, y], s=groesse, c=f, linewidths=0.5,
                   edgecolors=t["surface"], zorder=4)


def matrix(W, namen, treffer, t, quelle, toleranz, anteile):
    """Unteres Dreieck ohne Diagonale: Zeile = y, Spalte = x."""
    m = len(namen)
    feld = 1.45 if m <= 8 else 1.2
    fig, achsen = plt.subplots(m - 1, m - 1, figsize=(feld * (m - 1) + 1.5,
                                                      feld * (m - 1) + 1.6),
                               squeeze=False)
    klein = dict(fontsize=NAME_KLEIN, color=t["text2"], zorder=6,
                 bbox=dict(boxstyle="round,pad=0.1", linewidth=0,
                           facecolor=t["surface"], alpha=0.75))
    for zeile in range(m - 1):
        i = zeile + 1                       # y-Groesse
        for j in range(m - 1):              # x-Groesse
            ax = achsen[zeile, j]
            if j >= i:
                ax.axis("off")
                continue
            gefunden = treffer[(j, i)]
            zeichne_feld(ax, W, j, i, gefunden, t)
            rahmen(ax, t, ticks_innen=False)
            if not gefunden:
                for seite in ax.spines.values():
                    seite.set_color(t["grid"])
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.text(0.04, 0.95, "%d Serie%s" % (len(gefunden), "" if len(gefunden) == 1 else "n")
                    if gefunden else "keine", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color=t["text"] if gefunden else t["muted"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", linewidth=0,
                              facecolor=t["surface"], alpha=0.8))
            ax.text(0.5, 0.015, namen[j], transform=ax.transAxes,
                    ha="center", va="bottom", **klein)
            ax.text(0.015, 0.5, namen[i], transform=ax.transAxes, rotation=90,
                    ha="left", va="center", **klein)
            if zeile == m - 2:
                ax.set_xlabel(namen[j], fontsize=7.5, rotation=35, ha="right",
                              rotation_mode="anchor", labelpad=2)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(namen[i], fontsize=7.5, rotation=0, ha="right",
                              va="center", labelpad=4)
            else:
                ax.set_yticklabels([])
            ax.locator_params(nbins=3)

    breite, hoehe = fig.get_size_inches()
    oben = lambda zoll: 1.0 - zoll / hoehe
    links = lambda zoll: zoll / breite
    fig.text(links(0.12), oben(0.10), "Nur eine Groesse aendert sich", ha="left",
             va="top", fontsize=15, fontweight="semibold", color=t["text"])
    fig.text(links(0.12), oben(0.42),
             "Farbig: Serien, in denen alle Groessen ausser x und y konstant sind "
             "(Toleranz %.3g sd%s) · grau: alle Zeilen · Zeile = y, Spalte = x"
             % (toleranz, ", uebrige im gleichen Verhaeltnis" if anteile else ""),
             ha="left", va="top", fontsize=10, color=t["text2"])
    if quelle:
        fig.text(0.01, 0.0, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout(rect=(0, 0.01, 1, oben(0.75)), h_pad=0.25, w_pad=0.25)
    return fig


def einzelfeld(W, namen, x, y, gefunden, zeilen, t, quelle, toleranz, anteile):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    zeichne_feld(ax, W, x, y, gefunden, t, groesse_bg=14, groesse=34, linie=1.6)
    rahmen(ax, t)
    ax.set_xlabel(namen[x])
    ax.set_ylabel(namen[y])
    farben = t["series"]
    for k, g in enumerate(gefunden[:12]):
        ax.plot([], [], "-o", color=farben[k % len(farben)], markersize=5,
                label="Serie %d · %d Punkte · Zeilen %s" % (
                    k + 1, len(g), ", ".join(map(str, zeilen[g][:4]))
                    + (" ..." if len(g) > 4 else "")))
    if len(gefunden) > 12:
        ax.plot([], [], " ", label="+ %d weitere Serien (siehe Excel)" % (len(gefunden) - 12))
    if gefunden:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    kopf(ax, "%s gegen %s bei sonst gleichen Bedingungen" % (namen[y], namen[x]),
         "%d Serien · Toleranz %.3g sd%s · grau: alle Zeilen"
         % (len(gefunden), toleranz, " · uebrige im gleichen Verhaeltnis" if anteile else ""),
         t)
    if quelle:
        fig.text(0.01, 0.0, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def tabellen(W, namen, treffer, ds, toleranz):
    sd = W.std(axis=0)
    uebersicht, serien_zeilen, punkte = [], [], []
    for (x, y), gefunden in treffer.items():
        werte = [kennwerte(W[g, x], W[g, y], toleranz * sd[x], toleranz * sd[y])
                 for g in gefunden]
        steig = np.array([s for _, s, _ in werte], dtype=float)
        mit = steig[~np.isnan(steig)]
        uebersicht.append({
            "x": namen[x], "y": namen[y], "Serien": len(gefunden),
            "Punkte": int(sum(len(g) for g in gefunden)),
            "Serien_mit_Steigung": len(mit),
            "Steigung_Median": np.median(mit) if len(mit) else np.nan,
            "Steigung_min": mit.min() if len(mit) else np.nan,
            "Steigung_max": mit.max() if len(mit) else np.nan,
        })
        for k, (g, (aendert, s, r)) in enumerate(zip(gefunden, werte), start=1):
            eintrag = {"x": namen[x], "y": namen[y], "Serie": k, "Punkte": len(g),
                       "aendert_sich": aendert,
                       "x_von": W[g, x].min(), "x_bis": W[g, x].max(),
                       "y_von": W[g, y].min(), "y_bis": W[g, y].max(),
                       "Steigung_dy_dx": s, "r": r,
                       "Excel_Zeilen": ", ".join(map(str, ds.zeilen.to_numpy()[g]))}
            if ds.labels is not None:
                eintrag[ds.label_name] = ", ".join(sorted(set(ds.labels.to_numpy()[g])))
            serien_zeilen.append(eintrag)
            for zeile in g:
                p = {"x": namen[x], "y": namen[y], "Serie": k,
                     "Excel_Zeile": int(ds.zeilen.iloc[zeile]),
                     "x_Wert": W[zeile, x], "y_Wert": W[zeile, y]}
                if ds.labels is not None:
                    p[ds.label_name] = ds.labels.iloc[zeile]
                for spalte in ds.meta.columns[:2]:
                    p[spalte] = ds.meta[spalte].iloc[zeile]
                punkte.append(p)
    uebersicht = pd.DataFrame(uebersicht).sort_values(["Serien", "Punkte"],
                                                      ascending=False)
    return uebersicht, pd.DataFrame(serien_zeilen), pd.DataFrame(punkte)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = basis_parser("Streumatrix nur mit Zeilen, in denen alle uebrigen Groessen "
                     "konstant bleiben.")
    p.add_argument("--toleranz", type=float, default=0.1,
                   help="'konstant' = Unterschied hoechstens so viele "
                        "Standardabweichungen der jeweiligen Groesse (0 = exakt gleich)")
    p.add_argument("--min-punkte", type=int, default=3,
                   help="kleinste Serie (2 = jede Zweiergruppe zaehlt)")
    p.add_argument("--anteile", default="auto", choices=["auto", "ja", "nein"],
                   help="Groessen sind Anteile mit Summe 100: uebrige im gleichen "
                        "Verhaeltnis statt gleichen Werten; auto = erkennen")
    p.add_argument("--spalten", nargs="+", default=None,
                   help="nur diese Groessen zeigen (gesucht wird trotzdem mit allen)")
    p.add_argument("--max-groessen", type=int, default=12,
                   help="ohne --spalten: hoechstens so viele Groessen zeigen, die "
                        "mit den meisten Serien")
    p.add_argument("--paar", nargs=2, default=None, metavar=("X", "Y"),
                   help="zusaetzlich dieses Feld gross zeichnen, mit Legende")
    args = lies_argumente(p, "konstanz")
    if args.toleranz < 0:
        p.error("--toleranz darf nicht negativ sein")
    if args.min_punkte < 2:
        p.error("--min-punkte muss mindestens 2 sein")

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    namen = list(ds.X.columns)
    W = ds.X.to_numpy(dtype=float)
    anteile = ist_anteil(W) if args.anteile == "auto" else args.anteile == "ja"
    # Exakt gleich soll auch bei Rundungsrauschen gleich bleiben.
    toleranz = max(args.toleranz, 1e-9)

    print("\nSerien suchen: %d Groessen -> %d Paare, Toleranz %.3g sd%s"
          % (len(namen), len(namen) * (len(namen) - 1) // 2, args.toleranz,
             ", Anteile (uebrige im gleichen Verhaeltnis)" if anteile else ""))
    treffer = alle_serien(W, toleranz, anteile, args.min_punkte)
    mit = sum(1 for g in treffer.values() if g)
    print("  %d von %d Paaren mit Serien, %d Serien insgesamt"
          % (mit, len(treffer), sum(len(g) for g in treffer.values())))
    if not mit:
        print("  Hinweis : keine Serien - --toleranz erhoehen oder --min-punkte senken")

    uebersicht, serien_tab, punkte = tabellen(W, namen, treffer, ds, toleranz)
    ziel = out / "konstanz_serien.xlsx"
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        uebersicht.to_excel(writer, sheet_name="Uebersicht", index=False)
        serien_tab.to_excel(writer, sheet_name="Serien", index=False)
        punkte.to_excel(writer, sheet_name="Punkte", index=False)
    print("  gespeichert: %s" % ziel)

    # Welche Groessen zeigen
    if args.spalten:
        fehlt = [s for s in args.spalten if s not in namen]
        if fehlt:
            raise DatenFehler("--spalten: unbekannt %s. Vorhanden: %s"
                              % (", ".join(fehlt), ", ".join(namen)))
        zeigen = [namen.index(s) for s in args.spalten]
    else:
        je_groesse = np.zeros(len(namen))
        for (x, y), g in treffer.items():
            je_groesse[[x, y]] += len(g)
        zeigen = sorted(np.argsort(-je_groesse, kind="stable")[:args.max_groessen])
        if len(namen) > args.max_groessen:
            print("  Hinweis : %d von %d Groessen gezeigt (die mit den meisten "
                  "Serien, --max-groessen); Auswahl mit --spalten"
                  % (len(zeigen), len(namen)))
    if len(zeigen) >= 2:
        auswahl = {(a, b): treffer[(zeigen[a], zeigen[b])]
                   for a, b in itertools.combinations(range(len(zeigen)), 2)}
        fig = matrix(W[:, zeigen], [namen[k] for k in zeigen], auswahl, t, quelle,
                     args.toleranz, anteile)
        speichere(fig, out / "konstanz_matrix.png")
        zeige_oder_schliesse(fig, args.zeigen)

    if args.paar:
        fehlt = [s for s in args.paar if s not in namen]
        if fehlt or args.paar[0] == args.paar[1]:
            raise DatenFehler("--paar: zwei verschiedene Groessen noetig, unbekannt: %s. "
                              "Vorhanden: %s" % (", ".join(fehlt) or "-", ", ".join(namen)))
        x, y = (namen.index(s) for s in args.paar)
        # Neu suchen statt aus der Matrix: dort muss die kleinere Spaltennummer variieren.
        gefunden = serien(W, x, y, toleranz, anteile, args.min_punkte)
        fig = einzelfeld(W, namen, x, y, gefunden, ds.zeilen.to_numpy(), t, quelle,
                         args.toleranz, anteile)
        datei = "konstanz_%s_%s.png" % tuple(
            "".join(c if c.isalnum() else "_" for c in s) for s in args.paar)
        speichere(fig, out / datei)
        zeige_oder_schliesse(fig, args.zeigen)


if __name__ == "__main__":
    starte(main)
