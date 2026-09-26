"""Hauptkomponentenanalyse (PCA) inklusive Biplot mit Richtungspfeilen.

Erzeugt im Ausgabeordner:
  pca_biplot_PC1_PC2.png  Streudiagramm der Datenpunkte + Pfeile der Messgroessen
  pca_scree.png           erklaerte Varianz je Komponente (einzeln und kumuliert)
  pca_ladungen.png        Ladungsmatrix als Heatmap
  pca_ergebnis.xlsx       Scores, Ladungen und Varianzanteile als Tabelle

Mit --cluster zusaetzlich (fuer Daten ohne bekannte Gruppen):
  pca_gruppen_PC1_PC2.png Biplot, eingefaerbt nach den gefundenen Gruppen
  pca_gruppenprofil.png   Heatmap: welche Messgroessen machen jede Gruppe aus
  pca_gruppen.xlsx        Zuordnung jeder Excel-Zeile, Profile, Vertreter,
                          Stabilitaet, Zusammenhang mit vorhandenen Spalten

Die Pfeile sind die Ladungsvektoren: Richtung = in welche Richtung der Plot
diese Groesse zunehmen laesst, Laenge = wie stark die Groesse in der gezeigten
Ebene vertreten ist. Zwei Pfeile im spitzen Winkel bedeuten positiv
korrelierte Groessen, entgegengesetzte Pfeile negativ korrelierte, ein
rechter Winkel heisst weitgehend unabhaengig.

Aufruf:  python run_pca.py [--datei data/meine.xlsx] [--pfeile 10] ...
         python run_pca.py --cluster hdbscan
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from hda import cluster as hc
from hda import gruppen as hg
from hda.cli import ausgabeordner, basis_parser, blatt_wert, starte
from hda.data_io import lade_daten, skaliere
from hda.plotstyle import (gitter, kopf, rahmen, scatter_nach_gruppe, set_style,
                           speichere, zeige_oder_schliesse)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
# Pfeile und ihre Namen im Biplot: schlank, damit sie die Punkte nicht
# erschlagen. Die Schrift wird mit denselben Werten vermessen und gezeichnet,
# sonst stimmt die Kollisionspruefung nicht.
PFEIL_LINIE = 0.9
PFEIL_SPITZE = 8
NAME_GROESSE = 8.0
NAME_GEWICHT = 350        # Segoe UI Semilight; faellt sonst auf "normal" zurueck

def biplot(scores, ladungen, merkmale, labels, t, pc_x, pc_y, varianz,
           n_pfeile=None, label_name=None, quelle="", neutral=None, titel=None):
    """Scores als Punkte, Ladungen als Pfeile - beides im selben Koordinatensystem."""
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    gitter(ax, t)

    x, y = scores[:, pc_x], scores[:, pc_y]
    gruppen = scatter_nach_gruppe(ax, x, y, labels, t, groesse=42, alpha=0.75,
                                  neutral=neutral)

    # Nullachsen als stille Orientierung
    ax.axhline(0, color=t["grid"], linewidth=1.0, zorder=1)
    ax.axvline(0, color=t["grid"], linewidth=1.0, zorder=1)

    # Pfeile auf die Punktwolke skalieren. Ein gemeinsamer Faktor fuer beide
    # Achsen, sonst waeren die Winkel zwischen den Pfeilen verfaelscht.
    L = ladungen[:, [pc_x, pc_y]]
    laenge = np.hypot(L[:, 0], L[:, 1])
    reichweite = np.abs(np.column_stack([x, y])).max()
    skala = 0.80 * reichweite / max(laenge.max(), 1e-12)

    reihenfolge = np.argsort(laenge)[::-1]
    if n_pfeile:
        reihenfolge = reihenfolge[:n_pfeile]

    for i in reihenfolge:
        dx, dy = L[i, 0] * skala, L[i, 1] * skala
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=PFEIL_SPITZE,
                                    linewidth=PFEIL_LINIE, color=t["text2"],
                                    shrinkA=0, shrinkB=0), zorder=5)

    ax.set_xlabel("PC%d  (%.1f %% der Varianz)" % (pc_x + 1, varianz[pc_x] * 100))
    ax.set_ylabel("PC%d  (%.1f %% der Varianz)" % (pc_y + 1, varianz[pc_y] * 100))
    # adjustable="box" statt "datalim": nur so bleiben gesetzte Achsengrenzen
    # stehen. Mit "datalim" rechnet matplotlib sie beim Zeichnen neu, und die
    # vermessenen Textkaesten der Beschriftung waeren sofort wieder veraltet.
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.16)          # Platz fuer die Beschriftung der Pfeilspitzen
    rahmen(ax, t)

    gezeigt = len(reihenfolge)
    unter = ("Pfeile = Richtung steigender Messwerte (%d von %d Groessen) · "
             "PC%d und PC%d zeigen zusammen %.1f %% der Gesamtvarianz"
             % (gezeigt, len(merkmale), pc_x + 1, pc_y + 1,
                (varianz[pc_x] + varianz[pc_y]) * 100))
    kopf(ax, (titel or "PCA-Biplot") + " · PC%d vs. PC%d" % (pc_x + 1, pc_y + 1),
         unter, t)

    if gruppen:
        ax.legend(title=label_name, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  title_fontsize=9, labelspacing=0.7, handletextpad=0.5)
    if quelle:
        fig.text(0.01, -0.01, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()

    # Erst jetzt beschriften: das Layout steht, damit sind die gemessenen
    # Textkaesten belastbar und die Kollisionspruefung stimmt.
    beschrifte_pfeile(fig, ax, L * skala, merkmale, reihenfolge, t)
    return fig


def beschrifte_pfeile(fig, ax, pfeile, merkmale, reihenfolge, t):
    """Namen laengs der Pfeilrichtung setzen, ohne dass sie sich ueberdecken.

    Die Schrift folgt dem Pfeil (immer von links nach rechts lesbar) und sitzt
    an der Spitze. Ueberlappt ein Name einen schon gesetzten, rutscht er
    entlang seiner eigenen Richtung weiter nach aussen - der laengere Pfeil
    behaelt seinen Platz, weil in der Reihenfolge abnehmender Laenge
    beschriftet wird.
    """
    fig.canvas.draw()          # Renderer holen, Seitenverhaeltnis anwenden
    zeichner = fig.canvas.get_renderer()
    ax.set_autoscale_on(False)

    # Platzieren und Rahmen weiten haengen voneinander ab. Weil die
    # Ausweichpositionen in Punkten statt in Datenkoordinaten gemessen werden,
    # aendert das Weiten den Abstand Spitze-Name nicht - beides konvergiert.
    namen = []
    for runde in range(4):
        for n in namen:
            n.remove()
        namen = _setze_namen(ax, zeichner, pfeile, merkmale, reihenfolge, t)
        # Letzter Schritt muss das Setzen sein: ein Weiten danach wuerde die
        # Pfeilspitzen verschieben und die gepruefte Platzierung entwerten.
        if runde == 3 or not _weite_rahmen(fig, ax, zeichner, namen):
            break


def _ausweichraster(laengs_schritte=6, quer_schritte=5,
                    laengs=13.0, quer=11.0):
    """Ausweichpositionen relativ zur Pfeilspitze, in Punkten.

    Laengs des Pfeils weiter nach aussen und quer dazu, nach Abstand zur
    Spitze sortiert - so gewinnt immer die dichteste freie Stelle. Quer ist
    unverzichtbar: liegen mehrere Pfeile fast auf einer Linie (im Biplot der
    Regelfall), wird reines Nach-aussen-Schieben nie frei.
    """
    raster = [(a * laengs, q * quer)
              for a in range(laengs_schritte)
              for q in range(-quer_schritte, quer_schritte + 1)]
    return sorted(raster, key=lambda v: np.hypot(v[0], v[1] * 1.15))


AUSWEICHEN = _ausweichraster()
# Ab diesem Abstand steht der Name sichtbar neben statt an seinem Pfeil und
# bekommt eine Fuehrungslinie.
FUEHRUNG_AB = 20.0


def _ecken(anker, richtung, breite, hoehe, rand=2.0):
    """Die vier Ecken eines gedrehten Textkastens in Bildpunkten."""
    u = np.asarray(richtung, dtype=float)
    q = np.array([-u[1], u[0]])
    b, h = breite + 2 * rand, hoehe + 2 * rand
    start = np.asarray(anker, dtype=float) - u * rand - q * (h / 2)
    return np.array([start, start + u * b, start + u * b + q * h, start + q * h])


def _ueberlappt(A, B) -> bool:
    """Trennachsensatz: ueberschneiden sich zwei gedrehte Rechtecke?

    Die achsenparallele Huelle aus get_window_extent taugt dafuer nicht -
    bei schraeg stehender Schrift ist sie ein Vielfaches der Schrift selbst,
    und es waere nie eine Position frei.
    """
    for ecken in (A, B):
        for i in range(2):
            kante = ecken[i + 1] - ecken[i]
            achse = np.array([-kante[1], kante[0]])
            norm = np.hypot(*achse)
            if norm < 1e-9:
                continue
            achse /= norm
            a, b = A @ achse, B @ achse
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def _setze_namen(ax, zeichner, pfeile, merkmale, reihenfolge, t):
    """Setzt alle Namen und liefert die erzeugten Textelemente zurueck."""
    from matplotlib.font_manager import FontProperties

    schriftart = FontProperties(size=NAME_GROESSE, weight=NAME_GEWICHT)
    punkt = ax.figure.dpi / 72.0          # Punkte -> Bildpunkte
    belegt, namen = [], []

    for i in reihenfolge:
        dx, dy = pfeile[i, 0], pfeile[i, 1]
        spitze = ax.transData.transform((dx, dy))
        winkel = np.degrees(np.arctan2(dy, dx))
        bogen = np.radians(winkel)
        laengs = np.array([np.cos(bogen), np.sin(bogen)])     # Pfeilrichtung
        quer = np.array([-np.sin(bogen), np.cos(bogen)])      # senkrecht dazu

        # Kopfueber stehende Schrift um 180 Grad drehen und die Ausrichtung
        # spiegeln, damit der Text trotzdem nach aussen laeuft.
        if winkel > 90:
            schrift, ha = winkel - 180, "right"
        elif winkel < -90:
            schrift, ha = winkel + 180, "right"
        else:
            schrift, ha = winkel, "left"

        breite, hoehe, _ = zeichner.get_text_width_height_descent(
            merkmale[i], schriftart, False)
        # Bei ha="right" haengt der Text rueckwaerts an seinem Ankerpunkt;
        # der Kasten laeuft dann in die Gegenrichtung der Schrift.
        richtung = np.array([np.cos(np.radians(schrift)),
                             np.sin(np.radians(schrift))])
        if ha == "right":
            richtung = -richtung

        gewaehlt = AUSWEICHEN[-1]
        for aus, quer_versatz in AUSWEICHEN:
            versatz = laengs * (5 + aus) + quer * quer_versatz
            kasten = _ecken(spitze + versatz * punkt, richtung, breite, hoehe)
            if not any(_ueberlappt(kasten, b) for b in belegt):
                gewaehlt = (aus, quer_versatz)
                break

        aus, quer_versatz = gewaehlt
        versatz = laengs * (5 + aus) + quer * quer_versatz
        belegt.append(_ecken(spitze + versatz * punkt, richtung, breite, hoehe))

        fuehrung = None
        if np.hypot(aus, quer_versatz) > FUEHRUNG_AB:
            # Duenne Linie zur Spitze, sonst ist die Zuordnung Name-Pfeil weg.
            fuehrung = dict(arrowstyle="-", linewidth=0.8, color=t["grid"],
                            shrinkA=2, shrinkB=1)
        namen.append(ax.annotate(
            merkmale[i], xy=(dx, dy), xytext=tuple(versatz),
            textcoords="offset points", rotation=schrift,
            rotation_mode="anchor", ha=ha, va="center", color=t["text"],
            fontsize=NAME_GROESSE, fontweight=NAME_GEWICHT, zorder=6,
            annotation_clip=False,
            arrowprops=fuehrung,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=t["surface"],
                      edgecolor="none", alpha=0.7)))
    return namen


def _weite_rahmen(fig, ax, zeichner, artefakte) -> bool:
    """Zieht die Achsen auf, bis alle Namen innerhalb des Rahmens liegen.

    Sonst haengen die aeusseren Beschriftungen ueber dem Rand - mit einem
    geschlossenen Rahmen faellt das sofort unangenehm auf. Rueckgabe: ob
    ueberhaupt geweitet wurde.
    """
    umkehr = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs, ys = [x0, x1], [y0, y1]
    for a in artefakte:
        k = a.get_window_extent(zeichner)
        ecken = umkehr.transform([(k.x0, k.y0), (k.x1, k.y1),
                                  (k.x0, k.y1), (k.x1, k.y0)])
        xs += list(ecken[:, 0])
        ys += list(ecken[:, 1])

    px, py = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
    neu_x = (min(xs) - px, max(xs) + px)
    neu_y = (min(ys) - py, max(ys) + py)
    if (neu_x[0] > x0 - px and neu_x[1] < x1 + px
            and neu_y[0] > y0 - py and neu_y[1] < y1 + py):
        return False
    ax.set_xlim(*neu_x)
    ax.set_ylim(*neu_y)
    fig.canvas.draw()
    return True


def scree(varianz, t, quelle=""):
    """Wie viel Information steckt in wie vielen Komponenten?"""
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    gitter(ax, t, achse="y")

    k = np.arange(1, len(varianz) + 1)
    einzeln = varianz * 100
    kumuliert = np.cumsum(einzeln)

    ax.bar(k, einzeln, width=0.55, color=t["series"][0], zorder=3,
           label="je Komponente")
    ax.plot(k, kumuliert, color=t["series"][1], marker="o", markersize=5,
            markeredgecolor=t["surface"], markeredgewidth=1.2, zorder=4,
            label="kumuliert")

    # Direktbeschriftung nur dort, wo sie etwas beitraegt - nicht an jedem Punkt.
    for xi, yi in zip(k, einzeln):
        if yi < 4:
            continue
        if yi > 15:
            # In den Balken hinein - oberhalb liegt die kumulierte Linie.
            ax.text(xi, yi - 2.4, "%.0f%%" % yi, ha="center", va="top",
                    fontsize=8.5, color=t["surface"], fontweight="semibold")
        else:
            ax.text(xi, yi + 1.8, "%.0f%%" % yi, ha="center", va="bottom",
                    fontsize=8.5, color=t["text2"])
    for schwelle in (80, 95):
        if (kumuliert >= schwelle).any():
            treffer = int(np.argmax(kumuliert >= schwelle)) + 1
            ax.annotate("%d %% ab PC%d" % (schwelle, treffer),
                        xy=(treffer, kumuliert[treffer - 1]),
                        xytext=(7, -15), textcoords="offset points",
                        fontsize=8.5, color=t["text2"])

    ax.set_xticks(k)
    ax.set_xlabel("Hauptkomponente")
    ax.set_ylabel("erklaerte Varianz (%)")
    ax.set_ylim(0, 106)
    rahmen(ax, t)
    ax.tick_params(which="minor", bottom=False, top=False)  # Balken sind diskret
    ax.legend(loc="center right")
    kopf(ax, "Wie viele Komponenten braucht es?",
         "Anteil an der Gesamtvarianz je Komponente und aufsummiert", t)
    if quelle:
        fig.text(0.01, -0.02, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()
    return fig


def ladungs_heatmap(ladungen, merkmale, t, n_pc=4, quelle=""):
    """Ladungen als divergierende Heatmap: welche Groesse praegt welche Komponente."""
    n_pc = min(n_pc, ladungen.shape[1])
    M = ladungen[:, :n_pc]
    cmap = LinearSegmentedColormap.from_list("hda_div", t["diverging"])
    grenze = float(np.abs(M).max())

    hoehe = max(4.4, 0.34 * len(merkmale) + 2.0)
    fig, ax = plt.subplots(figsize=(1.3 * n_pc + 5.0, hoehe))
    bild = ax.imshow(M, cmap=cmap, vmin=-grenze, vmax=grenze, aspect="auto")

    ax.set_xticks(range(n_pc), ["PC%d" % (i + 1) for i in range(n_pc)])
    ax.set_yticks(range(len(merkmale)), merkmale)
    ax.tick_params(length=0)
    rahmen(ax, t, ticks_innen=False)
    # 2px-Fuge zwischen den Zellen, damit die Felder nicht verschmelzen
    ax.set_xticks(np.arange(-0.5, n_pc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(merkmale), 1), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(merkmale)):
        for j in range(n_pc):
            wert = M[i, j]
            ax.text(j, i, "%+.2f" % wert, ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(wert) > 0.62 * grenze else t["text"])

    leiste = fig.colorbar(bild, ax=ax, fraction=0.03, pad=0.02)
    leiste.outline.set_visible(False)
    leiste.set_label("Ladung", color=t["text2"], fontsize=9)
    kopf(ax, "Ladungen der Messgroessen",
         "positiv = gleichlaeufig mit der Komponente, negativ = gegenlaeufig", t)
    if quelle:
        fig.text(0.01, -0.01, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def main() -> None:
    p = basis_parser("PCA mit Biplot: Datenpunkte plus Pfeile fuer die Richtung, "
                     "in der jede Messgroesse zunimmt.")
    p.add_argument("-k", "--komponenten", type=int, default=None,
                   help="Anzahl berechneter Hauptkomponenten (Standard: alle)")
    p.add_argument("--pfeile", type=int, default=None,
                   help="nur die N laengsten Pfeile zeichnen (Standard: alle)")
    p.add_argument("--paare", nargs="*", default=["1,2"],
                   help="zu zeichnende Komponentenpaare, z. B. 1,2 1,3 2,3")
    hg.cluster_optionen(p)
    p.add_argument("--cluster-varianz", type=float, default=0.9,
                   help="auf so vielen PCs clustern, wie fuer diesen "
                        "Varianzanteil noetig sind")
    args = p.parse_args()
    if args.komponenten is not None and args.komponenten < 2:
        p.error("--komponenten muss mindestens 2 sein (Biplot braucht zwei Achsen)")

    ds = lade_daten(args.datei, blatt_wert(args.blatt), args.label_spalte,
                    args.id_spalten, args.nan)
    X, _ = skaliere(ds.X, args.skalierung)
    out = ausgabeordner(args.out)
    t = set_style(args.theme)
    quelle = ds.quelle.name if ds.quelle else ""

    k = min(args.komponenten or min(X.shape), min(X.shape))
    pca = PCA(n_components=k, random_state=args.seed)
    scores = pca.fit_transform(X)
    varianz = pca.explained_variance_ratio_
    # Ladungen in Korrelationsform: Eigenvektor mal Wurzel des Eigenwerts.
    ladungen = pca.components_.T * np.sqrt(pca.explained_variance_)

    print("\nPCA   : %d Komponenten, Skalierung '%s'" % (k, args.skalierung))
    print("        PC1 %.1f %% | PC2 %.1f %% | PC1+PC2 %.1f %%"
          % (varianz[0] * 100, varianz[1] * 100, (varianz[0] + varianz[1]) * 100))
    for pc in range(min(3, k)):
        stark = np.argsort(np.abs(ladungen[:, pc]))[::-1][:4]
        text = ", ".join("%s (%+.2f)" % (ds.merkmale[i], ladungen[i, pc]) for i in stark)
        print("        PC%d wird getragen von: %s" % (pc + 1, text))

    print("\nPlots:")
    paare = lies_paare(args.paare, k)
    for a, b in paare:
        fig = biplot(scores, ladungen, ds.merkmale, ds.labels, t, a, b, varianz,
                     args.pfeile, ds.label_name, quelle)
        speichere(fig, out / ("pca_biplot_PC%d_PC%d.png" % (a + 1, b + 1)))
        zeige_oder_schliesse(fig, args.zeigen)

    fig = scree(varianz, t, quelle)
    speichere(fig, out / "pca_scree.png")
    zeige_oder_schliesse(fig, args.zeigen)

    fig = ladungs_heatmap(ladungen, ds.merkmale, t, min(4, k), quelle)
    speichere(fig, out / "pca_ladungen.png")
    zeige_oder_schliesse(fig, args.zeigen)

    # Zahlen zum Weiterrechnen
    spalten = ["PC%d" % (i + 1) for i in range(k)]
    df_scores = pd.DataFrame(scores, columns=spalten)
    if ds.labels is not None:
        df_scores.insert(0, ds.label_name, ds.labels)
    df_ladungen = pd.DataFrame(ladungen, index=ds.merkmale, columns=spalten)
    df_varianz = pd.DataFrame({
        "Komponente": spalten,
        "Varianzanteil_pct": varianz * 100,
        "kumuliert_pct": np.cumsum(varianz) * 100,
    })
    ziel = out / "pca_ergebnis.xlsx"
    with pd.ExcelWriter(ziel, engine="openpyxl") as writer:
        df_varianz.to_excel(writer, sheet_name="Varianz", index=False)
        df_ladungen.to_excel(writer, sheet_name="Ladungen")
        df_scores.to_excel(writer, sheet_name="Scores", index=False)
    print("  gespeichert: %s" % ziel)

    if args.cluster != "aus":
        gruppen_analyse(args, ds, scores, ladungen, varianz, paare, t, out, quelle)


def lies_paare(angaben, k):
    """'1,2' -> (0, 1); ungueltige oder nicht vorhandene Paare mit Hinweis weg."""
    paare = []
    for paar in angaben:
        try:
            a, b = [int(v) - 1 for v in str(paar).replace(" ", "").split(",")]
        except ValueError:
            print("  uebersprungen: '%s' - erwartet wird z. B. 1,2" % paar)
            continue
        if max(a, b) >= k or min(a, b) < 0:
            print("  uebersprungen: PC%d,PC%d (nur %d Komponenten vorhanden)"
                  % (a + 1, b + 1, k))
            continue
        paare.append((a, b))
    return paare


def gruppen_analyse(args, ds, scores, ladungen, varianz, paare, t, out, quelle):
    """Gruppen ohne Labels finden und beschreiben, was sie unterscheidet."""
    cl = hc.finde_gruppen(scores, varianz, args.cluster, args.gruppen,
                          args.min_gruppe, args.cluster_varianz, args.seed)
    d = cl.dimensionen
    print("\nGruppen: %s auf PC1-PC%d (%.1f %% der Varianz), %s"
          % (cl.methode.upper(), d, varianz[:d].sum() * 100,
             ", ".join("%s=%s" % kv for kv in cl.parameter.items())))
    pcs = pd.DataFrame(scores[:, :d], columns=["PC%d" % (i + 1) for i in range(d)])
    namen = hg.beschreibe(cl, scores[:, :d], ds, t, out, "pca", "PCA",
                          args.stabilitaet, args.seed, quelle, args.zeigen, zusatz=pcs)
    if namen is None:
        return
    for a, b in paare:
        fig = biplot(scores, ladungen, ds.merkmale, namen, t, a, b, varianz,
                     args.pfeile, "Gruppe", quelle, neutral=hc.RAUSCHEN,
                     titel="Gefundene Gruppen")
        speichere(fig, out / ("pca_gruppen_PC%d_PC%d.png" % (a + 1, b + 1)))
        zeige_oder_schliesse(fig, args.zeigen)


if __name__ == "__main__":
    starte(main)
