"""Gruppen ohne Vorwissen finden und beschreiben, was sie unterscheidet.

Ablauf:
  1. Gruppen im Raum der ersten Hauptkomponenten suchen - so vielen, wie fuer
     einen Grossteil der Varianz noetig sind. Nur PC1/PC2 zu nehmen waere zu
     wenig: zwei Gruppen, die im Biplot uebereinander liegen, koennen sich auf
     PC3 sauber trennen.
  2. Pruefen, ob die Gruppen echt sind (Stabilitaet unter Teilstichproben).
     Ein Clusterverfahren findet IMMER Gruppen, auch in reinem Rauschen.
  3. Jede Gruppe beschreiben: welche Messgroessen weichen wie stark ab,
     welche Zeilen sind typisch, welche Grenzfaelle, und haengen die Gruppen
     mit vorhandenen Textspalten (Charge, Pruefstand, Datum ...) zusammen.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .plotstyle import kopf, rahmen

RAUSCHEN = "ohne Gruppe"


@dataclass
class Clustering:
    """Ergebnis von :func:`finde_gruppen`."""

    nummern: np.ndarray               # 1, 2, ... nach Groesse; -1 = keine Gruppe
    methode: str
    dimensionen: int                  # geclustert auf PC1..PC<dimensionen>
    parameter: dict
    auswahl: pd.DataFrame | None = None   # Kennwerte je Gruppenzahl (KMeans/GMM)

    @property
    def gruppen(self) -> list[int]:
        return sorted(set(self.nummern.tolist()) - {-1})

    def name(self, nummer: int) -> str:
        if nummer < 0:
            return RAUSCHEN
        # Ab 10 Gruppen mit fuehrender Null, sonst sortiert "Gruppe 10" vor "Gruppe 2".
        breite = 2 if len(self.gruppen) >= 10 else 1
        return "Gruppe %0*d" % (breite, nummer)

    @property
    def namen(self) -> np.ndarray:
        return np.array([self.name(int(g)) for g in self.nummern])


# --------------------------------------------------------------------------- #
# Gruppen finden
# --------------------------------------------------------------------------- #
def dimensionen_fuer(varianz, anteil: float = 0.9) -> int:
    """Wie viele Komponenten braucht es fuer ``anteil`` der Varianz (mind. 2)?"""
    d = int(np.searchsorted(np.cumsum(varianz), anteil - 1e-9) + 1)
    return max(2, min(d, len(varianz)))


def _rechne(Z, methode: str, parameter: dict, seed: int) -> np.ndarray:
    if methode == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=parameter["min_groesse"],
                       copy=True).fit_predict(Z)
    if methode == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=parameter["k"], n_init=10,
                      random_state=seed).fit_predict(Z)
    if methode == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=parameter["k"], n_init=3,
                               random_state=seed).fit(Z).predict(Z)
    raise ValueError(f"Unbekanntes Verfahren: {methode}")


def _waehle_k(Z, methode: str, seed: int, k_max: int = 8):
    """Gruppenzahl vorschlagen: Silhouette (KMeans) bzw. BIC (GMM)."""
    from sklearn.metrics import silhouette_score

    zeilen = []
    for k in range(2, min(k_max, len(Z) - 1) + 1):
        if methode == "gmm":
            from sklearn.mixture import GaussianMixture
            modell = GaussianMixture(n_components=k, n_init=3,
                                     random_state=seed).fit(Z)
            lab, bic = modell.predict(Z), modell.bic(Z)
        else:
            lab, bic = _rechne(Z, methode, {"k": k}, seed), np.nan
        sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        zeilen.append({"Gruppen": k, "Silhouette": sil, "BIC": bic})
    tabelle = pd.DataFrame(zeilen)
    if methode == "gmm":
        k = int(tabelle.loc[tabelle["BIC"].idxmin(), "Gruppen"])
    else:
        k = int(tabelle.loc[tabelle["Silhouette"].idxmax(), "Gruppen"])
    return k, tabelle


def _nach_groesse(roh: np.ndarray) -> np.ndarray:
    """Gruppen nach Groesse durchnummerieren: Gruppe 1 ist die groesste."""
    werte, anzahl = np.unique(roh[roh >= 0], return_counts=True)
    rang = {w: i + 1 for i, w in enumerate(werte[np.argsort(-anzahl, kind="stable")])}
    return np.array([rang.get(v, -1) for v in roh])


def finde_gruppen(scores, varianz, methode: str = "hdbscan", k: int | None = None,
                  min_groesse: int | None = None, anteil: float = 0.9,
                  seed: int = 42) -> Clustering:
    """Sucht Gruppen auf den fuehrenden Hauptkomponenten."""
    d = dimensionen_fuer(varianz, anteil)
    return finde_gruppen_in(np.asarray(scores)[:, :d], methode, k, min_groesse, seed)


def finde_gruppen_in(Z, methode: str = "hdbscan", k: int | None = None,
                     min_groesse: int | None = None, seed: int = 42) -> Clustering:
    """Sucht Gruppen in beliebigen Koordinaten (PCA-Scores, UMAP, t-SNE)."""
    Z = np.asarray(Z)
    d = Z.shape[1]
    auswahl = None
    if methode == "hdbscan":
        parameter = {"min_groesse": int(min_groesse or max(5, len(Z) // 50))}
    else:
        if k is None:
            k, auswahl = _waehle_k(Z, methode, seed)
        parameter = {"k": int(k)}
    nummern = _nach_groesse(_rechne(Z, methode, parameter, seed))
    return Clustering(nummern, methode, d, parameter, auswahl)


def stabilitaet(scores, cl: Clustering, laeufe: int = 30, anteil: float = 0.8,
                seed: int = 42) -> pd.Series:
    """Mittlerer Jaccard-Wert je Gruppe unter Teilstichproben (Hennig 2007).

    In jedem Lauf werden 80 % der Zeilen neu geclustert; fuer jede Gruppe
    zaehlt, wie gut sie sich darin wiederfindet (1 = identisch). Echte
    Gruppen ueberstehen das, zufaellige zerfallen.
    """
    Z = np.asarray(scores)[:, :cl.dimensionen]
    rng = np.random.default_rng(seed)
    n = len(Z)
    m = int(anteil * n)
    parameter = dict(cl.parameter)
    if "min_groesse" in parameter:
        # Auf die kleinere Stichprobe umrechnen, sonst fallen Gruppen nur
        # deshalb weg, weil weniger Punkte da sind.
        parameter["min_groesse"] = max(2, round(parameter["min_groesse"] * anteil))

    werte = {g: [] for g in cl.gruppen}
    for lauf in range(laeufe):
        idx = rng.choice(n, m, replace=False)
        neu = _rechne(Z[idx], cl.methode, parameter, seed + lauf + 1)
        ref = cl.nummern[idx]
        kandidaten = [neu == h for h in set(neu.tolist()) - {-1}]
        for g in cl.gruppen:
            A = ref == g
            if not A.any():
                continue
            werte[g].append(max(((A & B).sum() / (A | B).sum() for B in kandidaten),
                                default=0.0))
    return pd.Series({cl.name(g): float(np.mean(v)) if v else np.nan
                      for g, v in werte.items()}, name="Stabilitaet")


def benenne(nummern, breite: int = 1) -> np.ndarray:
    """Gruppennummern -> Namen, mit einheitlicher Breite ueber mehrere Verfahren."""
    return np.array([RAUSCHEN if g < 0 else "Gruppe %0*d" % (breite, g)
                     for g in np.asarray(nummern)])


def angleichen(ref: np.ndarray, andere: np.ndarray) -> np.ndarray:
    """Nummeriert ``andere`` so um, dass sie zu ``ref`` passen.

    Zwei Verfahren nummerieren unabhaengig: "Gruppe 1" bei UMAP kann "Gruppe 3"
    bei PCA sein. Die Zuordnung mit der groessten Gesamtueberschneidung
    (ungarische Methode) gibt uebereinstimmenden Gruppen dieselbe Nummer;
    Gruppen ohne Partner bekommen neue Nummern hinter den vorhandenen.
    """
    from scipy.optimize import linear_sum_assignment

    r_gr = sorted(set(ref.tolist()) - {-1})
    a_gr = sorted(set(andere.tolist()) - {-1})
    if not r_gr or not a_gr:
        return andere.copy()
    tab = np.array([[np.sum((ref == r) & (andere == a)) for a in a_gr] for r in r_gr])
    zeilen, spalten = linear_sum_assignment(-tab)
    abbild = {a_gr[s]: r_gr[z] for z, s in zip(zeilen, spalten) if tab[z, s] > 0}
    frei = max(r_gr) + 1
    for a in a_gr:
        if a not in abbild:
            abbild[a], frei = frei, frei + 1
    return np.array([abbild.get(v, -1) for v in andere])


def uebereinstimmung(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index auf den Punkten, die bei beiden eine Gruppe haben.

    1 = identische Aufteilung, 0 = nicht besser als Zufall. Punkte "ohne
    Gruppe" (HDBSCAN) zaehlen nicht mit - sonst wuerde das Rauschen als
    eigene Gruppe den Wert verzerren.
    """
    from sklearn.metrics import adjusted_rand_score

    beide = (a >= 0) & (b >= 0)
    if beide.sum() < 2:
        return np.nan
    return float(adjusted_rand_score(a[beide], b[beide]))


def bewertung(jaccard: float) -> str:
    if np.isnan(jaccard):
        return "-"
    if jaccard >= 0.85:
        return "sehr stabil"
    if jaccard >= 0.75:
        return "stabil"
    if jaccard >= 0.6:
        return "unsicher"
    return "zerfaellt"


# --------------------------------------------------------------------------- #
# Gruppen beschreiben
# --------------------------------------------------------------------------- #
def profil(X: pd.DataFrame, namen) -> pd.DataFrame:
    """Abweichung jeder Gruppe vom Gesamtmittel, in Standardabweichungen.

    Bewusst auf den Rohdaten gerechnet, nicht auf den PCA-Scores: so ist die
    Aussage direkt "Gruppe 2 hat +1.8 sigma Vibration" - unabhaengig davon,
    welche Skalierung fuer die PCA gewaehlt wurde.
    """
    streu = X.std(ddof=0).replace(0, np.nan)
    Zs = (X - X.mean()) / streu
    return Zs.groupby(np.asarray(namen)).mean().T


def mediane(X: pd.DataFrame, namen) -> pd.DataFrame:
    """Mediane je Gruppe in Originaleinheiten, dazu der Gesamtmedian."""
    tab = X.groupby(np.asarray(namen)).median().T
    tab["alle"] = X.median()
    return tab


def kennzeichen(prof: pd.DataFrame, n: int = 3, ab: float = 0.5) -> dict:
    """Je Gruppe die staerksten Abweichungen nach oben und unten."""
    ergebnis = {}
    for g in prof.columns:
        s = prof[g].dropna()
        hoch = s[s >= ab].sort_values(ascending=False).head(n)
        tief = s[s <= -ab].sort_values().head(n)
        ergebnis[g] = (list(hoch.items()), list(tief.items()))
    return ergebnis


def einordnung(scores, cl: Clustering):
    """Silhouette je Punkt (Grenzfall-Mass) und Abstand zur Gruppenmitte."""
    from sklearn.metrics import silhouette_samples

    Z = np.asarray(scores)[:, :cl.dimensionen]
    zugeordnet = cl.nummern >= 0
    sil = np.full(len(Z), np.nan)
    if len(cl.gruppen) >= 2:
        sil[zugeordnet] = silhouette_samples(Z[zugeordnet], cl.nummern[zugeordnet])
    abstand = np.full(len(Z), np.nan)
    for g in cl.gruppen:
        m = cl.nummern == g
        # Median statt Mittelwert: einzelne Ausreisser verschieben die Mitte nicht.
        abstand[m] = np.linalg.norm(Z[m] - np.median(Z[m], axis=0), axis=1)
    return sil, abstand


def rollen(cl: Clustering, sil, abstand, n: int = 5) -> np.ndarray:
    """Markiert je Gruppe die n typischsten Zeilen und die n unsichersten."""
    rolle = np.full(len(cl.nummern), "", dtype=object)
    for g in cl.gruppen:
        idx = np.flatnonzero(cl.nummern == g)
        if not np.isnan(sil[idx]).all():
            rolle[idx[np.argsort(sil[idx])[:n]]] = "Grenzfall"
        rolle[idx[np.argsort(abstand[idx])[:n]]] = "typisch"
    return rolle


def kreuztabellen(meta: pd.DataFrame, namen, max_werte: int = 30):
    """Haengen die Gruppen mit einer vorhandenen Spalte zusammen?

    Liefert ``[(spalte, tabelle, cramers_v), ...]``, staerkster Zusammenhang
    zuerst. Cramers V: 0 = unabhaengig, 1 = Spalte und Gruppe decken sich.
    Spalten mit sehr vielen Werten (IDs) und Zeitstempel werden uebersprungen.
    """
    from scipy.stats import chi2_contingency

    namen = pd.Series(np.asarray(namen), name="Gruppe")
    ergebnis = []
    for spalte in meta.columns:
        werte = meta[spalte].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(werte):
            continue
        if not 2 <= werte.nunique(dropna=True) <= max_werte:
            continue
        tab = pd.crosstab(werte.astype(str), namen)
        if min(tab.shape) < 2:
            continue
        chi2 = chi2_contingency(tab.to_numpy(), correction=False)[0]
        v = float(np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1))))
        ergebnis.append((spalte, tab, v))
    return sorted(ergebnis, key=lambda e: -e[2])


def zeitraeume(meta: pd.DataFrame, namen) -> pd.DataFrame:
    """Frueheste und spaeteste Zeitangabe je Gruppe, fuer jede Datumsspalte."""
    teile = []
    for spalte in meta.columns:
        werte = meta[spalte].reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(werte):
            continue
        gruppiert = werte.groupby(np.asarray(namen))
        teile.append(pd.DataFrame({spalte + "_von": gruppiert.min(),
                                   spalte + "_bis": gruppiert.max()}))
    return pd.concat(teile, axis=1) if teile else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def profil_heatmap(prof: pd.DataFrame, anzahl: pd.Series, stab: pd.Series, t,
                   quelle: str = "", verfahren: str = ""):
    """Gruppen x Messgroessen: wo liegt jede Gruppe ueber, wo unter dem Schnitt?"""
    # Staerkste Unterscheider nach oben - dort faengt man an zu lesen.
    reihenfolge = prof.abs().max(axis=1).sort_values(ascending=False).index
    M = prof.loc[reihenfolge]
    merkmale, gruppen = list(M.index), list(M.columns)
    cmap = LinearSegmentedColormap.from_list("hda_div", t["diverging"])
    # Mindestens +-1 sigma als Skala, sonst wirkt Rauschen knallig.
    grenze = max(1.0, float(np.nanmax(np.abs(M.to_numpy()))))

    hoehe = max(4.4, 0.34 * len(merkmale) + 2.4)
    fig, ax = plt.subplots(figsize=(1.35 * len(gruppen) + 5.0, hoehe))
    bild = ax.imshow(M.to_numpy(), cmap=cmap, vmin=-grenze, vmax=grenze,
                     aspect="auto")

    kopfzeilen = []
    for g in gruppen:
        z = "%s\nn=%d" % (g, anzahl.get(g, 0))
        if g in stab.index and not np.isnan(stab[g]):
            z += "\nJ=%.2f" % stab[g]
        kopfzeilen.append(z)
    ax.set_xticks(range(len(gruppen)), kopfzeilen, fontsize=8.5)
    ax.set_yticks(range(len(merkmale)), merkmale)
    ax.tick_params(length=0)
    rahmen(ax, t, ticks_innen=False)
    ax.set_xticks(np.arange(-0.5, len(gruppen), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(merkmale), 1), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(merkmale)):
        for j in range(len(gruppen)):
            wert = M.iat[i, j]
            if np.isnan(wert):
                continue
            ax.text(j, i, "%+.1f" % wert, ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(wert) > 0.62 * grenze else t["text"])

    leiste = fig.colorbar(bild, ax=ax, fraction=0.03, pad=0.02)
    leiste.outline.set_visible(False)
    leiste.set_label("Abweichung vom Gesamtmittel (σ)", color=t["text2"], fontsize=9)
    kopf(ax, "Was macht die Gruppen aus?"
             + (" · Gruppen aus %s" % verfahren if verfahren else ""),
         "rot = ueber dem Schnitt, blau = darunter · J = Stabilitaet "
         "(ab 0.75 belastbar)", t)
    if quelle:
        fig.text(0.01, -0.01, "Datenquelle: " + quelle, fontsize=8,
                 color=t["muted"], ha="left", va="top")
    fig.tight_layout()
    return fig
