# Hochdimensionale Datenanalyse

Skripte zum Einlesen einer Excel-Tabelle aus `data/` und zum Projizieren der
Daten auf zwei Dimensionen — per **PCA** (mit Richtungspfeilen), **UMAP** und
**t-SNE**.

## Datenformat

Erste Zeile = Spaltennamen, darunter die Daten:

| Pruefling_ID | Betriebszustand | Drehzahl_rpm | Drehmoment_Nm | … |
|---|---|---|---|---|
| P-0001 | Normalbetrieb | 1452.3 | 96.1 | … |

* **Numerische Spalten** werden als Merkmale verwendet.
* **Textspalten** sind Beschriftungen. Die Spalte mit den wenigsten
  verschiedenen Werten wird automatisch zum Einfärben genutzt
  (IDs fallen dadurch heraus); mit `--label-spalte` lässt sich das überschreiben.
* Zahlen, die Excel als Text abgelegt hat (`"1,23"`), werden automatisch
  umgewandelt. Konstante Spalten fliegen raus, Lücken werden per Median
  gefüllt (`--nan drop` verwirft stattdessen die Zeile).

## Einrichtung

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.10–3.12 verwenden — `umap-learn` hängt an `numba`, das für 3.13/3.14
noch keine Räder liefert. PCA und t-SNE laufen auch ohne `umap-learn`.

## Benutzung

Alles auf einmal:

```powershell
python run_all.py                                  # PCA mit Gruppen, UMAP, t-SNE, Vergleich, Streumatrix
python run_all.py --datei data/meine.xlsx --cluster hdbscan --sweep
python run_all.py --nur pca vergleich              # nur ausgewählte Schritte
python run_all.py --testdaten                      # vorher Beispieldaten erzeugen
```

`run_all.py` versteht dieselben gemeinsamen Optionen wie die Einzelskripte
(siehe unten) und reicht sie an jeden Schritt weiter, dazu `--cluster`,
`--gruppen`, `--sweep` und `--nur`. Scheitert ein Schritt, laufen die anderen
weiter; am Ende steht eine Zusammenfassung mit Laufzeiten. Fehlt `umap-learn`,
wird UMAP übersprungen.

Einzeln:

```powershell
python generate_testdata.py          # erzeugt data/beispiel_messdaten.xlsx
python run_pca.py --cluster kmeans
python run_umap.py --sweep
python run_tsne.py --sweep
python run_vergleich.py
```

Ohne `--datei` nimmt jedes Skript die neueste Excel in `data/`. Alle Plots und
Ergebnistabellen landen in `output/`.

### Gemeinsame Optionen

| Option | Wirkung |
|---|---|
| `-d, --datei` | Pfad zur Excel-Datei |
| `-b, --blatt` | Tabellenblatt (Name oder Index) |
| `-l, --label-spalte` | Spalte zum Einfärben |
| `--id-spalten` | numerische Spalten, die keine Merkmale sind (z. B. Seriennummern) |
| `-s, --skalierung` | `zscore` (Standard), `minmax`, `robust`, `keine` |
| `--nan` | `median` (Standard) oder `drop` |
| `-o, --out` | Ausgabeordner, Standard `output` |
| `--theme` | `light` (Standard) oder `dark` |
| `--zeigen` | Fenster öffnen statt nur speichern |
| `--seed` | Zufallsstartwert |

Die Standardisierung ist kein Detail: ohne sie dominiert schlicht die Spalte
mit den größten Zahlen (Drehzahl in rpm schlägt Vibration in mm/s um drei
Größenordnungen). `robust` ist die Wahl, wenn einzelne Ausreißer die Skala
verzerren.

## Die drei Verfahren

### `run_pca.py` — linear, interpretierbar

```powershell
python run_pca.py --pfeile 10 --paare 1,2 1,3
```

| Option | Wirkung |
|---|---|
| `-k, --komponenten` | Anzahl berechneter Komponenten |
| `--pfeile N` | nur die N längsten Pfeile zeichnen (gegen Pfeilsalat bei vielen Spalten) |
| `--paare` | Komponentenpaare, z. B. `1,2 1,3 2,3` |

Ausgabe: `pca_biplot_PC1_PC2.png`, `pca_scree.png`, `pca_ladungen.png`,
`pca_ergebnis.xlsx` (Varianz, Ladungen, Scores).

**Die Pfeile lesen.** Jeder Pfeil ist der Ladungsvektor einer Messgröße:

* **Richtung** — dorthin nimmt diese Größe zu. Punkte am Pfeilende haben hohe
  Werte, Punkte in der Gegenrichtung niedrige.
* **Länge** — wie stark die Größe in *dieser* Ebene vertreten ist. Ein kurzer
  Pfeil heißt nicht „unwichtig", sondern „zeigt woanders hin" (dann `--paare 1,3`
  probieren oder in `pca_ladungen.png` nachsehen).
* **Winkel zwischen zwei Pfeilen** — spitz: gleichläufige Größen; entgegen­gesetzt:
  gegenläufige; rechter Winkel: weitgehend unabhängig.

Die Achsen des Biplots sind bewusst gleich skaliert (`aspect="equal"`), sonst
wären die Winkel verzerrt und damit die Korrelationsaussage falsch.

Die Namen stehen **längs ihres Pfeils** und werden automatisch entzerrt: Der
längere Pfeil behält seinen Platz, kürzere weichen quer oder weiter nach außen
aus, bis nichts mehr überlappt. Geprüft wird dabei das gedrehte Rechteck der
Schrift (Trennachsensatz) — die achsenparallele Hülle von matplotlib ist bei
schräger Schrift ein Vielfaches der Schrift selbst und damit unbrauchbar.
Anschließend weitet sich der Rahmen so weit, dass alle Namen innerhalb liegen.

#### Gruppen finden, wenn es keine Labels gibt

```powershell
python run_pca.py --cluster hdbscan
python run_pca.py --cluster kmeans --paare 1,2 1,3
```

| Option | Wirkung |
|---|---|
| `--cluster` | `hdbscan` (keine Gruppenzahl nötig, Einzelgänger bleiben „ohne Gruppe“), `kmeans`, `gmm` |
| `--gruppen K` | feste Gruppenzahl für `kmeans`/`gmm`; ohne Angabe per Silhouette bzw. BIC gewählt |
| `--min-gruppe N` | `hdbscan`: kleinste zulässige Gruppe, Standard max(5, n/50) |
| `--cluster-varianz` | geclustert wird auf so vielen PCs, wie für diesen Varianzanteil nötig sind (Standard 0.9) — nicht nur auf PC1/PC2 |
| `--stabilitaet N` | Teilstichproben für die Stabilitätsprüfung, Standard 30, `0` = aus |

Ausgabe: `pca_gruppen_PC1_PC2.png` (Biplot nach gefundener Gruppe),
`pca_gruppenprofil.png` und `pca_gruppen.xlsx` mit den Blättern

* **Uebersicht** — Größe, Stabilität und kennzeichnende Messgrößen je Gruppe
* **Zuordnung** — jede Zeile mit `Excel_Zeile`, allen Text-/ID-Spalten,
  Gruppe, Messwerten und Scores; zum Abgleich mit Informationen, die nicht in
  der Datei stehen (Charge, Umbau, Schicht …)
* **Vertreter** — je Gruppe die 5 typischsten Zeilen und die 5 Grenzfälle
* **Daten_nach_Gruppe** — alle Rohdaten mit allen Spalten, Gruppe für Gruppe
  untereinander (Lücken bleiben leer, keine Median-Füllung)
* **Profil_sigma** / **Profil_Median** — Abweichung in σ bzw. Mediane in Originaleinheiten
* **Kreuztabellen** — Gruppen gegen vorhandene Textspalten mit Cramér V
  (1 = deckungsgleich); Datumsspalten erscheinen als Zeitraum je Gruppe

**Stabilität zuerst lesen.** Ein Clusterverfahren findet immer Gruppen, auch in
reinem Rauschen. Die Stabilität J sagt, wie gut sich eine Gruppe wiederfindet,
wenn 80 % der Zeilen neu geclustert werden: ab 0.75 belastbar, unter 0.6
nicht interpretieren. `hdbscan` meldet bei strukturlosen Daten gar keine
Gruppe; `kmeans`/`gmm` teilen immer auf — dort warnt zusätzlich eine
Silhouette unter 0.25.

### `run_umap.py` — nichtlinear, lokal **und** global

```powershell
python run_umap.py --nachbarn 15 --min-dist 0.1 --sweep
```

| Option | Wirkung |
|---|---|
| `-n, --nachbarn` | `n_neighbors`: klein (5–15) = lokale Feinstruktur, groß (50+) = globale Form |
| `-m, --min-dist` | 0.0–0.1 = kompakte Cluster, 0.5–0.9 = gleichmäßig verteilt |
| `--metrik` | `euclidean`, `manhattan`, `cosine`, … |
| `--sweep` | Raster mehrerer Parameterkombinationen |

### `run_tsne.py` — nichtlinear, rein lokal

```powershell
python run_tsne.py --perplexity 30 --sweep
```

| Option | Wirkung |
|---|---|
| `-p, --perplexity` | effektive Nachbarzahl, typisch 5–50 (muss < Anzahl Datenpunkte sein) |
| `--iterationen` | Optimierungsschritte, Standard 1000 |
| `--tsne-start` | Startpositionen: `random` (Standard, unabhängig von der PCA, über `--seed` reproduzierbar) oder `pca` (PC1/PC2 als Start) |
| `--sweep` | Raster mehrerer Perplexity-Werte |

**Gruppen auf der Karte:** `run_umap.py` und `run_tsne.py` verstehen dieselben
Optionen wie `run_pca.py` (`--cluster`, `--gruppen`, `--min-gruppe`,
`--stabilitaet`). Gesucht wird direkt auf der 2D-Karte; heraus kommen
`umap_gruppen.png` / `tsne_gruppen.png` (Karte nach Gruppe), das
Gruppenprofil `*_gruppenprofil.png` und `*_gruppen.xlsx` mit denselben
Blättern wie bei der PCA. Das Profil ist bei diesen Verfahren der einzige Weg
zu sehen, was eine Insel ausmacht - die Kartenachsen selbst sind nicht
deutbar. Die Gruppennummern der drei Skripte sind unabhängig vergeben;
„Gruppe 2“ bei UMAP muss nicht „Gruppe 2“ bei PCA sein - für abgeglichene
Nummern `run_vergleich.py` nehmen.

**Vorsicht beim Lesen von UMAP und t-SNE:** die Achsen haben keine Einheit
(deshalb sind sie unbeschriftet), die Abstände *zwischen* Clustern bedeuten bei
t-SNE nichts, und Clustergrößen sind verzerrt. Belastbar ist nur: „diese Punkte
liegen beieinander". Der ausgegebene **Trustworthiness**-Wert sagt, wie viel
der ursprünglichen Nachbarschaft erhalten blieb — ab etwa 0.90 sind die
sichtbaren Gruppen vertrauenswürdig. Was im `--sweep` über alle Einstellungen
zusammenbleibt, ist echte Struktur; was nur bei einer Einstellung auftaucht,
ist ein Artefakt.

### `run_vergleich.py` — finden alle drei Verfahren dieselben Gruppen?

```powershell
python run_vergleich.py                       # KMeans, Gruppenzahl je Verfahren frei
python run_vergleich.py --cluster hdbscan
```

Rechnet PCA, UMAP und t-SNE auf denselben Daten, sucht in jeder Darstellung
**unabhängig** Gruppen und vergleicht sie. Die Nummern werden an PCA
ausgerichtet (ungarische Methode), damit „Gruppe 2“ überall dieselbe Gruppe
meint.

| Option | Wirkung |
|---|---|
| `--cluster` | `kmeans` (Standard), `hdbscan`, `gmm` — für alle drei gleich |
| `--gruppen K` | feste Gruppenzahl; ohne Angabe wählt jedes Verfahren selbst (strengerer Test) |
| `--nachbarn`, `--min-dist` | UMAP; `min-dist` steht hier auf 0.0, das packt Gruppen dicht |
| `--perplexity`, `--tsne-start` | t-SNE |

Ausgabe: `vergleich_ueberschneidung.png` (Kreuztabelle je Verfahrenspaar,
Übereinstimmung auf der Diagonalen, ARI im Titel), `vergleich_karten.png`
(3×3-Raster: in jeder Zeile gibt ein Verfahren links die Gruppen vor, rechts
daneben stehen die Karten der beiden anderen Verfahren in denselben Farben;
ein Ring heißt, das gezeigte Verfahren ordnet den Punkt selbst anders zu),
`vergleich_gruppen.xlsx` (Gruppe jeder Zeile unter jedem Verfahren, Blatt
`Uneinig` mit den strittigen Zeilen).

**Lesen:** ARI ≥ 0.8 zwischen allen Paaren heißt, die Struktur steckt in den
Daten und nicht im Verfahren. Eine Gruppe, die nur ein Verfahren findet, ist
verdächtig. Uneinige Punkte sind fast immer Grenzfälle zwischen zwei Gruppen.
Bei reinem Rauschen liegt der ARI um 0.1–0.2.

### `run_streumatrix.py` — jede Messgröße gegen jede

```powershell
python run_streumatrix.py --cluster kmeans
python run_streumatrix.py --spalten Drehzahl_rpm Strom_A Vibration_mm_s
```

Unteres Dreieck samt Diagonale (gestrichelt), Originaleinheiten: Zeile = y-Achse, Spalte = x-Achse, jedes Feld
mit winzigen Achsennamen. Der Rahmen zeigt |r| in fünf Stufen (< 0.3 grau,
dann 0.3 / 0.5 / 0.7 / 0.9 immer dunkleres Rot und dickerer Strich). Auf der
Diagonalen steht jede Größe gegen sich selbst, die Punkte liegen dort auf einer
Geraden. Oben links in jedem Feld die Korrelation r (fett ab |r| ≥ 0.8).
Eingefärbt wird nach der Label-Spalte oder mit `--cluster` nach den Gruppen
aus der PCA. `--spalten` wählt Größen aus (ab etwa 20 wird es eng),
`--max-punkte` begrenzt die gezeichneten Zeilen (Standard 3000).
Ausgabe: `streumatrix.png`.

## Testdaten

`generate_testdata.py` erzeugt einen simulierten Antriebsprüfstand: 15
Messgrößen plus `Pruefling_ID` und `Betriebszustand`, Standard 300 Zeilen.

```powershell
python generate_testdata.py --zeilen 500 --seed 7
```

Die Daten sind so gebaut, dass in den Plots etwas zu sehen ist:

* **Drehmoment, Strom, Leistung** hängen physikalisch zusammen → ihre Pfeile
  zeigen fast in dieselbe Richtung.
* **Wirkungsgrad** läuft den Temperaturen entgegen → Pfeil in Gegenrichtung.
* **Vibration, Körperschall, Schalldruck** bilden eine eigene Richtung, die
  „Lagerschaden" von „Überlast" trennt → zwei getrennte Cluster.
* **Spannung und Betriebsstunden** sind fast reines Rauschen → kurze Pfeile.

Zum Vergleich erzeugt `python generate_testdata.py --rauschen` die Datei
`data/rauschen_messdaten.xlsx`: dieselben 15 Messgrößen mit denselben
Mittelwerten und Streuungen, aber jede Spalte unabhängig gezogen. So sieht es
aus, wenn es nichts zu finden gibt: flacher Scree-Plot (jede PC ~6–9 %),
Pfeile in alle Richtungen, Silhouette unter 0.25, Stabilität unter 0.75,
ARI zwischen den Verfahren um 0.1–0.2.

## Echte Beispieldaten: NASA C-MAPSS

`cmapss_zu_excel.py` (eigenständig, nur pandas/numpy/openpyxl) macht aus den
Triebwerksdaten in `data/CMAPSSData/` eine Excel im Stil der Testdaten:
Sensoren mit sprechenden Namen und Einheiten (°C, bar), dazu die Textspalten
`Pruefling_ID`, `Triebwerk`, `Betriebspunkt` (Flugzustand) und `Zustand`
(nach Restlebensdauer). Zyklus und Restlebensdauer stehen auf dem zweiten
Blatt, damit sie nicht als Messgrößen zählen.

```powershell
python cmapss_zu_excel.py                            # FD002, Training, 3000 Zeilen
python cmapss_zu_excel.py --satz FD001 --alle
python cmapss_zu_excel.py --ohne-einstellungen       # Sensoren müssen die Flugzustände allein verraten
python run_all.py --datei data/cmapss_FD002_train.xlsx --label-spalte Betriebspunkt -o output_cmapss
```

FD002/FD004 enthalten 6 Betriebspunkte — ein guter Test, ob die Gruppensuche
bekannte Zustände wiederfindet. FD001/FD003 haben nur einen Betriebspunkt;
dort bleibt nur der Verschleiß als Struktur. Standard ist eine Stichprobe von
3000 Zeilen, weil t-SNE und die Stabilitätsprüfung bei zehntausenden Zeilen
sehr langsam werden.

Achtung: Ohne `--datei` nehmen die Skripte die *neueste* Excel in `data/` —
nach dem Umwandeln also die C-MAPSS-Datei.

## Projektstruktur

```
hda/
  data_io.py     Excel einlesen, Merkmale/Labels trennen, skalieren
  plotstyle.py   Farben, rcParams, Streudiagramm nach Gruppe
  embedding.py   t-SNE-Lauf, gemeinsame Darstellung für UMAP/t-SNE, Trustworthiness
  cluster.py     Gruppen ohne Labels finden, prüfen, vergleichen
  gruppen.py     Gruppen beschreiben: Profil, Vertreter, Excel (für alle Verfahren)
  cli.py         gemeinsame Kommandozeilenoptionen
generate_testdata.py
cmapss_zu_excel.py   NASA-Triebwerksdaten -> Excel
run_all.py       alle Schritte nacheinander
run_pca.py  run_umap.py  run_tsne.py  run_vergleich.py  run_streumatrix.py
data/    Eingabe-Excel
output/  Plots und Ergebnistabellen
```

Alle Plots tragen einen geschlossenen Rahmen mit nach innen zeigenden Haupt-
und Nebenteilstrichen — im hellen Schema schwarz, im dunklen weiß, weil ein
fest schwarzer Rahmen auf dunklem Grund unsichtbar wäre.

Die Farbpalette ist auf Farbfehlsichtigkeit geprüft. Gruppen werden zusätzlich
über die **Markerform** unterschieden, damit die Zuordnung nie allein an der
Farbe hängt — wichtig, sobald die Plots ausgedruckt oder in einen Bericht
übernommen werden. Ab neun Gruppen werden die überzähligen zu „Sonstige"
zusammengefasst, statt neue Farben zu erfinden.
