"""Alle Analysen nacheinander: PCA (mit Gruppensuche), UMAP, t-SNE, Vergleich,
Streudiagramm-Matrix.

Die gemeinsamen Optionen (Datei, Skalierung, Theme, ...) gelten fuer alle
Schritte. Jeder Schritt laeuft als eigener Prozess - scheitert einer, laufen
die anderen trotzdem, und am Ende steht, was geklappt hat.

Aufruf:  python run_all.py
         python run_all.py --datei data/meine.xlsx --cluster hdbscan --sweep
         python run_all.py --nur pca vergleich
         python run_all.py --testdaten
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

from hda.cli import basis_parser

HIER = Path(__file__).resolve().parent
SCHRITTE = ["pca", "umap", "tsne", "vergleich", "streumatrix"]


def gemeinsame_optionen(args) -> list[str]:
    """Die Optionen aus basis_parser so zusammensetzen, wie jedes Skript sie erwartet."""
    # Pfade absolut machen: die Schritte laufen im Projektordner, der Aufruf
    # kann aber von woanders kommen.
    opt = ["--blatt", str(args.blatt), "--skalierung", args.skalierung,
           "--nan", args.nan, "--out", str(Path(args.out).resolve()),
           "--theme", args.theme, "--seed", str(args.seed)]
    if args.datei:
        opt += ["--datei", str(Path(args.datei).resolve())]
    if args.label_spalte:
        opt += ["--label-spalte", args.label_spalte]
    if args.id_spalten:
        opt += ["--id-spalten", *args.id_spalten]
    if args.zeigen:
        opt.append("--zeigen")
    return opt


def befehle(args) -> list[tuple[str, list[str]]]:
    basis = gemeinsame_optionen(args)
    gruppen = ["--cluster", args.cluster]
    if args.gruppen:
        gruppen += ["--gruppen", str(args.gruppen)]
    sweep = ["--sweep"] if args.sweep else []
    alle = {
        "pca": ["run_pca.py", *basis, *gruppen, "--paare", "1,2", "1,3"],
        "umap": ["run_umap.py", *basis, *sweep],
        "tsne": ["run_tsne.py", *basis, *sweep],
        "vergleich": ["run_vergleich.py", *basis, *gruppen],
        "streumatrix": ["run_streumatrix.py", *basis, *gruppen],
    }
    return [(name, alle[name]) for name in SCHRITTE if name in args.nur]


def main() -> None:
    p = basis_parser("Alle Analysen nacheinander ausfuehren.")
    p.add_argument("--nur", nargs="+", default=SCHRITTE, choices=SCHRITTE,
                   help="nur diese Schritte ausfuehren")
    p.add_argument("--cluster", default="kmeans",
                   choices=["hdbscan", "kmeans", "gmm"],
                   help="Gruppensuche fuer PCA und Vergleich")
    p.add_argument("--gruppen", type=int, default=None,
                   help="feste Gruppenzahl fuer kmeans/gmm (Standard: automatisch)")
    p.add_argument("--sweep", action="store_true",
                   help="UMAP und t-SNE zusaetzlich mit Parameterraster")
    p.add_argument("--testdaten", action="store_true",
                   help="vorher data/beispiel_messdaten.xlsx neu erzeugen")
    args = p.parse_args()

    plan = befehle(args)
    if args.testdaten:
        plan.insert(0, ("testdaten", ["generate_testdata.py"]))

    ergebnisse = []
    for name, befehl in plan:
        if name == "umap" and importlib.util.find_spec("umap") is None:
            print("\n=== umap: uebersprungen (umap-learn nicht installiert) ===")
            ergebnisse.append((name, "uebersprungen", 0.0))
            continue
        print("\n=== %s: %s ===" % (name, " ".join(befehl)), flush=True)
        start = time.perf_counter()
        rc = subprocess.run([sys.executable, *befehl], cwd=HIER).returncode
        dauer = time.perf_counter() - start
        ergebnisse.append((name, "ok" if rc == 0 else "FEHLER (Code %d)" % rc, dauer))

    print("\n=== Zusammenfassung ===")
    for name, status, dauer in ergebnisse:
        print("  %-12s %-18s %6.1f s" % (name, status, dauer))
    print("  Ergebnisse in: %s" % Path(args.out).resolve())
    if any(s.startswith("FEHLER") for _, s, _ in ergebnisse):
        sys.exit(1)


if __name__ == "__main__":
    main()
