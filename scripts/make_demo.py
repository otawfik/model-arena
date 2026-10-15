"""Generate screenshots/demo.svg: a hand-built mockup of the arena UI.

Kept small on purpose: the GitHub push path caps single file args at 128KB.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "screenshots", "demo.svg")

ROWS = [
    ("1", "Naive Bayes", "1305.8", "1.000", "9-0-6"),
    ("2", "Linear SVM", "1303.9", "0.999", "9-0-6"),
    ("3", "Logistic Regression", "1302.8", "1.000", "9-0-6"),
    ("4", "k-Nearest Neighbors", "1162.7", "0.995", "6-9-0"),
    ("5", "Random Forest", "1095.4", "0.984", "3-12-0"),
    ("6", "Dummy (majority class)", "1029.4", "0.200", "0-15-0"),
]

BG, PANEL, GOLD, TEXT, DIM = "#0d1117", "#161b22", "#f0b429", "#e6edf3", "#8b949e"


def row_svg(i, rank, name, elo, acc, rec):
    y = 120 + i * 44
    name_w = "700" if i == 0 else "400"
    return (
        f'<rect x="40" y="{y}" width="720" height="38" rx="6" fill="{PANEL}"/>'
        f'<text x="60" y="{y + 24}" fill="{DIM}" font-size="14" '
        f'font-family="monospace">{rank}</text>'
        f'<text x="100" y="{y + 24}" fill="{TEXT}" font-size="14" '
        f'font-weight="{name_w}" font-family="sans-serif">{name}</text>'
        f'<text x="420" y="{y + 24}" fill="{GOLD}" font-size="14" '
        f'font-weight="700" font-family="monospace">{elo}</text>'
        f'<text x="530" y="{y + 24}" fill="{TEXT}" font-size="14" '
        f'font-family="monospace">{acc}</text>'
        f'<text x="640" y="{y + 24}" fill="{DIM}" font-size="14" '
        f'font-family="monospace">{rec}</text>'
    )


def main():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480" '
        f'viewBox="0 0 800 480">',
        f'<rect width="800" height="480" fill="{BG}"/>',
        f'<text x="40" y="48" fill="{TEXT}" font-size="26" font-weight="700" '
        f'font-family="sans-serif">\U0001F3C6 Model Arena</text>',
        f'<text x="40" y="72" fill="{DIM}" font-size="13" '
        f'font-family="sans-serif">Six classifiers battle head-to-head on '
        f'topic classification. Winners climb the ELO ladder.</text>',
        f'<text x="60" y="104" fill="{DIM}" font-size="11" '
        f'font-family="monospace">#   MODEL                 ELO      CV ACC   W-L-T</text>',
    ]
    for i, r in enumerate(ROWS):
        parts.append(row_svg(i, *r))
    parts += [
        f'<rect x="40" y="404" width="720" height="52" rx="8" fill="{PANEL}"/>',
        f'<text x="60" y="426" fill="{TEXT}" font-size="13" '
        f'font-family="sans-serif">Head-to-head: \u201cthe striker scored the '
        f'winning goal\u201d</text>',
        f'<text x="60" y="446" fill="{DIM}" font-size="12" '
        f'font-family="sans-serif">Logistic Regression: <tspan fill="#3fb950">'
        f'sports \u2713</tspan>  vs  Dummy: <tspan fill="#f85149">travel '
        f'\u2717</tspan> \u2014 \U0001F3C6 Logistic Regression wins</text>',
        '</svg>',
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
