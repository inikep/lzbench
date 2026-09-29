#!/usr/bin/env python3
"""Draw the single-threaded results of a JSON file made by lzbench_json.py as a
static SVG chart for README.md: compression ratio against compression and against
decompression speed, with the Pareto frontier. The chart follows the viewer's
light or dark color scheme.

    doc/results/lzbench_svg.py doc/results/lzbench24_9555p.json \\
        --subtitle "AMD EPYC 9555P, single thread" > doc/results/lzbench24_9555p.svg
"""

import argparse
import json
import math
from xml.sax.saxutils import escape

W, H = 880, 420
PANEL_TOP, PANEL_BOTTOM = 86, 56
LEFT, GAP, RIGHT = 48, 48, 30
GROUPS = [('LZ', 'lz'), ('SYMMETRIC', 'sym'), ('MISC', 'misc')]

STYLE = """
  .bg { fill: #fcfcfb; stroke: rgba(11,11,11,0.10); }
  .grid { stroke: #e1e0d9; }
  .axis { stroke: #c3c2b7; }
  .front { stroke: #808080; }
  text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; fill: #7a7873; font-size: 12px; }
  .title { fill: #0b0b0b; font-size: 16px; font-weight: 650; }
  .sub, .panel, .key { fill: #52514e; font-size: 13px; }
  .panel { font-weight: 600; }
  .label { fill: #52514e; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; }
  .link { fill: #2a78d6; font-size: 13px; font-weight: 600; }
  .ring { stroke: #fcfcfb; }
  .lz { fill: #2a78d6; } .sym { fill: #eb6834; } .misc { fill: #1baf7a; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; stroke: rgba(255,255,255,0.10); }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .front { stroke: #ffffff; }
    text { fill: #9a988f; }
    .title { fill: #ffffff; }
    .sub, .panel, .key, .label { fill: #c3c2b7; }
    .link { fill: #6da7ec; }
    .ring { stroke: #1a1a19; }
    .lz { fill: #3987e5; } .sym { fill: #d95926; } .misc { fill: #199e70; }
  }
"""


def frontier(rows, metric):
    """Results that no other result is both faster and better (lower ratio) than, fastest first."""
    out, best = [], math.inf
    for r in sorted(rows, key=lambda r: (-r[metric], r['s'])):
        if r['s'] < best:
            out.append(r)
            best = r['s']
    return out


def fmt_int(v):
    return f'{v:,}'


def panel(out, rows, metric, x0, width, y_lo, y_hi, name):
    top, height = PANEL_TOP, H - PANEL_TOP - PANEL_BOTTOM
    speeds = [r[metric] for r in rows]
    lx0, lx1 = math.floor(math.log10(min(speeds))), math.ceil(math.log10(max(speeds)))
    X = lambda v: x0 + (math.log10(v) - lx0) / (lx1 - lx0) * width
    Y = lambda v: top + (v - y_lo) / (y_hi - y_lo) * height

    out.append(f'<text class="panel" x="{x0}" y="{top - 12}">{name}</text>')
    for e in range(lx0, lx1 + 1):
        x = X(10 ** e)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + height}"/>')
        out.append(f'<text x="{x:.1f}" y="{top + height + 17}" text-anchor="middle">{fmt_int(10 ** e) if e >= 0 else 10 ** e}</text>')
    for v in range(y_lo, y_hi + 1, 10):
        out.append(f'<line class="grid" x1="{x0}" x2="{x0 + width}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x0 + width}" y1="{top + height}" y2="{top + height}"/>')
    out.append(f'<text x="{x0 + width / 2:.1f}" y="{H - 22}" text-anchor="middle">MB/s, log scale; faster to the right</text>')

    front = frontier(rows, metric)
    on = {r['n'] for r in front}
    for r in sorted(rows, key=lambda r: r['n'] in on):
        cls = dict(GROUPS)[r['g']]
        rad = 4.5 if r['n'] in on else 3.5
        out.append(f'<circle class="{cls} ring" stroke-width="1.5" cx="{X(r[metric]):.1f}" cy="{Y(r["r"]):.1f}" r="{rad}"/>')
    pts = ' '.join(f'{X(r[metric]):.1f},{Y(r["r"]):.1f}' for r in front)
    out.append(f'<polyline class="front" fill="none" stroke-width="2" stroke-linejoin="round" opacity="0.85" points="{pts}"/>')
    # label the smallest result; the fastest ones sit among other points
    small = front[-1]
    out.append(f'<text class="label" x="{X(small[metric]) + 8:.1f}" y="{Y(small["r"]) - 9:.1f}">{escape(small["n"])}</text>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json', help='a JSON file made by lzbench_json.py')
    ap.add_argument('--subtitle', default='', help='shown after the title, e.g. the machine')
    args = ap.parse_args()
    with open(args.json) as f:
        data = json.load(f)
    rows = [r for r in data['single']['rows'] if r['n'] != 'memcpy']
    ratios = [r['r'] for r in rows]
    y_lo, y_hi = math.floor(min(ratios) / 10) * 10, math.ceil(max(ratios) / 10) * 10

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'role="img" aria-label="{escape(data["title"])}: compression ratio against compression and decompression speed">',
           f'<style>{STYLE}</style>',
           f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12"/>',
           f'<text class="title" x="{LEFT - 28}" y="30">{escape(data["title"])}</text>']
    if args.subtitle:
        out.append(f'<text class="sub" x="{LEFT - 28}" y="50">{escape(args.subtitle)}; '
                   f"ratio in %, lower is better</text>")
    out.append(f'<text class="link" x="{W - 16}" y="30" text-anchor="end">Open the interactive version →</text>')

    # legend
    x = W - 16
    items = [(g, c, sum(r['g'] == g for r in rows)) for g, c in GROUPS]
    legend = [f'{g} ({n})' for g, _, n in items]
    widths = [len(t) * 7 + 22 for t in legend] + [118]
    x -= sum(widths)
    for (g, c, n), t, w in zip(items, legend, widths):
        out.append(f'<circle class="{c}" cx="{x + 5}" cy="46" r="4.5"/><text class="key" x="{x + 14}" y="50">{t}</text>')
        x += w
    out.append(f'<line class="front" stroke-width="2" x1="{x}" x2="{x + 16}" y1="46" y2="46"/>'
               f'<text class="key" x="{x + 22}" y="50">Pareto frontier</text>')

    top, height = PANEL_TOP, H - PANEL_TOP - PANEL_BOTTOM
    for v in range(y_lo, y_hi + 1, 10):
        y = top + (v - y_lo) / (y_hi - y_lo) * height
        out.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{v}%</text>')
    width = (W - LEFT - GAP - RIGHT) / 2
    panel(out, rows, 'c', LEFT, width, y_lo, y_hi, 'Compression ratio vs. compression speed')
    panel(out, rows, 'd', LEFT + width + GAP, width, y_lo, y_hi, 'Compression ratio vs. decompression speed')
    out.append('</svg>')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
