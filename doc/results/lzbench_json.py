#!/usr/bin/env python3
"""Convert lzbench results into the JSON file read by doc/results/index.html.

Single-threaded results come from one run of lzbench with -o1 (markdown) or
-o4 (CSV) output, for example:

    lzbench -eALL -t8,8 -o4 silesia.tar > st.csv

"-eALL" expands to LZ/LZ+ENTROPY/SYMMETRIC and each of these aliases starts
with memcpy, so the memcpy rows split the results into groups named by
--groups. With --aliases bench/lzbench.h, each result is put in the alias of
--groups that lists its codec and level instead, which also regroups results
from lzbench versions with other aliases (e.g. 2.4: LZ/SYMMETRIC/MISC).
Multi-threaded results are one file per thread count, for example:

    lzbench -eALL -b4096 -T8 -t8,8 -o1 silesia.tar > mt8.md

and their groups are taken from the single-threaded results of the same codec.

Example:

    doc/results/lzbench_json.py --title "lzbench 2.4 on Silesia Corpus" \\
        --single st.csv --single-note "..." \\
        --multi 1=mt1.csv --multi 8=mt8.csv --multi 32=mt32.csv --multi-note "..." \\
        > doc/results/lzbench24_9555p.json

In the notes, text in `backticks` is shown as code.
"""

import argparse
import csv
import json
import re
import sys

# -T# adds a "C,D Threads" column: the threads used for compression and decompression
MD_ROW = re.compile(r'^\|\s*(?P<n>[^|]+?)\s*\|(?:\s*(?P<ct>\d+),\s*(?P<dt>\d+)\s*\|)?'
                    r'\s*(?P<c>[\d.]+) MB/s\s*\|\s*(?P<d>[\d.]+) MB/s\s*\|'
                    r'\s*(?P<s>\d+)\s*\|\s*(?P<r>[\d.]+)\s*\|\s*(?P<f>[^|]+?)\s*\|')


def read_results(path):
    """Return [(name, compr MB/s, decompr MB/s, compressed size, ratio %, filename, threads)]
    in run order, where threads is (compression, decompression) or None if not shown."""
    rows = []
    with open(path, newline='') as f:
        text = f.read()
    if text.lstrip().startswith('Compressor name,'):
        for rec in csv.DictReader(text.splitlines()):
            rows.append((rec['Compressor name'].strip(), float(rec['Compression speed']),
                         float(rec['Decompression speed']), int(rec['Compressed size']),
                         float(rec['Ratio']), rec['Filename'].strip(), None))
    else:
        for line in text.splitlines():
            m = MD_ROW.match(line)
            if m:
                threads = (int(m['ct']), int(m['dt'])) if m['ct'] else None
                rows.append((m['n'], float(m['c']), float(m['d']), int(m['s']), float(m['r']), m['f'], threads))
    if not rows:
        sys.exit(f'{path}: no lzbench results found (expected -o1 or -o4 output)')
    files = {r[5] for r in rows}
    if len(files) > 1:
        sys.exit(f'{path}: results for more than one file: {", ".join(sorted(files))}')
    return rows


def single_threaded(path, groups):
    out, seen, group = [], set(), -1
    for n, c, d, s, r, _, _ in read_results(path):
        if n == 'memcpy':
            group += 1
        # -o1c# prints a sorted copy of the table after the results; a repeated
        # compressor or one memcpy too many means it has started
        if group >= len(groups) or (n != 'memcpy' and n in seen):
            break
        if n in seen:
            continue
        if group < 0:
            sys.exit(f'{path}: the results do not start with memcpy, so the groups are unknown')
        seen.add(n)
        out.append({'n': n, 'g': '' if n == 'memcpy' else groups[group], 'c': c, 'd': d, 's': s, 'r': r})
    return out


# a codec in comp_desc: { "name", "name_version", ["algorithm",] first_level, last_level, ...
CODEC_ROW = r'\{\s*"([^"]+)",\s*"([^"]+)",\s*(?:"([^"]*)",\s*)?(-?\d+),\s*(-?\d+),'


def split_name(name):
    """A result name "<name_version> -<level>" or "<name_version>" -> (name_version, level or None)."""
    n, _, level = name.rpartition(' -')
    try:
        return n, int(level)
    except ValueError:
        return name, None


def algorithms(header):
    """Return a function (name_version, level) -> algorithm, from comp_desc and algorithm_by_level."""
    h = open(header).read()
    by_version = {m[1]: (m[0], m[2]) for m in re.findall(CODEC_ROW, h)}
    seg = h[h.index('algorithm_by_level[]'):]
    seg = seg[:seg.index('};')]
    per_level = re.findall(r'\{\s*"([^"]+)",\s*(-?\d+),\s*(-?\d+),\s*"([^"]+)"\s*\}', seg)
    def algorithm(version, level):
        name, alg = by_version.get(version, (None, None))
        for n, lo, hi, a in per_level:
            if n == name and level is not None and int(lo) <= level <= int(hi):
                return a
        return alg
    return algorithm


def add_algorithms(rows, algorithm):
    missing = []
    for r in rows:
        r['a'] = algorithm(*split_name(r['n']))
        if not r['a']:
            missing.append(r['n'])
    if missing:
        sys.exit('--algorithms: no algorithm for ' + ', '.join(missing))


def alias_groups(header, names):
    """Map (codec, level) and (codec, None) -> group, from the aliases in lzbench.h."""
    h = open(header).read()
    versions = {m[0]: m[1] for m in re.findall(CODEC_ROW, h)}
    seg = h[h.index('alias_desc[]'):]
    aliases = {}
    for m in re.finditer(r'\{\s*"([A-Za-z_0-9+]+)",\s*"[^"]*",((?:\s|\\|/\*.*?\*/|"[^"]*")+)\}', seg, re.S):
        body = re.sub(r'/\*.*?\*/', '', m.group(2), flags=re.S)
        aliases[m.group(1).upper()] = ''.join(re.findall(r'"([^"]*)"', body))
    def expand(a):
        out = []
        for part in aliases[a].split('/'):
            n, *levels = part.split(',')
            if n.upper() in aliases and n not in versions:
                out += expand(n.upper())
            elif n != 'memcpy':
                out += [(n, int(l)) for l in levels] or [(n, None)]
        return out
    group = {}
    for g in names:
        if g.upper() not in aliases:
            sys.exit(f'{header}: no alias {g}')
        for n, l in expand(g.upper()):
            group.setdefault((versions[n], l), g)
    return group


def regroup(rows, group):
    """Set each row's group from its name ("<name_version>" or "<name_version> -<level>")."""
    missing = []
    for r in rows:
        if r['n'] == 'memcpy':
            continue
        key = split_name(r['n'])
        g = group.get(key) or group.get((key[0], None))
        if g is None and key[1] is not None:
            # a level no alias lists (e.g. from an older lzbench): use the codec's nearest listed level
            listed = [l for (v, l) in group if v == key[0] and l is not None]
            if listed:
                g = group[(key[0], min(listed, key=lambda l: (abs(l - key[1]), l)))]
        if g is None:
            missing.append(r['n'])
        else:
            r['g'] = g
    if missing:
        sys.exit('--aliases: in none of the --groups aliases: ' + ', '.join(missing))


def codec(name):
    return name.split(' ')[0]


def multi_threaded(specs, single):
    by_name = {r['n']: r['g'] for r in single}
    by_codec = {}
    for r in single:
        by_codec.setdefault(codec(r['n']), r['g'])
    out = {}
    for spec in specs:
        threads, _, path = spec.partition('=')
        if not threads.isdigit() or not path:
            sys.exit(f'--multi {spec}: expected THREADS=FILE')
        seen = set()
        for n, c, d, s, r, _, used in read_results(path):
            # keep the first memcpy, and stop at the sorted copy that -o1c# prints
            if n in seen:
                if n == 'memcpy':
                    continue
                break
            seen.add(n)
            row = out.setdefault(n, {'n': n, 'g': by_name.get(n, by_codec.get(codec(n), '')) if n != 'memcpy' else '',
                                     's': s, 'r': r, 'c': {}, 'd': {}})
            row['c'][threads] = c
            row['d'][threads] = d
            # codecs without multithreading (e.g. crush, tornado) run on fewer threads
            if used and used != (int(threads), int(threads)):
                row.setdefault('th', {})[threads] = list(used)
    counts = {len(r['c']) for r in out.values()}
    if len(counts) > 1:
        sys.exit('--multi: the files do not have the same compressors')
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--title', required=True)
    ap.add_argument('--single', required=True, metavar='FILE', help='single-threaded results (-o1 or -o4)')
    ap.add_argument('--groups', default='LZ,LZ+ENTROPY,SYMMETRIC',
                    help='names of the memcpy-separated groups in --single (default: %(default)s)')
    ap.add_argument('--aliases', metavar='LZBENCH_H',
                    help='take the groups from the aliases in this lzbench.h instead of the memcpy rows')
    ap.add_argument('--algorithms', metavar='LZBENCH_H',
                    help="add each result's algorithm from comp_desc and algorithm_by_level in this lzbench.h")
    ap.add_argument('--single-note', default='', help='how the single-threaded results were made')
    ap.add_argument('--multi', action='append', default=[], metavar='THREADS=FILE',
                    help='multi-threaded results for one thread count (repeat for each)')
    ap.add_argument('--multi-note', default='', help='how the multi-threaded results were made; # is the thread count')
    ap.add_argument('--multi-threads-only', action='store_true',
                    help='leave out multi-threaded results of codecs that ran on fewer threads (no multithreading)')
    ap.add_argument('--exclude', action='append', default=[], metavar='NAME',
                    help='leave out a result, e.g. "tornado 0.6a -1" (repeat for each)')
    args = ap.parse_args()

    single = [r for r in single_threaded(args.single, args.groups.split(',')) if r['n'] not in args.exclude]
    if args.aliases:
        regroup(single, alias_groups(args.aliases, args.groups.split(',')))
    data = {
        'title': args.title,
        'orig': next(r['s'] for r in single if r['n'] == 'memcpy'),
        'file': read_results(args.single)[0][5],
        'single': {'note': args.single_note, 'rows': single},
    }
    if args.algorithms:
        add_algorithms(single, algorithms(args.algorithms))
    if args.multi:
        data['multi'] = {'note': args.multi_note,
                         'rows': [r for r in multi_threaded(args.multi, single) if r['n'] not in args.exclude
                                  and not (args.multi_threads_only and 'th' in r)]}
        if args.algorithms:
            add_algorithms(data['multi']['rows'], algorithms(args.algorithms))
    json.dump(data, sys.stdout, separators=(',', ':'))
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
