#!/usr/bin/env python3
"""Check every codec in lzbench for a newer upstream version.

Each mk/<name>.mk describes where its codec comes from with one line:

    NAME_UPSTREAM := <method> <source> [pattern=<regex>] [desc=<name>] [readme=<name>] [prerelease]

The methods are:

    git-tags   <git URL>        newest tag of a git repository (git ls-remote)
    gh-release <owner/repo>     tag of the latest GitHub release, for repositories with tags that
                                are not releases
    http       <URL>            newest version found by pattern= in a web page, directory listing
                                or RSS feed
    git-head   <git URL>        date of the last commit (in UTC), for codecs taken from a commit
                                (fetches that commit only, without its files)
    manual     <URL>            cannot be checked automatically: print the URL to check by hand
    frozen     <reason...>      no longer developed: not checked

pattern= is a regular expression whose first group is the version. For git-tags and gh-release it
must match the whole tag; the default, v?(\\d+(?:\\.\\d+)*), accepts release tags such as v1.2.3 or
1.2 and skips tags like v1.2.0rc1 or zstd-0.4.2. prerelease allows tags such as v1.4-dev4. The mk
files are read by make too, so patterns cannot contain '$' or '#'; the whole tag is matched instead
of using '^' and '$'.

lzbench's version is the second word of the codec's version string in bench/lzbench.h, from the
entry named by desc= (default: the codec name in lower case, with '-' for '_'). A codec taken from
a commit has a date there, or else the "Last update" date of its row in README.md (the row is found
by desc=, the codec name, or readme=). The README version is also compared with bench/lzbench.h.

No method uses the GitHub API, so no token is needed and there is no rate limit to hit.

Exit status: 2 when an upstream could not be checked or a codec has no NAME_UPSTREAM line; with
--exit-code, 1 when an update or a mismatch was found; otherwise 0.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

USER_AGENT = 'lzbench-check-updates'
TIMEOUT = 60
DEFAULT_TAGS = r'v?(\d+(?:\.\d+)*)'
DEFAULT_PRERELEASE_TAGS = r'v?(\d[\w.\-]*)'
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
OPTIONS = ('pattern', 'desc', 'readme')
FLAGS = ('prerelease',)


def read_upstreams(root):
    """NAME -> {method, source, options...} from the NAME_UPSTREAM lines of mk/*.mk."""
    codecs = {}
    mk = os.path.join(root, 'mk')
    for f in sorted(os.listdir(mk)):
        if not f.endswith('.mk'):
            continue
        text = open(os.path.join(mk, f)).read()
        names = re.findall(r'^\s*CODECS\s*\+=\s*(\S+)', text, re.M)
        for m in re.finditer(r'^\s*([A-Z0-9_]+)_UPSTREAM\s*:?=\s*(.*?)\s*$', text, re.M):
            words = m.group(2).split()
            u = {'name': m.group(1), 'file': 'mk/' + f, 'method': words[0] if words else '',
                 'source': words[1] if len(words) > 1 else '', 'args': words[2:]}
            for w in words[2:]:
                k, eq, v = w.partition('=')
                if k in OPTIONS and eq:
                    u[k] = v
                elif w in FLAGS:
                    u[w] = True
            codecs[u['name']] = u
        for n in names:
            if n not in codecs:
                codecs[n] = {'name': n, 'file': 'mk/' + f, 'method': None}
    return codecs


def read_versions(root):
    """comp_desc name -> version string, from bench/lzbench.h."""
    h = open(os.path.join(root, 'bench', 'lzbench.h')).read()
    h = h[h.index('comp_desc[]'):h.index('alias_desc[]')]
    return dict(re.findall(r'\{\s*"([^"]+)",\s*"([^"]+)",', h))


def read_readme(root):
    """name -> (version, date) from the table of compressors in README.md."""
    rows = {}
    for line in open(os.path.join(root, 'README.md')):
        m = re.match(r'\|\s*\[([^\]]+)\]\([^)]*\)\s*\|\s*([^|]*?)\s*\|', line)
        if not m:
            continue
        for group in m.group(1).split(', '):  # "wlz4/wlz4hc 1.3.0, wzip 1.3.0"
            words = group.split()
            if len(words) >= 2:
                for name in words[0].split('/'):
                    rows[name] = (words[1].lstrip('v'), m.group(2))
    return rows


def version_key(v):
    """Sort key for versions: 1.10 > 1.9, 1.4 > 1.4-dev3 > 1.4-dev2, 1.1.0 == 1.1."""
    key = [(2, int(p)) if p.isdigit() else (0, p.lower()) for p in re.findall(r'\d+|[A-Za-z]+', v)]
    while key and key[-1] == (2, 0):
        key.pop()
    return key + [(1, '')]


def newest(versions):
    versions = list(versions)
    return max(versions, key=version_key) if versions else None


def http_get(url):
    """Return the page and its URL after redirects."""
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode('utf-8', 'replace'), r.geturl()


def git(*args):
    out = subprocess.run(['git'] + list(args), capture_output=True, text=True, timeout=TIMEOUT,
                         env=dict(os.environ, GIT_TERMINAL_PROMPT='0', TZ='UTC'))
    if out.returncode:
        err = out.stderr.strip().splitlines()
        raise RuntimeError(err[-1] if err else 'git ' + args[0] + ' failed')
    return out.stdout


def match_tags(tags, u):
    pattern = re.compile(u.get('pattern') or (DEFAULT_PRERELEASE_TAGS if u.get('prerelease') else DEFAULT_TAGS))
    return [m.group(1) for m in map(pattern.fullmatch, tags) if m]


def latest_git_tags(u):
    out = git('ls-remote', '--tags', '--refs', u['source'])
    tags = [line.split('refs/tags/', 1)[1] for line in out.splitlines() if 'refs/tags/' in line]
    return newest(match_tags(tags, u))


def latest_gh_release(u):
    # github.com/<owner>/<repo>/releases/latest redirects to .../releases/tag/<tag>
    url = http_get(f"https://github.com/{u['source']}/releases/latest")[1]
    if '/releases/tag/' not in url:
        raise RuntimeError('no GitHub release')
    tag = urllib.parse.unquote(url.split('/releases/tag/', 1)[1])
    found = match_tags([tag], u)
    if not found:
        raise RuntimeError(f'latest release tag {tag!r} does not match the pattern')
    return found[0]


def latest_http(u):
    if not u.get('pattern'):
        raise RuntimeError('http needs pattern=')
    return newest(m.group(1) for m in re.finditer(u['pattern'], http_get(u['source'])[0]))


def latest_git_head(u):
    with tempfile.TemporaryDirectory(prefix='lzbench-check-') as d:
        git('init', '-q', '--bare', d)
        git('-C', d, 'fetch', '-q', '--depth=1', '--filter=tree:0', u['source'], 'HEAD')
        date, sha = git('-C', d, 'log', '-1', '--format=%cd %H', '--date=format-local:%Y-%m-%d', 'FETCH_HEAD').split()
    return date, sha[:10]


def check(u, versions, readme):
    """Return a result dict for one codec, MISMATCH when it is up to date but README.md disagrees."""
    r = check_upstream(u, versions, readme)
    if r.get('readme'):
        note = f"README.md has {r['readme']}"
        r['note'] = f"{r['note']}; {note}" if r.get('note') else note
        if r['status'] in ('ok', 'frozen'):
            r['status'] = 'MISMATCH'
    return r


def check_upstream(u, versions, readme):
    r = {'codec': u['name'], 'file': u['file'], 'method': u.get('method'), 'source': u.get('source', '')}
    if not u.get('method'):
        return dict(r, status='MISSING', note=f"no {u['name']}_UPSTREAM line in {u['file']}")
    desc = u.get('desc') or u['name'].lower().replace('_', '-')
    words = versions.get(desc, '').split()
    current = words[1].lstrip('v') if len(words) > 1 else None
    row = None
    for name in (u.get('readme'), desc, u['name'].lower().replace('_', '-')):
        if name and name in readme:
            row = readme[name]
            break
    r['current'] = current
    if row and current and not DATE.fullmatch(current) and version_key(row[0]) != version_key(current):
        r['readme'] = row[0]
    method = u['method']
    if method == 'frozen':
        return dict(r, status='frozen', note=' '.join(w for w in [u['source']] + u['args'] if w.partition('=')[0] not in OPTIONS))
    if method == 'manual':
        return dict(r, status='MANUAL', note='check ' + u['source'])
    if current is None:
        return dict(r, status='ERROR', note=f'no entry "{desc}" in bench/lzbench.h (set desc=)')
    try:
        if method == 'git-head':
            date = current if DATE.fullmatch(current) else (row[1] if row else None)
            if not date or not DATE.fullmatch(date):
                return dict(r, status='ERROR', note='no date in bench/lzbench.h or README.md')
            head_date, sha = latest_git_head(u)
            r.update(current=date, latest=f'{head_date} ({sha})')
            return dict(r, status='NEWER' if head_date > date else 'ok')
        latest = {'git-tags': latest_git_tags, 'gh-release': latest_gh_release, 'http': latest_http}.get(method)
        if latest is None:
            return dict(r, status='ERROR', note=f'unknown method {method!r}')
        r['latest'] = latest(u)
    except Exception as e:  # network errors, API limits, unexpected pages
        return dict(r, status='ERROR', note=f'{type(e).__name__}: {e}')
    if r['latest'] is None:
        return dict(r, status='ERROR', note='no version matches the pattern')
    if version_key(r['latest']) > version_key(current):
        return dict(r, status='OUTDATED')
    return dict(r, status='ok')


ORDER = ['OUTDATED', 'NEWER', 'MISMATCH', 'MISSING', 'ERROR', 'MANUAL', 'ok', 'frozen']


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('codecs', nargs='*', help='codec names as in mk/*.mk (default: all)')
    p.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                   help='lzbench source tree (default: the parent of this script)')
    p.add_argument('-v', '--verbose', action='store_true', help='also list codecs that are up to date or frozen')
    p.add_argument('--json', action='store_true', help='print the results as JSON')
    p.add_argument('--exit-code', action='store_true', help='exit with 1 when an update or a mismatch was found')
    p.add_argument('-j', '--jobs', type=int, default=16, help='upstreams checked in parallel (default: 16)')
    a = p.parse_args()

    upstreams = read_upstreams(a.root)
    versions = read_versions(a.root)
    readme = read_readme(a.root)
    wanted = [n.upper().replace('-', '_') for n in a.codecs] or sorted(upstreams)
    unknown = [n for n in wanted if n not in upstreams]
    if unknown:
        sys.exit('unknown codecs: ' + ', '.join(unknown))
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda n: check(upstreams[n], versions, readme), wanted))
    results.sort(key=lambda r: (ORDER.index(r['status']), r['codec']))

    if a.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    else:
        for r in results:
            if r['status'] in ('ok', 'frozen') and not a.verbose:
                continue
            newer = r.get('latest') and r['status'] in ('OUTDATED', 'NEWER')
            change = f"{r['current']} -> {r['latest']}" if newer else (r.get('current') or '-')
            print(f"{r['status']:<9} {r['codec']:<11} {change:<34} {r.get('note') or r['source']}")
        counts = {s: sum(r['status'] == s for r in results) for s in ORDER}
        print(', '.join(f'{n} {s}' for s, n in counts.items() if n))
    if any(r['status'] in ('MISSING', 'ERROR') for r in results):
        return 2
    return 1 if a.exit_code and any(r['status'] in ('OUTDATED', 'NEWER', 'MISMATCH') for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
