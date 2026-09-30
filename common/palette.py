"""Visual encoding for the growth chart, shared by the CLI (`main.generate_chart`) and the
dashboard (`app.py`), so the two can never drift into different pictures of the same data.

THE PROBLEM THIS SOLVES. The old chart drew every line with `cm.tab20(linspace(0, 1, n))`:
a continuous colormap sampled n ways. With twenty-odd entries that produces neighbouring
strategies in almost the same colour, all of them solid, all the same width, identified only
by a legend in the corner that you have to cross-reference by eye. Two lines that overlap
for fifteen years and diverge at the end were, in practice, unidentifiable.

THE ENCODING. Three redundant channels, so no single one has to carry the whole load:

1. **Colour = FAMILY.** A fixed, colourblind-safe hue per family (Okabe-Ito), assigned by
   name rather than by position, so BAA is the same blue whether you plot three strategies
   or twenty-five. Colour now means something instead of merely being different.
2. **Dash pattern + marker = VARIANT within the family.** The G12/G4/G3-levered members of
   a family share a hue and differ in stroke. This survives greyscale printing, which a
   pure-colour encoding does not.
3. **Direct labels at the right edge.** The name is written at the end of its own line, in
   its own colour, de-overlapped vertically. This is the channel that actually removes the
   cross-referencing work; the legend is kept only as a fallback for lines that end early.

Benchmarks are deliberately NOT given a competing hue: they are drawn in grey, thick and
translucent, so they read as the reference the eye measures against rather than as another
contestant. Controls are drawn thin and dotted for the same reason.
"""

import re

import numpy as np

#: Okabe-Ito, the standard colourblind-safe qualitative set, plus two neutrals for the
#: non-strategy roles. Keyed by FAMILY so the mapping is stable under any subset.
FAMILY_COLORS = {
    'HAA':  '#0072B2',   # blue
    'BAA':  '#D55E00',   # vermillion
    'DAA':  '#009E73',   # bluish green
    'VAA':  '#CC79A7',   # reddish purple
    'PAA':  '#E69F00',   # orange
    'DM':   '#56B4E9',   # sky blue
    'GTAA': '#A6761D',   # brown
    'Benchmark': '#4D4D4D',
    'Control':   '#8A8A8A',
}
FALLBACK_COLOR = '#666666'

#: Display order — families first (roughly by publication date), references last.
FAMILY_ORDER = ['HAA', 'BAA', 'DAA', 'VAA', 'PAA', 'DM', 'GTAA', 'Benchmark', 'Control']

_DASHES = [
    (None, None),          # solid
    (5, 2),                # dashed
    (1.5, 1.5),            # dotted
    (7, 2, 1.5, 2),        # dash-dot
    (3, 1, 3, 1, 8, 2),    # dash-dash-long
]
_MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def family_of(name, role='strategy'):
    """Family label for a registry key.

    Role wins over the name: `SPY_Benchmark` is a Benchmark, not a family called "SPY".
    Everything else is the token before the first underscore, which is how the registry
    names are built (`HAA_G12`, `BAA_G3_Leveraged_2X`).
    """
    if role == 'benchmark':
        return 'Benchmark'
    if role == 'control':
        return 'Control'
    # Trailing digits belong to the VARIANT, not the family: the registry key for Keller &
    # Butler's high-protection model is `PAA2_G12`, where the 2 is the protection factor a.
    # Splitting on '_' alone put it in a family of its own called "PAA2", separate from its
    # own leveraged siblings and with no colour assigned.
    head = str(name).split('_')[0]
    match = re.match(r'^([A-Za-z]+)', head)
    fam = match.group(1) if match else head
    # GEM is Antonacci's flagship single-module dual momentum — same family as DM, under
    # its own famous name. Without this alias it would form a one-entry family with no
    # assigned colour, drawn in fallback grey beside its sky-blue siblings.
    return 'DM' if fam == 'GEM' else fam


def family_sort_key(family):
    """Sort key putting known families in FAMILY_ORDER and unknown ones after, A-Z."""
    return (FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER),
            family)


def group_by_family(entries):
    """`[(name, role), ...]` -> ordered `[(family, [name, ...]), ...]`.

    Both levels are sorted deterministically, so the same registry always produces the same
    colour and dash assignment. That matters: a chart whose encoding shifts when you tick a
    box is not a chart you can compare against last month's.
    """
    groups = {}
    for name, role in entries:
        groups.setdefault(family_of(name, role), []).append(name)
    return [(fam, sorted(groups[fam]))
            for fam in sorted(groups, key=family_sort_key)]


#: Line width per execution multiple — the THIRD encoding channel, added 2026-07-29.
#: Ratio is the thing most worth seeing on a growth chart of levered variants and it was not
#: encoded at all: a 1x, a 2x and a 3x HAA drew as three lines of identical width, told apart
#: only by dash pattern, which also carries the universe.
_RATIO_WIDTH = {1.0: 1.4, 2.0: 2.2, 3.0: 3.2}


def ratio_of(name, leverage=None):
    """Execution multiple for a registry key. Prefers the declared `leverage`.

    `leverage` should come from `BaseStrategy.leverage`. The name suffix is only a fallback for
    callers that hold a key and nothing else — and it is a poor one, which is why the attribute
    exists: `family_of('PAA2_G12')` once returned `"PAA2"` because it read a trailing digit that
    was Keller's protection factor, not a ratio.
    """
    if leverage is not None:
        try:
            return float(leverage)
        except (TypeError, ValueError):
            pass
    head = str(name).upper()
    for suffix, ratio in (('_3X', 3.0), ('_2X', 2.0)):
        if head.endswith(suffix):
            return ratio
    return 1.0


def line_styles(entries, universe=None, ratios=None):
    """`[(name, role), ...]` -> `{name: matplotlib kwargs}`.

    Three independent channels, so two entries differing in any one of them stay distinguishable:

        colour       = FAMILY      (Okabe-Ito, stable by name)
        dash+marker  = UNIVERSE    (the variant index within the family)
        LINE WIDTH   = RATIO       1x thin -> 2x medium -> 3x thick

    `universe` is the FULL registry, also as `[(name, role), ...]`. When given, the variant
    index is taken from each family's membership in the universe rather than in `entries`,
    so a strategy keeps its dash pattern and marker when you untick its siblings. Without
    it, unticking BAA_G12 would promote BAA_G4 from dotted to solid and two runs of the
    dashboard could no longer be read against each other. Callers that plot a user-chosen
    subset should always pass it; `entries` alone is fine when you plot everything.

    `ratios` is an optional `{name: leverage}` read from `BaseStrategy.leverage`. Without it the
    ratio is inferred from the key suffix, which works for the current naming but is the weaker
    source — see `ratio_of`.

    Width previously encoded ROLE (2.6 benchmark / 1.2 control / 1.8 strategy). Role now drives
    alpha and zorder only. That trade is deliberate: with HAA holding seven entries, `_DASHES`
    has five patterns and would collide, and the ratio is more worth a dedicated channel than
    the role — a benchmark is already named in the legend and sits behind everything.
    """
    index_source = group_by_family(universe) if universe is not None \
        else group_by_family(entries)
    order = {}
    for _family, names in index_source:
        for i, name in enumerate(names):
            order[name] = i

    ratios = ratios or {}
    styles = {}
    for family, names in group_by_family(entries):
        color = FAMILY_COLORS.get(family, FALLBACK_COLOR)
        role = 'benchmark' if family == 'Benchmark' else (
            'control' if family == 'Control' else 'strategy')
        for fallback, name in enumerate(names):
            i = order.get(name, fallback)
            dashes = _DASHES[i % len(_DASHES)]
            ratio = ratio_of(name, ratios.get(name))
            styles[name] = {
                'color': color,
                'dashes': dashes if dashes[0] is not None else (),
                'linewidth': _RATIO_WIDTH.get(ratio, 1.4 + 0.9 * (ratio - 1.0)),
                'alpha': 0.75 if role == 'benchmark' else (0.65 if role == 'control' else 0.95),
                'marker': _MARKERS[i % len(_MARKERS)],
                'markersize': 4.5,
                'markerfacecolor': 'none',
                'markeredgewidth': 1.1,
                # Stagger the marker phase per variant so same-family lines that sit on top
                # of each other still show alternating glyphs instead of one merged blob.
                'markevery': (0.04 + 0.03 * (i % 5), 0.17),
                'zorder': 1 if role == 'benchmark' else (2 if role == 'control' else 3),
                '_family': family,
                '_role': role,
                '_ratio': ratio,
            }
    return styles


def plot_kwargs(style):
    """Strip the private `_family`/`_role` keys before handing a style to matplotlib."""
    return {k: v for k, v in style.items() if not k.startswith('_')}


def family_legend_handles(entries):
    """Proxy handles for a FAMILY legend — one swatch per family, not one per strategy.

    With every line named at its own right end, a 25-row legend restates what the labels
    already say and eats a quarter of the plot doing it. What the labels cannot say is what
    the colours MEAN, so that is what the legend is reduced to.
    """
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=FAMILY_COLORS.get(fam, FALLBACK_COLOR), lw=2.6, label=fam)
            for fam, _ in group_by_family(entries)]


def label_lines(ax, endpoints, styles, min_gap=0.030, x_offset=1.012):
    """Write each series' name at the right edge of the axes, colour-matched, de-overlapped.

    `endpoints` is `{name: final_y}` in DATA units; the axes must already be scaled (call
    this last). Labels are placed at the line's own final height and then pushed apart to
    `min_gap` of the axes height, which keeps the vertical ORDER of the labels identical to
    the vertical order of the lines — the property that makes the label readable as "this
    one ended above that one" rather than as an arbitrary list.

    Series that end before the right edge still get a label here; their true endpoint is
    earlier, so the legend remains the fallback for those.
    """
    if not endpoints:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == 'log'

    def to_frac(y):
        if y is None or not np.isfinite(y) or (log and y <= 0):
            return None
        if log:
            return (np.log10(y) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return (y - lo) / (hi - lo)

    placed = [(name, to_frac(y)) for name, y in endpoints.items()]
    placed = [(n, f) for n, f in placed if f is not None]
    if not placed:
        return
    placed.sort(key=lambda t: t[1], reverse=True)

    # Single downward pass, then a corrective upward pass. Two passes are enough because the
    # first only ever moves labels DOWN and the second only ever moves them UP, so neither
    # can undo the other's spacing — it converges rather than oscillating.
    fracs = [f for _, f in placed]
    for i in range(1, len(fracs)):
        fracs[i] = min(fracs[i], fracs[i - 1] - min_gap)
    for i in range(len(fracs) - 2, -1, -1):
        fracs[i] = max(fracs[i], fracs[i + 1] + min_gap)
    # The upward pass can push the top label past the axes; slide the whole stack back down
    # by the overflow rather than clipping it, which would silently drop a name.
    shift = max(0.0, fracs[0] - 1.0)
    fracs = [f - shift for f in fracs]

    for (name, _), frac in zip(placed, fracs):
        st = styles.get(name, {})
        ax.text(x_offset, frac, name, transform=ax.transAxes,
                va='center', ha='left', fontsize=7.5,
                color=st.get('color', FALLBACK_COLOR),
                fontweight='bold' if st.get('_role') == 'benchmark' else 'normal')
