"""
Base strategy class providing the template for all strategies in this repo.
"""
import pandas as pd


class BaseStrategy:
    #: How closely this implementation follows its source paper. Declared as a CLASS
    #: attribute so a subclass overrides it by stating one line, and so a leveraged wrap
    #: cannot silently inherit its parent's claim.
    #:
    #:   'faithful' — the paper's algorithm on the paper's universe and tickers.
    #:   'proxy'    — the paper's algorithm and universe, different funds (VAA_G4).
    #:   'custom'   — anything else: a universe or parameterisation nobody published.
    #:
    #: The default is the WEAK claim on purpose. A strategy that forgets to declare is
    #: reported as custom, which is at worst unfair to it; the reverse would put an
    #: unearned fidelity claim in a report, which is the failure mode that matters.
    fidelity = 'custom'

    #: WHERE THE FIDELITY CLAIM COMES FROM — paper, and section or figure. Required for
    #: every registered entry; `tests/test_anchors.py` asserts it is present and non-empty.
    #:
    #: This exists because `fidelity` on its own is an unfalsifiable string. The label was
    #: wrong three separate times — DM_G8_Composite, HAA_G1_Simple and VAA_G4 — always in
    #: the under-claiming direction, and the only test on it checked that the value was one
    #: of three legal words. A citation converts an opinion into something a reader can
    #: check against the PDF in academic-papers/, which is the difference between a claim
    #: and an assertion.
    #:
    #: A 'custom' entry cites what it DEPARTS from, so the departure is reviewable too.
    source = ''

    #: What this entry is FOR. The organising question is *"would you hold it?"* — because that
    #: is what decides whether it counts as a TRIAL in the best-of-N and participation-ratio
    #: machinery. A thing nobody would hold is not a trial anybody ran.
    #:   'strategy'    — a candidate you might actually allocate to. THE ONLY role counted as
    #:                   a trial (`main.py` filters the selection statistics on it).
    #:   'control'     — a degenerate case (single-asset) that exists to show how much of a
    #:                   family's result comes from its universe rather than its timing rule.
    #:                   Reported, excluded from the selection statistics.
    #:   'benchmark'   — a passive reference: no signal, no de-risking rule. Three of them ARE
    #:                   levered as of 2026-07-29 (SPY_2X, SPY_3X, RiskParity_3X) — a levered
    #:                   wrap compared only against 1x references cannot be decomposed into
    #:                   "what the timing rule added" and "what the leverage added".
    #:   'exploratory' — added 2026-07-29 for the 3x variants. A real strategy with a real
    #:                   signal, measured and reported in full, that NOBODY IN THIS PROJECT
    #:                   INTENDS TO HOLD. Same treatment as 'control' and for the same reason,
    #:                   but a different reason for not holding it: not degeneracy, but that
    #:                   **no 3x product predates 2008-11**, so its worst case is structurally
    #:                   unobservable and its drawdown column is a bull-market number.
    #:
    #:                   This role is what makes the 3x expansion safe. Twelve 3x entries with
    #:                   20%+ CAGRs would otherwise sit at the top of a Sortino-ranked table and
    #:                   be absorbed as twelve more trials, inflating the multiple-comparison
    #:                   correction against the entries that ARE candidates. With it, the trial
    #:                   count moves 18 -> 19 while the registry grows by twelve.
    #:
    #: DERIVED, not assigned, for anything with `leverage >= 3`: see the property below. A
    #: subclass may still override with a plain class attribute (`role = 'benchmark'`), which
    #: shadows the property as usual.
    _role = 'strategy'

    @property
    def role(self):
        """`'exploratory'` whenever the execution multiple is 3 or more; `_role` otherwise.

        Derived on the BASE class rather than on `LeveragedWrapMixin`, because `DMLeveraged`
        does not use that mixin — it subclasses `BaseStrategy` directly, having no canonical
        parent to wrap. When the rule lived on the mixin, `DM_G3_Leveraged_3X` and
        `DM_G5_Leveraged_3X` silently came out as `role='strategy'`: two 3x entries counted as
        trials and ranked as portfolios one might hold. Six of eight got the role and two did
        not, which is exactly the failure mode a derived rule is supposed to prevent.

        A rule that each subclass must remember will be forgotten by whichever subclass is
        shaped differently from the others.
        """
        if float(getattr(self, 'leverage', 1.0) or 1.0) >= 3.0:
            return 'exploratory'
        return self._role

    #: Execution multiple. 1.0 for everything that trades its own tickers; a leveraged WRAP
    #: overrides it with 2 or 3 in its constructor.
    #:
    #: Declared here so the UI and the chart palette read ONE attribute instead of parsing
    #: `_2X` / `_3X` off the end of a registry key. Name-parsing was how `family_of` stranded
    #: `PAA2_G12` in a family of its own — the trailing digit there is Keller's protection
    #: factor, not a ratio — and the same trap is waiting for anything that reads a key suffix.
    #:
    #: It records the EXPOSURE an entry carries, not whether it performs a mapping. A
    #: benchmark holding SSO directly sets `leverage = 2` even though it maps nothing — the
    #: growth chart's width channel and the picker's ratio filter both read this as exposure,
    #: and an earlier version that left such a benchmark at 1.0 drew a 100%-SSO line at 1x
    #: width and filed it under "1x".
    #:
    #: `letf_mapper.holds_leveraged_product` answers a related but different question by
    #: reading the TICKERS, and is the one used to bar an entry from setting the ranked window.
    #: Keep both: a fidelity label cannot defeat a ticker check, and a wrap declares 1x tickers
    #: while carrying 2x or 3x exposure, so neither test subsumes the other.
    leverage = 1.0

    def __init__(self, name, is_active=True, score_type='weighted'):
        self.name = name
        self.is_active = is_active
        self.score_type = score_type

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        """
        Returns a DataFrame (dates x assets) with allocation weights.
        - prices: DataFrame of monthly prices.
        - scores_13612w: DataFrame of 13612W scores (weighted momentum).
        - ret_12m, ret_3m: optional, strategy-dependent.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Sleeve declaration — MANDATORY
    # ------------------------------------------------------------------ #
    def sleeves(self):
        """Declare, explicitly, which tickers this strategy can hold and in which role.

        Returns ``{'offensive': set, 'defensive': set, 'canary': list}``.

        * ``offensive`` — every ticker that can ever be held as a risk position. For a
          leveraged wrap this MUST include the LETF images actually traded, not just the
          1x signal assets.
        * ``defensive`` — every ticker that can be held as the de-risked bucket. An asset
          may appear in both (TLT/DBC/LQD in BAA, IEF in HAA); that is a dual role and is
          resolved per month by the canary.
        * ``canary`` — signal-only tickers, never traded, but required for the signal to
          mean anything. Coverage counts them.

        There is no default and no inference. The base implementation raises. Until
        2026-07-28 the engine sniffed five different attribute names to guess a strategy's
        cash bucket (``defensive``, ``cash_assets``, ``cash_proxy``, ``cash``,
        ``cash_weights``) and silently returned an empty set when a strategy used a sixth —
        which is exactly what happened to RAA's ``risky_assets`` (audit finding M6). A
        vocabulary that has to be guessed will be guessed wrong again; declaring it is the
        only fix that does not recur.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must declare sleeves(). See BaseStrategy.sleeves '
            'for the contract; there is deliberately no inferred default.')

    def _sleeves(self):
        """(defensive, offensive, canary) tuple form used internally by the mask helpers.

        The shape is checked here rather than trusted. A declaration that returns a bare
        string instead of a set is the kind of mistake that produces a silently wrong
        defensive mask — `set('SPY')` is `{'S', 'P', 'Y'}` — and the margin logic would
        then de-lever against tickers that do not exist instead of the one that does.
        """
        s = self.sleeves()
        if not isinstance(s, dict):
            raise TypeError(f'{type(self).__name__}.sleeves() must return a dict, '
                            f'got {type(s).__name__}')
        missing = {'offensive', 'defensive'} - set(s)
        if missing:
            raise ValueError(f'{type(self).__name__}.sleeves() is missing {sorted(missing)}')
        for key in ('offensive', 'defensive', 'canary'):
            val = s.get(key)
            if val is None and key == 'canary':
                continue
            if isinstance(val, str) or not isinstance(val, (set, frozenset, list, tuple)):
                raise TypeError(f'{type(self).__name__}.sleeves()[{key!r}] must be a set or '
                                f'list of tickers, got {type(val).__name__}')
            bad = [t for t in val if not isinstance(t, str)]
            if bad:
                raise TypeError(f'{type(self).__name__}.sleeves()[{key!r}] holds non-ticker '
                                f'entries: {bad}')
        return set(s['defensive']), set(s['offensive']), list(s.get('canary') or [])

    def defensive_mask(self, alloc, scores=None):
        """
        Boolean DataFrame (same shape as `alloc`) flagging weight held DEFENSIVELY.

        Assets declared defensive but NOT also in the offensive universe are defensive
        unconditionally. Dual-role assets — held offensively in good months and
        defensively in bad ones (TLT/DBC/LQD in BAA, IEF in HAA) — are resolved by the
        canary: they count as defensive only in months the canary is down.

        Without a canary to disambiguate, dual-role assets are counted as OFFENSIVE. That
        is the conservative direction: it understates de-escalation rather than claiming
        protection the signal cannot prove.
        """
        defensive, offensive, canary = self._sleeves()
        mask = pd.DataFrame(False, index=alloc.index, columns=alloc.columns)

        unambiguous = sorted((defensive - offensive) & set(alloc.columns))
        if unambiguous:
            mask[unambiguous] = True

        dual = sorted((defensive & offensive) & set(alloc.columns))
        if dual and canary and scores is not None:
            avail = [c for c in canary if c in scores.columns]
            if avail:
                c = scores[avail].reindex(alloc.index)
                risk_off = ((c <= 0).any(axis=1) | c.isna().any(axis=1)).fillna(True)
                for col in dual:
                    mask[col] = risk_off
        return mask

    def offensive_weight(self, alloc, scores=None):
        """Per-month portfolio weight sitting in the offensive sleeve, clipped to [0, 1].

        Uninvested residue counts as neither: it is idle cash and is never borrowed against.
        """
        defensive_wt = alloc.where(self.defensive_mask(alloc, scores), 0.0).sum(axis=1)
        return (alloc.sum(axis=1) - defensive_wt).clip(0.0, 1.0)


class LeveragedWrapMixin:
    """`sleeves()` for a leveraged wrap.

    A wrap runs the canonical 1x signal and then translates the offensive sleeve into LETF
    tickers, so the book it actually holds contains UPRO/TQQQ/UWM/UGL, not SPY/QQQ/IWM/GLD.
    Coverage and the ledger both look at what is held, so the declaration must name the
    images — otherwise the coverage guard would happily measure a 2x strategy over years in
    which its products did not exist, which is precisely how 32 months of 100%-in-a-
    nonexistent-ticker got reported as +0.00% (audit finding C3).

    Must be listed BEFORE the canonical strategy in the bases so that `super().sleeves()`
    resolves to the canonical declaration.
    """

    #: A wrap is never faithful, whatever its parent claims. The signal math is the
    #: paper's; the restricted universe it runs on was invented here to satisfy leverage
    #: homogeneity, and no author ever published it. Declared on the MIXIN so the claim
    #: cannot be lost by forgetting a line in a subclass.
    fidelity = 'custom'

    # NOTE: the 3x -> role='exploratory' rule lives on `BaseStrategy`, not here. It was written
    # on this mixin first, and `DMLeveraged` does not use the mixin — so two of the eight 3x
    # entries silently kept `role='strategy'`. See `BaseStrategy.role`.

    @property
    def source(self):
        """The parent's citation, marked as departed from.

        Derived rather than restated so a wrap can never cite a different paper from the
        signal it actually runs, and so adding a wrap cannot forget the citation. What it
        departs from is the universe, never the algorithm — which is exactly the sentence
        a reader needs.
        """
        parent = super().source
        return (f'departs from {parent} — same signal on a restricted universe chosen for '
                f'uniform-ratio LETF execution, which no author published') if parent else ''

    def sleeves(self):
        from common.letf_mapper import LETFMapper
        base = super().sleeves()
        offensive, defensive = set(base['offensive']), set(base['defensive'])
        mapper = LETFMapper(leverage=self.leverage)
        # Dual-role assets are held at 1x by design and are never mapped, so only the
        # strictly-offensive names contribute images.
        images = {mapper.get_letf(a) for a in (offensive - defensive)}
        return {'offensive': offensive | {i for i in images if i},
                'defensive': defensive,
                'canary': list(base.get('canary') or [])}
