"""
HAA_G12 signal ticker -> Canadian-listed execution ticker, and the rules that govern it.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is an EXECUTION layer, not a strategy. The signal is unchanged and is computed on the
US tickers Keller published: `strategies/haa.py::HAA_12` ranks SPY/QQQ/IWM/VGK/EWJ/VWO/VNQ/
DBC/GLD/IEF/TLT/LQD on 13612U, reads the TIP canary, and defends into BIL or IEF. Nothing
here is allowed to reach any of that. `HAA_G12_CA` is deliberately ABSENT from
`main.ALL_STRATEGIES`: it is not another variant, it does not enter the selection statistics,
and it has no separate backtest. Adding it to the registry would also break
`tests/fixtures/run_facts.json`, which pins the registry count — so the constraint is
enforced structurally and not by remembering it.

WHY IT EXISTS
-------------
The owner wants execution confined to securities listed in Canada. The tax reasoning is the
owner's and their advisor's; this module takes it as a REQUIREMENT and checks only what code
can check (see RULE 3).

THE RULES
---------
Written down before any order was produced, in the same spirit as `common/letf_mapper.py`:
the only thing separating a mapping table from a convenience is a rule that can be checked
by reading it, with no returns in front of you.

RULE 1 — EVERY EXECUTION TICKER IS LISTED IN CANADA
    `exchange` must be one of `CANADIAN_EXCHANGES`. `assert_no_us_tickers` refuses an order
    list containing anything else, and it refuses by DEFAULT: a ticker it does not recognise
    is rejected rather than passed through. A guard that only rejects a known-bad list is a
    guard that approves every future mistake.

RULE 2 — UNHEDGED ONLY
    A Canadian holding SPY carries USD risk. Reproducing the backtest means carrying it too,
    so every execution ticker declares `hedged=False` and `assert_unhedged` enforces it.
    This is what excludes XSU (CAD-hedged US small cap) and XEH (CAD-hedged Europe), both of
    which are otherwise perfectly good funds. See `REJECTED_MAPPINGS`.

RULE 3 — THE GUARD CHECKS LISTING AND DOMICILE, NOT A TAX CONCLUSION
    Every fund below is a Canadian-resident trust listed on a Canadian exchange, and that is
    the whole of what is verified here. Whether that satisfies any particular provision of
    the Income Tax Act is a question for the owner's advisor, and this module must never be
    read as answering it.

    THE CONSTRAINT, in the owner's own words on 2026-09-17: *products HELD in Canada*, not
    the absence of American securities. That distinction was checked before any of this was
    written, because the two readings do not have the same answer:
      * the SECURITY HELD is a Canadian trust in all thirteen cases — this is the constraint,
        and the mapping satisfies it;
      * the funds' LOOK-THROUGH holdings are US and foreign securities in every case, and no
        mapping on this list changes that. Had the constraint been this one instead, the
        whole layer would have been pointless, and it is recorded here so nobody has to
        re-derive which question was being answered.
    Separately, the trading CURRENCY of a unit class says nothing about the issuer. `ZSP.U`
    is the same BMO trust as `ZSP`, priced in USD. It is not a US security, and holding it is
    not holding one.

RULE 4 — THE MAPPING MUST COVER THE SLEEVES EXACTLY
    `validate_covers(strategy)` fails when a tradeable ticker has no image, and fails when
    the table carries an image for a ticker the strategy cannot hold. A missing image must
    not silently drop a position from the book — that is the failure mode that produced
    audit finding C3, where 32 months held 100% of a ticker that did not exist and reported
    +0.00%. Canary tickers are signal-only and must NOT be mapped: TIP is read, never traded.

RULE 5 — A MISSING LIQUIDITY FIGURE IS AN ALARM, NEVER A PASS
    `avg_daily_volume` and `avg_spread_pct` are `None` for ZCOM, because its ETF Facts says
    in so many words that the information is not available. `liquidity_flags` returns a
    `unknown` flag for that, distinct from `ok`. An `ok` that means "I could not check" is
    the fail-open shape this repository keeps designing against.

WHERE THE NUMBERS COME FROM
---------------------------
Every figure below was read out of the issuer's regulatory ETF FACTS document, which is
where average daily volume and average bid-ask spread are disclosed. THE REPORTING PERIODS
DIFFER BY ISSUER, so each row carries its own `facts_as_of` and `source`. A table of
liquidity figures with one implied date would be prose caching a computation, which is the
disease `common/facts.py` exists to treat.

Nothing here is inferred. A figure the document does not state is `None` and says so.

THE DEVIATION REGISTER — SEVEN OF THIRTEEN
------------------------------------------
`deviation` is non-None wherever the EXECUTED asset is not the SIGNALLED asset. It was
initially believed that only VNQ->CGR and DBC->ZCOM deviated; reading the documents put the
count at seven, and the two bond lines worry more than the REIT line does. VNQ is held only
in months the signal ranks it top-six; a duration mismatch on IEF is paid in every defensive
month, which for HAA is a large fraction of them.

    IWM -> ZSML   Russell 2000          vs S&P SmallCap 600 (a profitability screen the
                                           Russell index does not apply)
    VWO -> XEC    FTSE Emerging         vs MSCI EM IMI (FTSE classifies South Korea as
                                           developed and excludes it; MSCI does not)
    VGK -> XEU    FTSE Dev Europe       vs MSCI Europe IMI
    VNQ -> CGR    US REITs              vs Global Realty Majors — the largest deviation
    DBC -> ZCOM   DBIQ Optimum Yield    vs Bloomberg Commodity TR (different roll)
    IEF -> ZTM    7-10y Treasuries      vs 5-10y Treasuries
    LQD -> ZIC    broad-maturity US IG  vs mid-term US IG

Measuring these is `tools/ca_execution_gap.py`'s job. They are named here so that the
report has something to be checked against, rather than discovering its own scope.
"""

from dataclasses import dataclass, field

#: The two exchanges an execution ticker may be listed on. Spelled as the ETF Facts
#: documents spell them, so a row can be checked against its own source without translation.
CANADIAN_EXCHANGES = frozenset({'TSX', 'Cboe CA'})

#: Fraction of average daily volume above which an order is flagged as too large to place
#: at market. Below the 33% in the owner's brief on purpose: the flag is advisory and cheap,
#: and a single monthly rotation that moves a third of a day's volume is already unusual.
ADV_PARTICIPATION_WARN = 0.33

#: Average bid-ask spread above which a product is flagged. 25 bps, the owner's threshold.
SPREAD_WARN_PCT = 0.25


@dataclass(frozen=True)
class CAExecution:
    """One signal ticker's Canadian execution image.

    `exec_ticker_cad` and `exec_ticker_usd` are two UNIT CLASSES of the same fund wherever
    both are present — same portfolio, same MER, different trading currency. They are not
    two positions, and `ca_execution` nets them into one sleeve. Where only the USD class
    exists (BIL -> UBIL.U), `exec_ticker_cad` is None and deploying CAD cash into that
    sleeve requires a conversion; that is a real recurring cost for HAA, whose canary sends
    the book defensive often, and it is named rather than smoothed over.
    """

    signal_ticker: str
    exec_ticker_cad: str | None
    exec_ticker_usd: str | None
    exchange: str
    trade_currency: str
    hedged: bool
    issuer: str
    inception_date: str
    mer_pct: float | None
    total_value: str | None
    avg_daily_volume: int | None
    avg_spread_pct: float | None
    #: Same three figures for the USD unit class, which trades far thinner than the CAD one
    #: on every fund here. Kept separate rather than averaged: `prefer_usd_units` is a
    #: routing decision, and routing into a class that trades 161 days out of 251 is a
    #: different risk from routing into one that trades every day.
    usd_avg_daily_volume: int | None
    usd_avg_spread_pct: float | None
    usd_days_traded: str | None
    #: The index the EXECUTED fund tracks, in its own document's words where they are short.
    tracks: str
    #: None when the executed asset is the signalled asset. A sentence naming the difference
    #: otherwise. See THE DEVIATION REGISTER above.
    deviation: str | None
    min_lot: int
    facts_as_of: str
    source: str
    notes: str = ''
    #: True when the ETF Facts states the figures are unavailable rather than omitting them.
    #: Distinguishes "the issuer says it cannot yet report" from "nobody looked".
    liquidity_unreported: bool = False


#: The table. Keyed by SIGNAL ticker, which is the only key the strategy knows about.
#:
#: IEF appears ONCE although it is dual-role in HAA — offensive candidate and defensive
#: candidate both. The canary resolves the role per month; the instrument is the same either
#: way, so a second entry would be two names for one position and would break netting.
#:
#: TIP is deliberately absent. It is a canary: read every month, never traded. RULE 4's
#: check asserts its absence rather than tolerating it.
CA_MAPPING: dict[str, CAExecution] = {
    'SPY': CAExecution(
        signal_ticker='SPY',
        exec_ticker_cad='ZSP', exec_ticker_usd='ZSP.U',
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2012-11-14', mer_pct=0.09, total_value='CAD 24.5 billion',
        avg_daily_volume=1_146_487, avg_spread_pct=0.02,
        usd_avg_daily_volume=102_300, usd_avg_spread_pct=0.04,
        usd_days_traded='251 out of 251',
        tracks='S&P 500', deviation=None, min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/BMO_S%26P_500_Index_ETF-EN-CAD_Units.pdf',
        notes='Chosen over VFV, which has no USD unit class. VFV is otherwise equivalent '
              '(MER 0.09%, spread 0.018%) but cannot absorb USD cash without a conversion, '
              'and this is the largest sleeve in the book.'),

    'QQQ': CAExecution(
        signal_ticker='QQQ',
        exec_ticker_cad='ZNQ', exec_ticker_usd='ZNQ.U',
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2019-02-12', mer_pct=0.39, total_value='CAD 2.1 billion',
        avg_daily_volume=128_126, avg_spread_pct=0.03,
        usd_avg_daily_volume=5_704, usd_avg_spread_pct=0.07,
        usd_days_traded='251 out of 251',
        tracks='Nasdaq-100', deviation=None, min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/'
               'BMO_Nasdaq_100_Equity_Index_ETF-EN-CAD_Units.pdf',
        notes='The USD class started later than the CAD class (2021-02-11).'),

    'IWM': CAExecution(
        signal_ticker='IWM',
        exec_ticker_cad='ZSML', exec_ticker_usd='ZSML.U',
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2020-02-05', mer_pct=0.22, total_value='CAD 119.1 million',
        avg_daily_volume=31_578, avg_spread_pct=0.22,
        usd_avg_daily_volume=957, usd_avg_spread_pct=0.24,
        usd_days_traded='161 out of 251',
        tracks='S&P SmallCap 600',
        deviation='IWM tracks the Russell 2000; ZSML tracks the S&P SmallCap 600, whose '
                  'ETF Facts describes constituents as "liquid and financially viable". '
                  'That profitability screen has no counterpart in the Russell index, and '
                  'it is a persistent factor tilt, not a tracking wobble.',
        min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/'
               'BMO_S%26P_US_Small_Cap_Index_ETF-EN-CAD_Units.pdf',
        notes='The CAD class trades ~31.6k units a day, not the ~1.3k first believed — that '
              'figure belongs to the USD class, which trades 161 days out of 251. Routing '
              'this sleeve through ZSML.U is the one place prefer_usd_units is questionable.'),

    'VGK': CAExecution(
        signal_ticker='VGK',
        exec_ticker_cad='XEU', exec_ticker_usd=None,
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BlackRock Asset Management Canada Limited',
        inception_date='2014-04-15', mer_pct=0.28, total_value='CAD 648.1 million',
        avg_daily_volume=55_006, avg_spread_pct=0.11,
        usd_avg_daily_volume=None, usd_avg_spread_pct=None, usd_days_traded=None,
        tracks='MSCI Europe IMI',
        deviation='VGK tracks FTSE Developed Europe All Cap; XEU tracks MSCI Europe IMI. '
                  'Both are all-cap developed Europe and the constituent overlap is high; '
                  'this is the mildest deviation on the list, and is listed so the gap '
                  'report measures it rather than assuming it away.',
        min_lot=1,
        facts_as_of='2026-04-30',
        source='https://www.blackrock.com/ca/investors/en/literature/etf-summary/'
               'xeu-facts-en-ca.pdf',
        notes='XEH is the CAD-hedged sibling and is refused under RULE 2.'),

    'EWJ': CAExecution(
        signal_ticker='EWJ',
        exec_ticker_cad='ZJPN', exec_ticker_usd=None,
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2022-01-24', mer_pct=0.39, total_value='CAD 496.7 million',
        avg_daily_volume=18_362, avg_spread_pct=0.22,
        usd_avg_daily_volume=None, usd_avg_spread_pct=None, usd_days_traded=None,
        tracks='MSCI Japan (ETF Facts: "approximately 85% of the free float-adjusted market '
               'capitalization in Japan")',
        deviation=None, min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/BMO_Japan_Index_ETF-EN-CAD_Units.pdf',
        notes='EWJ also tracks MSCI Japan, so this is a true like-for-like. Youngest fund '
              'on the list after ZCOM: history starts 2022-01, which bounds the gap report.'),

    'VWO': CAExecution(
        signal_ticker='VWO',
        exec_ticker_cad='XEC', exec_ticker_usd='XEC.U',
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BlackRock Asset Management Canada Limited',
        inception_date='2013-04-10', mer_pct=0.28, total_value='CAD 4,038.7 million',
        avg_daily_volume=429_853, avg_spread_pct=0.06,
        usd_avg_daily_volume=1_882, usd_avg_spread_pct=0.34,
        usd_days_traded='246 out of 251',
        tracks='MSCI Emerging Markets IMI',
        deviation='VWO tracks FTSE Emerging, which classifies South Korea as a developed '
                  'market and excludes it. MSCI EM IMI includes South Korea. That is a '
                  'country-level difference in the index, not a tracking error, and it is '
                  'the second-largest deviation here.',
        min_lot=1,
        facts_as_of='2026-04-30',
        source='https://www.blackrock.com/ca/investors/en/literature/etf-summary/'
               'xec-facts-en-ca.pdf',
        notes='The USD class spreads 0.34%, above SPREAD_WARN_PCT.'),

    'VNQ': CAExecution(
        signal_ticker='VNQ',
        exec_ticker_cad='CGR', exec_ticker_usd=None,
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BlackRock Asset Management Canada Limited',
        inception_date='2008-08-26', mer_pct=0.72, total_value='CAD 286.9 million',
        avg_daily_volume=26_664, avg_spread_pct=0.16,
        usd_avg_daily_volume=None, usd_avg_spread_pct=None, usd_days_traded=None,
        tracks='Global Realty Majors Index',
        deviation='THE LARGEST DEVIATION. VNQ is US REITs; CGR is a concentrated GLOBAL '
                  'real-estate index. The signal ranks US real estate and the book buys '
                  'world real estate, so the executed sleeve can move for reasons the '
                  'signal never saw. MER is also 0.72% against VNQ\'s 0.12%.',
        min_lot=1,
        facts_as_of='2026-04-30',
        source='https://www.blackrock.com/ca/investors/en/literature/etf-summary/'
               'cgr-facts-en-ca.pdf'),

    'DBC': CAExecution(
        signal_ticker='DBC',
        exec_ticker_cad='ZCOM', exec_ticker_usd=None,
        exchange='Cboe CA', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2025-10-21', mer_pct=0.30, total_value='CAD 635.0 million',
        avg_daily_volume=None, avg_spread_pct=None,
        usd_avg_daily_volume=None, usd_avg_spread_pct=None, usd_days_traded=None,
        tracks='Bloomberg Commodity Index Total Return',
        deviation='DBC tracks the DBIQ Optimum Yield Diversified Commodity Index, whose '
                  'roll rule is chosen to reduce contango drag. ZCOM tracks the Bloomberg '
                  'Commodity Index TR, which rolls on a fixed schedule. Same asset class, '
                  'materially different roll, and roll is most of what a commodity index '
                  'return is.',
        min_lot=1,
        facts_as_of='2025-08-31',
        source='https://fundfacts.bmo.com/EtfEnglish/BMO_Broad_Commodity_ETF-EN-CAD_Units.pdf',
        liquidity_unreported=True,
        notes='WEAKEST LINK. (1) Its ETF Facts states volume, spread, MER and total value '
              'are "not available because it is a new ETF"; the 0.30% MER and CAD 635.0M '
              'are from the 2026-08-31 BMO monthly factsheet and the MER there is an '
              'estimate. (2) It is an ALTERNATIVE MUTUAL FUND: its own ETF Facts says it '
              'may "employ leverage and borrow cash to use for investment purposes". DBC '
              'does not. (3) Inception 2025-10 bounds any DBC-sleeve gap measurement to '
              'under a year.'),

    'GLD': CAExecution(
        signal_ticker='GLD',
        exec_ticker_cad='CGL.C', exec_ticker_usd=None,
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BlackRock Asset Management Canada Limited',
        inception_date='2011-03-31', mer_pct=0.55, total_value='CAD 2,093.6 million',
        avg_daily_volume=81_549, avg_spread_pct=0.09,
        usd_avg_daily_volume=None, usd_avg_spread_pct=None, usd_days_traded=None,
        tracks='LBMA Gold Price PM ($/ozt), physical bullion',
        deviation=None, min_lot=1,
        facts_as_of='2025-08-31',
        source='https://www.blackrock.com/ca/individual/en/literature/etf-summary/'
               'cgl-c-summ-doc-en-ca.pdf',
        notes='CGL.C is the NON-HEDGED unit class. CGL is the hedged one and is refused '
              'under RULE 2 — the .C suffix is load-bearing, not cosmetic.'),

    'TLT': CAExecution(
        signal_ticker='TLT',
        exec_ticker_cad='ZTL', exec_ticker_usd='ZTL.U',
        exchange='Cboe CA', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2017-02-21', mer_pct=0.22, total_value='CAD 333.0 million',
        avg_daily_volume=38_721, avg_spread_pct=0.22,
        usd_avg_daily_volume=3_091, usd_avg_spread_pct=0.42,
        usd_days_traded='205 out of 252',
        tracks='Bloomberg U.S. Treasury 20+ Year',
        deviation=None, min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/'
               'BMO_Long-Term_US_Treasury_Bond_Index_ETF-EN-CAD_Units.pdf',
        notes='Listed on Cboe Canada, not the TSX. The USD class spreads 0.42%, the widest '
              'on this table and well above SPREAD_WARN_PCT.'),

    'IEF': CAExecution(
        signal_ticker='IEF',
        exec_ticker_cad='ZTM', exec_ticker_usd='ZTM.U',
        exchange='Cboe CA', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2017-02-21', mer_pct=0.22, total_value='CAD 95.8 million',
        avg_daily_volume=7_553, avg_spread_pct=0.12,
        usd_avg_daily_volume=1_857, usd_avg_spread_pct=0.11,
        usd_days_traded='166 out of 252',
        tracks='U.S. Treasuries with "between five and ten years to maturity"',
        deviation='IEF is the 7-10 year band; ZTM is the 5-10 year band, so the executed '
                  'sleeve is SHORTER duration than the signalled one. This is paid in every '
                  'month the sleeve is held, and HAA holds IEF both as an offensive '
                  'candidate and as one of two defensive candidates.',
        min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/'
               'BMO_Mid-Term_US_Treasury_Bond_Index_ETF-EN-CAD_Units.pdf',
        notes='DUAL-ROLE in HAA — offensive candidate and defensive candidate. One entry, '
              'one position; the canary decides the role, not the instrument. Listed on '
              'Cboe Canada, not the TSX. Smallest fund on the list at CAD 95.8M, and the '
              'thinnest CAD class at ~7.6k units a day.'),

    'LQD': CAExecution(
        signal_ticker='LQD',
        exec_ticker_cad='ZIC', exec_ticker_usd='ZIC.U',
        exchange='TSX', trade_currency='CAD', hedged=False,
        issuer='BMO Asset Management Inc.',
        inception_date='2013-03-19', mer_pct=0.28, total_value='CAD 3.5 billion',
        avg_daily_volume=119_491, avg_spread_pct=0.14,
        usd_avg_daily_volume=4_032, usd_avg_spread_pct=0.36,
        usd_days_traded='192 out of 251',
        tracks='Mid-term U.S. investment-grade corporate bonds, capped at the three largest '
               'issues from each issuer',
        deviation='LQD is broad-maturity US investment grade; ZIC is the MID-TERM band. '
                  'Shorter duration than the signalled asset, same direction as the '
                  'IEF -> ZTM mismatch.',
        min_lot=1,
        facts_as_of='2025-11-30',
        source='https://fundfacts.bmo.com/EtfEnglish/'
               'BMO_Mid-Term_US_IG_Corporate_Bond_Index_ETF-EN-CAD_Units.pdf',
        notes='The USD class spreads 0.36%, above SPREAD_WARN_PCT.'),

    'BIL': CAExecution(
        signal_ticker='BIL',
        exec_ticker_cad=None, exec_ticker_usd='UBIL.U',
        exchange='TSX', trade_currency='USD', hedged=False,
        issuer='Global X Investments Canada Inc.',
        inception_date='2023-04-13', mer_pct=0.13, total_value='USD 429.9 million',
        avg_daily_volume=None, avg_spread_pct=None,
        usd_avg_daily_volume=127_592, usd_avg_spread_pct=0.02,
        usd_days_traded='251 out of 251',
        tracks='U.S. Treasury bills with remaining maturities generally under 3 months',
        deviation=None, min_lot=1,
        facts_as_of='2026-03-31',
        source='https://www.globalx.ca/wp-content/uploads/2025/08/UBIL.U-ETF_Fact_Sheet-EN.pdf',
        notes='USD-ONLY: there is no CAD unit class, so the CAD figures are None by '
              'construction, not by omission. Consequence, and it is structural rather than '
              'an edge case: every time the TIP canary dies and the book defends into BIL, '
              'CAD cash must be converted. HAA defends often. ZUS.U is the recorded '
              'alternative (see ALTERNATIVES) and is a worse BIL proxy on every axis.'),
}

#: Signal tickers that are READ and never traded. Asserted absent from CA_MAPPING by RULE 4.
CANARY_ONLY = frozenset({'TIP'})

#: Products considered and NOT used, with the reason. Kept for the same purpose as
#: `letf_mapper.REJECTED_MAPPINGS`: so the audit trail survives and a rejected product is
#: not quietly reintroduced by somebody who only sees that it is cheap and liquid.
REJECTED_MAPPINGS = {
    'XSU': 'CAD-hedged US small cap. Refused under RULE 2 — hedging away the USD exposure '
           'the backtest carries makes the executed portfolio a different portfolio.',
    'XEH': 'CAD-hedged Europe. Same refusal as XSU.',
    'CGL': 'The HEDGED unit class of the iShares gold fund. CGL.C is the unhedged one.',
    'CBIL': 'Global X 0-3 Month T-Bill ETF — CANADIAN T-bills. It would remove the recurring '
            'FX conversion on the defensive sleeve, and it is refused anyway: the signal '
            'defends into US T-bills, and substituting Canadian ones is a deviation nobody '
            'asked for. It would also be the only deviation CHOSEN here rather than forced '
            'by what exists. If the conversion cost is later judged worse than the '
            'substitution, that is a decision to take explicitly, not a default.',
    'VFV': 'Vanguard S&P 500 Index ETF. Admissible in every respect and marginally tighter '
           'on spread (0.018%), but it has no USD unit class, so USD cash cannot reach the '
           'largest sleeve in the book without a conversion. ZSP/ZSP.U wins on that alone.',
}

#: Alternatives that ARE admissible and are simply not the first choice, with the numbers
#: that decided it. Separate from REJECTED_MAPPINGS because "worse" and "refused" are
#: different verdicts and collapsing them loses the reason.
ALTERNATIVES = {
    'ZUS.U': 'BMO Ultra Short-Term US Bond ETF, the recorded alternative for the BIL sleeve. '
             'Worse than UBIL.U on every published axis: MER 0.17% vs 0.13%, spread 0.10% vs '
             '0.02%, USD 133.4M vs 429.9M, 12.6k units/day vs 127.6k. It also holds '
             'short-term BONDS rather than pure T-bills, so it carries credit the signal '
             'does not. (ETF Facts as of 2025-11-30.)',
    'VFV': 'See REJECTED_MAPPINGS — admissible, not chosen, and the reason is the missing '
           'USD unit class rather than any defect.',
}


# --------------------------------------------------------------------------------------- #
# The rules, as callables. Each raises; none returns a bare False that a caller can ignore.
# --------------------------------------------------------------------------------------- #

def execution_tickers(entry):
    """Both unit classes of one entry, as a tuple with the Nones dropped."""
    return tuple(t for t in (entry.exec_ticker_cad, entry.exec_ticker_usd) if t)


def all_execution_tickers():
    """Every ticker any order may name, across both unit classes. Frozen."""
    out = set()
    for entry in CA_MAPPING.values():
        out.update(execution_tickers(entry))
    return frozenset(out)


def assert_listed_in_canada(mapping=None):
    """RULE 1, on the TABLE: every entry declares a Canadian exchange."""
    mapping = CA_MAPPING if mapping is None else mapping
    bad = {k: e.exchange for k, e in mapping.items() if e.exchange not in CANADIAN_EXCHANGES}
    if bad:
        raise ValueError(
            f'execution tickers must be listed in Canada (RULE 1); these declare something '
            f'else: {bad}. Legal values are {sorted(CANADIAN_EXCHANGES)}.')


def assert_unhedged(mapping=None):
    """RULE 2, on the TABLE: no entry is currency-hedged."""
    mapping = CA_MAPPING if mapping is None else mapping
    bad = sorted(k for k, e in mapping.items() if e.hedged)
    if bad:
        raise ValueError(
            f'{bad} map to CAD-hedged products (RULE 2). A Canadian holding the US original '
            f'carries USD risk; hedging it away executes a different portfolio from the one '
            f'the backtest measured.')


def assert_no_us_tickers(order_tickers, mapping=None):
    """RULE 1, on an ORDER LIST: refuse anything not in the table.

    Rejects by default rather than matching a known-bad list. A guard written as "reject
    these US tickers" approves every ticker nobody thought of, which is the entire class of
    mistake this is here to prevent — and the list of US tickers is unbounded while the list
    of admissible Canadian ones is thirteen long.
    """
    mapping = CA_MAPPING if mapping is None else mapping
    allowed = set()
    for entry in mapping.values():
        allowed.update(execution_tickers(entry))
    unknown = sorted(set(order_tickers) - allowed)
    if unknown:
        raise ValueError(
            f'{unknown} are not Canadian execution tickers (RULE 1). Orders may only name '
            f'one of {sorted(allowed)}. If a US ticker reached an order list, the signal '
            f'leaked into execution and the whole point of this layer is defeated.')


def validate_covers(strategy, mapping=None):
    """RULE 4: the table maps exactly what `strategy` can hold, no more and no less.

    Reads `sleeves()`, the mandatory declaration, rather than the `offensive`/`defensive`
    attributes. Those attributes carry SIGNAL tickers for a wrapped strategy while the book
    holds images, and sniffing them is how the live path spent two days labelling every
    leveraged entry "N/A" (see `main.size_positions`). One declaration, one source of truth.
    """
    mapping = CA_MAPPING if mapping is None else mapping
    s = strategy.sleeves()
    tradeable = set(s['offensive']) | set(s['defensive'])
    canary = set(s.get('canary') or ())

    missing = sorted(tradeable - set(mapping))
    if missing:
        raise ValueError(
            f'{type(strategy).__name__} can hold {missing}, which have no Canadian execution '
            f'image (RULE 4). A missing image must never silently drop the position: that is '
            f'audit finding C3, where months held 100% of a nonexistent ticker and reported '
            f'+0.00%.')

    extra = sorted(set(mapping) - tradeable)
    if extra:
        raise ValueError(
            f'the mapping carries images for {extra}, which {type(strategy).__name__} cannot '
            f'hold (RULE 4). A stale row is a route to an order nothing authorised.')

    mapped_canary = sorted(canary & set(mapping))
    if mapped_canary:
        raise ValueError(
            f'{mapped_canary} are CANARY tickers — read every month, never traded — and must '
            f'not have an execution image (RULE 4). Mapping one creates a path to buying the '
            f'thermometer.')


def liquidity_flags(entry, shares=None, use_usd_class=False):
    """Advisory flags for one entry, optionally against a proposed order size.

    Returns a list of ``(severity, message)`` where severity is 'warn' or 'unknown'.

    RULE 5 lives here. When `avg_daily_volume` is None the result is an 'unknown' flag, never
    an empty list: ZCOM publishes no volume and no spread, and an order sized against it must
    say that it could not be checked. An empty list means "checked, clean"; it never means
    "nothing to check".
    """
    flags = []
    if use_usd_class:
        adv, spread, label = entry.usd_avg_daily_volume, entry.usd_avg_spread_pct, 'USD class'
    else:
        adv, spread, label = entry.avg_daily_volume, entry.avg_spread_pct, 'CAD class'

    if adv is None:
        flags.append(('unknown',
                      f'{entry.signal_ticker}: no average daily volume is published for the '
                      f'{label}'
                      + (' — its ETF Facts states the information is not available because '
                         'the fund is new' if entry.liquidity_unreported else '')
                      + '. Order size could NOT be checked against turnover.'))
    elif shares is not None and adv > 0 and (shares / adv) > ADV_PARTICIPATION_WARN:
        flags.append(('warn',
                      f'{entry.signal_ticker}: {shares:,.0f} units is '
                      f'{shares / adv:.0%} of the {label} average daily volume '
                      f'({adv:,} units), above the {ADV_PARTICIPATION_WARN:.0%} threshold.'))

    if spread is None:
        flags.append(('unknown',
                      f'{entry.signal_ticker}: no average bid-ask spread is published for '
                      f'the {label}. Execution cost could NOT be checked.'))
    elif spread > SPREAD_WARN_PCT:
        flags.append(('warn',
                      f'{entry.signal_ticker}: the {label} average bid-ask spread is '
                      f'{spread:.2f}%, above the {SPREAD_WARN_PCT:.2f}% threshold.'))

    return flags


def deviations(mapping=None):
    """`{signal_ticker: sentence}` for every line whose executed asset is not the signalled
    one. This is the scope `tools/ca_execution_gap.py` must cover; a gap report that measures
    fewer lines than this is measuring the wrong thing."""
    mapping = CA_MAPPING if mapping is None else mapping
    return {k: e.deviation for k, e in sorted(mapping.items()) if e.deviation}
