"""
Momentum filters, each named by its author's convention.

The two SMA filters here count DIFFERENT things, and the difference is not cosmetic:

* Keller's ``SMA(12)`` counts **lags**. BAA (SSRN 4166845) n.5: "The SMA(12) momentum (with
  lag 12 months) equals the present price pt divided by the average of the last 13 asset
  prices including the present (also noted as SMA13), minus 1." Thirteen prices.
* Faber's ``SMA(10)`` counts **prices**. Ten.

Averaging twelve prices instead of thirteen is a one-character error that survives every
self-consistency test, because the result still looks like a momentum score: it is smooth,
correctly signed most of the time, and ranks assets plausibly. What it is not is the
paper's filter. Keller & Butler state the same definition from the other side (PAA, SSRN
2759734, §1): SMA(12) "is based on a linearly decreasing weight filter over the previous 12
monthly returns (so the most recent month return is weighted 12 times ... up to the oldest
(12th) month one time)". With thirteen prices the weights come out (12, 11, ..., 1) exactly;
with twelve they come out (11, 10, ..., 1, 0) and the oldest month silently drops out.

`tests/test_paper_rules.py::TestSMAWindows` pins both windows against arithmetic written out
in the test, from both directions, so neither can be "made consistent" with the other.

`fill_method=None` is passed explicitly at every `pct_change` call. It is the default from
pandas 2.2 onward and the only legal value in 3.x, but through 2.1 the default was `'pad'` —
which forward-fills a gap before differencing and would resurrect a delisted product's last
price into a live momentum score, undoing exactly what `_apply_stale_policy` writes NaN for.
Explicit costs nothing and removes the version dependence.
"""

import pandas as pd


def calc_13612w(monthly_prices):
    """
    13612W (weighted): 12*r1 + 4*r3 + 2*r6 + 1*r12.
    Returns a scores DataFrame (same index/columns as monthly_prices).

    Keller's papers divide the sum by 4; this does not, because the omission is
    rank- and sign-preserving and every consumer here uses the score only to rank
    or to test against zero. The printed "Momentum Score" in the live report is
    therefore 4x the paper's figure — see `strategy_specs/` and KNOWN_GAPS.md.
    """
    r1 = monthly_prices.pct_change(1, fill_method=None)
    r3 = monthly_prices.pct_change(3, fill_method=None)
    r6 = monthly_prices.pct_change(6, fill_method=None)
    r12 = monthly_prices.pct_change(12, fill_method=None)
    return (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)


def calc_13612u(monthly_prices):
    """
    13612U (unweighted): average of 1,3,6,12 month returns.
    """
    r1 = monthly_prices.pct_change(1, fill_method=None)
    r3 = monthly_prices.pct_change(3, fill_method=None)
    r6 = monthly_prices.pct_change(6, fill_method=None)
    r12 = monthly_prices.pct_change(12, fill_method=None)
    return (r1 + r3 + r6 + r12) / 4.0


def calc_sma12_ratio(monthly_prices):
    """Keller's SMA(12): price divided by the average of the last **13** prices, minus 1.

    Thirteen, not twelve. The name counts lags — p_t against p_t..p_t-12 — and Keller notes
    it is "also noted as SMA13" for exactly that reason (BAA n.5). The function keeps the
    paper's name; the window follows the paper's definition.

    Used by BAA (offensive and defensive ranking, and the absolute test against BIL) and by
    PAA (the breadth count that sets the bond fraction). A wrong window here moves all four.
    """
    sma13 = monthly_prices.rolling(window=13).mean()
    return (monthly_prices / sma13) - 1


def calc_sma10_ratio(monthly_prices):
    """Faber's SMA(10): price divided by the average of the last **10** prices, minus 1.

    Ten prices, and deliberately NOT the same convention as `calc_sma12_ratio` above.
    Faber (2006) counts prices where Keller counts lags; making the two "consistent" would
    break GTAA. `tests/test_paper_rules.py::test_sma10_still_averages_ten_prices` guards it.
    """
    sma10 = monthly_prices.rolling(window=10).mean()
    return (monthly_prices / sma10) - 1
