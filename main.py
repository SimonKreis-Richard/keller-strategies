#!/usr/bin/env python3
"""
Keller Strategies - Centralized Dashboard & Execution Engine

This file consolidates all parameters and execution logic. Simply update
the variables in the "USER DASHBOARD" section below to control the script.
"""

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common.data_engine import PriceStore, MAX_STALE_DAYS, REFRESH_MIN_HOURS, get_live_prices
from common.momentum import calc_13612w, calc_13612u
from common.metrics import calculate_metrics, build_rf_series, SEGMENT_MIN_MONTHS
from common.ledger import (ExecutionConfig, run_ledger, validate_row,
                           WeightInvariantError)
from common.coverage import coverage_report
from common import eras
from common.user_config import load_user_config
from strategies.haa import HAA_Simple, HAA_Balanced, HAA_12
from strategies.daa import DAA_G12, DAA_G4, DAA_G6
from strategies.vaa import VAA_G12, VAA_G4
from strategies.haa_leveraged import (HAA_G3_Leveraged, HAA_G4_Leveraged,
                                      HAA_G3_Leveraged_3X, HAA_G5_Leveraged_3X)
from strategies.daa_leveraged import (DAA_G3_Leveraged, DAA_G4_Leveraged,
                                      DAA_G3_Leveraged_3X, DAA_G5_Leveraged_3X)
from strategies.passive import (Sixty_Forty_1X, SPY_2X_Benchmark, SPY_3X_Benchmark,
                                RiskParity_3X)
from strategies.baa_leveraged import (BAA_G3_Leveraged, BAA_G4_Leveraged,
                                      BAA_G3_Leveraged_3X, BAA_G4_Leveraged_3X)
from strategies.gem_leveraged import (DM_G3_Leveraged, DM_G3_Leveraged_3X,
                                      DM_G5_Leveraged_3X)
from common.letf_mapper import (LETFMapper, assert_unlevered_defensive,
                                holds_leveraged_product)
from common.margin_sizing import NotCalculable, intraperiod_max_drawdown
from strategies.baa import BAA_G12, BAA_G4, BAA_SPY
from strategies.paa import PAA2
from strategies.gem import DMComposite as GEM, GEMClassic
from strategies.gtaa import GTAA_5
from strategies.base import BaseStrategy

# =========================================================================================
# ================================== USER DASHBOARD =======================================
# =========================================================================================
# The values below are the PROJECT DEFAULTS. Your personal settings (account balances,
# strategy picks, dates, leverage...) belong in `user_config.json` at the project root —
# gitignored, so they never reach the public repository. Copy `user_config.example.json`
# to get started; any key present there overrides the default shown here. The GUI
# (`python app.py`) reads and writes the same file.
_UC = load_user_config()

# 1. EVALUATION ERA AND GENERAL PARAMETERS
# THE EVALUATION WINDOW IS NOT A SETTING. It used to be: a START_DATE and an END_DATE you
# typed in. Every headline number was then conditional on a choice nobody could defend —
# 2015 flatters trend following, 2009 flatters buy-and-hold, 2007 flatters anything holding
# bonds. What replaced it is a fixed era cut into pre-registered market regimes whose
# boundaries were dated by the NBER, the FOMC, the BLS and the S&P 500 itself. See
# common/eras.py; the report prints every strategy in every segment.
START_DATE             = eras.COMMON_ERA_START  # derived floor; see common/eras.py
END_DATE               = None                   # None = the last COMPLETE month in the store
LEVERAGE_FACTOR        = _UC.get('LEVERAGE_FACTOR', 1.0)
                                       # Global MARGIN leverage applied to active strategies' returns
                                       # (borrowed money, e.g. brokerage margin / personal credit line).
                                       # This is SEPARATE from leveraged-ETF products: a 2x LETF strategy
                                       # run at LEVERAGE_FACTOR=1.3 results in ~2.6x effective exposure.
MARGIN_BORROW_RATE     = _UC.get('MARGIN_BORROW_RATE', 0.06)
                                       # Annual interest rate paid on the BORROWED margin portion only
                                       # (i.e. on LEVERAGE_FACTOR - 1.0). The financing cost embedded in
                                       # LETFs is already reflected in their real price history, so it is
                                       # NOT charged again here. Set to 0.0 to ignore margin borrowing cost.
MARGIN_FOLLOWS_SIGNAL  = _UC.get('MARGIN_FOLLOWS_SIGNAL', True)
                                       # True  = borrow ONLY against the offensive sleeve. Effective leverage
                                       #         is 1 + (LEVERAGE_FACTOR-1) x offensive weight, so the loan is
                                       #         repaid as the signal goes risk-off and the defensive sleeve is
                                       #         held unlevered at 1x. This mirrors the leveraged-ETF design,
                                       #         where rotating into the defensive asset de-levers by itself.
                                       # False = flat leverage every month, including risk-off months. The loan
                                       #         stays drawn, so bonds/cash are held with borrowed money during
                                       #         drawdowns. Kept for comparison and for reproducing old results.
                                       # Inert while LEVERAGE_FACTOR == 1.0 (no borrowing at all).
COST_PCT_PER_SIDE      = _UC.get('COST_PCT_PER_SIDE', _UC.get('TRANSACTION_COST_PCT', 0.001))
                                       # ONE-WAY transaction cost, charged PER LEG on the
                                       # notional actually traded. Keller's HAA paper states
                                       # "we assume a one-way transaction costs of 0.1%", so
                                       # 0.001 here means a full A->B rotation costs 0.2%.
                                       # The old key TRANSACTION_COST_PCT still works (with a
                                       # deprecation notice) but meant HALF of one-way.
EXECUTION_CONVENTION   = _UC.get('EXECUTION_CONVENTION', 'next_open')
                                       # 'next_open'   = fill at the open of the session AFTER
                                       #                 the signal. Matches after-hours GTC
                                       #                 orders, and is the only convention
                                       #                 that describes a tradeable portfolio.
                                       # 'next_close'  = fill at that session's close.
                                       # 'signal_close'= fill at the signal's own close. This
                                       #                 is look-ahead by construction (worth
                                       #                 +2.09 pp of CAGR on HAA-G8) and exists
                                       #                 only to reproduce pre-audit numbers.
CASH_TICKER            = _UC.get('CASH_TICKER', 'BIL')
                                       # Uninvested weight earns this asset's return. None =>
                                       # 0%, which is a modelling choice, not a market fact.
COVERAGE_POLICY        = _UC.get('COVERAGE_POLICY', 'trim')
                                       # What to do when a strategy's window starts before its
                                       # assets existed. 'trim' = shorten the window and say so
                                       # loudly; 'strict' = refuse to measure it at all.
RF_ANNUAL_FALLBACK     = _UC.get('RF_ANNUAL_FALLBACK', 0.03)
                                       # Assumed annual cash rate for months before BIL's
                                       # 2007-05 inception, where no realised T-bill total
                                       # return exists. Reported alongside every Sharpe.
LOOKBACK_MONTHS        = 13            # Momentum warm-up, in complete months. Used by the
                                       # coverage guard, NOT by truncating the result series.
DATA_START_DATE        = '2000-01-01'  # Master start date for yfinance downloads

# 2. EXECUTION MODE
# Modify EXECUTION_MODE to switch from historical analysis to exact daily signals.
# True = Live signals for CURRENT_EXECUTION_DATE.
# False = Full backtest over the frozen era, reported segment by segment (common/eras.py).
EXECUTION_MODE         = _UC.get('EXECUTION_MODE', False)
CURRENT_EXECUTION_DATE = _UC.get('CURRENT_EXECUTION_DATE', '2026-07-01')

# 3. LIVE BROKER ACCOUNT CONFIGURATION
# Used only when EXECUTION_MODE = True. Set YOUR real account names/balances in
# user_config.json (gitignored) — the placeholders below are examples only.
BROKER_ACCOUNTS = _UC.get('BROKER_ACCOUNTS', [
    {"account_name": "TFSA", "account_balance": 10000.0, "account_priority": 1},
    {"account_name": "RRSP", "account_balance": 10000.0, "account_priority": 2},
])
# LIVE ORDER-SIZING KNOBS (used only when EXECUTION_MODE = True). These shape how target
# weights become whole-share orders across your accounts; NONE of them affect the backtest.
# - SAFETY_MARGIN_PCT: ~2.0% keeps a cash reserve per account so orders aren't rejected on price moves mid-execution.
# - FLEXIBILITY_BAND_PCT: ~5.0%. One-sided DOWNWARD tolerance — accept UNDER-filling a position by up to X% rather than splitting the last sliver into a lower-priority account.
# - FLUSH_ROUND_UP_BAND_PCT: 0 = off. One-sided UPWARD tolerance — round the last lot UP to deploy idle cash, overshooting a position's target by at most X%, and only when the extra lot fits the account's usable balance (never touches the safety reserve). Handy for small accounts without fractional shares.
# - PRICE_CAP_MARGIN_PCT: 0 = off. Limit-price ceiling for after-hours GTC orders (e.g. IBKR
#   Midprice with a price cap): cap = live quote × (1 + X%). Shares are SIZED AT THE CAP
#   (worst-case cost), so the broker's funds check passes by construction — no rejections —
#   while the cap still sits above the market, so next-morning execution is near-certain.
#   Fills below the cap (the normal midprice case) leave a small cash residue = your price
#   improvement. ~1.5% covers a typical overnight gap on liquid ETFs.
# - MINIMUM_TRADE_PCT: ~2.0-3.0%. Skips tiny residual buys to avoid fees on small slices.
# - SHARE_LOT_SIZE: 1 = exact integer shares (standard for modern retail brokers).
SAFETY_MARGIN_PCT = _UC.get('SAFETY_MARGIN_PCT', 2.0)            # Reserve X% cash per account to prevent order rejections
FLEXIBILITY_BAND_PCT = _UC.get('FLEXIBILITY_BAND_PCT', 5.0)      # Accept UNDER-filling a position by up to X% (downward only) to avoid fragmenting it across accounts
FLUSH_ROUND_UP_BAND_PCT = _UC.get('FLUSH_ROUND_UP_BAND_PCT', 0.0)  # Round last lot UP to flush idle cash, overshooting target by at most X% (upward only; 0 = off). Bounded by usable balance.
PRICE_CAP_MARGIN_PCT = _UC.get('PRICE_CAP_MARGIN_PCT', 1.5)      # Limit-price cap = quote × (1 + X%); shares sized at the cap so GTC orders can't be rejected for funds (0 = off)
MINIMUM_TRADE_PCT = _UC.get('MINIMUM_TRADE_PCT', 2.0)            # Minimum target weight (%) required to trigger a trade slice (avoids tiny transactions)
SHARE_LOT_SIZE = _UC.get('SHARE_LOT_SIZE', 1)                    # Round shares to this lot size. 1 = exact integer shares (recommended for modern brokers)
FRACTIONAL_SHARES = _UC.get('FRACTIONAL_SHARES', False)          # Set to True if your broker allows fractional shares (makes FLUSH_ROUND_UP_BAND_PCT unnecessary)

# 4. LIST OF STRATEGIES TO EXECUTE (project default / catalog)
# Add or remove a hashtag (#) in front of a strategy to enable or disable it.
# NOTE: user_config.json overrides this catalog, and the CLI flag `--strategy` overrides
# everything. Two config forms are accepted, in this order:
#   "STRATEGIES"          — legacy allow-list of registry keys (`--list`), still honoured
#   "EXCLUDED_STRATEGIES" — the dashboard's deny-list: run everything except these
# The dashboard writes the deny-list, so unticking a box there changes what `python main.py`
# runs too. That is deliberate: the two must never report different line-ups.
#
# NOTE — OUR TAKE (the project creators', not an official or impartial ranking):
# We favor the HAA family for STRUCTURAL reasons: a single external canary (TIP) rather than a
# breadth count, a defensive sleeve that picks between BIL and IEF instead of defaulting to one,
# and the most recent of Keller's papers. G12 is preferred over G8 for asset-class breadth —
# twelve sleeves spanning US/ex-US equity, REITs, commodities, gold and three bond tenors — not
# for its backtested rank.
#
# That distinction matters, because the performance claim this note used to make was false twice
# over. On the owner's own window HAA_G1_Simple scored a HIGHER Sharpe and Sortino than HAA_G12;
# and in Keller's own Dec-1970..Dec-2022 results HAA-8 beats HAA-12 on BOTH max drawdown (-9.7%
# vs -10.7%) and Sharpe (1.21 vs 1.19) at identical 15.9% CAGR. There is no support for a G12
# performance preference in the source paper either.
#
# More generally: do not choose a variant from the ranked table. The report prints the rank
# correlation of that table between disjoint sub-periods, and it is approximately zero. The table
# is a description of what each regime rewarded, not a forecast.
# This is a personal preference/bias, NOT a scientific or impartial recommendation, and NOT
# financial advice. Do your own research; 0 liability.
STRATEGIES_TO_RUN = [

    # ---- HAA (Hybrid Asset Allocation) ----
    # HAA_Simple(),            # CONTROL — single asset, not a strategy
    # HAA_Balanced(),          # 8-asset "balanced" variant (equal-weight top 4)
    HAA_12(),

    # ---- BAA (Bold Asset Allocation) ----
    # BAA_G12(),
    # BAA_G4(),
    # BAA_SPY(),      # control

    # ---- Leveraged BAA ----
    # Only universes that execute at a UNIFORM multiple survive (see common/letf_mapper.py),
    # and only at 2x: the 3x twin of every universe measured rho ~= 0.999 against its own 2x
    # sibling, so it carried no information the 2x does not already show — while having no
    # bear-market history at all (no 3x product predates 2008-11).
    # BAA_G3_Leveraged(2),
    # BAA_G4_Leveraged(2),

    # ---- DAA (Defensive Asset Allocation) ----
    # DAA_G12(), 
    # DAA_G4(), 
    # DAA_G6(), 
    
    # ---- Leveraged DAA ----
    # DAA_G3_Leveraged(2),
    # DAA_G4_Leveraged(2),

    # ---- VAA (Vigilant Asset Allocation) ----
    # VAA_G12(),
    # VAA_G4(),

    # ---- Leveraged VAA ----
    # VAA_G3_Leveraged(2),
    # VAA_G4_Leveraged(2),

    # ---- DM (Antonacci's four-module composite dual momentum) ----
    # GEM(),
    # ---- Leveraged DM (single module, custom universe) ----
    # DM_G3_Leveraged(2),

    # ---- PAA (Protective Asset Allocation) ----
    # PAA2(),

    # ---- Leveraged PAA ----
    # PAA_G3_Leveraged(2),
    # PAA_G4_Leveraged(2),

    # ---- GTAA (Global Tactical Asset Allocation) ----
    # GTAA_5(),
]

# 5. OUTPUT CONFIGURATION
SAVE_FILES_TO_DISK = _UC.get('SAVE_FILES_TO_DISK', True)         # If False, outputs are only shown in the terminal (no files generated).
STRATEGIES_TO_DISPLAY = _UC.get('STRATEGIES_TO_DISPLAY', [])     # Strategy names to include in output. Empty = all. Example: ['HAA_G12', 'SPY_Benchmark']
TOP_N_COUNT = _UC.get('TOP_N_COUNT', 100)                        # Number of top strategies to show in the summary
# Hours before the daily cache re-checks Yahoo. 0 = every run, which is what it used to do
# unconditionally: 9.1s of network and a 10MB CSV rewrite per launch, to learn nothing.
# Signals come from COMPLETE months and live sizing quotes prices separately, so an
# intraday-stale daily cache cannot change a decision. `--refresh` forces it.
CACHE_REFRESH_HOURS = _UC.get('CACHE_REFRESH_HOURS', REFRESH_MIN_HOURS)
# Refuse a ticker with an interior gap longer than MAX_STALE_DAYS instead of forward-filling
# past it. True since 2026-07-30: ARCHITECTURE.md described this guard as enforced, but the
# key was never placed in the production config, so every CLI and GUI run silently built the
# store with strict_gaps=False. (The committed cache has no long gap — provenance() shows
# long_gaps: [] — so no number changed; the guard now guards.) Set False to proceed through
# a data outage; the report header then lists every gap carried, so the override is loud.
STRICT_GAPS = _UC.get('STRICT_GAPS', True)
# --- sustainable-leverage sizing (common/margin_sizing.py via common/leverage_advice.py) --- #
# The ONE preference parameter of the sizing module, in MULTIPLES OF DRAWDOWN: k=3 means the
# position must survive three times the stressed drawdown before the broker liquidates.
# MarginPolicy.PRESETS names 5 / 3 / 2 as prudent / balanced / aggressive.
SAFETY_FACTOR_K = _UC.get('SAFETY_FACTOR_K', 3.0)
# Assumed house maintenance requirement on an ordinary long ETF position. FINRA Rule 4210(c)
# floors it at 25%; a leveraged product is charged this times the FUND'S OWN multiple. An
# assumption about a broker nobody has chosen yet — see common/leverage_advice.py.
MAINTENANCE_BASE = _UC.get('MAINTENANCE_BASE', 0.30)
# Maximum leverage the broker or credit line permits, if it is known. None => that cap is
# reported unbounded and the fact is carried in every row's assumptions, which is the honest
# state until a real account exists. It must be the TAXABLE account's equity if it is ever
# filled in: registered accounts do not permit margin.
BORROWING_CAPACITY_LEVERAGE = _UC.get('BORROWING_CAPACITY_LEVERAGE', None)
# How many strategies to list under each regime segment in the per-regime leaderboards.
SEGMENT_TOP_N = _UC.get('SEGMENT_TOP_N', 5)
RANK_BY = _UC.get('RANK_BY', 'sortino')                          # Metric to rank by. Options: 'cagr', 'sharpe', 'sortino', 'vol', 'max_dd'. (Sortino is recommended)
# Whose inception may shorten the SHARED ranked window.
#   'strategies' (default) — only entries somebody published (fidelity != 'custom'). The
#     leverage-forced G4 wraps were dragging the headline table from 2008-06 to 2010-02
#     for all 25 entries, so the flagship drawdown column excluded the 2008 crisis. They
#     are still measured, over their own history, in a separate block underneath.
#   'all' — one window across everything, as before. Comparable rows everywhere, at the
#     cost of the months the earlier entries could otherwise be read through.
RANKED_WINDOW_POLICY = _UC.get('RANKED_WINDOW_POLICY', 'strategies')

# 6. PASSIVE BENCHMARKS
# They will be plotted on charts and measured if desired.
class Golden_Butterfly(BaseStrategy):
    fidelity = 'faithful'
    source = ("Tyler, portfoliocharts.com — the published Golden Butterfly: 20% each of "
              "total market, small-cap value, long treasuries, short treasuries and gold")
    role = 'benchmark'

    def __init__(self):
        super().__init__("Golden_Butterfly", is_active=False)
        self.assets = ['VTI','IJS','TLT','SHY','GLD']
    def sleeves(self):
        # A static 20/20/20/20/20 portfolio never de-risks; SHY/TLT are structural
        # allocations, not a defensive switch. Declaring them defensive would make the
        # margin logic think the benchmark de-levers, which it does not.
        return {'offensive': set(self.assets), 'defensive': set(), 'canary': []}
    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w = 0.2
        for a in self.assets:
            alloc[a] = w
        return alloc

class SPY_Benchmark(BaseStrategy):
    fidelity = 'faithful'
    source = 'buy and hold SPY — no rule to be faithful or unfaithful to'
    role = 'benchmark'

    def __init__(self):
        super().__init__("SPY_Benchmark", is_active=False)
    def sleeves(self):
        return {'offensive': {'SPY'}, 'defensive': set(), 'canary': []}
    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        alloc['SPY'] = 1.0
        return alloc

# Add these benchmarks to the list of strategies to run if you wish.
STRATEGIES_TO_RUN.extend([
    Golden_Butterfly(), 
    SPY_Benchmark()
])

# 7. DATA ENGINE CONFIGURATION
# Cache file to speed up execution without re-downloading from Yahoo Finance every time.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, 'data', 'cache')
# Kept as an alias for the legacy monthly cache path. Nothing reads it any more: the store
# writes daily_open.csv / daily_close.csv / daily_adj_close.csv into CACHE_DIR instead.
CACHE_FILE = os.path.join(CACHE_DIR, 'prices_master_cache.csv')

# Every ticker any REGISTERED strategy can hold or read as a signal, and nothing else.
# Fifteen names went with the 39 -> 25 registry cut: VUG/VTV/VBK/VBR and the nine SPDR
# sectors (only the deleted U6/U15 universes used them), IWD/SCZ (HAA_G16), and SHV/UST
# (DAA1_G12 — UST being the 2x treasury ETF that made its defence levered).
TICKERS = [
    # Universal offensive and defensive
    'SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VWO', 'VNQ', 'DBC', 'IEF', 'TLT', 'HYG', 'LQD',
    'VEA', 'BIL', 'SHY', 'BND', 'TIP', 'GSG', 'GLD', 'EEM', 'IYR',
    # GEM's all-world ex-US leg (VEU = Vanguard FTSE All-World ex-US, 2007-03)
    'VEU',
    # DM's four-module composite adds mortgage REITs
    'REM',
    # Benchmarks and GTAA
    'VTI', 'IJS', 'EFA',
    # History donors — see common/data_engine.HISTORY_BACKFILL / SYNTHETIC_CASH. These are
    # never traded and never ranked: they exist only to extend VEA/VWO/BND backwards past
    # their 2007 inceptions, and to build a T-bill return before BIL existed. Without them
    # NOTHING in this repository could be measured through the onset of the 2008 crisis.
    # EFA is already above (GTAA_G5 trades it); EEM is already above (VAA_G12 trades it).
    'AGG', '^IRX',
    # LETFs (3x) — the images MAP_3X points at. Held directly by SPY_3X_Benchmark (UPRO) and
    # RiskParity_3X (UPRO/TMF), and mapped onto by the 3x wraps.
    'UPRO', 'TQQQ', 'TNA', 'TMF', 'EDC',
    # LETFs (2x)
    'SSO', 'QLD', 'UWM', 'UGL',
    # Removed after the LETF audit: EURL, DRN, UBT, EET, EFO, URE (all under the $100M
    # liquidity floor -> issuer-closure risk) and UCO (2x WTI crude, wrong underlying for
    # the broad-commodity GSG signal, and no 3x commodity product exists).
]

TICKER_NAMES = {
    'SPY': 'S&P 500 ETF', 'IWM': 'Russell 2000', 'QQQ': 'Nasdaq 100', 'VGK': 'Europe ETF', 
    'EWJ': 'Japan ETF', 'VWO': 'Emerging Mkts', 'VNQ': 'Real Estate', 'DBC': 'Commodity ETF', 
    'IEF': '7-10 Yr Treas', 'TLT': '20+ Yr Treas', 'HYG': 'High Yield Bd', 'LQD': 'Inv Grade Bd',
    'VEA': 'Dev Mkts ex-US', 'VEU': 'All-World ex-US', 'VUG': 'US Growth', 'VTV': 'US Value', 'VBK': 'Small-Cap Gro',
    'VBR': 'Small-Cap Val', 'BIL': '1-3 Mo Treas', 'SHY': '1-3 Yr Treas', 'BND': 'Total Bond Mkt', 
    'TIP': 'TIPS Bond', 'GSG': 'Commodity Idx', 'GLD': 'Gold Trust', 'EEM': 'Emerg Mkts', 
    'IYR': 'US Real Estate', 'IWD': 'R1000 Value', 'SCZ': 'Dev Small-Cap', 'REM': 'Mortgage REIT', 
    'SHV': 'Short Treasury', 'UST': '2-10 Yr Treas', 'VTI': 'Total Stock Mkt', 'IJS': 'Small-Cap Val', 
    'EFA': 'EAFE (Dev Mkts)', 'XLB': 'Matls Select', 'XLE': 'Energy Select',
    'XLF': 'Fincl Select', 'XLI': 'Indus Select', 'XLK': 'Tech Select', 'XLP': 'Staples Select', 
    'XLU': 'Utils Select', 'XLV': 'Health Care', 'XLY': 'Cons Disc',
    'UPRO': 'UltraPro S&P500 3x', 'TQQQ': 'UltraPro QQQ 3x', 'TNA': 'Small Cap 3x',
    'TMF': '20+ Yr Treas 3x', 'EDC': 'Emerging Mkts 3x',
    'SSO': 'Ultra S&P500 2x', 'QLD': 'Ultra QQQ 2x', 'UWM': 'Ultra Russell2000 2x',
    'UGL': 'Gold 2x'
}

# =========================================================================================
# ========================= END OF DASHBOARD (ENGINE BELOW) ==========================
# =========================================================================================

# All recognized strategies mapped by name
# THE ADMISSION RULE, since 2026-07-28. An entry is registered only if it is one of:
#
#   1. a universe and parameterisation its AUTHOR published ('faithful'),
#   2. the same thing on substitute funds, stated as such ('proxy' — VAA_G4 only),
#   3. a universe this repo was FORCED to invent so that a leveraged sleeve executes at one
#      uniform multiple ('custom' — the G3 and G4 leveraged sizes, and nothing else),
#   4. a degenerate single-asset CONTROL, excluded from the selection statistics,
#   5. a passive benchmark.
#
# Everything else is a degree of freedom wearing a strategy's name. Sixteen entries were
# deleted under this rule; each deletion is documented where its class used to live.
ALL_STRATEGIES = {
    # HAA — the paper presents HAA-8 and HAA-12
    'HAA_G1_Simple': HAA_Simple,          # CONTROL
    'HAA_G8_Balanced': HAA_Balanced,
    'HAA_G12': HAA_12,

    # Leveraged HAA — CUSTOM universes, added 2026-07-29. HAA passes RULE 4 most cleanly of
    # any family: the TIP canary is exogenous, and the absolute-momentum filter is per-asset
    # with no denominator that moves when the universe shrinks. TO=NO/2 is Keller's own
    # ratio, so even the selection count is not a liberty here.
    'HAA_G3_Leveraged_2X': lambda: HAA_G3_Leveraged(2),
    'HAA_G4_Leveraged_2X': lambda: HAA_G4_Leveraged(2),

    # ---- 3x, role='exploratory' (added 2026-07-29) --------------------------------------- #
    # Measured and reported in FULL; excluded from the selection statistics, so these twelve
    # new registry rows cost exactly ONE extra trial (18 -> 19). The role is derived from
    # `leverage` on LeveragedWrapMixin, so a new 3x factory cannot forget it.
    #
    # The *_G3_3X entries are the point of the exercise: each twins an existing *_G3_2X on the
    # SAME universe and the SAME signal, so the leverage LEVEL is the only changed variable.
    # That pairing exists nowhere else here — GLD has no 3x product and TLT/EEM have no
    # admissible 2x one, so every wider universe is single-ratio by force.
    #
    # NOTHING BELOW IS A RECOMMENDATION. No 3x product predates 2008-11 (UPRO 2009-06,
    # TQQQ 2010-02, TMF 2009-04, EDC 2008-12), so not one of these drawdowns has been measured
    # through a bear market, and §7 of LEVERAGE.md puts the 2008 figure for 3x near -95%.
    'HAA_G3_Leveraged_3X': HAA_G3_Leveraged_3X,
    'HAA_G5_Leveraged_3X': HAA_G5_Leveraged_3X,

    # DAA
    'DAA_G12': DAA_G12,
    'DAA_G4': DAA_G4,
    'DAA_G6': DAA_G6,

    # Leveraged DAA — CUSTOM universe, uniform-ratio only (see common/letf_mapper.py) and
    # 2x only (see the note at the bottom of this dict).
    #
    # G3 was deleted on 2026-07-29 under RULE 5 (T=1 against B=2 collapsed the middle rung of
    # the cash ladder) and RESTORED the same day, because the fix belonged in the parameter,
    # not in the registry: `T = max(n//2, B)` gives G3 a T of 2 and the rung survives. See
    # strategies/daa_leveraged.py for the worked table.
    'DAA_G3_Leveraged_2X': lambda: DAA_G3_Leveraged(2),
    'DAA_G4_Leveraged_2X': lambda: DAA_G4_Leveraged(2),
    'DAA_G3_Leveraged_3X': DAA_G3_Leveraged_3X,
    'DAA_G5_Leveraged_3X': DAA_G5_Leveraged_3X,

    # VAA — no leveraged variant exists, and that is a finding rather than an omission.
    # VAA's protection is a breadth count over its OWN offensive universe, so restricting
    # that universe for LETF execution rewrites the protection rule (RULE 4). See the
    # deletion note at the bottom of strategies/vaa.py.
    'VAA_G12': VAA_G12,
    'VAA_G4': VAA_G4,                     # paper's universe, proxy tickers

    # BAA
    'BAA_G12': BAA_G12,
    'BAA_G4': BAA_G4,
    'BAA_G1_SPY': BAA_SPY,                # CONTROL

    # Leveraged BAA — CUSTOM universes. B=1 means the canary rule is binary in the paper
    # too, so there is no cash ladder for a small universe to collapse; G3 survives here
    # where DAA's G3 did not.
    'BAA_G3_Leveraged_2X': lambda: BAA_G3_Leveraged(2),
    'BAA_G4_Leveraged_2X': lambda: BAA_G4_Leveraged(2),
    # BAA_G4 at 3x is a DIFFERENT universe: [SPY,QQQ,IWM,EEM], because TLT and DBC sit in
    # BAA's defensive basket and a dual-role asset would be held 1x beside 3x siblings.
    'BAA_G3_Leveraged_3X': BAA_G3_Leveraged_3X,
    'BAA_G4_Leveraged_3X': BAA_G4_Leveraged_3X,

    # PAA — only the a=2 (most protective) variant is registered; PAA0/PAA1 measured
    # rho 0.974-0.982 against it and added nothing. No leveraged variant: same RULE 4
    # failure as VAA, spelled out at the bottom of strategies/paa.py.
    'PAA2_G12': PAA2,

    # DM — Antonacci's dual momentum. `DM_G8_Composite` is his 2012 paper's equally-weighted
    # four-module composite (equities / credit / REITs / stress), on ETF proxies for its index
    # series: rule 1, not a custom strategy. It was called `GEM_G8_FourModule_Custom` until
    # 2026-07-29, when reading SSRN 2042750 showed the three "invented" pairs were his own.
    # The leveraged sleeve is a different object: one module on [SPY, QQQ, IWM], a universe
    # chosen by what has 2x products, so it stays custom.
    'DM_G8_Composite': GEM,
    # GEM proper — the single-module flagship (added 2026-07-30, at Simon's request): SPY vs
    # VEU relative, absolute momentum gauged on SPY FIRST (the book's ordering, not the 2012
    # paper's), defending into BND. Rule 2 of the admission rule: published rules, ETF
    # stand-ins. NO leveraged variant is admissible — VEU has no LETF above the liquidity
    # floor (RULE 1), and a US-only restriction reproduces DM_G3_Leveraged, which is already
    # registered. The derivation is in strategies/gem.py::GEMClassic.
    'GEM_G2_Classic': GEMClassic,
    'DM_G3_Leveraged_2X': lambda: DM_G3_Leveraged(2),
    'DM_G3_Leveraged_3X': DM_G3_Leveraged_3X,
    'DM_G5_Leveraged_3X': DM_G5_Leveraged_3X,

    # GTAA — G5 is the faithful five-asset timing model and is now the only Faber entry
    'GTAA_G5': GTAA_5,

    # Benchmarks
    'Golden_Butterfly': Golden_Butterfly,

    # Passive references, added 2026-07-29 (strategies/passive.py). Every leveraged wrap here
    # was previously compared only against 1x references, so "how much of this is the timing
    # rule and how much is just leverage?" had no answer. These four supply the subtraction and
    # add zero degrees of freedom — no universe choice, no signal, no fitted parameter.
    # None may set the shared ranked window: `holds_leveraged_product` bars the three levered
    # ones structurally, so they cannot cost the entries they exist to be compared against.
    'Sixty_Forty_1X': Sixty_Forty_1X,
    'SPY_2X_Benchmark': SPY_2X_Benchmark,
    'SPY_3X_Benchmark': SPY_3X_Benchmark,
    'RiskParity_3X': RiskParity_3X,
    'SPY_Benchmark': SPY_Benchmark
}


def strategy_roster():
    """`[{name, role, fidelity, family}, ...]` for every registry entry, in registry order.

    Built by INSTANTIATING each factory rather than reading class attributes, because half
    the registry values are `lambda: X_Leveraged(2)` and the role/fidelity of a leveraged
    wrap is decided in `LeveragedWrapMixin`, not on the class the lambda closes over.
    Construction touches no data — it only builds ticker lists and runs
    `validate_universe` — so this is cheap enough to call at import time.

    Any UI that offers a choice of strategies should filter on `role` from here. Roles are
    defined on `BaseStrategy`: 'strategy' is a portfolio you might hold, 'benchmark' is a
    passive reference, 'control' is a degenerate diagnostic that exists to be subtracted
    from a family and is not something anyone would hold.
    """
    from common.palette import family_of

    roster = []
    for name, factory in ALL_STRATEGIES.items():
        obj = factory()
        role = getattr(obj, 'role', 'strategy')
        roster.append({'name': name, 'role': role,
                       'fidelity': getattr(obj, 'fidelity', 'custom'),
                       # Read from the object, never parsed off the key. The UI's ratio filter
                       # and the chart's width channel both consume this.
                       'leverage': float(getattr(obj, 'leverage', 1.0) or 1.0),
                       'family': family_of(name, role)})
    return roster


#: Registry keys whose role is 'control', read from the classes rather than listed by hand.
_CONTROL_KEYS = frozenset(d['name'] for d in strategy_roster() if d['role'] == 'control')

#: Registry keys whose role is 'exploratory' (the 3x variants). Hidden by default for the same
#: reason as the controls and behind their OWN switch — a 2x-vs-3x comparison is a legitimate
#: thing to want without also wanting the single-asset diagnostics.
_EXPLORATORY_KEYS = frozenset(d['name'] for d in strategy_roster()
                              if d['role'] == 'exploratory')

# THIRD registry change, 2026-07-29 (25 -> 22 -> 23), and it went both ways. Deleted: the four
# VAA/PAA leveraged wraps (RULE 4 — their protection is a breadth count over their OWN offensive
# universe, so restricting it for LETF execution rewrites the protection rule) and
# DAA_G3_Leveraged_2X (RULE 5). Added: HAA_G3/G4_Leveraged_2X. Then DAA_G3_Leveraged_2X was
# RESTORED the same day: RULE 5 had been written as a filter that deletes variants with T < B,
# and reads better as one that CHOOSES T = max(n//2, B). The variant was never defective — the
# parameter was. It ranks seventh of nine on Sortino, so the restoration is not a performance
# argument; see strategies/daa_leveraged.py for the b=0/b=1/b=2 decomposition.
#
# SECOND registry reduction, 2026-07-28 (39 -> 23), on the admission rule stated above the
# dict. Deleted: HAA_G4, HAA_G16 (custom sizes; G16 rho 0.935 vs G12 and started a year
# later); DAA_U6, DAA_U15, VAA_U6, VAA_U15 (custom US-only universes that also ranked BIL as
# an offensive momentum asset); DAA1_G12 (non-paper defensive sleeve containing UST, a 2x
# treasury ETF — see common/letf_mapper.assert_unlevered_defensive — and the single most
# expensive entry for coverage, dragging the comparison window from 2008-07 to 2011-04);
# BAA_G4_T2 (TO twiddle, rho 0.894); GTAA_G13_Moderate and GTAA_G13_Aggressive (no published
# Faber source for either, same universe, and the Aggressive one hybridised Keller's 13612U
# with Faber's SMA); and the four *_G2_Leveraged_2X (SPY+QQQ, rho 0.92 — a two-asset momentum
# choice between near-identical assets is not a choice).
#
# FIRST registry reduction, 2026-07-28 (64 -> 39), on two measured criteria kept deliberately
# separate.
#
# 1. REDUNDANCY. Over 156 common months (2012-01..2024-12), every leveraged 3X variant measured
#    rho ~= 0.996-0.999 against its own 2X sibling: same signal, same calendar, exposure merely
#    rescaled. The 3X entries were dropped (the factories still accept leverage=3 for ad-hoc
#    study; they are simply not registered, so they no longer inflate the ranking). Note the
#    honest caveat: rho ~= 0.999 means near-identical SHAPE, not magnitude — a 3x drawdown is far
#    worse than a 2x one. What was removed is a number that is both derivable and unrepresentative,
#    since no 3x product predates 2008-11 and therefore no 3x drawdown here has seen a bear market.
#    Also dropped on the same criterion: PAA_G12_V0/V1 (rho 0.974-0.982 vs V2), BAA_G12_T3
#    (rho 0.951 vs BAA_G12), and three of the four leveraged GEM sleeves (rho 0.95-0.999).
#
# 2. FIDELITY / NON-CONTRIBUTION. FAA, MAA, EAA, LAA, RAA and CAA were removed entirely. Across 8
#    named drawdown episodes, not one of them ever outperformed every retained strategy; the only
#    departures from the retained envelope were downside (MAA -13.3% in 2015-16, RAA -20.6% in the
#    2022 bear — worse than SPY). LAA measured rho 0.93 against the Golden_Butterfly benchmark that
#    is retained. MAA's low correlation with the rest was the decorrelation of a bug, not
#    information: MAA_G7 vs MAA_G7_TV was rho = 1.000, i.e. its shrinkage parameters did nothing.
#    None of the six implemented its source paper faithfully. See KNOWN_GAPS.md §6.

def parse_cli_args():
    """Parse CLI arguments to select strategies, list them, or switch modes."""
    import argparse
    parser = argparse.ArgumentParser(description="Keller Strategies Backtesting Engine")
    parser.add_argument('--strategy', type=str, nargs='+', help="Space-separated list of strategies to run.")
    parser.add_argument('--list', action='store_true', help="Show all available strategies and exit.")
    parser.add_argument('--live', action='store_true', default=None, help="Force execution in live signal generation mode.")
    parser.add_argument('--backtest', dest='live', action='store_false', default=None, help="Force backtest mode (overrides EXECUTION_MODE from user_config.json).")
    parser.add_argument('--refresh', action='store_true',
                        help="Force the trailing-window re-download even if the cache was "
                             "checked recently (see CACHE_REFRESH_HOURS).")
    return parser.parse_args()

def get_strategies_to_run(args):
    """Determine the strategies to run based on command line arguments or user dashboard."""
    if args.list:
        print("\nAvailable Strategies:")
        for name in sorted(ALL_STRATEGIES.keys()):
            print(f" - {name}")
        return None
        
    if args.strategy:
        # Support both space-separated and comma-separated elements
        selected = []
        for s in args.strategy:
            selected.extend([part.strip() for part in s.split(',') if part.strip()])
            
        run_list = []
        for name in selected:
            if name in ALL_STRATEGIES:
                run_list.append(ALL_STRATEGIES[name]())
            else:
                print(f"Warning: Strategy '{name}' is not recognized and will be skipped.")
        return run_list

    # user_config.json override: a LEGACY allow-list of registry keys (`--list`). Kept for
    # hand-written configs; the dashboard no longer writes it.
    if 'STRATEGIES' in _UC:
        run_list = []
        for name in _UC['STRATEGIES']:
            if name in ALL_STRATEGIES:
                run_list.append(ALL_STRATEGIES[name]())
            else:
                print(f"Warning: Strategy '{name}' in user_config.json is not recognized and will be skipped.")
        return run_list

    # The dashboard's format: a DENY-list. Everything runs unless it was unticked, so a
    # strategy added to the registry later appears on its own instead of being silently
    # frozen out by a list written months ago — which is exactly what STRATEGIES did.
    # Controls are excluded unless SHOW_CONTROLS, and the 3x variants unless SHOW_EXPLORATORY,
    # matching the GUI picker — so the CLI and the dashboard cannot disagree about what
    # "everything" means.
    if 'EXCLUDED_STRATEGIES' in _UC:
        excluded = set(_UC['EXCLUDED_STRATEGIES'] or ())
        unknown = sorted(excluded - set(ALL_STRATEGIES))
        if unknown:
            print(f"Warning: EXCLUDED_STRATEGIES names nothing in the registry: {unknown}")
        keep_controls = bool(_UC.get('SHOW_CONTROLS', False))
        keep_expl = bool(_UC.get('SHOW_EXPLORATORY', False))
        return [factory() for name, factory in ALL_STRATEGIES.items()
                if name not in excluded
                and (keep_controls or name not in _CONTROL_KEYS)
                and (keep_expl or name not in _EXPLORATORY_KEYS)]

    return STRATEGIES_TO_RUN

def size_positions(alloc, prices, signal_date, accounts, config, strat, s_w, live_prices=None):
    """Translate the latest target weights into whole-share orders across accounts.

    Pure computation — no printing — so both the CLI report and the GUI can consume
    the same result. Mutates `accounts` balances as fills are allocated.

    `live_prices` (optional pd.Series ticker→price): current market prices used for
    share-quantity math. The SIGNAL (which assets, target weights) is always monthly;
    only the sizing price is live, so orders match what the broker actually charges.
    Assets missing from `live_prices` fall back to the month-end close with a warning.

    With PRICE_CAP_MARGIN_PCT > 0, shares are sized AT the cap = quote × (1 + margin)
    (worst-case fill), so a GTC limit order at the cap can never be rejected for funds.

    Returns:
        {'orders': [ {asset, name, mode, rank, account, target_wt, actual_wt,
                      shares, value (= max cost at cap), price (quote),
                      cap (limit price or None), score, fill}, ... ],
         'warnings': [str, ...],
         'n_positions': int,
         'total_off_wt': float, 'total_def_wt': float}
    """
    target = alloc.iloc[-1]
    positions = target[target > 0]
    result = {'orders': [], 'warnings': [], 'n_positions': len(positions),
              'total_off_wt': 0.0, 'total_def_wt': 0.0}

    if len(positions) == 0:
        return result

    # Rank positions by momentum
    current_scores = s_w.iloc[-1]
    pos_scores = current_scores.reindex(positions.index).fillna(-np.inf)
    ranked_positions = pos_scores.sort_values(ascending=False).index.tolist()

    total_capital = sum(a['account_balance'] for a in accounts)

    rank = 1

    for asset in ranked_positions:
        target_wt = positions[asset]
        target_val = target_wt * total_capital

        # Verify price is available (live quote preferred, month-end close as fallback)
        if asset in prices.columns:
            price = prices.loc[signal_date, asset]
            if live_prices is not None:
                lp = live_prices.get(asset)
                if lp is not None and pd.notna(lp) and lp > 0:
                    price = float(lp)
                else:
                    result['warnings'].append(f"No live quote for {asset} — sized at month-end close ({signal_date.date()}).")
            if pd.isna(price) or price <= 0:
                result['warnings'].append(f"No valid price for {asset} on {signal_date.date()}, skipping sizing.")
                continue
        else:
            result['warnings'].append(f"{asset} not in price history, skipping sizing.")
            continue

        # Price cap for after-hours GTC limit orders: budget & size shares AT the cap
        # (worst-case fill), so the broker's funds check can never reject the order.
        # A fill below the cap (normal midprice case) just leaves the price improvement
        # as cash. cap None = feature off → size at the quote itself.
        cap_margin = config.get('PRICE_CAP_MARGIN_PCT', 0.0) / 100.0
        cap = round(price * (1.0 + cap_margin), 2) if cap_margin > 0 else None
        exec_price = cap if cap is not None else price

        remaining_target_val = target_val
        accumulated_val = 0.0
        
        for acc in accounts:
            if remaining_target_val <= 0.01:
                break
                
            usable_balance = acc['account_balance'] - (acc['initial_balance'] * (config['SAFETY_MARGIN_PCT'] / 100.0))
            if usable_balance <= 0.01:
                continue
                
            ideal_alloc_val = min(remaining_target_val, usable_balance)
            
            # Prevent execution of tiny cash slices
            if (ideal_alloc_val / total_capital) < (config['MINIMUM_TRADE_PCT'] / 100.0):
                continue
                
            # Assume we stop filling here, what's the hypothetical total weight?
            hypothetical_total_val = accumulated_val + ideal_alloc_val
            hypothetical_wt = hypothetical_total_val / total_capital
            
            stop_filling = False
            flexi_label = "Yes"
            
            if ideal_alloc_val >= remaining_target_val - 0.01:
                stop_filling = True
                flexi_label = "Yes"
            else:
                if hypothetical_wt >= (target_wt - (config['FLEXIBILITY_BAND_PCT'] / 100.0)):
                    stop_filling = True
                    flexi_label = "Yes"
                else:
                    stop_filling = False
                    flexi_label = "No"
                    
            shares = ideal_alloc_val / exec_price
            if not config['FRACTIONAL_SHARES']:
                lot = max(1, int(config['SHARE_LOT_SIZE']))
                shares = (shares // lot) * lot

                # Optional "flush" round-up (live sizing only): when this fill finishes the
                # position, bump the last lot UP to deploy idle cash — but only if the extra
                # lot (a) still fits this account's usable balance (never touches the safety
                # reserve) and (b) overshoots the target weight by at most FLUSH_ROUND_UP_BAND_PCT.
                flush_band = config.get('FLUSH_ROUND_UP_BAND_PCT', 0.0) / 100.0
                if flush_band > 0 and stop_filling:
                    leftover = ideal_alloc_val - shares * exec_price   # cash the floor would strand
                    shares_up = shares + lot
                    val_up = shares_up * exec_price
                    pos_wt_up = (accumulated_val + val_up) / total_capital
                    if leftover > 0.01 and val_up <= usable_balance and (pos_wt_up - target_wt) <= flush_band:
                        shares = shares_up
                        flexi_label = "Flush+"

            actual_alloc_val = shares * exec_price
            if shares > 0:
                actual_alloc_wt = actual_alloc_val / total_capital
                asset_score = pos_scores.get(asset, 0.0)
                asset_name = TICKER_NAMES.get(asset, asset)

                # Sleeve resolution goes through the mandatory declaration, NOT the raw
                # `offensive`/`defensive` attributes. For a leveraged wrap those attributes
                # hold the 1x SIGNAL tickers while the orders hold the LETF images
                # (SSO/QLD/UWM), so until 2026-07-30 every one of the fourteen leveraged
                # entries sized with mode "N/A" and an offensive/defensive split of 0/0 —
                # the attribute-sniffing failure mode `sleeves()` was created to remove
                # (2026-07-28), surviving on the live path. No hasattr guard: `sleeves()`
                # is mandatory and raises by design; catching it here would restore the
                # silent-empty-set behaviour the declaration exists to prevent.
                defensive, offensive, canary = strat._sleeves()
                in_off = asset in offensive
                in_def = asset in defensive
                mode_label = "N/A"
                if in_off and not in_def:
                    mode_label = "Offensive"
                elif in_def and not in_off:
                    mode_label = "Defensive"
                elif in_off and in_def:
                    # Dual-role: resolved the same way `BaseStrategy.defensive_mask` does —
                    # risk-off when any canary is non-positive OR unreadable — so the live
                    # label and the backtest mask cannot disagree about the same month.
                    avail = [c for c in canary if c in s_w.columns]
                    if avail:
                        c_scores = s_w.iloc[-1][avail]
                        risk_off = bool((c_scores <= 0).any() or c_scores.isna().any())
                        mode_label = "Defensive" if risk_off else "Offensive"
                    else:
                        mode_label = "Offensive"    # no canary: dual-role counts offensive

                if mode_label == "Offensive":
                    result['total_off_wt'] += actual_alloc_wt
                elif mode_label == "Defensive":
                    result['total_def_wt'] += actual_alloc_wt

                result['orders'].append({
                    'asset': asset, 'name': asset_name, 'mode': mode_label, 'rank': rank,
                    'account': acc['account_name'], 'target_wt': float(target_wt),
                    'actual_wt': float(actual_alloc_wt), 'shares': float(shares),
                    'value': float(actual_alloc_val), 'price': float(price),
                    'cap': float(cap) if cap is not None else None,
                    'score': float(asset_score), 'fill': flexi_label,
                })

                acc['account_balance'] -= actual_alloc_val
                accumulated_val += actual_alloc_val
                remaining_target_val -= actual_alloc_val

            if stop_filling:
                break

        # An unfillable position must be SAID, not inferred from a smaller number three
        # columns over. Every skip path above (`MINIMUM_TRADE_PCT`, an exhausted usable
        # balance) can leave a position short of target with no entry anywhere — a
        # live-money gap: the user reads the orders as the strategy, and silence here
        # means "fully deployed" when it is not. Tolerance: the FLEXIBILITY_BAND the fill
        # logic itself uses, or one whole lot — flooring strands at most one lot of cash,
        # which is the ordinary cost of whole shares, not an unfilled position.
        shortfall = target_val - accumulated_val
        band_val = (config['FLEXIBILITY_BAND_PCT'] / 100.0) * total_capital
        lot_val = 0.0 if config['FRACTIONAL_SHARES'] else \
            max(1, int(config['SHARE_LOT_SIZE'])) * exec_price
        if shortfall > max(band_val, lot_val, 0.01) and total_capital > 0:
            result['warnings'].append(
                f"{asset}: sized {accumulated_val / total_capital:.1%} of a "
                f"{target_wt:.1%} target — ${shortfall:,.0f} short (safety reserve, "
                f"minimum-trade floor, or whole-share rounding). The book is NOT fully "
                f"deployed at these orders.")
        rank += 1

    return result

def load_store(config):
    """Open the daily price store (downloading / refreshing as needed)."""
    cache_dir = config.get('CACHE_DIR') or os.path.join(ROOT_DIR, 'data', 'cache')
    # The absent-key default is True, matching the dashboard: a config dict that never
    # thought about gaps gets the guard, not the exemption. "Unknown must mean check" is
    # the same failure direction the refresh stamp already follows.
    return PriceStore(TICKERS, start=config['DATA_START_DATE'], cache_dir=cache_dir,
                      download=config.get('ALLOW_DOWNLOAD', True),
                      strict_gaps=config.get('STRICT_GAPS', True),
                      refresh_hours=config.get('CACHE_REFRESH_HOURS', REFRESH_MIN_HOURS))


def build_signal_panel(store, config):
    """Monthly signal panel and momentum scores, over the FULL available history.

    The panel deliberately starts at DATA_START_DATE, not at START_DATE. The old code
    sliced the prices at START_DATE and then dropped a further 13 rows for the momentum
    warm-up, so a report labelled `2015-01-01 -> 2025-01-01` actually measured
    2016-02 -> 2024-12 (audit finding M1). Signals are now computed over everything
    available and only the RESULT is sliced, which is the one ordering that cannot shift
    the window behind the reader's back.

    Index entries are REAL trading dates (`2023-12-29`), never synthetic month-end labels,
    and the trailing month is absent until it has actually finished.
    """
    end = (config['CURRENT_EXECUTION_DATE'] if config['EXECUTION_MODE']
           else config.get('END_DATE'))
    # END_DATE is None in normal use: the panel runs to the last COMPLETE month the store
    # holds, which is a fact about the data rather than a date anyone chose.
    #
    # `closed_by` is passed in LIVE MODE ONLY. A backtest has no standpoint in time and must
    # keep the strict rule — a month counts as finished only once a later observation proves
    # it, which is what makes a backtest independent of the day it was run. Live execution
    # does have a standpoint, and withholding a month the calendar has already closed would
    # size this month's orders from last month's decision. See
    # `PriceStore.month_end_dates` for the two conditions.
    month_ends = store.month_end_dates(start=config['DATA_START_DATE'],
                                       end=pd.to_datetime(end) if end else None,
                                       closed_by=end if config['EXECUTION_MODE'] else None)
    prices = store.monthly_adj_close(month_ends)
    return prices, calc_13612w(prices), calc_13612u(prices)


def load_data(config):
    """Fetch historical price data and pre-calculate momentum scores.

    Returns (prices, scores_13612w, scores_13612u, store). The store is part of the return
    value because execution is no longer implied by the monthly panel: pricing a fill at the
    session after the decision needs daily bars.
    """
    store = load_store(config)
    prices, scores_w, scores_u = build_signal_panel(store, config)
    return prices, scores_w, scores_u, store

def compute_live_signals(prices, scores_w, scores_u, strategies, config, broker_accounts,
                         store=None):
    """Compute live target orders for every active strategy — no printing.

    This is the presentation-agnostic core consumed by both the CLI report
    (run_live_mode) and the GUI's Live tab. Returns (signal_date, results) where
    results is a list of per-strategy dicts:
        {'name', 'error' (str or None), 'sizing' (see size_positions),
         'canary': [{'asset','state','score'}], 'accounts': post-fill account dicts,
         'pricing': {'mode': 'live'|'month-end', 'asof': str}}
    Returns (None, []) when there is not enough price history before the execution date.

    Sizing prices: the monthly signal decides WHAT to hold; share quantities are then
    computed at the latest market quote (get_live_prices) so orders match what the
    broker actually charges. If the live fetch fails (offline), sizing falls back to
    the month-end close — flagged in 'pricing' and in the warnings.
    """
    exec_dt = pd.to_datetime(config['CURRENT_EXECUTION_DATE'])
    available = prices.index[prices.index < exec_dt]
    if len(available) == 0:
        return None, []
    signal_date = available[-1]

    # --- is this signal the CURRENT one? ------------------------------------------------ #
    # `_complete_month_ends` withholds a month until a later-month observation proves it
    # finished, which is the right rule for a backtest and the wrong proxy on the one day
    # that matters. Run this on the 1st of a month before the new month's first bar exists
    # and the previous month-end is withheld too, so the report sizes orders from a signal
    # a full month old. `build_signal_panel` now passes `closed_by` so the calendar can
    # close the month; this says so out loud when it still could not.
    stale_months = (pd.Period(exec_dt, freq='M') - pd.Period(signal_date, freq='M')).n
    stale_note = None
    if stale_months >= 2:
        stale_note = (f"SIGNAL IS {stale_months} MONTHS OLD: the latest complete month-end in "
                      f"the store is {signal_date.date()}, but you are executing on "
                      f"{exec_dt.date()}. Refresh the price cache before trading — these "
                      f"orders are not this month's rotation.")

    # Live-quote memo shared across strategies (one batch fetch per new ticker set)
    live_state = {'cache': {}, 'asof': None, 'failed': False}

    def _live_quotes(assets):
        missing = [a for a in assets if a not in live_state['cache']]
        if missing and not live_state['failed']:
            try:
                fetched, asof = get_live_prices(missing)
                live_state['cache'].update(fetched.to_dict())
                live_state['asof'] = asof
            except Exception:
                live_state['failed'] = True
        quotes = pd.Series({a: live_state['cache'][a] for a in assets if a in live_state['cache']})
        return None if quotes.dropna().empty else quotes

    display_all = len(config['STRATEGIES_TO_DISPLAY']) == 0
    results = []

    for strat in strategies:
        if not strat.is_active:
            continue
        if not display_all and strat.name not in config['STRATEGIES_TO_DISPLAY']:
            continue

        s_w = scores_u.loc[:signal_date] if strat.score_type == 'unweighted' else scores_w.loc[:signal_date]

        # --- the backtest's guards, on the path that actually spends money -------------- #
        # These ran only in `run_backtest` until 2026-07-29. Being latently satisfied is not
        # the same as being enforced: `assert_unlevered_defensive`'s own docstring says "the
        # engine fails any strategy that breaks it", and the engine did not — here. A
        # strategy added through user_config.json's STRATEGIES list reached `size_positions`
        # unchecked.
        try:
            assert_unlevered_defensive(strat)
        except ValueError as e:
            results.append({'name': strat.name, 'error': str(e), 'stale': stale_note})
            continue

        try:
            alloc = strat.generate_allocations(prices.loc[:signal_date], s_w, None, None)
        except Exception as e:
            results.append({'name': strat.name, 'error': str(e), 'stale': stale_note})
            continue

        problems = validate_row(alloc.iloc[-1], prices.loc[signal_date],
                                label=strat.name, when=signal_date)
        # Nothing invented may be sized into an order. Every constructed span ends before
        # its ticker's real inception by construction, so this is cheap and makes the claim
        # structural rather than incidental — today it holds by ~18 years of margin.
        if store is not None:
            row = alloc.iloc[-1]
            for t in row[row > 1e-9].index:
                real_from = store.constructed_before(t)
                if real_from is not None and signal_date < real_from:
                    problems.append(f'  {t} is CONSTRUCTED before {real_from.date()}; the '
                                    f'signal date {signal_date.date()} predates the fund')
        # A price that disagrees with its own adjustment history must not be sized into an
        # order. Found 2026-09-01: a spliced vintage understated momentum and flipped a live
        # canary, so the book went to cash a month early. The blast radius is deliberately
        # THIS strategy's decision, not the whole registry — and it covers the sleeves and
        # the canary, not merely what is held: a corrupted score on an unheld candidate is
        # what changes the selection, and a dead canary sends the book to cash without ever
        # appearing in the allocation row.
        if store is not None:
            v = getattr(store, 'verification', None)
            if v is None or v.get('status') in ('not_applicable', 'skipped'):
                problems.append('  the price store was never verified against its own '
                                'adjustment history; live orders must not be sized from '
                                'data nothing has checked')
            elif v.get('status') == 'disagrees':
                sl = strat.sleeves()
                universe = set(alloc.iloc[-1][alloc.iloc[-1] > 1e-9].index)
                universe |= set(sl.get('offensive', ())) | set(sl.get('defensive', ()))
                universe |= set(sl.get('canary', ()))
                hit = sorted({x['ticker'] for x in v['violations']} & universe)
                if hit:
                    problems.append(
                        f'  {", ".join(hit)} carr{"ies" if len(hit) == 1 else "y"} more '
                        f'than one dividend-adjustment vintage, so momentum over windows '
                        f'crossing the seam is wrong (and wrong LOW). Re-run with '
                        f'--refresh, or delete data/cache to rebuild.')

        if problems:
            results.append({'name': strat.name, 'stale': stale_note,
                            'error': 'refusing to size this allocation:\n'
                                     + '\n'.join(problems)})
            continue

        # Fresh account copies per strategy, filled in priority order
        accounts = [dict(a, initial_balance=a['account_balance']) for a in broker_accounts]
        accounts.sort(key=lambda x: x['account_priority'])

        # Live quotes for the assets this strategy actually holds (signal stays monthly)
        target = alloc.iloc[-1]
        held = target[target > 0].index.tolist()
        quotes = _live_quotes(held) if held else None

        sizing = size_positions(alloc, prices, signal_date, accounts, config, strat, s_w,
                                live_prices=quotes)
        # DECISION (2026-07-30, EXEC-001): live sizing deliberately stays at 1x, and says
        # so on every levered run. `LEVERAGE_FACTOR` is a BACKTEST setting: deploying it
        # live requires knowing whether `account_balance` means equity or buying power at
        # the user's broker, and margin exists only in the taxable account — neither of
        # which this code can infer. Silent was the one wrong option: a user who calibrates
        # with common/margin_sizing.py and runs --live would receive orders that do not
        # implement the sizing they were just given. Both consumers (CLI report and GUI
        # Live tab) read sizing['warnings'], so one insertion covers both paths.
        if config.get('LEVERAGE_FACTOR', 1.0) != 1.0:
            sizing['warnings'].insert(0,
                f"LEVERAGE_FACTOR={config['LEVERAGE_FACTOR']} is a BACKTEST-ONLY setting: "
                f"these orders are sized at 1x of your account balances, borrowing nothing. "
                f"The backtest figures for this strategy assume up to "
                f"{config['LEVERAGE_FACTOR']}x. To deploy margin, size the borrowed slice "
                f"yourself against common/margin_sizing.py's recommendation.")
        pricing = ({'mode': 'live', 'asof': str(live_state['asof'].date())}
                   if quotes is not None else
                   {'mode': 'month-end', 'asof': str(signal_date.date())})
        if quotes is None and held:
            sizing['warnings'].insert(0, "Live quote fetch failed — ALL positions sized at "
                                         f"month-end close ({signal_date.date()}). Prices may be stale.")

        canary = []
        if hasattr(strat, 'canary') and strat.canary:
            c_scores = s_w.iloc[-1][strat.canary]
            for c_asset, c_score in c_scores.items():
                canary.append({'asset': c_asset,
                               'state': "ALIVE" if c_score > 0 else "DEAD",
                               'score': float(c_score)})

        results.append({'name': strat.name, 'error': None, 'sizing': sizing,
                        'canary': canary, 'accounts': accounts, 'pricing': pricing,
                        'stale': stale_note})

    return signal_date, results

def run_live_mode(prices, scores_w, scores_u, strategies, config, store=None):
    """CLI wrapper around compute_live_signals — prints the classic live report."""
    signal_date, results = compute_live_signals(prices, scores_w, scores_u, strategies,
                                                config, BROKER_ACCOUNTS, store=store)
    if signal_date is None:
        print("Not enough history.")
        return
    print(f"\n=== LIVE SIGNALS as of {signal_date.date()} ===")
    stale = next((r['stale'] for r in results if r.get('stale')), None)
    if stale:
        print(f"\n  *** {stale}\n")

    for res in results:
        print(f"\n--- {res['name']} ---")
        if res['error']:
            print(f"Error computing signal: {res['error']}")
            continue

        sizing = res['sizing']
        print(f"Required positions: {sizing['n_positions']}")
        p = res['pricing']
        print(f"Sizing prices: {'LIVE quotes as of ' + p['asof'] if p['mode'] == 'live' else 'month-end close ' + p['asof']}"
              f" (signal: month-end {signal_date.date()})")
        for w in sizing['warnings']:
            print(f"  Warning: {w}")

        if sizing['orders']:
            print(f"{'Position':<9} | {'Name':<15} | {'Type':<9} | {'Rank':<4} | {'Account':<8} | {'Target %':>8} | {'Actual %':>8} | {'Quantity':>10} | {'Cap Price':>10} | {'Max Cost':>13} | {'Momentum Score':>14} | {'Flexible':^10}")
            print("-" * 152)
            for o in sizing['orders']:
                cap_s = f"${o['cap']:>9.2f}" if o['cap'] is not None else f"{'-':>10}"
                print(f"{o['asset']:<9} | {o['name']:<15} | {o['mode']:<9} | {o['rank']:<4} | {o['account']:<8} | {o['target_wt']:>8.2%} | {o['actual_wt']:>8.2%} | {o['shares']:>10.2f} | {cap_s} | ${o['value']:>12.2f} | {o['score']:>14.2f} | {o['fill']:^10}")
            print("-" * 152)

        if res['canary']:
            print("Canary Status : " + " | ".join(
                f"{c['asset']}: {c['state']} ({c['score']:.2f})" for c in res['canary']))

        if sizing['total_off_wt'] > 0 or sizing['total_def_wt'] > 0:
            print(f"Portfolio Mode: Offensive {sizing['total_off_wt']:>5.2%} / Defensive {sizing['total_def_wt']:>5.2%}")
        elif sizing['orders']:
            # Orders with no resolvable sleeve should be impossible now that resolution
            # goes through _sleeves(); if it happens, the absence must not be silent —
            # a missing line reads as "nothing to report", not "the resolver failed".
            print("Portfolio Mode: UNRESOLVED — orders exist but no sleeve could be "
                  "assigned. Check the strategy's sleeves() declaration.")

        print("Remaining Cash per Account:")
        for acc in res['accounts']:
            print(f"  {acc['account_name']:<8}: ${acc['account_balance']:>9.2f} (Usable cutoff: ${acc['initial_balance'] * (config['SAFETY_MARGIN_PCT'] / 100.0):.2f})")

#: Composition of every STATIC entry, printed in the report's legend and the dashboard.
#: `Golden_Butterfly` is the reason this exists: at a glance it reads as a 60/40, and it is
#: five equal sleeves including gold and small-cap value. Now that a real `Sixty_Forty_1X` sits
#: beside it in the same table, telling them apart from the names alone is impossible.
_STATIC_COMPOSITIONS = {
    'Golden_Butterfly': '20% each VTI / IJS / TLT / SHY / GLD (monthly)',
    'Sixty_Forty_1X': 'SPY 60% / IEF 40% (monthly)',
    'SPY_Benchmark': 'SPY 100% (buy & hold)',
    'SPY_2X_Benchmark': 'SSO 100% — 2x S&P 500 (buy & hold)',
    'SPY_3X_Benchmark': 'UPRO 100% — 3x S&P 500 (buy & hold)',
    'RiskParity_3X': 'UPRO 55% / TMF 45% — 3x equity + 3x treasuries, QUARTERLY',
}


def may_set_ranked_window(strat, policy='strategies'):
    """May `strat`'s inception shorten the shared ranked window for everybody else?

    Two independent exclusions under the default `'strategies'` policy, and they catch different
    things — which is why both are needed:

    1. **`fidelity == 'custom'`** — an entry nobody published. Four custom leveraged G4 wraps
       sharing UGL's 2008-12 inception were dragging the headline table from 2008-06 to 2010-02
       for all 25 entries, so the flagship drawdown column excluded the crisis this project
       exists to measure (REPORT-002).
    2. **holds a leveraged product** — structural, read off the tickers. Every LETF launched
       late (UWM 2007-01 is the earliest 2x, TNA 2008-11 the earliest 3x), so such an entry
       always shortens the table and always by years.

    The second is not redundant. The first caught the 2x wraps only because they happen to be
    labelled `custom`; a levered BENCHMARK has as good a claim to `faithful` as `SPY_Benchmark`
    does ("no rule to be faithful or unfaithful to"), and labelling it so would hand 100% UPRO
    the power to cost every other row its 2008 — the same defect, re-entering through the label
    instead of the role.

    Nothing is hidden and nothing is dropped: an excluded entry is still measured over its own
    history, in a separated block. What changes is only whose inception may shorten other rows.
    `policy='all'` restores the old behaviour for anyone who wants one window over everything.
    """
    if policy == 'all':
        return True
    if getattr(strat, 'fidelity', 'custom') == 'custom':
        return False
    return not holds_leveraged_product(strat)


def _shared_binding_ticker(prepared, binding):
    """(ticker, inception) when every binding strategy is held back by the SAME product.

    Naming the strategy is only half an answer, and the less useful half: what a reader can
    act on is that four entries are waiting on UGL. Returns None when the tied strategies
    are bound by different tickers, because then there is no single fact to state.
    """
    if not binding:
        return None
    covs = [p['cov'] for p in prepared if p['strat'].name in set(binding)]
    tickers = {c.get('binding_ticker') for c in covs}
    if len(tickers) != 1 or None in tickers:
        return None
    return tickers.pop(), pd.Timestamp(covs[0]['binding_inception'])


def run_backtest(prices, scores_w, scores_u, strategies, config, store=None):
    """Execute backtest mode — calculate historical returns and metrics.

    Every performance number produced here passes through `run_ledger`, which refuses to
    price a rebalance whose weights do not sum to 1 or which holds a ticker with no price on
    its execution date. A strategy that violates either invariant is reported as a FAILURE
    and contributes no metrics: a month the engine cannot price is a month with no return,
    not a month with a 0% return.
    """
    if store is None:
        raise ValueError(
            'run_backtest now requires the PriceStore: execution is priced at the session '
            'AFTER the decision, which the monthly panel alone cannot express. Call '
            'load_data(config) and pass its fourth return value.')
    if config['LEVERAGE_FACTOR'] != 1.0:
        print(f"\n[NOTE] LEVERAGE_FACTOR={config['LEVERAGE_FACTOR']} (margin), borrow rate={config['MARGIN_BORROW_RATE']:.1%}/yr.")
        print("Margin leverage applies to ACTIVE strategies only (passive benchmarks stay at 1x).")
        if config['MARGIN_FOLLOWS_SIGNAL']:
            print("MARGIN_FOLLOWS_SIGNAL=True: borrowing is scaled by the OFFENSIVE weight, so the")
            print("loan is repaid as the signal goes risk-off (defence held at 1x, like the LETF design).")
            print(f"Effective leverage therefore varies between 1.00x and {config['LEVERAGE_FACTOR']:.2f}x — see the report.")
        else:
            print("MARGIN_FOLLOWS_SIGNAL=False: flat leverage — the loan stays drawn in risk-off")
            print("months, so defensive positions are held with borrowed money during drawdowns.")
        print("Interest accrues on the debit balance actually drawn, day-counted. For leveraged ETF products, use the G2/G3/G4 leveraged variants.")
        # The banner used to describe the financing cost in detail and stay silent on the
        # thing that actually ends a levered account. Interest is the small print; the
        # margin call is the failure mode, and the ledger does not model it at all.
        print("[WARNING] The ledger NEVER compares equity to a maintenance requirement — not")
        print("intra-month, not at month end, nowhere. Every levered figure below is an")
        print("UN-LIQUIDATED UPPER BOUND: it is what you would have earned had the broker")
        print("never called. Measured on daily paths, drawdowns run up to 60% deeper than the")
        print("month-end series these returns are built from. For the leverage a model can")
        print("actually carry, read the SUSTAINABLE MARGIN LEVERAGE section below, which sizes")
        print("every row against a maintenance requirement. See KNOWN_GAPS.md §3.")
        print("Live mode does NOT deploy this leverage: --live sizes orders at 1x of account")
        print("balances and says so in its own warnings. Backtest and live are described here")
        print("together so the difference cannot be discovered by surprise.\n")

    month_ends = prices.index
    start_floor = pd.to_datetime(config.get('START_DATE') or eras.COMMON_ERA_START)
    end_req = (pd.to_datetime(config['CURRENT_EXECUTION_DATE']) if config['EXECUTION_MODE']
               else pd.to_datetime(config['END_DATE']) if config.get('END_DATE')
               else month_ends[-1])
    convention = config.get('EXECUTION_CONVENTION', 'next_open')
    cash_ticker = config.get('CASH_TICKER', 'BIL')
    # Named for what it governs: a bare `policy` was later rebound to RANKED_WINDOW_POLICY
    # in the same function — correct today only because the first use completes first.
    coverage_policy = config.get('COVERAGE_POLICY', 'trim')
    warmup = config.get('LOOKBACK_MONTHS', 13)
    cost_per_side = config.get('COST_PCT_PER_SIDE', config.get('TRANSACTION_COST_PCT', 0.001))

    rf_series, rf_desc = build_rf_series(store, month_ends, cash_ticker,
                                         config.get('RF_ANNUAL_FALLBACK', 0.03))

    results, metrics_data, failures = {}, [], []

    print("\nCalculating backtests...")
    print(f"  execution: {convention} | cost: {cost_per_side:.4%}/side (one-way, per leg) | "
          f"cash: {cash_ticker or 'none (0%)'} | rf: {rf_desc}")

    # --- phase 1: signals and coverage, before a single fill is priced ------------------ #
    # The ranked table has to compare strategies over ONE window or its rows do not mean the
    # same thing — and that window must not be chosen by anybody. It is the latest date at
    # which EVERY strategy in this comparison can honestly be measured, floored at the era
    # start. Coverage answers that without pricing anything, so it runs first and the ledger
    # then runs once, already on the right window.
    prepared = []
    for strat in strategies:
        s_w = scores_u if strat.score_type == 'unweighted' else scores_w
        try:
            alloc = strat.generate_allocations(prices, s_w, None, None)
        except Exception as e:
            failures.append((strat.name, f'signal error: {e}'))
            print(f"  > FAILED {strat.name}: {e}")
            continue

        # --- defence is held at 1x, always ---------------------------------------------- #
        # A defensive sleeve containing a leveraged product raises the risk it exists to
        # cut. DAA1_G12 held UST (2x treasuries) as a defensive candidate and was deleted;
        # this refuses to price anything like it rather than reporting it as risk-off.
        try:
            assert_unlevered_defensive(strat)
        except ValueError as e:
            failures.append((strat.name, str(e)))
            print(f"  > FAILED {strat.name}: {e}")
            continue

        # --- coverage: never measure a strategy over years its assets did not exist ----- #
        try:
            cov = coverage_report(strat, store, start_floor, warmup)
        except KeyError as e:
            failures.append((strat.name, str(e)))
            print(f"  > FAILED {strat.name}: {e}")
            continue
        # The ERA FLOOR bounds what is REPORTED, not merely what is ranked. `coverage_report`
        # answers a question about the DATA — "from when can this be measured" — and correctly
        # ignores the request, so its `earliest` can predate the era: SPY is measurable from
        # 2001-03 and the era opens 2004-11. Left unclamped, that entry's full-history row and
        # its regime panel would reach 44 months outside the era the same report declares, and
        # its headline CAGR would be earned partly in months no other row can see.
        #
        # `eras.COMMON_ERA_START` and NOT `start_floor`, and the distinction is the whole
        # point: the floor is DERIVED and frozen, while `START_DATE` is whatever a caller
        # asked for. Clamping to the request would make the regime panel inherit the caller's
        # dates — a run starting in 2015 would print n/a for the GFC even for SPY — which is
        # precisely the date-selection bias the panel exists to remove.
        # `tests/test_guards.py::TestRegimePanelIgnoresTheRequestedWindow` fails on the
        # difference, and did, when this landed clamped to the wrong one.
        #
        # Clamped here rather than in `coverage_report` because `trimmed` and its message are
        # statements about the data and must keep their meaning — computed above, unaffected.
        if cov['earliest'] is not None:
            cov['earliest'] = max(pd.Timestamp(cov['earliest']),
                                  pd.to_datetime(eras.COMMON_ERA_START))
        if cov['earliest'] is None or (cov['trimmed'] and coverage_policy == 'strict'):
            reason = cov['message'] or 'insufficient coverage'
            failures.append((strat.name, reason))
            print(f"  > FAILED {strat.name}: {reason}")
            continue
        prepared.append({'strat': strat, 'alloc': alloc, 'scores': s_w, 'cov': cov})

    if not prepared:
        print("\n  Nothing could be measured.")
        return [], {}

    # The first RETURN lands one period after the first decision, so the decision that
    # produces the return in the era's opening month is the month-end before it.
    prior = month_ends[month_ends < start_floor]
    floor_decision = prior[-1] if len(prior) else month_ends[0]

    # --- WHOSE inception is allowed to set the shared window? --------------------------- #
    # 'strategies' (default): only entries somebody published. Four CUSTOM leveraged G4
    # wraps share UGL's 2008-12 inception and were dragging the headline table from 2008-06
    # to 2010-02 for all 25 entries — so the flagship drawdown column excluded the crisis
    # this project exists to measure, immediately after a commit whose stated purpose was to
    # put 2008 into the sample. That is the same coverage cost used to justify DELETING
    # DAA1_G12; four entries were doing it and were kept.
    #
    # Nothing is hidden and nothing is dropped: an entry that cannot cover the shared window
    # is still measured, over its own history, in a clearly separated block underneath. What
    # changes is only which entries are allowed to shorten everyone else's row.
    # 'all' restores the old behaviour for anyone who wants one window over everything.
    # A SECOND, structural exclusion on top of the fidelity one: nothing whose returns depend
    # on a LEVERAGED PRODUCT may set the window, whatever its fidelity label. Every LETF
    # launched late — the earliest 2x is UWM (2007-01), the earliest 3x is TNA (2008-11) — so
    # such an entry always shortens the table, and always by years.
    #
    # The fidelity filter alone does not cover this, and the gap is a trap rather than a
    # hypothetical. It caught the 2x wraps only because they happen to be labelled `custom`. A
    # levered *benchmark* — 100% UPRO, or Hedgefundie's UPRO/TMF — has as much claim to
    # `faithful` as `SPY_Benchmark` does ("no rule to be faithful or unfaithful to"), and
    # labelling it so would have handed it the power to drag every row from 2008-06 to 2010:
    # precisely the REPORT-002 defect this policy was written to prevent, re-entering through
    # the label rather than the role.
    #
    # Structural, so it cannot be defeated by a labelling choice: it reads the tickers.
    window_policy = config.get('RANKED_WINDOW_POLICY', 'strategies')
    if window_policy not in ('strategies', 'all'):
        raise ValueError(f"RANKED_WINDOW_POLICY must be 'strategies' or 'all', got {window_policy!r}")
    setters = [p for p in prepared if may_set_ranked_window(p['strat'], window_policy)]
    if not setters:                      # a run of nothing but custom wraps still needs one
        setters = prepared
    common_decision, binding = eras.common_window(
        {p['strat'].name: p['cov']['earliest'] for p in setters}, floor=floor_decision)
    decisions = month_ends[(month_ends >= common_decision) & (month_ends <= end_req)]
    if len(decisions) < 2:
        print("  > FAILED: fewer than two decision dates are common to these strategies")
        return [], {}

    # `binding` is the full TIED set, not one arbitrary name. The advice below is only
    # actionable if it names every strategy holding the window back: four leveraged G4
    # variants share UGL's 2008-12 inception, and dropping any one of them moves nothing.
    binding_ticker = _shared_binding_ticker(setters, binding)
    why = (f'set by {", ".join(binding)}, whose assets start later' if binding
           else f'the era floor ({start_floor:%Y-%m})')
    print(f"  common window: {decisions[1]:%Y-%m}..{decisions[-1]:%Y-%m} "
          f"({len(decisions) - 1} months) — {why}")
    if binding:
        if binding_ticker:
            print(f"  All of them are bound by the same ticker: {binding_ticker[0]} "
                  f"({binding_ticker[1]:%Y-%m-%d}).")
        print("  Every row of the ranked table is measured over exactly this window. Each "
              "strategy's FULL history is")
        drop = ('all of ' if len(binding) > 1 else '') + ' and '.join(binding)
        print(f"  reported separately, regime by regime, below it. Dropping {drop} from "
              "the comparison lengthens the")
        print("  window for everything else — which is a fair thing to do, and a dishonest "
              "thing to do twice until the")
        print("  table looks good.")

    # --- phase 2: price every strategy over the common window, then over its own history - #
    late = [p['strat'].name for p in prepared
            if pd.Timestamp(p['cov']['earliest']) > common_decision]
    if late:
        print(f"  {len(late)} entr(ies) cannot cover that window and are measured over their "
              f"OWN history, reported separately:")
        print(f"    {', '.join(sorted(late))}")

    for p in prepared:
        strat, alloc, s_w, cov = p['strat'], p['alloc'], p['scores'], p['cov']
        if cov['trimmed']:
            print(f"  ! {cov['message']}")

        # An entry that starts after the shared window gets its own decision calendar. Its
        # row is then NOT comparable with the ranked block, which is exactly why the report
        # keeps the two apart instead of printing them in one sorted list.
        in_ranked = pd.Timestamp(cov['earliest']) <= common_decision
        own_decisions = decisions if in_ranked else month_ends[
            (month_ends >= cov['earliest']) & (month_ends <= end_req)]
        if len(own_decisions) < 2:
            failures.append((strat.name, 'fewer than two decision dates in its own history'))
            print(f"  > FAILED {strat.name}: fewer than two decision dates")
            continue

        targets = alloc.reindex(own_decisions).fillna(0.0)

        # Margin leverage applies to active strategies only; passive benchmarks stay at 1x
        # so they remain a clean, unleveraged reference point.
        lev = config['LEVERAGE_FACTOR'] if strat.is_active else 1.0
        exec_cfg = ExecutionConfig(
            convention=convention,
            cost_bps_per_side=cost_per_side * 10_000.0,
            cash_ticker=cash_ticker,
            borrow_rate=config['MARGIN_BORROW_RATE'],
            leverage=lev,
            leverage_follows_signal=config['MARGIN_FOLLOWS_SIGNAL'],
        )
        sleeves = {'defensive_mask': strat.defensive_mask(targets, s_w)}

        try:
            led = run_ledger(targets, store, exec_cfg, sleeves, label=strat.name)
        except (WeightInvariantError, ValueError) as e:
            failures.append((strat.name, str(e).split('\n')[0]))
            print(f"  > FAILED {strat.name}: {str(e).splitlines()[0]}")
            for line in str(e).splitlines()[1:6]:
                print(f"      {line}")
            continue

        for w in led.warnings:
            print(f"  ! {strat.name}: {w}")

        # --- a SECOND pass over the strategy's FULL measurable history ------------------- #
        # The regime panel must not inherit the caller's choice of START_DATE, or it stops
        # being an antidote to date-selection bias and becomes another expression of it: a
        # run starting in 2015 would print `n/a` for the GFC even for SPY, which has data
        # back to 2000. The headline table stays on the requested window; the regime panel,
        # the coverage column and the publication split all run from the earliest date the
        # strategy can honestly be measured.
        returns_full = led.returns
        if config.get('REGIME_PANEL_FULL_HISTORY', True):
            full_dec = month_ends[(month_ends >= cov['earliest']) & (month_ends <= end_req)]
            if len(full_dec) > len(own_decisions):
                full_targets = alloc.reindex(full_dec).fillna(0.0)
                try:
                    led_full = run_ledger(
                        full_targets, store, exec_cfg,
                        {'defensive_mask': strat.defensive_mask(full_targets, s_w)},
                        label=f'{strat.name} (full history)')
                    returns_full = led_full.returns
                except (WeightInvariantError, ValueError) as e:
                    print(f"  ! {strat.name}: full-history pass failed, regime panel falls "
                          f"back to the requested window ({str(e).splitlines()[0]})")

        m = calculate_metrics(led.returns, rf=rf_series, start_label=own_decisions[0])
        eff_lev = led.effective_leverage

        # --- the KPIs sustainable-leverage sizing needs, measured on THIS window ---------- #
        # Attached here rather than recomputed by the sizing layer, and that is the whole
        # boundary: `common/leverage_advice.py` then needs no PriceStore and no strategy object,
        # only the metrics list. A sizing module that recomputes its own Sharpe can disagree
        # with the table beside it and nobody would know.
        #
        # `daily_max_dd` is the one that costs something (~40ms per entry): the drawdown of the
        # DAILY path with each month's allocation held, which is the only place in this
        # repository that looks INSIDE a month. It is allowed to fail — CAP 1 then reports
        # itself non-calculable rather than the row vanishing.
        try:
            daily_dd = intraperiod_max_drawdown(store.adj_close(), targets, cash_ticker)
        except (NotCalculable, ValueError, KeyError) as e:
            daily_dd = None
            print(f"  ! {strat.name}: no intra-period drawdown ({e}); its leverage cap will "
                  f"report non-calculable")
        # Held = what the book can TRADE. The canary is read and never held, so it carries no
        # maintenance requirement — the same exclusion `holds_leveraged_product` makes.
        _defensive, _offensive, _canary = strat._sleeves()
        held_tickers = tuple(sorted(set(_offensive) | set(_defensive)))

        metrics_data.append({
            'name': strat.name,
            'is_active': strat.is_active,
            # Declared, never inferred: 'faithful' | 'proxy' | 'custom', and
            # 'strategy' | 'control' | 'benchmark'. A report that prints a number without
            # saying whether anyone published the thing that produced it is half a report.
            'fidelity': getattr(strat, 'fidelity', 'custom'),
            'role': getattr(strat, 'role', 'strategy'),
            'cagr': m['cagr'], 'max_dd': m['max_dd'], 'sharpe': m['sharpe'],
            'sortino': m['sortino'], 'vol': m['vol'], 'upi': m['upi'],
            'ulcer_index': m['ulcer_index'], 'cum_ret': m['cum_ret'],
            # The rf PROVENANCE travels with the number it adjusted. Sharpe and Sortino are
            # net of this series, and 88 of its months are constructed from ^IRX rather than
            # realised — a report that prints the ratio without the provenance is asserting
            # something about the data that the data does not support.
            'rf_annual': m['rf_annual'], 'rf_desc': rf_desc,
            # --- inputs to the sustainable-leverage caps, none of them chosen -------------- #
            # Peak-to-TROUGH duration, because that is when the maintenance test fires; the
            # interest that capitalises over it is what `margin_sizing` charges to CAP 1.
            'max_dd_months': m['max_dd_months'],
            # The borrowed fraction under MARGIN_FOLLOWS_SIGNAL, so it scales the carry in CAP 2.
            'offensive_weight_mean': float(strat.offensive_weight(targets, s_w).mean()),
            'held_tickers': held_tickers,
            'holds_leveraged_product': holds_leveraged_product(strat),
            'daily_max_dd': daily_dd,
            'avg_lev': float(eff_lev.mean()), 'min_lev': float(eff_lev.min()),
            'max_lev': float(eff_lev.max()),
            # What was actually traded, not what was assumed (T4.4).
            'first_return': m['first_date'], 'last_return': m['last_date'],
            'n_periods': m['n_periods'],
            'annual_turnover': led.annual_turnover,
            'total_cost_bps': float(led.cost_paid.sum()) * 10_000.0,
            'n_trades_total': int(led.n_trades.sum()),
            'avg_cash_weight': float(led.cash_weight.mean()),
            'coverage_trimmed': bool(cov['trimmed']),
            'binding_ticker': cov['binding_ticker'],
            'binding_inception': cov['binding_inception'],
            # The common window, identical for every row — recorded so the report can state
            # it and the manifest can cite it, instead of the reader assuming it.
            'window_start': own_decisions[1], 'window_end': own_decisions[-1],
            'window_binding': binding,
            # False => this entry could not cover the shared window and is measured over its
            # own history. Its row is NOT comparable with the ranked block, and the report
            # prints the two separately rather than sorting them into one misleading list.
            'in_ranked_window': in_ranked,
            'returns': led.returns,
            # Full measurable history — used by the regime panel and the publication split
            # so neither depends on START_DATE. Equals `returns` when the requested window
            # already covers everything the strategy can be measured over.
            'returns_full': returns_full,
            'first_return_full': returns_full.index[0] if len(returns_full) else None,
            'last_return_full': returns_full.index[-1] if len(returns_full) else None,
            # The realised cash series this row's ratios were netted against. Carried on the
            # entry so `eras.partition_panel` can compute a segment's Sharpe/Sortino against the
            # rate that prevailed DURING that segment, without needing the store threaded down
            # to it. Same object for every row, so this costs one reference each.
            'rf_series': rf_series,
        })
        results[strat.name] = m['cum_ret']

    if failures:
        print(f"\n  {len(failures)} strategy(ies) produced NO metrics — they are absent from "
              f"the table below, not scored as zero:")
        for name, why in failures:
            print(f"    - {name}: {why}")

    return metrics_data, results

#: Re-exported from common.metrics, which is where it is defined. Three places rank on these
#: keys and a second copy of the tuple is a second chance for the orderings to disagree.
from common.metrics import RANK_KEYS  # noqa: E402  (kept here for import-order clarity)


def _fmt_window(d):
    a, b = d.get('first_return'), d.get('last_return')
    if a is None or b is None:
        return 'n/a'
    return f"{pd.Timestamp(a).strftime('%Y-%m')}..{pd.Timestamp(b).strftime('%Y-%m')}"


def _num(v, w, fmt):
    return f"{v:>{w}{fmt}}" if v is not None and not pd.isna(v) else f"{'N/A':>{w}}"


def _window_blind_spot(metrics, window_start):
    """One line naming what the ranked window cannot see, or '' when it sees everything.

    The ranked table is bounded by the LAST strategy to exist, so a handful of late entries
    can hold the headline back past a crash that most rows lived through. That is disclosed
    today only by implication — the window is printed, and the regime panels further down
    tell a different story. This states the cost directly, with the worst example measured:
    the strategy whose full-history drawdown is most understated by the ranked window.
    """
    from common.metrics import drawdown_series, wealth_curve

    window_start = pd.Timestamp(window_start)
    starts = [pd.Timestamp(d['first_return_full']) for d in metrics
              if d.get('first_return_full') is not None]
    if not starts or min(starts) >= window_start:
        return ''

    worst = None
    for d in metrics:
        full = d.get('returns_full')
        if full is None or full.empty or pd.Timestamp(full.index[0]) >= window_start:
            continue
        dd_full = float(drawdown_series(wealth_curve(full)).min())
        dd_win = d.get('max_dd')
        if dd_win is None or pd.isna(dd_win):
            continue
        gap = dd_win - dd_full                      # both negative; positive gap = understated
        if worst is None or gap > worst[0]:
            worst = (gap, d['name'], dd_full, float(dd_win))

    span = f"{min(starts):%Y-%m}..{window_start - pd.offsets.MonthEnd(1):%Y-%m}"
    if worst is None or worst[0] <= 0.0005:
        return f"{span}, which {len(starts)} measured strateg(ies) can otherwise be read through"
    _, name, dd_full, dd_win = worst
    return (f"{span} — e.g. {name}'s drawdown is {dd_win:.2%} here but {dd_full:.2%} over its"
            f" own full history")


def print_report(metrics_data, display_metrics, config, store=None):
    """The text report.

    Four blocks sit under the ranked table, and none of them is decoration:

    * **What was actually traded** — realised turnover, trade count, cost paid, average cash
      weight, and the window that was MEASURED rather than the one that was requested. A
      report labelled 2015-2025 used to measure 2016-02..2024-12 and never said so.
    * **Regime panel** — how each variant behaved in each named drawdown, printing
      `n/a (inception ...)` where the strategy's own products did not exist. This is the
      instrument the ranked table is genuinely useful as.
    * **Before vs after publication** — a rule's pre-publication record is its author's
      search space, and must never be blended into a headline.
    * **Selection context and rank stability** — where rank 1 sits relative to what ranking
      N zero-skill variants produces by itself, and whether the ordering persists at all.
    """
    report_lines = []

    rank_key = config['RANK_BY'] if config['RANK_BY'] in RANK_KEYS else 'sortino'
    is_reverse = False if rank_key == 'vol' else True

    def sort_key(d):
        val = d.get(rank_key, np.nan)
        if val is None or pd.isna(val):
            return -np.inf if is_reverse else np.inf
        return val

    # Two blocks, never one sorted list. An entry that could not cover the shared window was
    # measured over its own, shorter history, so ranking it against the others would compare
    # rows that do not mean the same thing — which is the failure the shared window exists to
    # prevent. It is reported in full underneath, with its own window stated.
    sorted_metrics = sorted([d for d in display_metrics if d.get('in_ranked_window', True)],
                            key=sort_key, reverse=is_reverse)
    off_window = sorted([d for d in display_metrics if not d.get('in_ranked_window', True)],
                        key=sort_key, reverse=is_reverse)
    top_n = int(config.get('TOP_N_COUNT') or 0)
    shown = sorted_metrics[:top_n] if 0 < top_n < len(sorted_metrics) else sorted_metrics

    def star(key, label):
        return f'{label}*' if rank_key == key else label

    show_lev = (config['LEVERAGE_FACTOR'] != 1.0) and config['MARGIN_FOLLOWS_SIGNAL']
    lev_hdr = f" | {'Avg Lev':<8} | {'Lev Range':<13}" if show_lev else ""

    # 28, not 24: a name plus its flags can reach 28 chars (`HAA_G3_Leveraged_3X ! (expl)`),
    # and at 24 the `(expl)` marker pushed every numeric column out of alignment on exactly
    # the rows it was added to warn about.
    header_line = (f"{'Rank':<5} | {'Strategy':<28} | {star('cagr', 'CAGR'):<8} | "
                   f"{star('max_dd', 'Max Drawdown'):<14} | {star('sharpe', 'Sharpe'):<7} | "
                   f"{star('sortino', 'Sortino'):<8} | {star('upi', 'UPI'):<7} | "
                   f"{star('vol', 'Volatility'):<10}{lev_hdr}")
    width = len(header_line)

    # ------------------------------- header ------------------------------------ #
    report_lines.append("\n" + "=" * width)
    report_lines.append("KELLER STRATEGIES - BACKTEST REPORT")
    win = next((d for d in display_metrics if d.get('window_start') is not None), None)
    if win is not None:
        bind = tuple(win.get('window_binding') or ())
        why = (f"set by {', '.join(bind)}, the last of these to exist" if bind
               else f"era floor {pd.Timestamp(config['START_DATE']):%Y-%m}")
        report_lines.append(f"Ranked window    : "
                            f"{pd.Timestamp(win['window_start']):%Y-%m}.."
                            f"{pd.Timestamp(win['window_end']):%Y-%m} — identical for every"
                            f" row, NOT selectable ({why})")
        # What the ranked window cannot see, stated rather than left to be inferred from
        # the regime panel forty lines further down. A headline drawdown measured over a
        # window that starts after a crash is not a claim about the crash.
        hidden = _window_blind_spot(metrics_data or display_metrics, win['window_start'])
        if hidden:
            report_lines.append(f"  ...which excludes  : {hidden}")
    report_lines.append("Regime panels    : each strategy's FULL history, cut into"
                        " pre-registered regimes (common/eras.py)")
    report_lines.append(f"Execution        : {config.get('EXECUTION_CONVENTION', 'next_open')}"
                        f" | cost {config.get('COST_PCT_PER_SIDE', 0.0):.3%}/side (one-way,"
                        f" per leg) | cash {config.get('CASH_TICKER') or 'none (0%)'}")
    rfs = sorted({round(d.get('rf_annual', 0.0), 6) for d in display_metrics})
    if rfs:
        desc = next((d.get('rf_desc') for d in display_metrics if d.get('rf_desc')),
                    f"realised {config.get('CASH_TICKER') or 'n/a'}")
        report_lines.append(f"Risk-free rate   : {desc}"
                            f" ({', '.join(f'{r:.2%}/yr' for r in rfs)} over the measured"
                            f" windows) — Sharpe and Sortino are net of it")
        if 'CONSTRUCTED' in desc:
            report_lines.append("                   CONSTRUCTED months are accrued from a"
                                " published yield, not a traded fund. Every ratio for a")
            report_lines.append("                   strategy opening before the fund's own"
                                " inception is net of that. See KNOWN_GAPS.md §2.")
    if config['LEVERAGE_FACTOR'] == 1.0:
        report_lines.append("Margin leverage  : 1.0x (off)")
    elif config['MARGIN_FOLLOWS_SIGNAL']:
        report_lines.append(f"Margin leverage  : up to {config['LEVERAGE_FACTOR']}x, scaled by"
                            f" offensive weight (signal-following; defence held at 1x)")
    else:
        report_lines.append(f"Margin leverage  : {config['LEVERAGE_FACTOR']}x flat (drawn in"
                            f" risk-off months too)")
    if config['LEVERAGE_FACTOR'] != 1.0:
        # The saved report outlives the console banner, so the caveat has to be in it.
        report_lines.append("                   NO MAINTENANCE TEST. The ledger accrues interest"
                            " on the debit balance and never")
        report_lines.append("                   compares equity to a maintenance requirement, so"
                            " every levered row is an")
        report_lines.append("                   un-liquidated upper bound. See KNOWN_GAPS.md §3"
                            " and common/margin_sizing.py.")
    if store is not None:
        prov = store.provenance()
        report_lines.append(f"Data             : {prov['source']} | {prov['rows']} daily rows"
                            f" | last complete month {prov['last_complete_month']}"
                            f" | sha256 {prov['sha256'][:12]}")
        # A cache that was not re-checked says so. Skipping the network is a performance
        # choice; hiding that you skipped it would be a correctness one.
        if getattr(store, 'refresh_skipped', None):
            report_lines.append(f"                   {store.refresh_skipped}")
        # The adjustment-vintage verdict. A backtest is barely moved by a seam ~90 days
        # behind the run date, so this reports rather than refuses — but it reports WHERE
        # the numbers are read, not into a log nobody opens.
        report_lines += getattr(store, 'verification_lines', lambda: [])()
        # With STRICT_GAPS=True (the default) a long interior gap refuses the run before
        # this line can print. This is the survivable path for a deliberate False override:
        # the run proceeds, and every gap it carried is named where the numbers are read.
        if prov.get('long_gaps'):
            report_lines.append(f"                   [WARNING] STRICT_GAPS is off and this data"
                                f" carries {len(prov['long_gaps'])} interior gap(s) longer than"
                                f" {MAX_STALE_DAYS} trading days: {prov['long_gaps']}. Returns"
                                f" spanning them are not trustworthy.")
    report_lines.append("=" * width)
    report_lines.append(header_line)
    report_lines.append("-" * width)

    for idx, d in enumerate(shown):
        lev_s = ""
        if show_lev:
            avg, lo, hi = d.get('avg_lev'), d.get('min_lev'), d.get('max_lev')
            lev_s = (f" | {avg:>7.2f}x | {f'{lo:.2f}-{hi:.2f}x':>13}"
                     if avg is not None and not pd.isna(avg)
                     else f" | {'N/A':>8} | {'N/A':>13}")
        flag = ' !' if d.get('coverage_trimmed') else ''
        # Both roles are measured and ranked here but excluded from the selection statistics,
        # so both must be visibly marked — an unflagged 3x row at the top of a Sortino ranking
        # reads as a recommendation, which is the one thing it is not.
        if d.get('role') == 'control':
            flag += ' (ctrl)'
        elif d.get('role') == 'exploratory':
            flag += ' (expl)'
        report_lines.append(
            f"{idx + 1:<5} | {d['name'] + flag:<28} | {_num(d['cagr'], 7, '.2%')} | "
            f"{_num(d['max_dd'], 13, '.2%')} | {_num(d['sharpe'], 6, '.2f')} | "
            f"{_num(d['sortino'], 7, '.2f')} | {_num(d.get('upi'), 6, '.2f')} | "
            f"{_num(d['vol'], 9, '.2%')}{lev_s}")

    report_lines.append("=" * width)
    report_lines.append(f"(* sorted by {rank_key.upper()}, "
                        f"{'descending' if is_reverse else 'ascending'})")
    if len(shown) < len(sorted_metrics):
        report_lines.append(f"(TOP_N_COUNT={top_n}: {len(sorted_metrics) - len(shown)} further"
                            f" strategies were measured but are not shown)")
    if any(d.get('coverage_trimmed') for d in shown):
        report_lines.append("(! = the strategy's assets did not exist for the whole era, so"
                            " it could not be measured from the era floor; see the next"
                            " table)")
    if show_lev:
        report_lines.append("(Avg/Range Lev = REALISED effective leverage. Benchmarks stay at"
                            " 1.00x by design.)")

    if off_window:
        report_lines.append("")
        report_lines.append("SHORTER HISTORY — measured, but NOT over the window above, so"
                            " these rows are not comparable with it")
        report_lines.append("-" * width)
        for d in sorted(off_window, key=lambda x: pd.Timestamp(x['window_start'])):
            report_lines.append(
                f"{'':<5} | {d['name']:<28} | {_num(d['cagr'], 7, '.2%')} | "
                f"{_num(d['max_dd'], 13, '.2%')} | {_num(d['sharpe'], 6, '.2f')} | "
                f"{_num(d['sortino'], 7, '.2f')} | {_num(d.get('upi'), 6, '.2f')} | "
                f"{_num(d['vol'], 9, '.2%')}   {_fmt_window(d)}")
        report_lines.append("-" * width)
        report_lines.append("(These entries' own products did not exist at the ranked"
                            " window's start. Under RANKED_WINDOW_POLICY='strategies'")
        report_lines.append(" they are measured over their own history instead of shortening"
                            " every other row; set it to 'all' to")
        report_lines.append(" force one window across everything, at the cost of the months"
                            " the earlier entries can otherwise be read through.)")

    report_lines += _execution_section(sorted_metrics + off_window)
    report_lines += _partition_sections(sorted_metrics + off_window)
    report_lines += _segment_leaderboard_sections(
        sorted_metrics + off_window,
        top_n=int(config.get('SEGMENT_TOP_N', 5) or 5),
        # Default the per-segment ordering to the SAME metric as the headline table, so the
        # report does not silently rank one table by Sortino and the panels by total return.
        # SEGMENT_RANK_BY overrides it; segments too short to annualise fall back to 'return'
        # and say so on their own line.
        rank_by=config.get('SEGMENT_RANK_BY') or rank_key)
    report_lines += _regime_section(sorted_metrics + off_window)
    report_lines += _post_publication_section(sorted_metrics + off_window)
    # EVERYTHING MEASURED, not everything displayed. The multiple-testing count is the
    # number of variants that were tried; narrowing STRATEGIES_TO_DISPLAY narrows what you
    # look at, never what you searched. Passing the display subset here understated the
    # count by up to a factor of five, in the one section whose entire job is to state it
    # honestly — and it did so in the direction that flatters rank 1.
    # n_shown counts BOTH blocks: a short-history row is displayed, just not ranked.
    report_lines += _selection_section(metrics_data or display_metrics, rank_key, is_reverse,
                                       n_shown=len(sorted_metrics) + len(off_window))
    advice = (_advice_from_config(metrics_data or display_metrics, config)
              if (metrics_data or display_metrics) else None)
    report_lines += _sustainable_leverage_section(metrics_data or display_metrics, config,
                                                  advice=advice)
    report_lines += _robustness_section(metrics_data or display_metrics, rank_key)
    report_lines += _leverage_frontier_section(metrics_data or display_metrics, config,
                                               advice)

    report_str = "\n".join(report_lines)
    print(report_str)
    return report_str


def _advice_from_config(metrics, config):
    """The sizing advice for one run, from the same knobs the CLI exposes.

    One producer, two consumers — the SUSTAINABLE MARGIN LEVERAGE section and the LEVERAGE
    FRONTIER — because advising twice would run the drawdown bootstrap twice for byte-
    identical output.
    """
    from common import leverage_advice as advice_mod
    return advice_mod.advise(
        metrics,
        k=float(config.get('SAFETY_FACTOR_K', 3.0)),
        borrow_rate_annual=float(config.get('MARGIN_BORROW_RATE', 0.06)),
        maintenance_base=float(config.get('MAINTENANCE_BASE',
                                          advice_mod.MAINTENANCE_BASE)),
        capacity_leverage=config.get('BORROWING_CAPACITY_LEVERAGE'),
        run_leverage=float(config.get('LEVERAGE_FACTOR', 1.0)))


def _sustainable_leverage_section(metrics, config, advice=None):
    """What margin each record could carry, and which real-world limit stopped it.

    The counterpart to every levered row above. Those say what leverage DID at a multiple
    somebody chose; this says what leverage SURVIVES — and the output that matters is not the
    number but `binding`, because a model capped by its own drawdown is a different object from
    one capped by a credit line and only the second can be relaxed by phoning a broker.

    Reproducibility, and the reason this section exists at all: the registry table in
    LEVERAGE.md section 5.4 was produced on 2026-07-30 by a script that was not kept, so the
    repository's answer to its own central question could not be regenerated from the
    repository. It can now.
    """
    from common import leverage_advice as advice_mod

    if not metrics:
        return []
    if advice is None:
        advice = _advice_from_config(metrics, config)
    if not advice.by_name:
        return []

    lines = ["", "SUSTAINABLE MARGIN LEVERAGE — what survives, not what optimises",
             "  " + advice_mod.headline(advice), ""]
    lines += ['  ' + row for row in advice.table().splitlines()]
    lines.append("")
    # The inputs that table consumes and does not show. Every figure below was computed on
    # every run since the module landed and rendered nowhere — including the deflated Sharpe
    # probability, which is this repository's own answer to "is the edge real given how many
    # variants were searched", and the block sensitivity, which exists specifically to make
    # the bootstrap's one arbitrary choice visible.
    lines.append("  EVIDENCE BEHIND THOSE NUMBERS — how much to believe the column above")
    lines += ['  ' + row for row in advice.evidence().splitlines()]
    lines.append("")
    lines.append("  Assumed, and none of it your broker's:")
    for a in advice_mod.assumption_lines(advice):
        lines.append(f"    - {a}")
    if advice.skipped:
        lines.append("  Not sized:")
        for name, why in advice.skipped:
            lines.append(f"    - {name}: {why}")
    lines.append("  The ledger NEVER tests a maintenance requirement, so the levered rows above"
                 " remain un-liquidated")
    lines.append("  upper bounds. This section is where that risk is priced — beside the"
                 " engine, not inside it.")
    return lines


def _robustness_section(metrics, rank_key):
    """How much of the ranking above survives a different sample.

    `_selection_section` already asks a version of this — is rank 1 above what pure search
    over N variants produces, and does the ordering persist between two or three disjoint
    sub-periods. This asks it exhaustively rather than on a handful of windows: every possible
    equal split of the history, not the two the calendar happened to suggest.

    Both stay. The rank-stability rhos describe SPECIFIC eras and are readable as history;
    PBO is a single number about the selection procedure and is readable as a verdict.
    """
    from common import robustness as rb

    if not metrics:
        return []
    try:
        frame, binding = rb.common_frame(metrics)
    except NotCalculable as exc:
        return ["", f"ROBUSTNESS — not computed: {exc}"]

    # The rate the leaderboard's Sharpes were already netted against — measured 2026-08-01,
    # leaving this at 0.0 flattered low-vol entries by up to +0.16 Sharpe (AUD-02).
    rf = rb.realised_rf(metrics)

    lines = []
    try:
        lines += rb.pbo_lines(rb.pbo(frame=frame, binding=binding, rf_annual=rf))
    except NotCalculable as exc:
        lines += ["", f"PBO — not computed: {exc}"]
    try:
        lines += rb.rank_lines(rb.rank_bootstrap(frame=frame, binding=binding,
                                                 rank_key=rank_key, rf_annual=rf))
    except NotCalculable as exc:
        lines += ["", f"RANK STABILITY — not computed: {exc}"]

    # The family view: same splits, same resampled histories, evidence POOLED across
    # variants — because "which mechanism is solid" is a different question from "which
    # variant won", and the variant answer has just been shown above to be unstable. A
    # family whose every variant scores well is harder to explain as luck than one lucky
    # variant: the noise explanation has to win N times instead of once.
    for grouping, fn in (('family', rb.family_of), ('mechanism', rb.mechanism_of)):
        try:
            lines += rb.group_pbo_lines(rb.group_pbo(frame=frame, binding=binding,
                                                     groups=fn, grouping=grouping,
                                                     rf_annual=rf))
        except NotCalculable as exc:
            lines += ["", f"{grouping.upper()}-LEVEL PBO — not computed: {exc}"]
        try:
            lines += rb.group_rank_lines(rb.group_rank_bootstrap(
                frame=frame, binding=binding, groups=fn, grouping=grouping,
                rank_key=rank_key, rf_annual=rf))
        except NotCalculable as exc:
            lines += ["", f"{grouping.upper()} RANK STABILITY — not computed: {exc}"]
    return lines


def _leverage_frontier_section(metrics, config, advice):
    """The margin decision as a curve: P(margin call), median and 5th-percentile CAGR at
    every constant leverage level, on the same resampled histories the robustness section
    ranked.

    This is the independent check on the SUSTAINABLE MARGIN LEVERAGE section above it: that
    section derives one number per model from a bootstrapped drawdown quantile and a closed
    form; this walks two thousand alternative histories month by month under the ledger's
    own monthly-reset policy and reports how often the broker would have called at each
    level — including AT the recommended one. The two methods share the maintenance
    requirement and the borrow rate and nothing else.
    """
    import math as _math

    from common import robustness as rb

    if not metrics or advice is None or not advice.by_name:
        return []
    try:
        frame, binding = rb.common_frame(metrics)
    except NotCalculable as exc:
        return ["", f"LEVERAGE FRONTIER — not computed: {exc}"]

    maintenance, recommended = {}, {}
    for name, rec in advice.by_name.items():
        m = rec.maintenance_margin_used
        if m is not None and not (isinstance(m, float) and _math.isnan(m)):
            maintenance[name] = float(m)
            recommended[name] = rec.recommended_leverage
    try:
        res = rb.leverage_frontier(
            frame=frame, binding=binding, maintenance=maintenance, recommended=recommended,
            borrow_rate_annual=float(config.get('MARGIN_BORROW_RATE', 0.06)))
    except NotCalculable as exc:
        return ["", f"LEVERAGE FRONTIER — not computed: {exc}"]
    return rb.frontier_lines(res)


def _execution_section(metrics):
    """T4.4 — what was actually traded, next to what was assumed, next to who published it."""
    head = (f"{'Strategy':<24} | {'Type':<9} | {'Measured window':<16} | {'Mo':>4} | "
            f"{'Turn/yr':>8} | {'Trades':>7} | {'Cost bps':>9} | {'Cash':>6} | Coverage")
    lines = ["", "WHAT WAS ACTUALLY TRADED (measured, not assumed)", head, "-" * len(head)]
    for d in metrics:
        cov = 'full'
        if d.get('coverage_trimmed'):
            inc = d.get('binding_inception')
            stamp = pd.Timestamp(inc).strftime('%Y-%m') if inc is not None else '?'
            cov = f"trimmed: {d.get('binding_ticker')} from {stamp}"
        role = d.get('role', 'strategy')
        kind = d.get('fidelity', 'custom') if role == 'strategy' else role
        lines.append(
            f"{d['name']:<24} | {kind:<9} | {_fmt_window(d):<16} | "
            f"{d.get('n_periods', 0):>4} | "
            f"{d.get('annual_turnover', float('nan')):>8.2f} | "
            f"{d.get('n_trades_total', 0):>7} | {d.get('total_cost_bps', 0.0):>9.1f} | "
            f"{d.get('avg_cash_weight', 0.0):>6.1%} | {cov}")
    lines.append("Turn/yr = one-way notional traded per year as a multiple of equity. Cost bps"
                 " is cumulative over the whole")
    lines.append("window and includes initial deployment and terminal liquidation.")
    lines.append("Type: faithful = the author's universe and parameters | proxy = same, on"
                 " substitute funds | custom = a")
    lines.append("universe nobody published (here: only the leverage-forced G3/G4 sleeves and"
                 " GEM) | control = a degenerate")
    lines.append("single-asset case, reported but excluded from the selection statistics"
                 " below | benchmark = a passive")
    # Corrected 2026-07-29: this line used to end "benchmark = passive, never levered", which
    # stopped being true the moment SPY_2X/SPY_3X/RiskParity_3X were added. A benchmark is
    # passive — no signal, no de-risking — and three of them are now deliberately levered,
    # because a levered wrap compared only against 1x references cannot be decomposed.
    lines.append("reference with no signal and no de-risking rule. Three benchmarks ARE levered"
                 " (SPY_2X, SPY_3X,")
    lines.append("RiskParity_3X): a levered wrap compared only against 1x references cannot be"
                 " separated into")
    lines.append("'what the timing rule added' and 'what the leverage added'. None of them may"
                 " set the ranked window.")
    lines.append("")
    lines.append("Static benchmark compositions, so the table is readable without opening the"
                 " code:")
    for name, comp in _STATIC_COMPOSITIONS.items():
        if name in {d['name'] for d in metrics}:
            lines.append(f"  {name:<20} {comp}")
    return lines


#: Panel column width. Every segment key in common/eras.py must fit, or the header lies.
CELL_W = 16


def _cell(cell):
    """One panel cell: a measured pair, a partial one, or an explicit n/a."""
    if 'na' in cell:
        return f"{'n/a ' + cell['na'][-7:]:^{CELL_W}}"
    text = f"{cell['return']:+.1%}/{cell['max_dd']:.1%}"
    return f"{('~' + text) if cell.get('partial') else text:^{CELL_W}}"


def _partition_panel_lines(metrics, seg, era_end):
    """One segmentation, one row per strategy, one column per segment."""
    segments = eras.resolved_segments(seg, era_end)
    panel = eras.partition_panel(metrics, seg, era_end)
    if not panel or not segments:
        return []

    cols = [s.key for s in segments]
    has_adverse = any(s.adverse for s in segments)
    if has_adverse:
        cols.append('ADVERSE')

    head = f"{'Strategy':<24} | " + " | ".join(f"{c[:CELL_W]:^{CELL_W}}" for c in cols)
    lines = ["", seg.title, f"  dated by: {seg.source}", head, "-" * len(head)]
    for d in metrics:
        row = panel.get(d['name'])
        if row is None:
            continue
        lines.append(f"{d['name']:<24} | " + " | ".join(_cell(row[c]) for c in cols))
    lines.append("-" * len(head))
    for s in segments:
        n = (s.end.to_period('M') - s.start.to_period('M')).n + 1
        mark = ' [adverse]' if s.adverse else ''
        lines.append(f"  {s.key:<17} {s.start:%Y-%m}..{s.end:%Y-%m} {n:>4} mo{mark}  {s.label}")
    if has_adverse:
        lines.append("  ADVERSE           every [adverse] month of the era compounded"
                     " together, and nothing else")
    if seg.note:
        lines.append(f"  note: {seg.note}")
    return lines


def _partition_sections(metrics):
    """Every strategy in every pre-registered regime — the report's centre of gravity.

    This is what replaced the START_DATE / END_DATE boxes. A window you type in is a result
    you chose; these boundaries were dated by the NBER, the FOMC, the BLS and the S&P 500,
    they tile the era end to end with no gap and no overlap, and no run can add, move or drop
    one. Each panel is measured over the strategy's own full history, so a run cannot shorten
    a regime either.
    """
    first, last = eras.era_bounds(metrics)
    if last is None:
        return []
    lines = ["", "=" * 110,
             "BEHAVIOUR BY PRE-REGISTERED MARKET REGIME  (segment return / worst drawdown"
             " inside it)",
             f"Era {eras.COMMON_ERA_START[:7]}..{last:%Y-%m}. Every strategy is shown in"
             f" EVERY segment, over its own history — not over the",
             "ranked window above, which would make these panels an expression of"
             " date-selection bias rather than a cure for it.",
             "Segment boundaries are public dated facts, frozen in common/eras.py. Nothing"
             " about this run can move one.",
             "=" * 110]
    for seg in eras.SEGMENTATIONS:
        lines += _partition_panel_lines(metrics, seg, last)
    lines.append("")
    lines.append("`~` = the strategy began INSIDE that segment, so the cell covers only the"
                 " part it could trade.")
    lines.append("`n/a inc YYYY-MM` = its own assets did not exist yet. That gap is the"
                 " answer, not a missing number. One ticker draws")
    lines.append("the line through 2008: HYG, the oldest US high-yield ETF (2007-04), so"
                 " every strategy holding high yield opens")
    lines.append("AFTER the October 2007 peak while the ones that do not (HAA, GTAA_G5, the"
                 " G4 sizes) can be read through the")
    lines.append("whole bear market. Nothing reaches the dot-com bust — the multi-asset"
                 " universes did not exist. See KNOWN_GAPS.md §1-2,")
    lines.append("including the four tickers whose pre-2008 history is CONSTRUCTED and how.")
    lines.append("These labels are ex-post. The NBER dated the 2020 trough in 2021; no"
                 " strategy could have known its regime at")
    lines.append("the time, and none of this is fed back into a signal.")
    return lines


def _seg_cells(d):
    """One ranked leaderboard row's numbers. Annualised fields print `n/a` when the segment
    was too short to carry them, never a plausible-looking figure computed from four months."""
    def f(key, fmt):
        v = d.get(key)
        return format(v, fmt) if v is not None and np.isfinite(v) else 'n/a'
    return (f"{d['return']:>8.1%}  dd {d['max_dd']:>7.1%}  "
            f"cagr {f('cagr', '>7.1%')}  vol {f('vol', '>6.1%')}  "
            f"shrp {f('sharpe', '>5.2f')}  sort {f('sortino', '>5.2f')}  "
            f"upi {f('upi', '>5.2f')}  {d['n_months']:>3}mo")


def _segment_leaderboard_sections(metrics, top_n=5, rank_by='return'):
    """One small ranked table per regime — the main leaderboard's logic, applied per segment.

    The matrix above answers "how did THIS strategy behave in each regime?". This answers
    the other question, which the matrix makes you compute by eye across twenty-five rows:
    "who actually led THIS regime, and by how much?".

    It inherits the ranked table's one non-negotiable property: **every row spans exactly
    the same months.** A strategy that entered the segment late is listed underneath with
    the months it covered, and is not ranked. That is not tidiness — ranking a strategy
    that missed the crash against one that lived through it would rebuild date-selection
    bias inside each panel, in the direction that flatters the latecomer, since arriving
    after the fall shows the recovery without it.
    """
    first, last = eras.era_bounds(metrics)
    if last is None:
        return []
    lines = ["", "=" * 110,
             f"WHO LED EACH REGIME  (ranked by {rank_by.upper()};"
             f" only strategies covering the segment IN FULL)",
             "Same rule as the ranked table at the top: comparable rows or no row. A strategy"
             " that entered a segment late is listed",
             "beneath it with what it did cover, and is NOT ranked — arriving after the crash"
             " shows the recovery without the fall.",
             f"Segments shorter than {SEGMENT_MIN_MONTHS} months carry no annualised metrics"
             f" (a CAGR over 2 months is that return raised to",
             "the sixth power) and are ranked by total RETURN instead. Each line states the key"
             " that actually ordered it.",
             "=" * 110]

    for seg in eras.SEGMENTATIONS:
        segments = eras.resolved_segments(seg, last)
        panel = eras.partition_panel(metrics, seg, last)
        if not panel or not segments:
            continue
        keys = [(s.key, s, ) for s in segments]
        if any(s.adverse for s in segments):
            keys.append(('ADVERSE', None))
        lines.append("")
        lines.append(seg.title.upper())
        for key, s in keys:
            lb = eras.segment_leaderboard(panel, key, rank_by=rank_by)
            ranked, partial, absent = lb.ranked, lb.partial, lb.absent
            if s is not None:
                span = (f"{s.start:%Y-%m}..{s.end:%Y-%m}  {(s.end.to_period('M') - s.start.to_period('M')).n + 1:>3} mo"
                        f"{' [adverse]' if s.adverse else ''}")
            else:
                span = 'every adverse month of the era, compounded'
            # Name the key that ORDERED the rows, not the one requested — they differ whenever
            # the segment was too short to annualise, and a caption that claims the requested
            # one would be describing a ranking that does not exist.
            by = (f"  [by {lb.rank_by.upper()}]" if lb.rank_by == rank_by
                  else f"  [by RETURN — too short for {rank_by.upper()}]")
            lines.append(f"  {key:<18} {span}{by}")
            if not ranked:
                lines.append(f"      nothing covers this segment in full "
                             f"({len(partial)} partial, {len(absent)} absent)")
                continue
            for i, d in enumerate(ranked[:top_n]):
                lines.append(f"      {i + 1:>2}. {d['name']:<24} "
                             f"{_seg_cells(d)}")
            if len(ranked) > top_n:
                worst = ranked[-1]
                lines.append(f"      ... {len(ranked) - top_n} more ranked; last is "
                             f"{worst['name']} {worst['return']:.1%} / dd {worst['max_dd']:.1%}")
            tail = []
            if partial:
                tail.append(f"{len(partial)} partial ("
                            + ', '.join(f'{n} {m}mo' for n, m, _ in partial[:3])
                            + (', ...' if len(partial) > 3 else '') + ')')
            if absent:
                tail.append(f"{len(absent)} absent")
            if tail:
                lines.append(f"      not ranked: {'; '.join(tail)}")
    lines.append("")
    lines.append("Read these as DESCRIPTIONS of what each regime rewarded, never as a"
                 " forecast. The segment labels are ex-post —")
    lines.append("the NBER dated the 2020 trough in 2021 — and the RANK STABILITY panel"
                 " below reports how little of any")
    lines.append("ordering survives into the next sub-period.")
    return lines


def _regime_section(metrics):
    """T5.1 — the panel the ranked table is actually useful as."""
    from common.regimes import EPISODES, episode_panel, coverage_fraction

    panel = episode_panel(metrics)
    if not panel:
        return []
    spans = [d.get('first_return_full') or d.get('first_return') for d in metrics]
    spans = [pd.Timestamp(x) for x in spans if x is not None]
    ends = [d.get('last_return_full') or d.get('last_return') for d in metrics]
    ends = [pd.Timestamp(x) for x in ends if x is not None]
    span = (f"{min(spans):%Y-%m}..{max(ends):%Y-%m}" if spans and ends else "n/a")
    head = f"{'Strategy':<24} | " + " | ".join(f"{k[:15]:^15}" for k, *_ in EPISODES) + " | Cov"
    lines = ["", "BEHAVIOUR BY HISTORICAL REGIME  (episode return / worst drawdown inside it)",
             f"Measured over each strategy's FULL available history ({span}). Unlike the "
             f"partitions above, these episodes are NOT",
             "exhaustive — they are named stress windows, and they reach back before the era "
             "for the few assets that existed.",
             head, "-" * len(head)]
    for d in metrics:
        row = panel.get(d['name'])
        if row is None:
            continue
        cells = []
        for key, *_ in EPISODES:
            cell = row[key]
            if 'na' in cell:
                cells.append(f"{'n/a ' + cell['na'][-7:]:^15}")
            else:
                cells.append(f"{cell['return']:>+7.1%}/{cell['max_dd']:>6.1%} ")
        lines.append(f"{d['name']:<24} | " + " | ".join(cells) +
                     f" | {coverage_fraction(row):>3.0%}")
    lines.append("-" * len(head))
    for key, start, end, label in EPISODES:
        lines.append(f"  {key:<16} {start[:7]} .. {end[:7]}   {label}")
    lines.append("`n/a (inception YYYY-MM)` means the strategy's own assets did not exist yet."
                 " That gap is the honest answer, not")
    lines.append("a missing number: no 3x product predates 2008-11 and most 2x products start"
                 " 2006-2007, so no leveraged variant")
    lines.append("here lived through the dot-com bust and only some lived through the GFC."
                 " Episode dates are frozen in")
    lines.append("common/regimes.py and were written from the historical record, not tuned to"
                 " results.")
    return lines


def _post_publication_section(metrics):
    """T5.4 — never blend in-sample with out-of-sample."""
    from common.metrics import calculate_metrics
    from common.regimes import post_publication_split, publication_date

    rows = []
    for d in metrics:
        # Full history again: a pre-publication column bounded by the caller's START_DATE
        # would compare an arbitrary slice of in-sample against out-of-sample.
        returns = d.get('returns_full')
        if returns is None or returns.empty:
            returns = d.get('returns')
        pub = publication_date(d['name'])
        if returns is None or returns.empty or pub is None:
            continue
        pre, post = post_publication_split(returns, d['name'])
        if post is None or len(post) < 12:
            continue
        pre_cagr = (calculate_metrics(pre)['cagr']
                    if pre is not None and len(pre) >= 12 else None)
        rows.append((d['name'], pub, pre_cagr, calculate_metrics(post)['cagr'], len(post)))
    if not rows:
        return []
    head = (f"{'Strategy':<24} | {'Published':<10} | {'Pre (in-sample)':>16} | "
            f"{'Post':>10} | {'Months post':>12}")
    lines = ["", "BEFORE vs AFTER PUBLICATION (CAGR)", head, "-" * len(head)]
    for name, pub, pre_cagr, post_cagr, n in rows:
        pre_s = f"{pre_cagr:>16.2%}" if pre_cagr is not None else f"{'n/a':>16}"
        lines.append(f"{name:<24} | {pub.strftime('%Y-%m'):<10} | {pre_s} | "
                     f"{post_cagr:>10.2%} | {n:>12}")
    lines.append("A rule's pre-publication record is its author's search space, however"
                 " carefully the rules were pre-registered.")
    lines.append("Only the post column is evidence about the future, and it is short.")
    return lines


def _selection_section(metrics, rank_key, is_reverse, n_shown=None):
    """T5.2 + T5.3 — what the top of the table is worth, and whether it persists.

    `metrics` must be everything MEASURED. `n_shown` is how many of them the ranked table
    displayed, and is printed alongside so the two counts can never be confused again.
    """
    from common.metrics import calculate_metrics
    from common.selection import (participation_ratio, rank_stability, selection_context,
                                  selection_trials)

    # Which entries count as trials — and WHY controls and 3x variants do not — is decided in
    # `selection.selection_trials`, because the Sharpe haircut in `common/leverage_advice.py`
    # asks the same question and two answers to it would be one too many.
    active = selection_trials(metrics)
    if len(active) < 3:
        return []

    values = [d.get(rank_key) for d in active]
    values = [v for v in values if v is not None]
    if not is_reverse:                      # 'vol' / 'max_dd': lower is better
        values = [-v for v in values]
    returns_by_name = {d['name']: d.get('returns') for d in active
                       if d.get('returns') is not None and not d['returns'].empty}

    pr = participation_ratio(returns_by_name)
    ctx = selection_context(values, n_effective=pr['participation_ratio'] if pr else None)
    if ctx is None:
        return []

    # The trial count is what was MEASURED. Narrowing STRATEGIES_TO_DISPLAY narrows what you
    # look at, never what you searched, so the two numbers are printed side by side and the
    # search count is the one the null threshold below is built from.
    shown_note = (f" | measured {len(metrics)}, displayed {n_shown}"
                  if n_shown is not None and n_shown < len(metrics) else '')
    lines = ["", "SELECTION CONTEXT — how much of rank 1 is skill and how much is search",
             f"  variants ranked on {rank_key.upper()} : {ctx['n']}{shown_note}"]
    if pr:
        lines.append(f"  effective independent bets       : {pr['participation_ratio']:.1f}"
                     f" (PC1 explains {pr['pc1_share']:.1%};"
                     f" {pr['n_pcs_for_90pct']} PCs for 90% of variance,"
                     f" {pr['n_months']} common months)")
    lines.append(f"  mean / sd across variants        : {ctx['mean']:.3f} / {ctx['sd']:.3f}")
    lines.append(f"  observed best                    : {ctx['observed_best']:.3f}")
    lines.append(f"  expected best of {ctx['n']} with NO skill : {ctx['threshold_nominal']:.3f}"
                 f"  (mean + {ctx['z_nominal']:.2f} sd)")
    if pr:
        lines.append(f"  ... using effective bets instead : {ctx['threshold_effective']:.3f}"
                     f"  (mean + {ctx['z_effective']:.2f} sd)  <- the fairer bar")
        verdict = 'ABOVE' if ctx['observed_best'] > ctx['threshold_effective'] else 'BELOW'
        lines.append(f"  rank 1 is {verdict} what pure selection over this suite produces.")
    lines.append("  The nominal count assumes every variant is an independent trial; these"
                 " trade overlapping universes on")
    lines.append("  the same momentum signal, so the effective row is the honest comparison.")

    # Rank stability across disjoint sub-periods.
    spans = [s.index for s in returns_by_name.values() if len(s)]
    if spans:
        lo, hi = min(s[0] for s in spans), max(s[-1] for s in spans)
        years = (hi - lo).days / 365.25
        if years >= 8:
            n_win = 3 if years >= 11 else 2
            edges = pd.date_range(lo, hi, periods=n_win + 1)
            # DISJOINT windows: sharing an endpoint would put one month in two sub-periods
            # and correlate them by construction, which is the one thing this must not do.
            windows = [(f"{edges[i]:%Y-%m}..{edges[i + 1]:%Y-%m}",
                        edges[i], edges[i + 1] - pd.Timedelta(days=1))
                       for i in range(n_win)]
            pairs, tops = rank_stability(
                returns_by_name, windows, lambda r: calculate_metrics(r)[rank_key])
            if pairs:
                lines.append("")
                lines.append(f"RANK STABILITY — does the {rank_key.upper()} ordering persist?")
                for p in pairs:
                    lines.append(f"  {p['a']} vs {p['b']}   rho = {p['rho']:+.2f}"
                                 f"  (p ~ {p['p']:.2f}, n = {p['n']})")
                common_top = (set.intersection(*(set(v) for v in tops.values()))
                              if tops else set())
                lines.append(f"  strategies in every sub-period top-5:"
                             f" {sorted(common_top) if common_top else 'NONE'}")
                lines.append("  rho near zero means the ordering carries no predictive content."
                             " It remains an excellent")
                lines.append("  DESCRIPTION of what each regime rewarded — which is what the"
                             " regime panel above is for.")
    return lines


def generate_chart(display_results, strategies):
    """Logarithmic growth comparison, encoded by FAMILY and labelled at the right edge.

    The encoding lives in `common/palette.py` and is shared with the dashboard, so the
    picture in `backtest_results/growth_log_*.png` and the picture in the GUI cannot drift
    apart. See that module for why a continuous colormap over n lines was unreadable.
    """
    from common import palette

    entries = [(name, next((getattr(s, 'role', 'strategy')
                            for s in strategies if s.name == name), 'strategy'))
               for name in display_results]
    # Index the strokes against the whole registry, not against this run's selection, so
    # two reports drawn from different STRATEGIES_TO_DISPLAY lists stay comparable.
    _roster = strategy_roster()
    styles = palette.line_styles(entries,
                                 universe=[(d['name'], d['role']) for d in _roster],
                                 ratios={d['name']: d['leverage'] for d in _roster})

    fig, ax = plt.subplots(figsize=(16, 9))
    for name, cum_ret in display_results.items():
        ax.plot(cum_ret.index, cum_ret.values, label=name,
                **palette.plot_kwargs(styles[name]))
    ax.set_yscale('log')
    ax.set_title('Keller Strategies – Performance Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (log)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(handles=palette.family_legend_handles(entries), title='Family',
              loc='upper left', fontsize=8, ncol=3, framealpha=0.85)
    fig.subplots_adjust(right=0.80)
    palette.label_lines(ax, {n: float(s.iloc[-1]) for n, s in display_results.items()
                             if len(s)}, styles)

def save_outputs(report_str, display_results, config, store=None, metrics_data=None):
    """Save the text report, the chart, and the manifest that makes them citable.

    The manifest is not optional decoration. The 92 pre-audit reports had to be quarantined
    wholesale precisely because nothing recorded which commit, which data, or which
    conventions produced them — there was no way to tell which were affected by what.
    """
    if not config['SAVE_FILES_TO_DISK']:
        if display_results:
            plt.show()
        return None

    from common.manifest import build_manifest, file_sha256, write_manifest

    out_dir = os.path.join(ROOT_DIR, 'backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    text_report_path = os.path.join(out_dir, f'report_{timestamp}.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write(report_str.strip())
    print(f"\nText report saved at: {text_report_path}")
    outputs = {os.path.basename(text_report_path): file_sha256(text_report_path)}

    if display_results:
        chart_path = os.path.join(out_dir, f'growth_log_{timestamp}.png')
        plt.savefig(chart_path, dpi=300)
        print(f"Chart saved at: {chart_path}")
        plt.close()
        outputs[os.path.basename(chart_path)] = file_sha256(chart_path)

    manifest_path = os.path.join(out_dir, f'report_{timestamp}.manifest.json')
    write_manifest(manifest_path, build_manifest(config, store, metrics_data or [],
                                                 outputs=outputs, repo_dir=ROOT_DIR))
    print(f"Manifest saved at: {manifest_path}")
    return manifest_path

def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    # Parse CLI arguments
    args = parse_cli_args()
    
    # Determine execution mode: CLI --live override, else dashboard default
    live_mode = args.live if args.live is not None else EXECUTION_MODE
    
    # Determine strategies to execute
    strategies = get_strategies_to_run(args)
    if strategies is None:
        # User ran with --list and list was shown, exit cleanly
        return

    # Configuration dictionary
    config = {
        'START_DATE': START_DATE,
        'END_DATE': END_DATE,
        'LEVERAGE_FACTOR': LEVERAGE_FACTOR,
        'MARGIN_BORROW_RATE': MARGIN_BORROW_RATE,
        'MARGIN_FOLLOWS_SIGNAL': MARGIN_FOLLOWS_SIGNAL,
        'EXECUTION_MODE': live_mode,
        'CURRENT_EXECUTION_DATE': CURRENT_EXECUTION_DATE,
        'SAFETY_MARGIN_PCT': SAFETY_MARGIN_PCT,
        'FLEXIBILITY_BAND_PCT': FLEXIBILITY_BAND_PCT,
        'FLUSH_ROUND_UP_BAND_PCT': FLUSH_ROUND_UP_BAND_PCT,
        'PRICE_CAP_MARGIN_PCT': PRICE_CAP_MARGIN_PCT,
        'MINIMUM_TRADE_PCT': MINIMUM_TRADE_PCT,
        'SHARE_LOT_SIZE': SHARE_LOT_SIZE,
        'FRACTIONAL_SHARES': FRACTIONAL_SHARES,
        'SAVE_FILES_TO_DISK': SAVE_FILES_TO_DISK,
        'STRATEGIES_TO_DISPLAY': STRATEGIES_TO_DISPLAY,
        'TOP_N_COUNT': TOP_N_COUNT,
        'RANK_BY': RANK_BY,
        # --refresh forces the trailing-window re-download regardless of when the cache
        # was last checked. Otherwise the configured interval applies.
        'CACHE_REFRESH_HOURS': 0.0 if getattr(args, 'refresh', False) else CACHE_REFRESH_HOURS,
        'STRICT_GAPS': STRICT_GAPS,
        'SAFETY_FACTOR_K': SAFETY_FACTOR_K,
        'MAINTENANCE_BASE': MAINTENANCE_BASE,
        'BORROWING_CAPACITY_LEVERAGE': BORROWING_CAPACITY_LEVERAGE,
        'SEGMENT_TOP_N': SEGMENT_TOP_N,
        'RANKED_WINDOW_POLICY': RANKED_WINDOW_POLICY,
        'COST_PCT_PER_SIDE': COST_PCT_PER_SIDE,
        'EXECUTION_CONVENTION': EXECUTION_CONVENTION,
        'CASH_TICKER': CASH_TICKER,
        'COVERAGE_POLICY': COVERAGE_POLICY,
        'RF_ANNUAL_FALLBACK': RF_ANNUAL_FALLBACK,
        'LOOKBACK_MONTHS': LOOKBACK_MONTHS,
        'DATA_START_DATE': DATA_START_DATE,
        'CACHE_DIR': CACHE_DIR,
    }

    # 1. Load data and scores
    prices, scores_w, scores_u, store = load_data(config)

    # 2. Execution
    if config['EXECUTION_MODE']:
        run_live_mode(prices, scores_w, scores_u, strategies, config, store=store)
    else:
        metrics_data, results = run_backtest(prices, scores_w, scores_u, strategies, config,
                                             store=store)
        
        display_all = len(config['STRATEGIES_TO_DISPLAY']) == 0
        display_metrics = [d for d in metrics_data if display_all or d['name'] in config['STRATEGIES_TO_DISPLAY']]
        display_results = {k: v for k, v in results.items() if display_all or k in config['STRATEGIES_TO_DISPLAY']}
        
        report_str = print_report(metrics_data, display_metrics, config, store=store)
        
        if display_results:
            generate_chart(display_results, strategies)
            
        save_outputs(report_str, display_results, config, store=store,
                     metrics_data=metrics_data)

if __name__ == "__main__":
    main()
