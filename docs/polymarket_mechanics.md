# Polymarket Mechanics — Ground-Truth Reference

**Written 2026-07-26.** Purpose: this repo has ~120 commits of statistical analysis
built on unverified assumptions about how Polymarket actually works. Several were
wrong. This file is the primary-source reference so a future session can check an
assumption in seconds instead of inferring it from the analysis code.

**Rule used throughout: where docs and live behaviour disagree, live behaviour wins,
and the evidence is shown.** Every non-obvious claim carries a source URL or the
exact API/RPC call that demonstrates it.

Confidence markers:

| Marker | Meaning |
|---|---|
| **[LIVE]** | Verified by me against the live API / Polygon mainnet on 2026-07-26. Evidence shown. |
| **[SRC]** | Verified by reading contract or SDK source. |
| **[DOC]** | Stated in official docs; not independently confirmed. |
| **[?]** | Unknown / could not establish. Not papered over. |

> **Scope note.** This is a research document. It deliberately does not change any
> analysis code, config, or frozen artifact. Where a mechanic invalidates an existing
> conclusion it is flagged in [§12](#12-assumptions-this-repo-has-made-that-are-wrong)
> and left for the owner to action.

---

## 1. Summary table — the key numbers

| Quantity | Value | Confidence |
|---|---|---|
| **Taker fee formula** | `fee = C × rate × p × (1−p)`, C = shares, p = price | **[LIVE]** exact to 7 s.f. on-chain |
| **Taker fee rate** | Crypto **0.07** · Sports **0.05** · Politics/Finance/Tech/Mentions **0.04** · Economics/Culture/Weather/Other **0.05** · **Geopolitics 0** | **[LIVE]** |
| **Maker fee** | **Zero.** `takerOnly: true` everywhere sampled | **[LIVE]** 89/90 txs, one fee'd leg each |
| **Fee rounding** | floor/truncate to 5 dp; min charge 0.00001 USDC | **[DOC]** |
| **Fee field to read** | `feeSchedule.{rate,exponent,takerOnly,rebateRate}` (Gamma) | **[LIVE]** |
| **Fee field to IGNORE** | `taker_base_fee` / `maker_base_fee` = **1000 = the contract ceiling**, not a rate | **[LIVE]+[SRC]** |
| **Fee collector** | `0x115f48dc2a731aa16251c6d6e1befc42f92accc9` | **[LIVE]** |
| **Min tick size** | Per-market config. Live observed: **0.001** (72%) and **0.01** (28%). Docs list 0.1/0.01/0.005/0.0025/0.001/0.0001 | **[LIVE]** 220/220 static==live |
| **Min order size** | **5 shares** on 470/470 live order-book markets (historically 15, and 0) | **[LIVE]** |
| **Price bounds** | `[tick, 1 − tick]` — a clamp, not a tick change | **[SRC]** |
| **Collateral (settlement)** | **pUSD** `0xC011a7E1…82DFB`, 6 dp (since 2026-04-28) | **[LIVE]** |
| **Collateral (`positionId` derivation)** | still **USDC.e** `0x2791Bca1…84174`; neg-risk uses **WCOL** | **[LIVE]** |
| **Outcome tokens** | ERC-1155 on Gnosis CTF `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`, 6 dp; 1 share = 1e6 units | **[LIVE]** |
| **Exchange (current)** | **CTFExchangeV2** `0xE111180000d2663C0091e4f400237545B87B996B`; NegRisk V2 `0xe2222d27…310F59` | **[LIVE]** 90/90 txs |
| **Gas paid by user** | **None.** An operator relayer submits and pays | **[LIVE]** |
| **UMA liveness** | **7200 s (2 h) default**; `customLiveness` of 600/900/1800 common | **[SRC]+[LIVE]** |
| **UMA bond** | Per-market **250 – 50,000** USDC (docs' "$750" is stale) | **[LIVE]** |
| **Undisputed resolution** | ~2 h after proposal | **[DOC]** |
| **Disputed → DVM** | 2–6 days (24 h commit + 24 h reveal + debate) | **[DOC]** |
| **50/50 payout** | `[1,1]` payout vector → each token redeems **$0.50** | **[SRC]** |
| **`end_date_iso`** | ⛔ **A DATE, not a time.** Midnight-floored. Do not use as a boundary | **[LIVE]** |
| **Rate limits (public)** | Gamma `/markets` 300/10 s · data-api `/trades` 200/10 s · CLOB `/book` 1500/10 s | **[DOC]** |

---

## 2. The instrument — what a "share" actually is

**A share is an ERC-1155 conditional token, not a contract-for-difference.** **[LIVE]+[SRC]**

- Outcome tokens live on the **Gnosis Conditional Tokens Framework (CTF)** at
  `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` on Polygon. Confirmed live: every
  trade transaction I decoded emits `TransferSingle` events from that contract.
- Each binary market has a `conditionId` and **two** `tokenId`s — YES and NO. Gamma
  returns them in `clobTokenIds` as a **JSON-encoded string** that must be
  double-parsed: `"[\"6932431…\", \"5179715…\"]"`. Index 0 = YES, index 1 = NO. **[LIVE]**
- **Split / merge:** `splitPosition` converts 1 USDC → 1 YES + 1 NO; `mergePositions`
  converts 1 YES + 1 NO → 1 USDC. This is what enforces `p_yes + p_no = 1`. **[SRC]**
- **Settlement** pays in collateral: a winning share redeems for **$1**, a losing
  share for **$0**. Both USDC and the outcome tokens use **6 decimals**, so "1 share"
  is `1_000_000` base units.

### The collateral changed — the repo probably doesn't know this

On **2026-04-28** Polymarket migrated the user-facing collateral from USDC.e to **pUSD**
(`CollateralToken`, `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB`), an ERC-20 wrapped 1:1
over USDC. **[LIVE]**

Evidence — decoded receipt of tx `0x9ddaccdf…82343d`:

```
0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb  Transfer x24   <- pUSD, the settlement asset
0x2791bca1f2de4661ed88a30c99a7a9449aa84174  Transfer x2    <- USDC.e, only the on/off-ramp
0x4d97dcd97ec945f40cf65f87097ace5ea0476045  TransferSingle x21  <- CTF outcome tokens
```

**But there are three distinct collateral answers depending on the layer, and conflating
them will break `tokenId` derivation:** **[LIVE]**

| Layer | Token | Address | Decimals |
|---|---|---|---|
| User balance / settlement | **pUSD** | `0xC011a7E1…82DFB` | 6 |
| **CTF `positionId` derivation (standard markets)** | **USDC.e** | `0x2791Bca1…84174` | 6 |
| CTF collateral (**neg-risk** markets) | **WCOL** (WrappedCollateral) | `0x3a3bd7bb…02e2` | 6 |

Verified against a market created on the research date: `getPositionId(USDC.e, …)`
reproduces the CLOB `token_id` exactly, while pUSD, native USDC and WCOL all fail to match.
Polymarket's own pUSD page claims it "settles all trading activity in native USDC" — that
prose contradicts its own code and live behaviour. **Trust the chain.**

### `tokenId` derivation **[SRC]+[LIVE]**

```
conditionId  = keccak256(oracle, questionId, outcomeSlotCount)
collectionId = getCollectionId(parentCollectionId, conditionId, indexSet)
positionId   = uint256(keccak256(collateralToken, collectionId))     # == CLOB token_id
```

- **`indexSet` is a bitmask, not an index**: for a binary market YES = `1`, NO = `2`.
- `collectionId` is an **alt_bn128 curve point**, not a hash — collections combine by
  elliptic-curve point addition, deliberately, to avoid birthday-attack collisions.
- **CLOB `token_id` == CTF ERC-1155 `positionId`.** Confirmed live.
- **1 share = 1,000,000 base units** (6 decimals). Confirmed: an API-reported
  `211.1111` shares reads `211111110` raw on-chain — a clean 1e6 ratio.

### When `p_yes + p_no = 1` does NOT hold

The identity is enforced by **arbitrage plus the mint/merge mechanic**, not by
construction. It breaks in the obvious places: across the bid-ask spread, on a
one-sided book, and — now — by the taker fee, which is charged on top of notional.

---

## 3. The order book — a real CLOB

**Confirmed: it is a central limit order book with off-chain matching and on-chain
settlement, not an AMM.** **[DOC]+[SRC]+[LIVE]**

Polymarket's docs call it "hybrid-decentralized": an operator matches orders off-chain,
settlement happens on-chain non-custodially
(<https://docs.polymarket.com/concepts/order-lifecycle>). The on-chain entry point is
`matchOrders(...)` guarded by `onlyOperator` — **only the operator can settle a match**
(<https://github.com/Polymarket/ctf-exchange-v2/blob/main/src/exchange/CTFExchange.sol>). **[SRC]**

The legacy `fpmm` field in CLOB `/markets` is a leftover from the old AMM era and is
an empty string on current markets. **[LIVE]**

### Order types **[SRC]**

`GTC`, `GTD`, `FOK`, `FAK` — from the SDK enum in
`py_clob_client/clob_types.py`. There is no separate IOC; **FAK is their IOC**.

- All orders are technically **limit** orders; a "market order" is a limit order priced
  to cross immediately, restricted to FOK/FAK, with `expiration = 0`. **[DOC]**
- `postOnly` is a separate boolean flag, not an order type. **[DOC]**
- **GTD trap:** GTD orders expire one minute *before* their stated expiration, and the
  expiration must be ≥3 minutes out. Minimum effective lifetime ≈2 minutes. **[DOC]**

### Matching **[SRC]**

`_deriveMatchType` in V2's `Trading.sol` resolves to three cases:

| Case | Condition | On-chain effect |
|---|---|---|
| **COMPLEMENTARY** | opposite sides, **same** tokenId | direct P2P transfer |
| **MINT** | both BUY, **complementary** tokenIds | collateral split into YES+NO |
| **MERGE** | both SELL, complementary tokenIds | YES+NO merged back to collateral |

Price improvement always accrues to the **taker**. **[DOC]**

### Tick size — how it actually varies

**Tick size is a per-market configuration value. It is NOT a function of the current
price at query time.** **[LIVE]**

I tested this directly: for 220 live markets I compared Gamma's static
`orderPriceMinTickSize` against the CLOB's live `GET /tick-size` and `GET /book`
`tick_size`:

```
static=0.001  live=0.001  ->  158
static=0.01   live=0.01   ->   62
disagreements: 0
markets with quotes off the live tick grid: 0/220
```

There **is** a strong correlation with price, but it is selection, not a rule — markets
that live at the extremes are *configured* with a finer tick:

```
LIVE tick vs price band (n=220)
  price [0,    0.01) -> {0.001: 60}
  price [0.01, 0.04) -> {0.001: 57}
  price [0.04, 0.10) -> {0.001: 24, 0.01: 22}
  price [0.10, 0.90) -> {0.01:  40, 0.001: 15}
  price [0.96, 0.99) -> {0.001:  2}
```

Note both ticks coexist in the mid-band, which rules out a pure price-band rule.

What *does* change near 0 and 1 is a **price clamp**, and this is very likely the origin
of the "tick changes at the extremes" folklore: **[SRC]**

```python
def price_valid(price: float, tick_size: TickSize) -> bool:
    return price >= float(tick_size) and price <= 1 - float(tick_size)
```

So on a 0.01-tick market valid prices are `[0.01, 0.99]`; on 0.001, `[0.001, 0.999]`.

**Tick size is also mutable at runtime** — the docs instruct integrators to listen for a
`tick_size_change` event, and the SDK caches it with a 300 s TTL. Never hardcode it. **[DOC]+[SRC]**

**Docs list six tick sizes** (0.1, 0.01, 0.005, 0.0025, 0.001, 0.0001) but **the SDKs
only type four** (`Literal["0.1","0.01","0.001","0.0001"]`) — 0.005 and 0.0025 markets
will not round correctly through the legacy SDK. **[SRC]** In a 30,000-market historical
sweep I also found two genuine oddities: `minimum_tick_size` of **0.02** and **0.04**. **[LIVE]**

### Minimum order size

**5 shares** on 470/470 live order-book markets. **[LIVE]** Historical sweep of 30,000
markets: `5` (25,545), `15` (4,105), `0` (346), plus junk values `14`, `14.96`, `14.97`, `25`.

⚠️ **The docs contradict themselves on the units.** The `/book` reference says
`min_order_size` is "the minimum number of **shares**"; the market-details page says
Gamma's `orderMinSize` is "Minimum order size in **USDC**". **[?]** — unresolved.

### Buying NO vs selling YES

**Mechanically the NO book is the exact mirror of the YES book, reflected at 1.0.** **[LIVE]**

Live proof — `will-kim-kardashian-win-the-2028-democratic-presidential-nomination`,
both tokens fetched simultaneously:

```
YES token: bids [(0.002, 1640195.9), (0.001, 2223085.8)]
           asks [(0.003,  573029.79), (0.004, 13000.0), (0.005, 393447.35), ...]

NO  token: bids [(0.997,  573029.79), (0.996, 13000.0), (0.995, 393447.35), ...]
           asks [(0.998, 1640195.9), (0.999, 2223085.8)]
```

Sizes match **exactly** at complementary prices. Buying NO at 0.998 *is* selling YES at
0.002. The engine crosses them via the MINT/MERGE paths above:
`BUY YES @ p` crosses `BUY NO @ q` iff `p + q ≥ 1`. **[SRC]**


### Self-trade prevention

**None found.** No STP in the docs, the OpenAPI spec, the error-code list, or the V2
`Trading.sol` (there is no `maker != taker` guard). **[?]/[SRC by absence]**

This is directly relevant to `manufactured_record_flag` in CLAUDE.md: **the protocol does
not prevent a wallet from trading against itself.** Self-dealt records are mechanically
possible, which supports keeping that flag as a live data-quality concern.

---

## 4. Negative risk / multi-outcome markets

- `neg_risk: true` marks a market as one leg of a mutually-exclusive multi-outcome event.
  **13,122 / 30,000** historical markets and **262 / 470** live order-book markets are
  neg-risk — it is the majority case for live markets, not an edge case. **[LIVE]**
- Each candidate/leg is **its own binary market with its own `conditionId`**, grouped by
  `neg_risk_market_id`. `neg_risk_request_id` identifies the originating request. **[LIVE]**
- Neg-risk markets settle through a **separate exchange contract**, `NegRiskCTFExchangeV2`
  `0xe2222d279d744050d28e00520010520000310F59` (8 of my 90 decoded txs). **[LIVE]**
- Legs are linked by an ID relation, not just a shared field: `question_id = neg_risk_market_id
  + question_index`, where `marketId = keccak256(oracle, feeBips, metadata) & ~0xff`. Every
  `neg_risk_market_id` therefore ends in `00`. **[SRC]+[LIVE]** The `uint8` index caps a
  neg-risk event at **256 outcomes**.
- ⭐ **`neg_risk_market_id` is the correct event-level clustering key** — better than
  Gamma's `eventSlug`, which was an empty string in every `data-api/trades` record sampled. **[LIVE]**
- `neg_risk_request_id` is the **UMA request id, unique per outcome**, not per event. Do not
  cluster on it. **[LIVE]**

### The convert mechanic, exactly **[SRC]**

```solidity
function convertPositions(bytes32 _marketId, uint256 _indexSet, uint256 _amount) external
```

> "If the market has `n` questions, and the user converts `_amount` of `m` NO tokens, the
> user receive `_amount * (m-1)` collateral and `_amount` each of the complimentary YES tokens."

The NO tokens are burned and the adapter synthetically mints **WrappedCollateral** to back
the resulting YES tokens — which is *why* neg-risk markets are collateralised in WCOL. The
popular simplification "a NO share converts into 1 YES share in every other market" is just
the `m = 1` special case where the collateral term is zero.

This is the arbitrage that pins Σ(YES) near 1: minting a complete YES set costs exactly $1
(split $1 on each of `n` legs, then convert the full index set to burn all `n` NOs and
recover `n−1` collateral).

**The sum is only approximately 1, and deviations are large.** **[LIVE]** Live sums of
YES prices across event legs:

| legs | Σ YES | event |
|---:|---:|---|
| 51 | 1.0040 | democratic-presidential-nominee-2028 |
| 35 | **0.9265** | presidential-election-winner-2028 |
| 15 | **1.0695** | lol-lpl-2026-season-winner |
| 10 | 1.0190 | lol-lck-2026-season-winner |
| 8 | 0.9810 | how-many-gold-cards-will-trump-sell-in-2026 |
| 7 | **0.8950** | min-arctic-sea-ice-extent-this-summer |
| 6 | 1.0365 | harvey-weinstein-prison-time |

Deviations run to ±10%. Any analysis that assumes legs sum to 1 (or treats each leg as an
independent bet) is mis-specified. **Losing legs resolve NO and pay $0**; the winning leg
pays $1. The contracts *require* exactly one YES: a second YES reverts `reportOutcome` and
can leave the event unresolvable. **[SRC]**

### ⭐ Why Σ(YES) > 1 is usually NOT free money — and why this corroborates the arb NO-GO

Measured live on the 5-outcome `fed-decision-in-july-181` event: **Σ best YES bids = 1.0100,
Σ best YES asks = 1.0150.** That looks like a 1¢ arbitrage. It is not. At the Economics
taker rate 0.05, the fee to lift all five legs is `0.05 × Σ p(1−p) ≈ 0.0165` — **~1.65¢ of
fees against a 1.0¢ apparent edge.** **[LIVE]**

> **The no-arb band is fee-wide.** The raw sum of legs routinely exceeds 1 without being
> exploitable. This gives a *mechanism* for the repo's already-recorded
> "cross-market arb tested live = NO-GO" (commit `d9af3fd`): the screen was seeing the
> width of the fee band, not mispricing. Any future arb screen must net the per-leg fee
> before declaring an edge.

⚠️ **A 50/50 tie is NOT a valid outcome under the NegRiskOperator** — the adapter
**reverts** on a `[1,1]` UMA outcome. **[SRC]** Behaviour/recovery in that case is **[?]**.
Also documented: if all questions resolve NO, a converted position is worth *less* than the
original.

---

## 5. Fees — RESOLVED

This was the repo's biggest open question. It is now closed, from docs **and** on-chain.

### The formula **[DOC]+[LIVE]**

<https://docs.polymarket.com/trading/fees> (fetch as `.md`; the HTML URL returns a JS shell):

```
fee = C × feeRate × p × (1 - p)
```

> **Makers are never charged fees.** Only takers pay fees.
> Fees are rounded to 5 decimal places. The smallest fee charged is 0.00001 USDC.

**The repo's formula shape `shares × k × price × (1−price)` is CORRECT.** What was wrong
is `k`: the repo used `DEFAULT_TAKER_FEE_K = 0.0` as the base case.

### The rates **[LIVE]**

Live `feeSchedule` from Gamma across 300 live markets, cross-tabulated by `feeType`:

| `feeType` | `rate` | n |
|---|---:|---:|
| `politics_fees` | 0.04 | 195 |
| `sports_fees_v2` | 0.05 | 25 |
| `general_fees` | 0.05 | 14 |
| `tech_fees` | 0.04 | 11 |
| `crypto_fees_v2` | 0.07 | 8 |
| `culture_fees` | 0.05 | 7 |
| `weather_fees` | 0.05 | 7 |
| `economics_fees` | 0.05 | 5 |
| *(null — geopolitics)* | `feesEnabled: false` | 28 |

`feeSchedule` shape: `{"exponent": 1, "rate": 0.04, "rebateRate": 0.25, "takerOnly": true}`.
The `exponent` reshapes the parabola; it is **1** everywhere sampled.

### On-chain proof the fee is real **[LIVE]**

I decoded 90 live transactions from the public tape and solved for
`k = fee / (shares × p × (1−p))`:

```
implied k, rounded to 3dp:
  0.070 -> 73    0.050 -> 7    0.040 -> 3
  (0.049, 0.069, 0.067, 0.062 -> 1-2 each; artifacts of the tape's rounded VWAP price)
  0.000 -> 1     (geopolitics / fee-free)
```

Worked example, tx `0x9ddaccdf…82343d` (`btc-updown-5m`, crypto, rate 0.07):

```
shares = 580.677591, price = 0.527058567
predicted: 580.677591 × 0.07 × 0.527058567 × 0.472941433 = 10.132090
on-chain fee event data                                  = 10132090 (6 dp) = 10.132090  ✓
```

Exact to 7 significant figures.

**Maker legs pay zero, confirmed:** in **89 of 90** transactions exactly one `OrderFilled`
leg carried a nonzero fee; **zero** transactions had more than one. That is `takerOnly: true`
demonstrated empirically.

### `taker_base_fee: 1000` — what it actually is **[SRC]+[LIVE]**

It is the **protocol's maximum signable fee rate**, in basis points out of 10,000 — i.e.
10% — **not the charged rate.** From V1 `Fees.sol`:

```solidity
uint256 internal constant MAX_FEE_RATE_BIPS = 1000; // 1000 bips or 10%
```

Three independent proofs it is a ceiling:
1. It is identically `1000` across categories whose real rates are 0.04, 0.05 and 0.07. **[LIVE]**
2. `maker_base_fee` is also `1000`, yet makers demonstrably pay **zero**. **[LIVE]**
3. `1000` is verbatim `MAX_FEE_RATE_BIPS`. **[SRC]**

There is an **open Polymarket GitHub issue** on exactly this confusion:
<https://github.com/Polymarket/py-clob-client/issues/326>.

**Read `feeSchedule`. Never read `*_base_fee`.**

### When fees started **[DOC]**

Per the official changelog (<https://docs.polymarket.com/changelog/predictions>):
**2026-01-05** first taker fees (15-min crypto) → **2026-03-30** "Fee Structure V2"
extends to nearly all categories → **2026-07-10** sports rate raised 0.03 → 0.05.

**Polymarket was genuinely zero-fee before 2026-01-05.** This is why the repo's historical
ledger shows no fees and why its conclusions are *gross* numbers.

### V1 vs V2 — a live discrepancy worth knowing **[SRC]**

The **archived V1** contract computed fees on-chain as
`baseRate × min(p, 1−p) × shares` — a *tent*, not the parabola. **V2 removed
price-dependent fee computation from the contract entirely**: `Fees.sol` now only
*validates* that an operator-supplied fee doesn't exceed `maxFeeRateBps` (default 500 =
5%), and `feeRateBps` was dropped from the signed order struct. Fees are now
**operator-set at match time**. The `p(1−p)` parabola is enforced by the operator, not the
contract — so it can change without a contract upgrade. Track the docs, not the source.

### Gas, deposits, withdrawals **[LIVE]+[DOC]**

- **The user pays no gas.** In every decoded transaction the `from` address was an
  operator relayer, not either trader (e.g. `0xb420249f9b7584ce0f0bf5d79c3587ec07f867c0`,
  gasUsed 2,304,499). Polymarket's relayer pays POL.
- **Exception:** allow-listed EOA traders submit directly and *do* pay their own gas. **[DOC]**
- **No Polymarket deposit or withdrawal fee**; third-party on-ramps may charge their own. **[DOC]**
- **No redemption/settlement fee** found. **[DOC by absence]**

### Rebates (they partially offset the above) **[DOC]**

- **Maker rebates**: funded by taker fees, paid daily in pUSD, $1 minimum. Crypto 20%,
  Sports 15%, everything else 25% (`feeSchedule.rebateRate`).
- **Taker rebates**: tiered by trailing-30-day weighted volume, live since 2026-05-28.
- **Liquidity rewards**: the `rewards` field (`rates`, `min_size`, `max_spread`); quadratic
  scoring on spread from midpoint, paid daily at midnight UTC.

---

## 6. Resolution

### The oracle **[SRC]**

Markets resolve via **UMA's OptimisticOracleV2** through Polymarket's `UmaCtfAdapter`.
Lifecycle: `initialize` → `prepareCondition` + `requestPrice` → a proposer posts an answer
with a bond → **liveness window** → if undisputed, anyone calls the *permissionless*
`resolve(questionID)` which does `settleAndGetPrice` then `ctf.reportPayouts`.

**Nothing resolves itself.** There is unbounded, variable human/bot latency between
"price available" and "market resolved on CTF".

### Timings

| Phase | Duration | Confidence |
|---|---|---|
| Event → proposal | **unbounded**, permissionless, no SLA | **[?]** |
| Liveness | **7200 s (2 h) default** | **[SRC]** `defaultLiveness()` on-chain = 7200 |
| — but `customLiveness` overrides are common | 600 / 900 / 1800 s | **[LIVE]** |
| 1st dispute | re-request + a **full fresh liveness** (does NOT go to DVM) | **[SRC]** |
| 2nd dispute | escalates to UMA **DVM** | **[SRC]** |
| DVM commit + reveal | 24 h + 24 h, needs 65% of staked UMA | **[DOC]** |
| Disputed total | **2–6 days** | **[DOC]** |
| `resolve()` → redeemable | permissionless, unbounded | **[SRC]** |

**Bond is per-market and the docs are stale.** Docs say "$750"; live `umaBond` values are
**250, 500, 25000, 50000** — a 200× range, and *no* sampled market had 750. **[LIVE]**

### Outcome encoding (UMIP-107) **[SRC]**

| Key | Meaning | Value |
|---|---|---|
| `p1` | **NO** | 0 |
| `p2` | **YES** | 1 |
| `p3` | **cannot be determined** → 50/50 | 0.5 |
| `p4` | **too early** — re-requests, never pays out | `type(int256).min` |

Payout array order is `[YES, NO]`. A `p4` return silently resets the request, so **a market
can sit in a "too early" loop indefinitely with no resolution and no error signal.**

### The 50/50 case **[SRC]**

Payout vector `[1,1]` instead of `[1,0]`. In Gnosis `ConditionalTokens.sol`,
`payoutDenominator` is the **sum** of numerators (= 2), and `redeemPositions` pays
`stake × numerator / denominator` = **$0.50 per token**, for both YES and NO. Collateral
stays fully backed.

`reportPayouts` requires `payoutDenominator == 0`, so **resolution is one-shot and
irreversible on-chain.** A wrong resolution cannot be corrected.

### Redemption **[SRC]+[DOC]**

**There is no automatic payout.** Winning tokens sit in the wallet until `redeemPositions`
is called — via a **Claim** button in the UI, or an **opt-in** auto-redeem toggle.

> **Consequence for wallet analytics:** unredeemed winnings are invisible as USDC. Any PnL
> computed from observed USDC flows systematically lags true PnL by an amount that depends
> on a per-wallet setting. **Score on resolution outcome, never on observed USDC receipt.**

### Field semantics — which flags are reliable

| Field | What it actually means | Reliable? |
|---|---|---|
| `end_date_iso` (CLOB) | ⛔ a **calendar date** rendered as `T00:00:00Z` | **NO** |
| `endDateIso` (Gamma) | ⛔ bare date, **492/492 date-only** in my sample **[LIVE]** | **NO** |
| `endDate` (Gamma) | full timestamp, but **87% are still midnight UTC**; **11% already in the past** for live markets **[LIVE]** | weak |
| `gameStartTime` | full timestamp, sports only. **The reliable sports event anchor** | **YES** |
| `active` | only "deployed and not archived" — **NOT** "tradeable" | **NO** |
| `closed` | resolved/closed. Reliable-positive; `closed=False` is *not* evidence of live | one-way |
| `accepting_orders` | documented as "book is open" — but ~28% of *closed* markets still report `True` | **NO** |
| `enable_order_book` | effectively an era flag (CLOB vs legacy AMM) | context |
| `umaEndDate` | **the actual resolution timestamp** (== `closedTime`, 300/300) despite the name | **YES** |
| `umaResolutionStatus` | `proposed` / `disputed` / `resolved` — best resolution-stage signal | **YES** |
| `resolvedBy` | the **adapter contract address**, not a person | **YES** |
| `resolutionSource` | frequently an empty string; real criteria live in `description` | low value |

Polymarket's own prescribed liveness test is
`active && !closed && acceptingOrders` — and even that is unreliable per an
[open bug report](https://github.com/Polymarket/rs-clob-client/issues/199).

⚠️ **Format gotcha:** `gameStartTime` comes back as `'2025-12-19 23:40:00+00'` — a space
separator and a `+00` offset, **not** the `...T...Z` used by every other timestamp field. **[LIVE]**

### Resolution risk is real and systemic **[NEWS]**

Notable disputes: the **Zelenskyy suit** market (~$237M volume, July 2025, flipped YES→NO
after nine days amid whale-rigging allegations); the **Ukraine mineral deal** market
(March 2025, a UMA whale using three accounts cast ~25% of the vote; Polymarket called it
"unprecedented" and **refused refunds**); the **MicroStrategy BTC sale** dispute (2026).

> **This is a confound for this repo's core metric.** The ground-truth label is a
> token-weighted vote outcome, not reality, and reporting suggests substantial overlap
> between UMA voters and Polymarket traders. A wallet showing persistent edge on large,
> disputable markets may be measuring **governance influence, not forecasting skill** — and
> that would survive out-of-sample validation exactly the way the favorite-longshot effect
> did. Per CLAUDE.md's additive-metadata rule, a non-destructive `resolution_dispute_flag`
> from `umaResolutionStatus == 'disputed'` would let the user filter this without dropping
> any row.

---

## 7. Liquidity and microstructure **[LIVE]**

Measured across 260 live order-book markets by walking the real ask side. The depth
and market-impact tables are not reproduced here; they are about the cost of acting,
which this repo does not do. The one point that bears on measurement is that depth
varies by orders of magnitude between the deepest in-play sports lines and the median
market, so **any statement about book depth has to be per market**. Both the original
assumption in this repo ("a $100 order moves the price") and its blanket correction
were wrong for the same reason: they aggregated.

Liquidity is provided by independent market makers incentivised via the **Liquidity
Rewards** program plus the **maker rebate** funded from taker fees. There is no
protocol AMM.

---

## 8. How `/trades` actually reports fills — critical for copy-trading **[LIVE]**

This is the single most important section for anything wallet-following.

### One tape row = one taker's aggregated fill across many maker orders

Decoding tx `0x9ddaccdf…82343d`, which appears as **exactly one row** in
`data-api.polymarket.com/trades`:

```
21 × OrderFilled events   (20 maker legs + 1 taker leg)
 1 × OrdersMatched
tape shows: ONE row, size=580.677591, price=0.527058567
```

Across 90 decoded transactions, `OrderFilled` events per tape row:

```
2 -> 60    3 -> 21    4 -> 4    5 -> 2    6 -> 1    7 -> 1    8 -> 1
```

**Every tape row aggregates 2–8+ on-chain fills.**

### The reported `price` is a size-weighted average, not a book level

Consequence: **34% of tape prices are off the tick grid** (506/1500 sampled):

```
0.6299999915   0.4588865729   0.6378737409   0.2089070022   0.9399999624
```

`price` is effectively `usdcSize / size`. It is a **VWAP that already includes the
slippage** of walking the book. It is the taker's realized average, not a quote.

### Market BUY orders are denominated in USDC, not shares

Explains the odd non-round sizes: `size=4.166665, price=0.239999856` → exactly **$1.00**.
`size=14.285713 @ 0.07` → exactly **$1.00**. Confirmed by the SDK: `MarketOrderArgs.amount`
is documented as "BUY orders: $$$ Amount to buy".

### The tape has no fee field

Keys are exactly:
`proxyWallet, side, asset, conditionId, size, price, timestamp, title, slug, icon,
eventSlug, outcome, outcomeIndex, name, pseudonym, bio, profileImage,
profileImageOptimized, transactionHash`.

**Fees must be modelled from `feeSchedule` or decoded on-chain.** And since fees are charged
*on top of* notional for a market BUY, the reported `price` **understates the true cost**.
For the example above: 306.05 notional + 10.13 fee over 580.68 shares → true all-in
**0.5445 vs a reported 0.5271 — a 1.74¢ gap**, which is the same order of magnitude as
every edge this repo has been measuring.

### Other fields

- `outcomeIndex` is **unreliable in `/trades`**: values were `0` (253), `1` (172), and a
  sentinel **`999`** (75) in a 500-row sample. `/activity` returns proper 0/1.
- `/activity` is strictly richer than `/trades`: it adds **`type`** (`TRADE`, …) and
  **`usdcSize`** (true USDC notional). Prefer it.
- `/positions` exposes `redeemable`, `mergeable`, `negativeRisk`, `oppositeAsset`,
  `realizedPnl`, `avgPrice`.

---

## 9. API limits **[DOC]**

> ⚠️ **Analytical consequence worth recording: some order flow is invisible on-chain.**
> Polymarket US (QCX LLC) is a separate product from `polymarket.com` — CFTC-designated,
> **fiat-based, off-chain, no wallets, no CTF, no Polygon**, with accounts explicitly not
> connected to the on-chain venue. If flow migrates there, the on-chain tape thins with no
> visible on-chain cause: a silent regime change of exactly the kind Metric D's
> `regime_flag` exists to catch.

- **Rate limits (public, documented) [DOC]:**

  | API | limit |
  |---|---|
  | Gamma `/markets` | 300 req / 10 s |
  | Gamma general | 4,000 req / 10 s |
  | data-api `/trades` | 200 req / 10 s |
  | data-api `/positions` | 150 req / 10 s |
  | CLOB `/book` | 1,500 req / 10 s |
  | CLOB `/prices-history` | 1,000 req / 10 s |

- ⚠️ **A new per-signer token-bucket limiter is in warning mode now and enforces from
  ~2026-08-07.** Standard tier: 40 orders/s, burst 60. Watch the `Poly-RateLimit-*`
  response headers. **[DOC]**

---

## 10. Contract addresses (current)

| Contract | Address | Confidence |
|---|---|---|
| CTF Exchange **V2** | `0xE111180000d2663C0091e4f400237545B87B996B` | **[LIVE]** |
| NegRisk CTF Exchange **V2** | `0xe2222d279d744050d28e00520010520000310F59` | **[LIVE]** |
| Conditional Tokens (CTF) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | **[LIVE]** |
| pUSD CollateralToken | `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB` | **[LIVE]** |
| USDC.e (underlying) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | **[LIVE]** |
| Fee collector | `0x115f48dc2a731aa16251c6d6e1befc42f92accc9` | **[LIVE]** |

⚠️ **`py-clob-client` still hardcodes the V1 exchange addresses** and
`github.com/Polymarket/ctf-exchange` is archived. Both are stale. **[SRC]**

---

## 11. What would invalidate the copy conclusions

Ordered by severity.

1. **Fees are live and were not in the backtest.** Every historical edge in this repo is a
   **gross** number from a zero-fee era that ended 2026-01-05. Forward copying pays
   **1.00–1.75¢/share at mid prices** as a taker. Against measured edges of ~1–3¢ this is
   not a haircut, it is most of the thesis.
2. **The tape price is a post-slippage VWAP, and the fee sits on top of it.** A wallet's
   recorded entry price is not a price you could have gotten — it is their realized
   average *after* consuming the book, and their true cost was ~1.7¢ higher still.
   Copying at the same *quoted* price is not reproducing their trade.
3. **Fee-as-fraction-of-stake is largest at low prices**: `rate × (1−p)`. A 5¢ crypto
   longshot pays **6.65% of stake**; a 98¢ favorite pays **0.1%**. Any longshot-tilted
   strategy is taxed punitively; the insurance-shaped high-price band is the cheapest to
   trade.
4. **Resolution is a token-weighted vote, not reality**, with documented governance
   attacks and no refunds. Edge on large disputable markets may be governance influence.
6. **One human ≠ one `proxyWallet`.** A signer maps deterministically to exactly one wallet
   (no nonce in the CREATE2 salt), but a person can hold unlimited wallets across
   emails/logins/eras with nothing linking them. **A wallet is a pseudonymous account, not a
   trader**, and "distinct wallets" is not evidence of distinct decision-makers.
7. **A wallet's visible trades may not be its complete activity. [?]** ERC-1155 outcome
   tokens can be transferred wallet-to-wallet without ever touching the exchange, so a
   position can appear or vanish with no row in `/trades`.
8. **No self-trade prevention** — manufactured records are mechanically possible.
10. **A copier is structurally a taker** (you react to their fill), so you always pay the
    taker fee and never earn the maker rebate. The sharp wallet may well be a *maker* —
    earning the rebate — in which case its edge is partly an economics you cannot copy.
11. **Sports books auto-clear at game start**, and "if a game starts earlier than scheduled,
    orders may not be cleared in time." **[DOC]** Directly relevant to the sports arm's
    in-play caveat.

---

## 12. Assumptions this repo has made that are WRONG

| # | Assumption | Truth | Affected |
|---|---|---|---|
| 1 | **`DEFAULT_TAKER_FEE_K = 0.0`** — fees effectively zero | **False.** k = 0.04 (politics) / 0.05 (sports) / 0.07 (crypto), live since 2026-01-05, verified on-chain to 7 s.f. Only geopolitics is 0 | `src/paper_trader.py:138`; every net-of-cost conclusion |
| 2 | **"The fee constant k is UNPINNED"** | **Resolved.** Read `feeSchedule.rate` from Gamma per market | `src/paper_trader.py:1150-1153`, `docs/project3_sports_metric_b.md` §5 |
| 3 | **`taker_base_fee: 1000` is in "unexplained units"** | **It is `MAX_FEE_RATE_BIPS` = 10%, the protocol ceiling — not a rate.** Identical across all categories; `maker_base_fee` is also 1000 while makers pay zero | `src/paper_trader.py:558-559` |
| 4 | **"Gamma reports `fee: None`"** ⇒ no fee | **Wrong field.** `fee` isn't the fee field; `feesEnabled` + `feeSchedule` are. `fee: None` ≠ fee-free | `docs/project3_sports_metric_b.md:313` |
| 5 | **"Tick size is 0.001, not 0.01"** (stated as a global) | **Neither is global.** Per-market config; live split is 0.001 (72%) / 0.01 (28%), and **22/24 live sports markets use 0.01**. Docs list six legal values incl. 0.005/0.0025 | `docs/project4_value_gate.md:130` |
| 6 | **"The fee curve vanishes at 100¢"** ⇒ costs are negligible there | **True in absolute USDC, misleading as a fraction of stake.** Fee/stake = `rate × (1−p)`, which is *largest at low prices*. Directionally the Project 4 conclusion survives; the magnitude reasoning does not | `docs/project4_value_gate.md:144`, Project 4 NO-GO |
| 7 | **"Costs take ~25% of the edge"** | Recompute — that used k≈0. At k=0.04–0.07 the entry fee alone is 1.00–1.75¢/share at mid | `docs/project4_value_gate.md:26` |
| 8 | **`end_date_iso` is a real end time** | **Already caught, and confirmed here.** It is a bare calendar date; CLOB appends `T00:00:00Z`. **492/492** live `endDateIso` are date-only. Even `endDate` is midnight 87% of the time and already past for 11% of live markets | `src/market_meta.py`, `src/paper_trader.py` (already handled) |
| 9 | **`price` in the tape is an entry price** | **It is a size-weighted average across 2–8+ fills**, already including slippage. 34% of prints are off the tick grid because of it | all of `features.py` / edge computation |
| 10 | **One trade row = one trade** | **One row = one taker's aggregated match**, 2–8+ on-chain fills. Maker-side activity is not separately visible | ingest, sample-size and clustering assumptions |
| 11 | **Settlement is in USDC** | Settlement now moves **pUSD**, a wrapper. Raw USDC.e appears only at the on/off-ramp | any on-chain USDC-flow reasoning |
| 12 | **V1 exchange contracts** | **CTFExchangeV2** is live (`0xE1111800…`), V1 repo archived, `py-clob-client` addresses stale | any contract-address assumption |
| 13 | **Ranking treats neg-risk legs as independent binaries** | 262/470 live markets are neg-risk; leg YES prices sum to 0.895–1.07, not 1. Legs are mechanically coupled and `question_id = neg_risk_market_id + index`. **`neg_risk_market_id` is the right event-level cluster key** — the repo's breadth/clustering may be counting one event as many | breadth, cluster counts, `validate.py` cluster-robust split |
| 14 | **A `proxyWallet` is a trader** | It is a smart-contract proxy, not the signing EOA. One signer → one wallet, but **one human → unlimited wallets**. A wallet is a pseudonymous account, not a person | the unit of analysis everywhere |
| 16 | **`eventSlug` groups markets into events** | It was an **empty string** in every `data-api/trades` record sampled. Use `neg_risk_market_id`, or Gamma's `events[0].slug` | any event-level grouping |

**The single highest-impact correction is #1 + #9 together:** the backtest measured gross
edge at a price that was already a post-slippage average, in a zero-fee era, and forward
copying pays 1.0–1.75¢/share on top. That is the same order of magnitude as every edge the
repo has found.

---

## 13. Open questions

Honest unknowns. Do not treat any of these as settled.

1. **Can distinct `proxyWallet`s be linked to one human?** *Partly resolved:* one signer maps
   to exactly one wallet (deterministic CREATE2, no nonce), so a login cannot fan out.
   But **nothing bounds how many logins one person creates**, and no public linkage exists.
   **Unresolved: what fraction of the ranked wallets are independent decision-makers.**
2. **Are a wallet's visible `/trades` its complete activity?** ERC-1155 outcome tokens can
   be transferred wallet-to-wallet without touching the exchange, and would not appear in
   the tape. **Unquantified — and measurable**, by diffing CTF `TransferSingle` events
   against the tape for a sample of ranked wallets. Worth doing.
3. **`min_order_size` units — shares or USDC?** Two official docs pages contradict each
   other. Live value is `5` either way; the ambiguity only bites at extreme prices.
4. **`is_50_50_outcome`** — in the API schema with **no description anywhere**. 66/30,000
   markets have it `true`. Meaning unknown.
6. **Maker-vs-taker attribution in the tape.** Since each row is a taker aggregate, can a
   wallet's *maker* fills be recovered at all from public data? If sharp wallets are
   predominantly makers, the entire dataset may be biased toward their aggressive trades.
7. **`/fee-rate` endpoint** returns 403 live and its doc example (`base_fee: 30`)
   matches neither the docs nor the wild. Appears legacy post-V2.
8. **Behaviour of a 50/50 tie under the NegRiskOperator**, which the adapter source says
   is not a valid outcome.
9. **The dominant UMA adapter** (`0x65070BE9…`, used by most current markets) points at an
   **undocumented proxied OOv2** (`0x2c0367a9…`) rather than the canonical one. Governance
   and provenance unknown.
10. **Exact historical fee timeline per market.** Backtests spanning 2026-01-05 → 2026-03-30
    straddle a partial rollout; `feeSchedule` reflects *today*, not the fee in force at the
    time of each historical trade. **[?]** whether historical fee state is recoverable at all.
11. **A third settlement contract exists.** `scripts/verify_mechanics.py --onchain` observed
    a trade settling through `0xe3333700ca9d93003f00f0f71f8515005f6c00aa`, which is neither
    CTFExchangeV2 nor NegRiskCTFExchangeV2 and is not in the docs' contract table. Likely the
    Combos / `NegRiskModule` path that supersedes the deprecated neg-risk adapter, but
    **unconfirmed.** Anything keying on exchange address must not assume only two.

---

## 14. How to check things yourself

```bash
# tick size, min order size, neg_risk, live book
curl -s "https://clob.polymarket.com/book?token_id=<TOKEN_ID>"
curl -s "https://clob.polymarket.com/tick-size?token_id=<TOKEN_ID>"

# the ONLY correct fee source
curl -s "https://gamma-api.polymarket.com/markets?condition_ids=<CONDITION_ID>" \
  | python3 -c "import json,sys; m=json.load(sys.stdin)[0]; \
    print(m['slug'], m.get('feesEnabled'), m.get('feeType'), m.get('feeSchedule'))"

# the tape (note: price is a VWAP, no fee field, one row = many fills)
curl -s "https://data-api.polymarket.com/trades?limit=5"
curl -s "https://data-api.polymarket.com/activity?user=<PROXY>&limit=5"   # richer: usdcSize, type

# ground-truth a fee on-chain
#   fee word = index 4 of the taker's OrderFilled data; or the dedicated fee event
#   topic 0x55bb3cade9d43b798a4fe5ffdd05024b2d7870df53920673bfc7e68047cd0ab1
curl -s -X POST https://polygon.drpc.org -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"eth_getTransactionReceipt","params":["<TXHASH>"]}'
```

Docs are served as clean markdown by appending `.md` to any docs URL — the HTML URL
returns a JS shell. Index: <https://docs.polymarket.com/llms.txt>.

Primary sources used: <https://docs.polymarket.com/trading/fees> ·
<https://docs.polymarket.com/concepts/order-lifecycle> ·
<https://docs.polymarket.com/concepts/resolution> ·
<https://docs.polymarket.com/market-data/market-details> ·
<https://docs.polymarket.com/changelog/predictions> ·
<https://github.com/Polymarket/ctf-exchange-v2> ·
<https://github.com/Polymarket/ctf-exchange> (archived) ·
<https://github.com/Polymarket/py-clob-client> ·
<https://github.com/Polymarket/uma-ctf-adapter> ·
<https://github.com/gnosis/conditional-tokens-contracts> ·
<https://github.com/UMAprotocol/UMIPs/blob/master/UMIPs/umip-107.md>
