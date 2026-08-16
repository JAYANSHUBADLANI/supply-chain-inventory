# Progress

All five phases complete. Every number in the README and the case write-up comes from a
full clean pipeline run (`rm -rf data/interim data/processed reports/` then `make all`).

## Done

**Phase 1: cleaning and the data quality audit**

- Raw extract read as `latin-1`, 180,519 rows, 53 columns confirmed against the file.
- Exclusion ledger reconciling row for row: 7,754 cancelled or suspected fraud, 8,193 past
  the structural break, 164,572 kept. The ledger asserts in code and the pipeline fails if
  it does not balance.
- Clean window fixed at 2015-01-01 to 2017-09-30, 144 weeks, 100 products.
- Lead-time audit established the column is generated: First Class has zero realised
  variance across 27,814 rows, Second and Standard Class are discrete uniform over {2..6}
  and statistically indistinguishable from each other (chi-square p = 0.295) despite
  scheduling two days and four days, and region explains 0.05% of variance (eta squared
  0.000503).
- Structural break located at 2017-10: order lines halve, catalogue turns over, quantity
  collapses to a constant 1 from November 2017.
- Market time-slicing found: no market is active in more than 41.7% of weeks, which ruled
  out product by market as a planning grain.
- Aggregate demand autocorrelation at most 0.051 in absolute value across lags 1 to 4.
- `Late_delivery_risk` shown to be derived, 97.5% reproducible from real versus scheduled.

**Phase 2: demand, lead time, segmentation**

- Empirical weekly demand statistics with active-span handling, so launch and delist weeks
  are excluded rather than counted as zero demand.
- Demand exclusion ledger: 100 units seen, 46 thin sample, 14 intermittent, 40 carried,
  covering 95.7% of unit demand and 97.1% of revenue.
- Lead-time parameters carry an `is_proxy` flag and a source string through every
  downstream consumer, so the substitution cannot be lost.
- ABC by revenue (80/95 cuts), XYZ by weekly CV (0.25/0.50 cuts). Six of nine cells
  populated. Concentration profile records that 7 of 100 units carry 80% of revenue against
  the textbook expectation of 20%, and the degeneracy check flags that the A tier sits
  entirely in one XYZ class.

**Phase 3: policy**

- Daily time base chosen after confirming complete daily coverage, no day of week effect
  (1.018 ratio between highest and lowest weekday mean) and weekly sigma at 0.947 of the
  independent-draws prediction.
- Combined safety stock, reorder point and EOQ for all 40 carried units.
- Variance decomposition is the headline: lead time drives 87.9% of variance on AX and
  10.1% on CZ, so the demand-only simplification understates by 2.87x on the former and
  1.05x on the latter.
- Sensitivity sweep across sigma_L from 0 to 2x. A fixed lead-time assumption holds 44% of
  the required buffer, and that conclusion survives the whole range.
- Bootstrap against the closed form, 20,000 draws. The normal approximation over-buffers AX
  by 30% at 99.9% and under-buffers CZ by 33%, in opposite directions and for different
  reasons.
- Service level curve for one representative unit per segment, plus the differentiation
  basis (buffer cost per 1,000 of revenue protected) after finding the curve knee is
  scale invariant and cannot differentiate segments on its own.

**Phase 4: backtest**

- Fit on 724 days, simulate on a held-out 280 days, first 30 excluded as warmup.
- Continuous review (s, Q), daily steps, lost sales, common random numbers across all three
  policies so no policy wins on a luckier delivery sequence.
- Segmented 99.990% fill and 97.89% cycle service, demand-only 99.267% and 82.01%, flat
  baseline 99.817% and 90.07%.
- Segment split confirms the mechanism: the advantage is concentrated on AX (100% versus
  85.2% cycle service) and vanishes on CZ, exactly as the variance decomposition predicted.
- Cycle service level is suppressed for any segment completing fewer than 30 cycles, which
  is BX, BZ and CZ.

**Phase 5: documentation**

- README leads on the data quality finding rather than burying it, and carries the
  limitations as a full section.
- Case write-up in `reports/case_write_up.md` for an operations manager persona.
- Eight figures, all regenerated from the same tables the numbers come from.

**Testing**

- 130 tests. Audit detectors are tested against both data they should flag and data they
  should not. The simulation is tested for pipeline accounting, receipt ordering, warmup
  exclusion and monotonicity in buffer size.

## Fixed during the build

- `weekly_demand_panel` built its column index from observed weeks only, so a week with no
  activity anywhere would vanish rather than count as a coverage gap. Caught by a test,
  fixed to use a complete calendar index.
- The regional lead-time test originally decided on a p value, which at 107,752 rows
  reported a signal that was 0.05% of variance. Changed to decide on eta squared.
- The service level curve knee is scale invariant and lands at 98% for every unit, so it
  cannot justify differentiated targets. Added `differentiation_basis` for the argument
  that actually separates the segments, and documented the property rather than leaving it
  to be discovered.

## Pending, and things worth deciding

**Nothing blocking. The project runs end to end and is demoable as it stands.**

Things I would add if this goes further, in the order I would do them:

- **Repeated-seed backtest.** The comparison uses one test window and one lead-time
  realisation per unit. Common random numbers make it fair but do not put an interval
  around the differences. Repeating across seeds would say whether the 587 unit gap is
  stable or noise. This is the weakest claim in the project as it stands.
- **Tiered ordering cost.** A flat 250 per order gives the slow movers 228 days of cover.
  Either a per-unit ordering cost or a maximum cover constraint would fix it.
- **Periodic review (R, S) as an alternative.** The current policy is continuous review.
  Most distribution businesses review on a cycle, which changes the protection interval to
  R plus L and would change the numbers.

**Two things I still need to settle:**

- The cost assumptions in `config/config.yaml` (25% annual holding rate, 250 per order, 65%
  cost to price ratio) are conventional placeholders. If I find figures I would rather
  defend, they change every cost number but no conclusion.
- The lead-time proxy currently uses Standard Class only, on the reasoning that inbound
  bulk replenishment would move on standard freight rather than same-day courier. Setting
  `leadtime.proxy_mode` to null pools all four modes instead, which raises sigma slightly
  and introduces bimodality from the Same Day rows.

Not pushed yet. Local files only for now.
