# V5 Numerical Verification — Audit and Corrections

**Input verified:** `V5_output (1).zip` (eight CSVs). All reanalyses use original run-level CSVs; there is no rerunning or revising the ABM. Statistical inference uses matched seed 1–100 and two-sided Student-*t* tests with df = 99 and 95% t critical intervals; population contrast multiplicity adjustment uses Holm across 25 cells. Computed exact values are available in the four companion CSVs.

## Input coverage and consistency

| Experiment | Raw rows | Summary rows | Uniqueness / coverage |
|:--|--:|--:|:--|
| Baseline factorial | 400 | 400 run-level ex post | Four treatments × 100 seeds; no duplicates |
| Factorial timeseries | 40,000 | — | 4 × 100 seeds × 100 rounds; no duplicates |
| Population composition | 10,000 | 100 | 5 × 5 × 4 × 100; no duplicates |
| Six univariate sweeps | 16,400 | 164 | 41 parameter levels × 4 × 100; no duplicates |
| Rewiring × incentive grid | 16,800 | 168 | 6 × 7 × 4 × 100; no duplicates |

Final-round time-series values agree exactly with run-level final observations for ρ_C, q_C, μ and Ω. This is internal output consistency, not independent source-code validation.

## Confirmed, corrected, or qualified claims

1. **Baseline:** round-100 cooperator shares 0.0752/0.1284/0.6447/0.4254; q_C 0.268907/0.873786/0.797355/0.993409; μ 1.1777/1.4417/2.8684/2.6947; Ω 93.1535/93.1535/760.4423/534.9074 all confirmed.
2. **Time indexing:** round 1 strategy, neighborhood and mean-tier aggregates match, but Ω in T0/T1 = 93.1535 and in T2/T3 = 99.8535. Hence round 1 is not fully reward-free, and measurement timing must be described carefully.
3. **Monotonicity:** T0 mean cooperation does **not** monotonically decrease over all 100 rounds. It declines in early rounds with later fluctuations.
4. **Paired confidence intervals:** use Student-*t* rather than 1.96×SEM. For T3−T2, ρ_C −0.2193 [−0.240914, −0.197686], q_C +0.196055 [+0.184530, +0.207579], μ −0.1737 [−0.258554, −0.088846], Ω −225.5349 [−244.6118, −206.4579]. For interaction, ρ_C −0.2725 [−0.297251, −0.247749]. The supplied outline's p values and t statistics largely match exactly.
5. **Composition 90% C, 75% Type A:** +0.0988, 95% Student-*t* CI [+0.030273, +0.167327], p=0.005156; Holm(25)=0.051556. It is not significant at 5% after this family-wise adjustment.
6. **Composition 90% C, 90% Type A:** +0.2148, CI [+0.133068, +0.296532], p=1.0103e-6; Holm(25)=1.4e-5. Stronger conditional exception.
7. **Five positive population cells:** +0.0015 (10% C, 90% Type A), +0.0004 (25%,90%), +0.0173 (90%,50%), +0.0988 (90%,75%), +0.2148 (90%,90%). Only the last remains Holm-significant. One cell has an exact zero mean (90%,10%).
8. **Six sweeps present:** 6+7+8+7+7+6=41 distinct univariate levels. Behavioral invariance across α confirmed at the matched-seed final-observation level, while the mean reported ROI scales from 1.6818 to 10.0909 for T2 and 1.9965 to 11.9789 for T3 across α=0.05–0.30.
9. **At λ_L=1:** exactly 400 undefined final Type A q* entries; do not discard valid cooperation/wealth outcomes. Calling the closed-form singularity a 'removable singularity' is incorrect as a claim of a generally finite q* limit: the threshold representation degenerates. Underlying utility comparison is evaluated separately.
10. **Tier 1 versus final thresholds:** using baseline Tier 1 marginal reward M(1)=19.89535 and k=15, λ_L=2.25: q*(1)=0.738915; baseline final Type A average q* in T2/T3 is 0.723664/0.726026. At λ_L=1.50, the *Tier 1* q*(1)=0.347287, whereas the *final mean* q* reported is 0.047545. At w=1.25, the Tier 1 q*(1)=0.473643 versus the final mean 0.323773; at w=1.50 the Tier 1 value is 0.208372 versus final mean 0.028527. Do not mislabel final means as entry thresholds.
11. **Gamma description:** the point exponent term p_base^(γμ) rises with γ for p_base>1, while the decreasing-reputation/newcomer component declines; the net marginal reward may fall. Under the stated reward formula and inferred b=10.5, γ=0.35 gives M(1)=19.5342, q*(1)≈0.7582, not M(1)=18.60 or q*(1)=0.805. Verify theoretical formula notation in your methodology before copying the illustrative derived number.
12. **Rewiring-rate sweep:** in T1 final ρ increases 0.0752→0.1469 across p=0→0.5. In T3 it decreases 0.6447→0.3879 while q_C rises to 0.9888 at p=0.5. Note that q_C T3 is not strictly increasing at the final two p levels: 0.993409 at p=0.3, 0.991685 at p=0.4, 0.988808 at p=0.5.
13. **Bivariate sign counts:** 5 positive, 15 negative, 22 zero T3−T2 cooperation contrasts. All positive cells at zero exchange rate and nonzero rewiring.
14. **Diagnostic transitions:** at baseline T2 has 461.20 C→D and 475.67 D→C transitions; T3 has 115.40 and 107.94. These are all-agent transition counts, not Type B-only counts, and not distinct individuals converted. Mean connected-round fractions: T0 1, T1 0.9549, T2 1, T3 0.7595.
15. **No-incentive balances:** private and public digital point balances are nonzero in T0/T1 despite w=0; their modeled euro-equivalent valuation and outlay is zero, not their issuance count.
16. **ROI semantics:** mean ROI is not ratio of mean quantities. T2 mean reported ratio 5.045438 but ratio of reconstructed mean synergy to mean outlay 5.078796. For T3 these are 5.989432 and 6.007322. Use 'modeled recorded-synergy-to-token-outlay ratio'.
17. **Economic quantities reconstructed run by run:** mean T2 outlay 122,243.63, ROI-implied recorded synergy 620,850.45, difference 498,606.82. T3 86,648.22, 520,523.79, and 433,875.57. Paired T3−T2 differences are −35,595.41, −100,326.66, and −64,731.25 respectively. The baseline model does not justify an additional *positive* rewiring-overhead allowance on this narrow metric.
18. **Important synergy scope qualification:** approximate C–C-conditioned synergy from 100×ρ_C,t×4×q_C,t×33.75 summed across rounds equals 44,703.23 in T0, 146,194.88 in T1, 618,825.83 in T2 and 517,646.03 in T3. The active-run ROI-implied numerator is slightly larger (620,850.45 / 520,523.79), consistent with heterogeneity. This construction measures cooperator-received benefits on C–C neighborhood configurations, **not all synergy accruing to defectors and cooperators**, although the payoff rule allows defectors to benefit from cooperative neighbors. Do not label it 'total realized agricultural synergy of the entire system'.
19. **Ostrom operationalization:** `monitoring_mechanism_implemented` and `graduated_sanction_mechanism_implemented` equal zero across run-level diagnostics. Streak resets and information observability are partial analogues rather than explicit graduated sanctions or accountable monitors.
20. **External validity:** no field data or direct real-world cost validation in these CSVs. α_min≈0.0297 (T2) and ≈0.0250 (T3) cover modeled token outlay conditional on unchanged behavior; they are not empirically supported minimum viable agricultural synergy levels.

## Use and scope

Use the full Word chapter for thesis prose and the companion statistical CSVs as a reproducible analytic audit trail. The equations for the baseline Type A reward schedule are reconstructed from previously agreed V3 specifications and the recorded baseline M(1), not verified afresh against Python source (not part of the uploaded ZIP). Other numbers in the chapter are recomputed directly from the eight supplied CSVs. Do not treat consistency checks as validation of every underlying simulation mechanic or externally calibrated economic relationship.
