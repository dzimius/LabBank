# bs_optimization (release subset)

Balance-sheet optimisation on top of the `optimize_prep` fast layer. This public folder holds only
what the joint balance-sheet + swap optimisation runs need; the optimisation report and the other
optimisers are not part of the release.

Every script puts the four group folders on `sys.path`, so modules import each other by bare name.

| Folder | Scripts |
|---|---|
| `optimizers/` | `joint_optimizer` (balance sheet + new par swaps), built on the LP core in `bs_optimizer`; `optimizer_io` (config) |
| `scenarios/` | `robust_curves` (alternative base curves), exact repricing (`anchor_eve_exact`, `anchor_sot_exact`, `bank_reprice_at_weights`), curve banks (`curve_scenario_bank`, `mc_scenario_basis`), Monte Carlo pricing (`mc_by_product`, `mc_packed`, NumPy) |
| `hedging/` | `per_scenario_hedge` (minimum hedge per curve), `swap_ladder`, `hedge_optimizer`, `dated_swaps` (swaps on their own trade dates), `monthly_swap_plan` (swap purchases month by month) |
| `reporting/` | `btrm_article_runs` (all optimisation runs), `btrm_robust_check`, `btrm_mc500` (out-of-sample Monte Carlo test) |
| `input/` | `optimizer_config.xlsx` |

How to run them: `docs/release_notes_2026-11.md`. Results go to `runs/<bank>/bs_optimization/`.
