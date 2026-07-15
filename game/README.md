# Data-collection game — training-side tooling

The Phoneme Hatchery web app (game + parent onboarding + reviewer app +
Supabase schema) **moved to the light-haven-sites repo** at
`light-haven-sites/apps/` — see the README there for setup and deploy.
PRD: [plans/prds/07-12-2026-PRD-data-collection-game.md](../plans/prds/07-12-2026-PRD-data-collection-game.md).

What stays in this repo is everything that touches the training pipeline:

- `export_tool/export_verified.py` (D5) — pulls `verified_good` clips out of
  Supabase into `recordings/<child_code>/<phoneme>/*.wav` + `child_meta.json`,
  and re-syncs deletions. Run with `poe game-export`; needs `SUPABASE_URL`
  and `SUPABASE_SECRET_KEY` (sb_secret_..., from Dashboard > Settings > API
  Keys) in the environment. Never commit that key anywhere.
- `evaluation/harness/holdout.py` + `evaluation/harness/learning_curve.py`
  (D6) — holdout-children split and the learning-curve stopping rule
  (`poe holdout-eval`, `poe learning-curve`).

The split is deliberate: the app repo never holds training data or secret
keys, and this repo never holds the site. The two meet only at Supabase
(app writes, export tool reads) and at `recordings/` (export tool writes,
harness reads).
