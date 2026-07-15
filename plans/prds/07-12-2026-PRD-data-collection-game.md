# PRD: Game for Data Collection — "Phoneme Hatchery" (working title)

**Status:** DRAFT — nothing started, nothing complete. Per CLAUDE.md, completion of any item below is declared by Daniel only, never inferred from code existing.
**Owner:** Daniel Hair
**Last updated:** 2026-07-12
**Source context:** Grill-me design session 2026-07-12 (this PRD's interview); [Evaluation Foundation PRD](./07-10-2026-PRD-models-trustworthy.md) (the "why"); planning notes on corpus sizing/age bands (breadth-beats-depth analysis, 2026-07-12).

> Sections marked **[PROPOSED]** were designed by Claude during the grill session but not explicitly confirmed by Daniel — redline freely. Everything else was decided in the interview.

---

## 1. Problem Statement

The Evaluation Foundation epic produced trustworthy numbers, and the numbers say the same thing from every angle:

- Honest LOSO accuracy tops out at **57.90–58.69% on Chloe** (WavLM-CTC / Wav2Vec2-CTC), the only target-age-band speaker (age 7). The confirmed conclusion of that PRD (§6): **"needs more target-age-band data."** This epic is that data.
- The corpus is **extremely deep, extremely narrow**: 5 speakers × ~1,200 clips. The models have extracted what they can from ~33 takes per phoneme per speaker; what they have never seen is the *variety* of child voices. A new child's 3rd take of "sh" is worth far less than a new child.
- The live-mic smoke test measured **mic/channel quality alone costing ~21–24 top-1 points** on every model. Device/mic diversity in collected data matters as much as speaker diversity. (A parallel channel-augmentation experiment is attacking the same factor synthetically — see `plans/channel-augmentation-experiment.md`.)
- Session-to-session shift was measured to cost roughly the entire benefit of being a known speaker (full-corpus models scored a fresh session from a *training* speaker at never-heard-speaker levels). Multiple sessions per child add real diversity, not just more clips.
- Chloe is currently an **n=1 evaluation**. The headline number has no error bars until several target-age children are held out of training entirely.

**Bottom line:** the binding constraint on model quality is no longer measurement, decoding bugs, or architecture — it is the absence of a broad child-voice corpus. This epic builds the machine that collects one, as a fun browser game, without compromising the label discipline the last epic taught us to respect (mislabeled speakers caused real fold-purity bugs).

## 2. Goals

1. **Corpus:** isolated single-phoneme utterances from children ages 4–8, labeled with the intended target phoneme from the existing 37-phoneme set, ~3s clips at 16kHz mono WAV, organized speaker-first (`recordings/<speaker_id>/<phoneme>/*.wav`), directly consumable by the existing pipeline and LOSO harness with zero format changes.
2. **Breadth over depth:** ~3 takes × 37 phonemes ≈ **~110 clips per child**, split across **two ~10-minute sessions on different days**. Many kids, few takes each.
3. **Scale tiers:** **≥20 children = minimum useful** (~2,200 clips; doubles the corpus with 100% target-age data); **40–60 children = target** (~4,500–6,500 clips). Beyond ~100 only if the learning curve (§8) is still climbing — the stopping point is *measured*, not guessed.
4. **Device diversity as a first-class goal:** kids play on their own family devices (phones, tablets, laptops), capturing channel variety by construction. A single-device corpus would train a "that-tablet specialist" — explicitly what we are avoiding.
5. **Label quality equal to the existing corpus:** every clip's label means "verified attempt at the target phoneme" (the Sky/Cassie supervised-labeling precedent), enforced by a human review pipeline (§7).
6. **Honest evaluation built in from day one:** ~20% of children held out of training entirely (stratified by age band), assigned at signup, so headline numbers get real error bars and can never leak.
7. **Fun is a requirement, not a nicety:** the retention target is exactly two sessions per child. A game a 4-year-old abandons at prompt 12 collects no corpus. Interaction budget: two ~10-minute sessions × ~55 prompt cycles each, ~10s per cycle.

## 3. Non-Goals (explicitly out of scope for this epic)

- **The product game.** The Unreal Engine game (open-world, mini-games per `MiniGameActivityideas.md`, local CTC inference) is a separate future epic. This is the *collection vehicle*. If it grows into something the company ships, great — but nothing here is gated on that.
- **Live model feedback / in-game classification.** Confirmed capture-only for v1. The recording itself always rewards; no clip is ever gated, scored pass/fail, or filtered by a model. (An optional server-side "bonus sparkle" using known-target top-3 — 81–97% accuracy, far stronger than open 37-way — is deferred; see §11. Rationale for caution: rewarding only model-recognized clips would bias the corpus toward what models already handle, and the clips models fail on are the most valuable training data.)
- **CVC / multi-phoneme sequences.** Isolated single phonemes are the product's confirmed core interaction, not a simplification. CVC collection is a future extension once this pipeline works.
- **Ages 9–15.** Phase 1 is 4–8, weighted toward 5–7. The metadata schema (age band per child) is designed so 9–12 and 13–15 (a two-population band: pre/post voice change) slot in later without rework — but no game design, recruitment, or evaluation work for those bands now.
- **Offline mode.** Online-first, confirmed. Offline (game + local ingest on a laptop for a no-internet venue) is a later nice-to-have, tracked in §11.
- **Model training/improvement work itself.** This epic delivers data and the learning-curve measurement; retraining decisions and any new model work are downstream.
- **GOP calibration.** Rejected-but-kept clips (§7) are future GOP-calibration material; building GOP is not this epic.

## 4. The Game

### 4.1 Core fantasy (decided)

**Creature hatchery.** Each of the 37 phonemes *is* a creature. The child says the creature's sound to hatch its egg; hatched critters fill a collection book. The collection book doubles as corpus-progress tracking — a completed collection = that child's corpus is done.

- Sound-symbol pairing is genuinely educational: the /b/ egg hatches a creature whose name starts with /b/.
- Collecting has strong appeal across ages 4–8 and both genders.
- Art is 2D, illustrated, web-cheap. The glowing-Unreal aesthetic belongs to the other product.

### 4.2 Core loop **[PROPOSED — redline]**

One prompt cycle (~10 seconds):

1. An egg appears with the grapheme shown and the critter's silhouette behind it.
2. Mascot voice plays the reference sound: "Say /sh/!" (pre-recorded human or high-quality TTS voice-over — the reference pronunciation the child imitates; it must be phonetically correct, so recorded by Daniel or vetted carefully).
3. Recording window opens with a clear visual cue (egg glows/pulses for ~3s); the child says the sound.
4. The take is saved and uploaded. **The egg cracks one stage — always.** Every utterance visibly progresses the hatch regardless of what was said. No pass/fail, ever (decided).
5. Three takes = the egg hatches; celebration moment; critter joins the collection book; next egg.

Supporting rules:

- **Skip button** (kid refuses a sound → egg goes back in the nest, offered again later; a child who never says /r/ still has a valid 36-phoneme corpus — partial data is acceptable, class balance is monitored in §8).
- Three takes are consecutive within a cycle (simplest; the egg-cracking metaphor demands it). Cross-take diversity comes from session 2 being a different day — which we measured to matter more than take-count anyway.
- Silence/no-speech detection is a **client-side nudge only** ("I didn't hear you — try again!" via simple energy threshold), never a discard: the clip still uploads, review decides its fate.
- **Session 1 ends when ~18–19 critters are hatched** ("the nest is sleepy — come back tomorrow, new eggs are wiggling!"); session 2 completes the collection. Sessions can end early and resume anytime; two sessions is the design target, not a hard gate.
- Reward economy: hatching always flows from recording itself (decided: "the recording itself will always give you something"). An accuracy-linked *bonus* layer (rarer color variants, sparkles) is deferred with the server-side scoring non-goal.

### 4.3 Prompt ordering **[PROPOSED — redline]**

Fixed curriculum order, early-acquired sounds first (/m/, /b/, /p/, vowels…) so the first minutes are guaranteed wins for the youngest players, late-acquired sounds (/r/, /th/, s-blends) mid-session once warmed up, never as the session opener. All 37 phonemes get equal takes (corpus balance) — no confusion-pair weighting of *quantity*; the confusion pairs (dh/th, s/sh/th, m/n) get special treatment in *review* (§7), not in collection.

### 4.4 Audio capture spec (decided format; capture path [PROPOSED])

- **Target artifact:** ~3s, 16kHz, mono, 16-bit PCM WAV — byte-format-consistent with `recordings/`.
- **Capture path:** raw PCM via Web Audio API (AudioWorklet), downsampled to 16kHz mono and encoded to WAV **client-side**, then uploaded (~94KB/clip). Explicitly **not** MediaRecorder's webm/opus default — lossy compression would make collected audio unlike both the existing corpus and the eventual deployment path. Device/mic character (the channel diversity we want) survives; codec artifacts (which we don't want) are avoided.
- Recorded per clip: anonymous child id, target phoneme, take number, session id, timestamp, user-agent/device info (see §6 for the full schema).

## 5. Platform & Stack (decided)

- **Browser web app** — runs on whatever device a family already has; no install; `getUserMedia` mic capture. This is also what delivers goal #4 (device diversity) by construction.
- **Vercel** (hosting, game frontend + reviewer app frontend) + **Supabase** (Postgres for metadata, Storage for wavs, Auth for parent accounts and reviewer accounts).
- Storage envelope is trivial: 60 kids × ~110 clips × ~94KB ≈ **~620MB** — comfortably inside cheap tiers.
- No ML anywhere in the v1 runtime (capture-only). The game has zero model dependencies.

## 6. Consent, Privacy, and Data Handling

### 6.1 Data minimization (decided)

Collected per child: **the voice clips + age band** (e.g. 4, 5, 6, 7, 8 — band, never birthdate). Optional, never required: **coarse region (state or country only)**. Nothing else — no child name, no photos, no emails for the child, no fine location. Speaker IDs are anonymous (`child_001`-style); real names never appear in paths or metadata (unlike the existing 5-speaker corpus, which uses family first names — acceptable for family, not for others' kids).

### 6.2 Two-phase consent (decided)

**Phase A — Facilitated (families Daniel knows; target 20–40 children):**
- In-person paper/PDF consent signed by the parent: what's recorded, purpose (training speech models), retention, who can hear it (reviewers), right to deletion.
- Parents are present during play; kids use their own family devices (this is also the channel-diversity mechanism).
- No lawyer required for this phase. **Collection can start as soon as the pipeline works.**

**Phase B — Distributed (public):**
- Gated on **verifiable parental consent (VPC)** per COPPA — an in-app "I agree" checkbox is *not* sufficient (a child can tap it). Working choice **[PROPOSED]**: signed-consent-form upload (print/sign/photograph) as the VPC method — no payment processor, cheap to build. Note: because reviewers (third parties) hear the clips and clips are retained, the lighter "email-plus" method is likely unavailable; the FTC audio exemption (transcribe-and-delete-immediately) does not apply since we retain audio.
- Gated on **lawyer review** of the privacy policy + consent flow. Budget expectation set in the grill session: ~$2,000–7,500 for a privacy-specialist one-time engagement (~$1,000–3,000 if they review drafts built from the FTC's free small-business COPPA guidance). Safe Harbor certification (PRIVO/kidSAFE, ~$5–15k+/yr) is explicitly deferred until real scale.
- Parent-first onboarding: the parent creates the account, consents, picks the age band, then hands the device to the child. The child's flow contains no text entry and no PII.
- **Do not launch the public call-to-action until Phase B's gates are met.** Phase A does not wait for Phase B.

### 6.3 Deletion & rights

A parent can request deletion; deletion removes clips from storage and marks the child id tombstoned in the DB **and in any exported training copies** — the export tool (§8) must support re-sync so a deletion propagates to `recordings/`. Consent records are retained.

## 7. Label Quality: the Review Pipeline (decided: public reviewer app with accounts)

The label on every training clip means **"a verified attempt at the target phoneme"** — the child attempted the prompted sound (even imperfectly; that's the Sky/Cassie precedent and it's what makes speech-therapy kids' data usable). Verification is a separate human step, possibly days later, possibly by different people.

**Reviewer app (v1 deliverable):**
- Real reviewer accounts (Supabase Auth) with a click-through reviewer agreement (they will hear children's voices; no downloading/sharing; work is logged).
- Task: see target phoneme + hear reference pronunciation → play clip → verdict.
- **Verdict taxonomy [PROPOSED]:** `verified_good` (clear attempt at target) / `wrong_sound` (clear attempt at a *different* sound) / `unusable` (silence, noise, unintelligible, multiple voices).
- **Nothing is ever deleted by review.** `wrong_sound` and `unusable` clips are kept, tagged, and excluded from training (`wrong_sound` is future GOP-calibration gold). Only `verified_good` clips are exported to training.
- **Quality controls (required, since reviewers are untrusted):** reviewer id recorded on every verdict; Daniel spot-checks ~10% of each reviewer's verdicts; the known confusion pairs (**dh/th, s/sh/th, m/n**) always get a second independent review, disagreement → Daniel adjudicates; per-reviewer agreement stats tracked; a small seeded set of known-good/known-bad "golden clips" mixed into queues to measure reviewer reliability.
- Throughput reality check: 60 kids ≈ 6,600 clips ≈ a few seconds each ≈ ~10–15 reviewer-hours total. Small — the app can be lean.

## 8. Data Flow Back Into Training

```text
game (browser) → Supabase Storage + Postgres row per clip
      → reviewer app (verdicts)
      → export tool: verified_good clips → recordings/<child_id>/<phoneme>/<child_id>_<phoneme>_ep-<ipa>_<date>_<time>_<take>.wav
                     + per-child metadata JSON (age band, region?, device(s), session dates, holdout flag, consent record id)
      → existing LOSO harness (evaluation/harness/) — manifest builds from recordings/ unchanged
```

- **Filename/layout convention matches the existing corpus exactly** (`recordings/<speaker>/<phoneme>/*.wav`, existing filename pattern) so `evaluation/harness/dataset.py` ingests new children with zero code changes. Per-child metadata lives in a JSON file per speaker dir (the path carries no age/PII).
- **Holdout discipline:** ~20% of children flagged `holdout=true` **at signup**, stratified by age band and **including the youngest band** (the model will be weakest there; better to know). Holdout kids' clips are exported but never enter any training fold. Small harness extension needed: a train/eval split by holdout flag alongside the existing per-speaker LOSO rotation.
- **Learning-curve protocol (the empirical stopping rule):** at N = 10, 20, 30, 40… ingested training children, train (existing 5-speaker corpus + N new kids) and evaluate on the held-out children; plot held-out-child accuracy vs N, reported per age band. Curve still climbing at 20 → recruit more; flattening at 40 → stop. This converts "how many kids do we need?" from a promise into a measurement — same philosophy as the Evaluation Foundation.
- **Age-stratified reporting from day one:** every evaluation reports per-age-band accuracy, never just an average — "78% on kids" must not be able to hide "61% on 5-year-olds."
- **Speech-therapy kids are wanted, not excluded** (with parental consent) — they are the actual end users. The review pipeline handles their productions correctly by design (label = attempted target, verified).

## 9. Deliverables & Status Tracking

Per CLAUDE.md: complete only when Daniel declares it. No mocked pieces may be called complete unless Daniel says so.

| # | Deliverable | Status |
|---|---|---|
| D1 | Game frontend: hatchery loop, 37 prompts, capture→WAV→upload, session structure, skip/resume | Not started |
| D2 | Parent onboarding: account, consent capture (Phase A: record of paper consent; Phase B: VPC flow), age band + optional region | Not started |
| D3 | Supabase schema + storage: clips, children, sessions, consent records, verdicts, holdout flag | Not started |
| D4 | Reviewer app: accounts + agreement, review queue, verdict taxonomy, confusion-pair double-review, spot-check & agreement tooling | Not started |
| D5 | Export tool: verified clips → `recordings/` convention + per-child metadata JSON; deletion re-sync | Not started |
| D6 | Harness extension: holdout-children eval split + learning-curve runs + per-age-band reporting | Not started |
| D7 | Phase A collection: ≥20 children recorded, reviewed, exported | Not started |
| D8 | Learning-curve report at ≥20 children (informs recruit-more vs stop) | Not started |
| D9 | Phase B gate package: lawyer-reviewed policy, VPC flow live, reviewer agreement — before any public call-to-action | Not started |
| D10 | Phase B collection toward 40–60 children (if D8 says the curve is climbing) | Not started |

## 10. Success Criteria

Separate bars — do not conflate (same discipline as the Evaluation Foundation PRD):

- **Pipeline done:** a family Daniel does not coach can consent, play two sessions on their own device, and their verified clips land in `recordings/` and appear in a harness run — end to end, no manual surgery. True regardless of how many families have done it.
- **Corpus minimum:** ≥20 children (≥2/3 in the 5–7 core band, youngest band represented, ≥10 distinct device models across the cohort), ~110 clips/child target (partial corpora count — see skip rule), ≥20% held out.
- **Corpus target:** 40–60 children, same stratification — pursued only if the learning curve is still climbing at 20.
- **Quality:** review completed on 100% of clips; spot-check disagreement rate per reviewer tracked (persistent outlier reviewers' verdicts re-reviewed); class balance monitored (no phoneme below ~60% of target take-count across the cohort without a plan).
- **What success is NOT:** hitting the 85% ship bar on the models. That's the *next* epic's question; this epic succeeds when the data and its honest measurement exist. A learning-curve result of "accuracy is flattening below the bar" would be a *successful* outcome of this epic (we'd know the next constraint isn't data volume).

## 11. Explicit Non-Decisions (deferred, tracked, not forgotten)

| Item | Deferred to | Trigger to revisit |
|---|---|---|
| Server-side known-target "bonus sparkle" scoring (top-3 is 81–97%) | v1.1 of the game | Kids' engagement in Phase A suggests the reward economy needs more juice; must never gate rewards or filter clips |
| Offline/facilitated-venue mode (laptop-hosted) | Later | A recruitment venue without reliable internet actually materializes |
| Ages 9–12, 13–15 collection (13–15 = two voice populations, pre/post change) | Phase 2/3 epics | Phase 1 corpus target met; metadata schema already supports it |
| CVC-carrier collection in-game | Future epic | Isolated-phoneme pipeline proven; product needs CVC data |
| GOP calibration using kept `wrong_sound` clips | Future epic | Enough reviewed wrong_sound volume exists |
| COPPA Safe Harbor certification (PRIVO/kidSAFE) | Post-scale | Public phase grows beyond hobby scale |
| Unreal product game & mini-games (`MiniGameActivityideas.md`) | Separate epic | A model meets the ship bar |
| Learned/confidence-weighted multi-model combiner (from the tandem analysis) | Future epic | Held-out tuning data exists — which this epic's holdout children will provide |

## 12. Open Risks

- **Recruitment shortfall** — 20–40 families may not all follow through; two-session completion rate is unknown. Mitigation: session 2 reminder flow, and the corpus degrades gracefully (a one-session child still yields ~55 useful clips).
- **The youngest band (4–5) may frustrate**: compliance and on-demand phoneme production drop hard under ~4½; expect higher `unusable`/`wrong_sound` rates and design review capacity for it. The band is still wanted — hold some out for evaluation especially.
- **Browser audio variance**: iOS Safari mic permission quirks, autoplay policies, AudioWorklet support on old devices. Mitigation: device smoke-test matrix before Phase A; the capture path is deliberately simple (raw PCM → WAV).
- **Mic distance/clipping on consumer devices** is uncontrolled by design (it's the diversity we want) — but a client-side level meter nudge ("come closer!") keeps clips usable without homogenizing the channel.
- **Legal timeline for Phase B** — lawyer review is a hard gate for the public phase; if it stalls, Phase A still delivers the minimum-useful corpus.
- **Reviewer supply/quality** — public reviewer accounts with golden-clip checks are unproven; fallback is Daniel + family reviewing (the volume is only ~10–15 hours).
- **Selection bias via fun** — if late-curriculum sounds land when kids are tired, those phonemes get worse takes. Monitor per-phoneme usable-rate by prompt position; shuffle curriculum order across sessions if it shows.

## 13. Coordination Notes

- A channel-augmentation experiment is running in a parallel session (see `plans/channel-augmentation-experiment.md`): do not touch `evaluation/harness_cache/`, `evaluation/full_models*/`, or `logs/full_train_channel_aug.log`; expect heavy CPU on this machine. Its outcome (channel robustness) and this epic (real device diversity) attack the same measured ~21–24-point mic effect from opposite ends — both results should be read together when the learning-curve report (D8) lands.
