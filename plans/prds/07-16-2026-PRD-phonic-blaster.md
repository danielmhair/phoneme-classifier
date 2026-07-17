# PRD: Phonic Blaster — Voice-Triggered Lane Shooter (v1, no classifier)

Source spec: `light-haven-sites/apps/src/phonic-blaster/prd.md`. This PRD adapts that spec to the existing Light Haven apps codebase, per decisions confirmed with Daniel on 07-16-2026.

## Problem Statement

Children practicing letter sounds need lots of short, repeated productions of a target phoneme — but drill-style repetition is boring, and the existing Phoneme Hatchery data-collection game is paced around deliberate one-at-a-time takes, not sustained practice. There is no play experience that rewards a child for producing many separate, rhythmically paced utterances of a prompted sound.

Separately, the project's long-term goal (trustworthy phoneme classification) is starved for child speech data. A game that children *want* to keep playing — and that can optionally save each detected utterance labeled with the sound the screen was prompting — is a natural future data source. But no classifier meets the ship bar yet, so v1 must be fun and useful **without** recognizing which phoneme was said.

## Solution

Phonic Blaster: a child-friendly, browser-based, three-lane shooter added as a new page in the existing Light Haven apps codebase. A large prompt shows the sound to practice (e.g. **/b/**). Every time the microphone detects a *separate* vocal sound, the player's character fires in its current lane. A long continuous sound fires exactly one shot — the child must pause briefly between sounds, which is exactly the clean-utterance behavior the future dataset wants.

Enemies descend the three lanes; the child steers with the keyboard and "shoots" with their voice, keeping a steady speaking rhythm. A rhythm bar rewards well-paced utterances (Perfect timing = bonus/stronger shot) and an overheat meter gently discourages rapid-fire babbling. All audio processing is local to the browser; nothing is uploaded. Optionally, each detected utterance can be saved locally as a 16 kHz WAV labeled with the **prompted** phoneme (explicitly not verified as correct), structured so an ONNX classifier can drop in later.

The game is fully testable without a microphone: spacebar simulates a detected utterance.

## User Stories

1. As a child player, I want to see one big friendly letter/sound prompt on screen, so that I know exactly which sound to practice.
2. As a child player, I want my character to shoot every time I make a sound, so that speaking feels powerful and fun.
3. As a child player, I want a long "aaaaah" to only fire once, so that I learn to make crisp, separate sounds.
4. As a child player, I want to move my character between three lanes with Left/Right arrows or A/D, so that I can line up with enemies.
5. As a child player, I want enemies to move down the lanes toward me, so that there is excitement and a reason to keep speaking.
6. As a child player, I want to see my score go up when I hit enemies, so that I feel rewarded for practicing.
7. As a child player, I want a small number of lives/health shown with friendly icons, so that mistakes feel survivable rather than scary.
8. As a child player, I want the game to get gradually harder in rounds, so that it stays challenging as I improve.
9. As a child player, I want a rhythm bar that shows when it's a good time to say the next sound, so that I learn a steady speaking pace.
10. As a child player, I want "Perfect" timing to give bonus points or a stronger shot, so that pacing well feels extra rewarding.
11. As a child player, I want gentle feedback like "Too soon!" instead of punishment, so that I never feel bad for trying.
12. As a child player, I want a heat meter that fills when I shoot too fast and a friendly cool-down moment when it overheats, so that I naturally slow down instead of babbling.
13. As a child player, I want a visible microphone activity indicator, so that I can tell the game hears me.
14. As a child player, I want bright, colorful, simple graphics, so that the game feels made for me.
15. As a child player, I want encouraging messages between rounds, so that I want to keep playing.
16. As a parent, I want a microphone setup screen with a live level meter and a test step, so that I can confirm the mic works before handing the device to my child.
17. As a parent, I want the game to work with a "Continue without microphone" option, so that we can still play (keyboard mode) when a mic isn't available or permission is declined.
18. As a parent, I want all audio processing to happen locally in the browser with no uploads and no external speech services, so that my child's voice stays private.
19. As a parent, I want a pause menu, so that I can stop the game instantly when real life interrupts.
20. As a parent, I want background chatter and quiet noise to be ignored where possible, so that the game responds to my child's voice and not the TV.
21. As a developer, I want spacebar to simulate a detected utterance, so that I can test every gameplay system without a microphone.
22. As a developer, I want timing, heat, rhythm, and sound-detection thresholds in one typed config, so that tuning game feel doesn't require hunting through code.
23. As a developer, I want the microphone system fully separated from gameplay logic behind a small event interface, so that gameplay can be developed and tested against simulated events.
24. As a developer, I want the sound-detection module structured so an ONNX phoneme classifier can be inserted later at a single seam, so that v2 (correctness scoring) is an addition rather than a rewrite.
25. As a developer, I want the game mounted as its own page/route in the existing apps codebase, so that multiple games can coexist as separate pages without new deployment targets.
26. As a developer, I want a progress.md tracked alongside the game code during the build, so that build status survives across sessions.
27. As a dataset curator, I want an optional mode that saves each detected utterance locally as a 16 kHz mono WAV labeled with the prompted phoneme, so that reviewed clips can later grow the classification corpus.
28. As a dataset curator, I want every saved clip clearly marked as carrying a *prompted* label (what the screen asked for), never treated as ground truth of what was said, so that the future review pipeline stays honest.
29. As a dataset curator, I want prompts drawn from the existing 37-phoneme curriculum (the `recordings/` directory names), so that any collected clips use the canonical class labels without remapping.
30. As a maintainer, I want the game to build and run with the existing `npm install` / `npm run dev` / `npm run build` scripts of the apps package, so that there is one workflow for the whole codebase.
31. As a maintainer, I want only open-source dependencies (Phaser, Meyda) and no paid APIs, so that the game stays freely distributable.
32. As a maintainer, I want a README section covering setup, controls, and the keyboard-only mode, so that contributors can run the game unaided.

## Implementation Decisions

- **Placement**: Phonic Blaster lives in a new `phonic-blaster` folder under the apps source tree and is exposed as a new route in the existing React Router shell. **The React wrapper is a doorway only** — a thin component that mounts and unmounts the Phaser game instance. All game logic is plain strict TypeScript + Phaser; no React inside the game. (Resolves the source spec's "no React" against the reality of the shared React codebase — decision confirmed by Daniel.)
- **Engine**: Phaser 4 with scenes as the top-level structure: a mic-setup scene, the play scene, a pause overlay, and a game-over/round-transition scene. Placeholder graphics are Phaser-drawn shapes (no copyrighted assets).
- **Microphone system — VoiceGate**: a self-contained module owning getUserMedia → AudioContext → AudioWorklet frame capture → **Meyda feature extraction from the start** (per Daniel's decision): per-frame RMS/energy (and spectral features as needed for noise robustness) feed an utterance state machine. The module emits a tiny event interface — `soundStart`, `soundEnd` (one pair per separate utterance), and continuous `level` for the activity meter — and gameplay consumes only these events. Onset/hangover/gap thresholds follow the values already proven in the Phoneme Hatchery recorder (~100 ms onset to count as an utterance, ~250 ms silence gap to separate bursts, hangover before closing), all configurable.
- **Simulated input adapter**: spacebar produces the same `soundStart`/`soundEnd` events as VoiceGate, so every downstream system (shots, rhythm, heat, recording hooks) is driven identically in keyboard-only mode. "Continue without microphone" simply runs this adapter.
- **RhythmJudge**: pure function from utterance-onset timestamps + tempo config to a verdict — Too soon / Good / Perfect / Late — plus a score/shot-strength modifier. The rhythm bar renders its state; Perfect grants bonus points or a stronger projectile.
- **HeatSystem**: pure reducer — each shot adds heat (more when the previous shot was recent), time cools it, a full meter enters a temporary overheat lockout with friendly feedback, then shooting re-enables. No permanent penalty.
- **WaveDirector**: deterministic round/spawn scheduler — enemy cadence, speed, and lane distribution per round, difficulty rising across rounds. Rounds also rotate the prompted phoneme.
- **Prompt source**: prompts come from the existing 37-phoneme curriculum module (ids identical to the `recordings/` directory names — the canonical class labels). No new phoneme list is created.
- **ClipStore (optional, off by default)**: when enabled, each detected utterance's samples are encoded as 16 kHz mono 16-bit WAV (reusing the existing WAV encoding utility) and saved locally (in-browser storage with a bulk export/download), with metadata: prompted phoneme id, timestamp, duration, energy. Stored strictly as *prompted-label* data. No uploads in v1 — this is deliberately narrower than the Supabase pipeline the Hatchery uses.
- **Classifier seam**: VoiceGate's `soundEnd` event carries the utterance's audio buffer. In v1 nobody consumes it except ClipStore; in v2 an ONNX classifier subscribes at this exact point. No other structure changes anticipated.
- **Config module**: one typed config object covering rhythm windows, heat rates, cool-down, detection thresholds, wave pacing, lives, and scoring — the single tuning surface the source spec requires.
- **HUD**: large phoneme prompt, score counter, lives display, rhythm bar, heat meter, mic activity indicator — all in the play scene.
- **Dependencies added**: `phaser` and `meyda` only; both open-source. No cloud speech, no paid APIs, no ML models in v1.
- **Build**: no new build target — the game rides the existing apps package's `npm run dev` / `npm run build`.

## Testing Decisions

- **Decision (Daniel, 07-16-2026): no unit tests yet.** This version is prototype-first; tests will be added once gameplay feel has stabilized.
- The architecture is nevertheless shaped for future testability: VoiceGate's state machine, RhythmJudge, HeatSystem, and WaveDirector are pure or frame-driven modules that can later be tested with synthetic inputs and no Phaser/browser dependency. A good future test exercises only external behavior (events in → events/verdicts out), never internal state.
- Interim verification is manual: the keyboard-only mode makes every system exercisable without a mic, and the mic-setup screen's meter + test step verifies the live path.
- Prior art for future tests: the Python side's `tests/` suite (temporal brain) demonstrates the house style of behavior-level unit tests; the apps package currently has no JS test runner, so one (e.g. Vitest) would be introduced when tests land.

## Out of Scope

- **Phoneme correctness recognition** — the game never judges whether the child produced the prompted sound; ONNX/classifier integration is v2.
- **Uploading audio anywhere** — no Supabase writes, no external speech services; ClipStore is local-only.
- **Review tooling for collected clips** — the existing reviewer flow is not extended to Phonic Blaster clips in v1.
- **Real art assets** — placeholder shapes only.
- **Mobile/touch controls** — keyboard + microphone; touch layout can come later.
- **Accounts/child profiles** — unlike the Hatchery's `/play/:childId` flow, Phonic Blaster v1 has no identity or persistence beyond optional local clips.
- **Replacing or modifying the Phoneme Hatchery game** — it remains untouched; games coexist as separate pages.

## Further Notes

- **progress.md discipline (Daniel's request)**: the build must maintain a `progress.md` inside the game folder, updated as milestones land, so progress is trackable across sessions. Suggested build order (matching the source spec): playable keyboard-only core → mic detection (VoiceGate + setup screen) → rhythm system → overheat system → optional local recordings.
- The single-utterance behavior (long sound = one shot, pause required between sounds) is not just a game mechanic — it trains exactly the clean, separated utterances the CTC training corpus needs (repeats within one clip break CTC alignment). Game design and data-collection goals are aligned by construction.
- The prompted-label caveat matters for honesty downstream: a saved clip proves what the screen asked, not what the child said. Any future ingestion into the training corpus must pass through review, consistent with the project's holdout/trust discipline.
- The source spec lists `recordings/callie` as the sound inventory; the curriculum module already encodes those 37 directory names with graphemes and hint words, so the prompt display can show the friendly grapheme (e.g. "b", "sh", "oo (book)") rather than raw IPA.
