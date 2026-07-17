# PRD: Light Haven Apps on Phones — Capacitor Wrap + Mobile Responsiveness

Builds on the Phonic Blaster PRD (`plans/prds/07-16-2026-PRD-phonic-blaster.md`), which explicitly deferred mobile/touch. Decisions below confirmed with Daniel on 07-16-2026.

**Prior art (Daniel's direction): mirror the light-haven-home repo.** That codebase (sibling repo `light-haven-home`) has a drafted Capacitor PRD (`plan/prds/2026-06-17-capacitor-native-wrapper.md`) and a working Supabase auth setup; this PRD adopts the same conventions — same Capacitor config shape, same platform-abstraction pattern, and the same Supabase client/auth configuration — rather than inventing parallel ones.

## Problem Statement

Children practice letter sounds on the devices families actually have — phones and tablets — but the Light Haven apps are desktop-web-only today. Phonic Blaster is literally unplayable on a phone: lane steering, firing in no-mic mode, and pausing are all keyboard-only. The wider app shell (home, onboarding, Phoneme Hatchery, reviewer, admin) has no responsive CSS at all — no media queries, `100vh` layouts that break under mobile browser chrome and notches, and touch targets sized for a mouse.

Beyond the browser, there is no installable app: nothing to put on a child's home screen, no native microphone permission flow, and Phonic Blaster's clip export (one browser download per WAV) cannot work inside a native WebView at all.

## Solution

Make the whole Light Haven apps shell phone-ready, then wrap it with Capacitor into a native app — Android buildable and testable now (Windows dev machine), iOS project scaffolded and committed for when a Mac is available.

Three layers, each independently useful:

1. **Touch controls for Phonic Blaster** — tap zones steer between lanes, an on-screen pause button replaces P/Esc, and a fire button replaces spacebar in no-mic mode. Touch produces the same semantic game intents the keyboard produces, through the same event-driven path; desktop keyboard play is unchanged.
2. **Responsive shell** — every route (home, onboarding, Hatchery play, Phonic Blaster, reviewer, admin) works on phone viewports: safe-area insets, dynamic viewport units, touch-target sizing, and scrollable-on-overflow layouts. The game canvas keeps its 540×720 design coordinates and FIT letterboxing, which already suit a portrait phone.
3. **Capacitor wrap** — the existing Vite build rides inside a native WebView with proper native mic permission flow, portrait lock, splash/icon, and a platform seam that swaps browser downloads for native filesystem + share-sheet export of saved clips. The web deployment on Vercel is unaffected; one codebase, one build, two delivery channels.

The mic pipeline (getUserMedia → AudioWorklet → Meyda) is standard web audio that WKWebView and Android WebView both support; it stays untouched, and the whole game remains playable in a plain mobile browser before Capacitor ever enters the picture.

## User Stories

1. As a child player, I want to move my rocket between lanes by tapping the left or right side of the screen, so that I can play Phonic Blaster on a phone with no keyboard.
2. As a child player, I want a big on-screen pause button, so that I can stop the game without knowing keyboard shortcuts.
3. As a child player, I want an on-screen "blast" button when playing without a microphone, so that keyboard-free devices can still play the practice mode.
4. As a child player, I want the game to fill my phone's screen nicely in portrait without stretched or blurry graphics, so that it feels made for my device.
5. As a child player, I want taps on the game to never scroll, zoom, or pull-to-refresh the page, so that playing never accidentally breaks the game.
6. As a child player, I want the game buttons big enough for small fingers, so that I never miss because a button was tiny.
7. As a child player, I want an app icon on the home screen that opens straight into Light Haven, so that I can start playing without a grown-up typing a web address.
8. As a child player, I want the app to stay in portrait orientation, so that the game never flips sideways mid-play.
9. As a child player playing the Phoneme Hatchery, I want the egg game to fit my phone screen with nothing cut off, so that I can hatch eggs on the couch, not just at a desk.
10. As a parent, I want the mic setup screen to trigger the phone's normal microphone permission prompt, so that granting access feels like every other app.
11. As a parent, I want the mic to keep working the same way inside the installed app as in the browser (local processing, nothing uploaded by Phonic Blaster), so that the privacy promise carries over to mobile.
12. As a parent, I want to export saved practice clips from the installed app through the phone's share sheet (to Drive, email, files), so that clip export works even though WebView browser downloads don't.
13. As a parent, I want the exported clips to keep their manifest and prompted-label caveat, so that the honesty rules survive the new export path.
14. As a parent, I want to sign in to the Hatchery with my email and password inside the app, so that the parent-first consent flow works identically on mobile.
15. As a parent, I want my sign-in session to persist across app restarts, so that I don't re-authenticate every time my child wants to play.
16. As a parent, I want onboarding (creating a child profile) to be comfortable on a phone keyboard and small screen, so that setup can happen entirely on the phone.
17. As a parent, I want the pause button reachable without covering gameplay, so that I can stop the game instantly with one thumb.
18. As a reviewer, I want the reviewer page usable on a phone (scrollable lists, tappable verdict buttons, playable audio), so that I can review clips away from a desktop.
19. As an admin, I want the admin page to at least render and scroll correctly on a phone, so that nothing in the app is broken on mobile even if deep admin work stays on desktop.
20. As a developer, I want touch input translated into the same semantic intents (move-left, move-right, fire, pause) that keyboard events produce, so that gameplay logic never knows which input device was used.
21. As a developer, I want the touch-intent mapping to be a pure module (pointer geometry in, intent out), so that it can be unit-tested later without Phaser or a browser.
22. As a developer, I want a single platform seam (web vs. native) that the rest of the code queries, so that Capacitor-specific behavior never leaks into game or page logic.
23. As a developer, I want the Capacitor app to load the exact output of the existing Vite build, so that there is one build pipeline and no mobile fork.
24. As a developer, I want `npx cap sync`-style steps wrapped in npm scripts, so that building the Android app is a documented one-liner.
25. As a developer, I want the game to remain fully playable in a desktop browser with keyboard exactly as before, so that mobile support is additive, not a migration.
26. As a developer, I want the mobile web experience verified in a browser (device viewport + touch emulation) before any native build, so that most iteration avoids the slow native cycle.
27. As a developer, I want the iOS project scaffolded and committed even though it can't be built on Windows, so that iOS work later starts from a known-good configuration instead of from scratch.
28. As a dataset curator, I want clips saved on mobile to be byte-identical in format to desktop clips (16 kHz mono WAV, prompted-label metadata), so that device diversity grows the future corpus without a new ingestion path.
29. As a dataset curator, I want phone-mic recordings to become possible at all, so that the corpus can eventually include the device class children actually use.
30. As a maintainer, I want the Vercel web deployment to remain the unchanged default target, so that adding native packaging risks nothing for existing users.
31. As a maintainer, I want native platform folders and any generated assets organized so the web-only workflow never touches them, so that web contributors can ignore Capacitor entirely.
32. As a maintainer, I want only open-source dependencies (Capacitor core + official plugins), so that the app stays freely distributable with no paid services.
33. As a maintainer, I want a README section covering Android build/run steps and the platform seam, so that contributors can produce a phone build unaided.
34. As a maintainer, I want the detection thresholds to remain tunable in the one typed config, so that phone-mic tuning sessions (a known follow-up) need no code hunting.

## Implementation Decisions

### TouchControls (new deep module, Phonic Blaster)

- A pure intent-mapping core: given pointer coordinates, playfield geometry, and current input mode, it returns a semantic intent — `laneLeft`, `laneRight`, `fire`, `pause`, or none. No Phaser types in the core.
- Tap zones: tapping the left/right half of the playfield moves one lane in that direction (matching the keyboard's one-tap-one-lane behavior). Zone boundaries come from design coordinates (540×720), not device pixels, so the FIT letterboxing does not skew them.
- A thin Phaser adapter subscribes to pointer events and dispatches intents into the exact same handlers the keyboard listeners call today (steering stays event-driven — the fix that replaced per-frame polling).
- On-screen pause button rendered in the play scene HUD; always visible, in a corner outside the main firing sightlines, sized ≥44 design px.
- On-screen fire button appears **only** in no-mic (simulated voice) mode and drives the existing SimulatedVoice adapter, so shots, cooldown, heat, and stats flow through the identical path as spacebar. In mic mode there is no fire button — the voice is the trigger.
- Touch and keyboard coexist unconditionally; no device sniffing to disable either.
- The mic setup, pause, and game-over scenes already use Phaser pointer buttons and need only touch-target size review, not new input handling.

### Responsive shell (all routes)

- Viewport meta gains `viewport-fit=cover`; safe-area insets (`env(safe-area-inset-*)`) pad fixed/absolute chrome (back links, headers, the game shell) so notches and home-indicator bars never cover content.
- All `100vh` usages move to dynamic viewport units (`dvh`) with a `vh` fallback, fixing the mobile-browser address-bar collapse.
- Media-query passes per route group, in priority order: Phonic Blaster shell, home, onboarding, Hatchery play, reviewer, admin ("Everything" scope — Daniel's decision). Reviewer/admin standard: everything rendns, scrolls, and is tappable on a phone; multi-column layouts stack; tables scroll horizontally in their own container.
- Touch targets across pages meet ~44px minimum; form inputs use font sizes that don't trigger iOS auto-zoom.
- The game mount area sets `touch-action: none`/`manipulation` as appropriate and suppresses overscroll/pull-to-refresh and double-tap zoom; text selection and long-press callouts are disabled on the canvas.
- The Phaser config keeps FIT scaling, 540×720 design coordinates, and the DPR-scaled backing store; on orientation/viewport changes the scale manager refreshes. No landscape layout is designed — the app is portrait-locked (below).

### Platform seam (new deep module, shared)

- One small interface consumed by the rest of the codebase: platform detection (`isNative`, backed by Capacitor's native-platform check — the same platform-abstraction module pattern light-haven-home's Capacitor PRD defines), clip export, and app-chrome hints (status bar style, orientation lock). Exactly two implementations: web (current behavior: anchor-click downloads) and Capacitor (native filesystem + share sheet).
- ClipStore keeps owning WAV encoding, IndexedDB, and the manifest; it hands a list of named blobs (manifest first) to the platform's export function and no longer knows how files leave the device.
- Native export: write manifest + WAVs into an app-scoped export folder, then invoke the system share sheet on the batch. The manifest's prompted-label caveat text is unchanged. Zip bundling remains deferred (same stance as v1).
- IndexedDB and localStorage (clip store, save-clips flag, theme, Supabase session) work as-is inside the WebView; no storage migration.

### Capacitor scaffolding (light-haven-home conventions)

- Capacitor core + CLI added to the existing apps package; a `capacitor.config.ts` at the package root with `webDir` pointing at the existing Vite build output and **`server.androidScheme: 'https'`** — the light-haven-home PRD's requirement for Supabase localStorage sessions to survive in the Android WebView. No separate mobile build — `npm run build` then `cap sync`, wrapped in a `build:native`-style npm script exactly as light-haven-home defines it. Vite config and the Vercel deployment are untouched.
- App identity follows the light-haven-home naming pattern (`com.lighthavenhome.app` there): a reverse-domain appId under the Light Haven name for this app (final string chosen at implementation with Daniel); display name "Light Haven Apps". Icon/splash generated from a single 1024×1024 source via the `@capacitor/assets` CLI, same as the home app's plan.
- Native platform folders (`android/`, `ios/`) are committed to git, per the home PRD's rationale: they carry manual configuration (permissions, URL scheme, orientation) and committing them means anyone can build without re-running `cap add`.
- **Android (buildable now)**: manifest declares `RECORD_AUDIO` (and `MODIFY_AUDIO_SETTINGS`); the WebView's `getUserMedia` permission request is granted automatically once the app holds the native permission, so the mic-enable button must first ensure the native runtime permission (native prompt on first use, then the in-page flow proceeds). Portrait orientation fixed in the manifest.
- **iOS (scaffolded only)**: platform committed with `NSMicrophoneUsageDescription`, portrait-only orientation, and a minimum iOS version that guarantees `getUserMedia` + AudioWorklet in WKWebView (iOS 15 floor). Explicitly not built or tested in this PRD — first real iOS run happens when a Mac is available.
- Keyboard plugin configured with `resize: 'body'` (config-only, from the home PRD) so form screens (auth, onboarding) reflow cleanly when the on-screen keyboard opens.
- Status bar styled to match the app background on launch (dark icons-on-dark-theme treatment, matching the home PRD's approach with this app's palette).
- Router: BrowserRouter is kept. The native app always cold-starts at the index; all further navigation is client-side, so no server-side SPA fallback is needed inside Capacitor. (The Vercel rewrite keeps handling deep links on the web.)
- The AudioWorklet module and other public assets load from the app bundle via Capacitor's local origin — same absolute paths as the web build.
- Environment variables: Vite inlines `VITE_*` values at build time, so the native build carries the same Supabase URL/key as the web build with no extra plumbing (home PRD decision, verified applicable here).

### Supabase auth (same setup as light-haven-home — Daniel's direction)

- The Supabase client adopts light-haven-home's explicit auth options: `autoRefreshToken: true`, `persistSession: true`, `detectSessionInUrl: true`. (These are supabase-js defaults the current client already gets implicitly; making them explicit keeps the two codebases' setups identical and intentional.)
- Auth method stays email + password via the existing parent-first gate — same as the home app; no OAuth providers are added. Sessions persist in WebView localStorage across app restarts (enabled by the `https` androidScheme above).
- **Auth email deep linking, home-app pattern**: Supabase auth emails (password reset, and email confirmation if enabled) must open the app rather than a dead browser tab. Same three steps as the home PRD: (1) register a custom URL scheme for this app in both native projects, (2) add the scheme's auth-callback URL to the Supabase project's allowed redirect URLs alongside the production web URL, (3) listen for Capacitor's `appUrlOpen` event and hand the URL's tokens to the Supabase client (`setSession` / existing `onAuthStateChange` flow). On the web, `detectSessionInUrl` keeps handling these links unchanged.
- The scheme is this app's own (not the home app's `lighthaven://`) so the two installed apps never fight over links; the *pattern* is identical.
- Hatchery uploads and reviewer/admin queries work as on the web over HTTPS, network-permitting.

## Testing Decisions

- **Decision (Daniel, 07-16-2026): no unit tests in this PRD.** Consistent with Phonic Blaster v1's prototype-first stance. The TouchControls intent core and the platform seam are deliberately pure/swappable so behavior-level tests (inputs in → intents/calls out, never internal state) can land cheaply once gameplay feel stabilizes.
- **Primary verification is scripted browser sessions via the playwright-cli skill (Daniel's direction)**: mobile viewport emulation (device dimensions, DPR, touch events) drives the touch controls, responsive layouts, safe-area behavior, and the no-mic fire button end-to-end in a desktop browser before any native build. This mirrors how v1 was verified (scripted keyboard play sessions).
- The mic path is verified in-browser with a fake audio device (as in v1), then on a real Android device via the Capacitor run loop: native permission prompt, live level meter, pop test, real play session, clip save, and share-sheet export.
- What playwright cannot cover — the native permission prompt, share sheet, portrait lock, WebView audio quirks — is verified manually on Daniel's Android hardware; per the project's completion rule, nothing is "complete" until Daniel has run it on a real phone and said so.
- Detection-threshold feel on real child + phone mic remains an open tuning task tracked from the v1 PRD; this PRD only requires that the thresholds stay reachable in the typed config.

## Out of Scope

- **iOS building, testing, or distribution** — the Xcode project is scaffolded and committed, nothing more. No Mac workflow, no TestFlight, no App Store.
- **Play Store publishing** — this PRD ends at a locally installable, debug/sideloadable Android app. Store listing, signing pipelines, and release tracks are a follow-up.
- **Landscape layouts** — the app is portrait-locked; no landscape design work.
- **PWA packaging** (manifest/service worker/offline) — Capacitor is the chosen installable path; offline support is not attempted.
- **Zip bundling for clip export** — still deferred from v1; native export shares individual files plus the manifest.
- **New gameplay features** — no changes to rounds, cooldown, heat, scoring, or detection logic; this PRD is input, layout, and packaging only.
- **Phoneme correctness / classifier integration** — unchanged v2 seam, untouched here.
- **Native performance optimization** — the WebView runs the same JS; no native rendering or audio-processing rewrite.
- **Reviewer/admin mobile redesign** — they get the responsive pass (usable, scrollable, tappable), not a rethought mobile UX.

## Further Notes

- **Sequencing intent**: touch controls + responsive CSS land first — at that point the whole app is playable in any mobile browser over LAN, which is the fast iteration loop — and the Capacitor wrap lands second, when the only remaining unknowns are native ones (permissions, share sheet, orientation, WebView audio).
- **progress.md discipline continues** (Daniel's standing request): the Phonic Blaster progress file tracks the touch/responsive/native milestones as they land, statuses meaning "implemented and awaiting Daniel's verification".
- The platform seam is deliberately the only place allowed to import Capacitor APIs. Everything else — game, ClipStore, pages — must behave identically under `vite dev` in a desktop browser, which keeps the v1 test harness (keyboard mode, fake mic) fully valid.
- Phone-mic clips are a corpus-diversity opportunity (new device channel class), but the same honesty rules apply: prompted labels only, human review before any training use, consistent with the project's holdout/trust discipline.
- The existing high-DPI camera work in the game transfers directly to phones (typically DPR 2–3), so no new rendering work is expected beyond verifying text sharpness on device.
