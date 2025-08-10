danielmhair: Okay, here is my theme.
https://www.notion.so/Theme-Automatic-Phoneme-Recognition-APR-22b502b4855680cba080ea07b68a3127
I want to start working on Epic 1
https://www.notion.so/Epic-Live-Phoneme-CTCs-22b502b4855680149d70eec42adf84d3

What would you suggest as the first task?

GitHub Copilot: Kicking off Epic “Live Phoneme CTCs,” the single highest‑leverage first task is a focused streaming inference spike to prove end‑to‑end viability (audio → framed chunks → model → CTC greedy decode → incremental phoneme output) with latency + quality instrumentation.

Proposed First Task (Spike): Minimal Streaming Phoneme Loop (Greedy CTC)

Goal:
- Capture live mic audio (or simulate with a wav file stream) in small frames (e.g. 20–40 ms).
- Feed buffered chunks into an existing acoustic/embedding model you already use (e.g. wav2vec2) without retraining.
- Run a lightweight CTC head (existing classifier or a temporary linear layer) to produce logits.
- Perform incremental greedy decoding (no beam, no LM) and emit provisional phoneme symbols as soon as stable.
- Measure per-step latency (capture→logit→decode) and overall real-time factor (RTF).
- Log basic accuracy proxy: compare concatenated stream decode vs. offline full-file decode for the same audio sample.

Why this first:
- De-risks architectural assumptions (chunk size, buffering, model warmup).
- Surfaces immediate gaps (need for state caching, normalization differences, device placement).
- Establishes performance baseline before investing in beam search, LM integration, or endpointing heuristics.

Acceptance Criteria:
1. Script (e.g. src/stream_ctc_spike.py) runs on a short test wav and (optionally) live mic.
2. Emits incremental phoneme outputs to console (e.g. with timestamps).
3. Logs: avg chunk latency, max latency, real-time factor, number of emitted phonemes.
4. Produces a final concatenated phoneme string; difference ratio vs. offline decode printed.
5. Clear TODO comments noting where future enhancements plug in (beam, LM, silence handling, confidence).
6. A minimal test (can be skipped if no mic) that streams a known wav and asserts the pipeline produces non-empty output within time bounds.

Definition of Done:
- Code committed (spike label) + brief README section or doc note.
- No refactors of core training code yet—purely additive.
- Identified next bottlenecks listed (e.g., GPU warmup, batching opportunities, phoneme smoothing).

Suggested Subtasks (executed inside the spike if small, else queue them):
- Choose / load model (re-use embedding extractor used in s2_extract_embeddings_for_phonemes.py).
- Implement chunked audio iterator (file + optional PyAudio/sounddevice backend).
- Frame buffering & overlap (e.g., keep 50% overlap if model needs context).
- Forward pass & logits stitching (handle left context if required).
- Greedy decode with collapse + blank removal per chunk; stabilization rule (only emit when a symbol persists across N consecutive chunks).
- Latency + RTF instrumentation.
- Offline full-file decode comparator.

Immediate Assumptions (state if wrong and adjust later):
- Existing embedding + classifier stack can process arbitrary-length tensors; if not, will need sliding window with stride.
- Phoneme inventory already finalized in tokenizer/phoneme_vocab.txt.
- Real-time target RTF < 0.5 initially acceptable.

Next Tasks (after spike, not now):
- Add beam search + optional LM.
- Confidence scoring & temporal smoothing.
- Endpoint detection (silence / stable tail).
- Packaging a streaming API (WebSocket / REST) and client CLI integration.

Let me know if you’d like me to scaffold the spike file now.

danielmhair: So is this spike your suggesting to makethe wav2vec2-ctc?

GitHub Copilot: Yes—this first spike is about standing up a minimal streaming wav2vec2 → CTC decoding loop (not full training yet). Think “can we feed live (or simulated) audio frames into a wav2vec2 backbone plus a skinny CTC head and incrementally emit phonemes with acceptable latency?”

Clarify the scope:
1. Backbone: Use an existing pretrained wav2vec2 (HF) or your current embedding extractor (if already wav2vec2-based). No fine‑tuning in the spike.
2. CTC head: A simple Linear(hidden_dim, phoneme_vocab_size) + log softmax. For the spike it can be:
   - Randomly initialized (outputs will be noisy, but lets you validate latency + mechanics), OR
      - Loaded from your existing classifier layer if it already matches the phoneme vocab, OR
         - Borrow a pretrained wav2vec2-CTC model and remap labels to your phoneme set if close enough.
3. Streaming emulation: Slice a wav file into overlapping chunks; later replace with mic input.
4. Greedy decode per chunk, collapse repeats + drop blanks, and only “commit” a phoneme once it’s stable across N chunks (simple stabilization heuristic).
5. Metrics: per-chunk latency, real-time factor, emitted phoneme sequence vs. an offline full-file decode.

Why not train yet? You first want to:
- Confirm chunk sizing / overlap needed for stable predictions.
- Measure baseline latency (Python I/O + model forward).
- Expose any buffer/state handling issues (wav2vec2 has convolutional front-end that benefits from overlap).

If you decide you want actual meaningful phoneme outputs in the spike (not just timings), fastest path:
Option A: Start from a wav2vec2 model already trained for phoneme CTC in a related language and adapt label mapping.
Option B: Load your current offline classifier (if it already gives per-frame phoneme logits) and simply wrap it with a CTC collapsing step (if outputs are frame-aligned without explicit CTC blank you may add a synthetic blank channel and do argmax with a spacing rule—slightly hacky).
Option C: Do a quick one-epoch fit of the linear head on a small subset to get non-random logits (optional; adds time).

Decision recommendation:

For the spike: Use a pretrained wav2vec2-base (or the model you already rely on) + freshly initialized linear head just to exercise the pipeline. Add a TODO to swap in trained weights next.
Parallel small task: export/derive a phoneme label mapping JSON (id → symbol) if not already explicit.
Next actions I can take if you want:

Scaffold src/stream_wav2vec2_ctc_spike.py with chunking, model load placeholder, greedy decoder, metrics hooks.
Add a simple CLI arg: --wav path.wav (and later --mic).
Add a label mapping loader from tokenizer/phoneme_vocab.txt.
Let me know if you’d like me to create the scaffold now.

Restore Checkpoint

Gathering context from existing requirements and relevant scripts to integrate with the new streaming spike scaffold. Reading tokenizer vocabulary to map phoneme IDs. Listing src directory to see existing module patterns for CLI scripts. Reading an existing CLI script for style conventions. Reading another script for patterns. Creating new streaming spike scaffold file with chunked audio reading, wav2vec2 model load, dummy linear CTC head, greedy decoding, stabilization, metrics, and offline comparison. 

Made changes.

Fixing lint/style issues and removing unused imports/variables in the new spike file. 
```


Made changes.


```
Reading the spike file to fix indentation and minor errors. Fixing indentation error and adding proper indentation for coverage calculation. 
```


Made changes.


```
Adjusting blank line spacing to satisfy linter and removing unused variable. 
```


Made changes.


```
Polishing the new spike script: fixing style (blank lines), mild refactor for device move, removing unused vars, and addressing type hints. Re-reading file to apply precise patch with accurate context. Applying targeted patch: add blank line before dataclasses, adjust device placement code, remove unused start_global variable. 
```


Made changes.


```
Refining the spike script to resolve style/type lint complaints. 
```


Made changes.


```
Silencing type checker complaints by adding type ignore comments and ensuring model is moved to device without linter errors. 
```


Made changes.


```
Implemented the streaming spike scaffold.

Checklist:
- Added new script stream_wav2vec2_ctc_spike.py
- Loads wav2vec2 backbone + random CTC head
- Reads phoneme vocab (adds blank)
- Chunked overlapping streaming over a wav file
- Greedy per-chunk CTC collapse + simple stabilization
- Emits incremental phonemes with timestamps
- Collects latency metrics (avg/max + RTF proxy)
- Performs offline full-file pass and coverage comparison
- Lists clear next-step TODOs

Notes:
- Head is untrained; outputs are meaningless for now—latency/mechanics only.
- Resampling & live mic input intentionally deferred.
- Added type: ignore comments to bypass HF typing quirks.

Try it (example):
python -m src.stream_wav2vec2_ctc_spike --wav path/to/sample.wav

Let me know if you want mic streaming, a tiny test, or integration with existing workflow next.

---

Add task MD file that is in the tasks folder. You will need to make that folder as well. And in that file, add all of the information that you described about this Spike task and then include the checklist that you have and check off what you have completed

---

Checking for the existing streaming spike script location before creating the task documentation. Creating tasks directory and adding a markdown file documenting the spike with checklist and completion status. 

Made changes.

Refining the markdown task file to satisfy style/lint rules (blank lines around headings/lists and code block language). Applying markdown formatting fixes: add blank lines before/after headings and lists; specify code block language. 
```


Made changes.


```
Adding a blank line before the list under Implementation Summary to satisfy markdown lint rule. 
```


Made changes.


```
Added the task markdown file with full spike description and checklist, created tasks/ directory, and fixed markdown lint issues. Let me know if you’d like separate issue files for each remaining unchecked item or to add mic mode next.