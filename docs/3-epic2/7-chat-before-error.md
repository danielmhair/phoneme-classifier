> Okay, now I want a testing script that will take each sound file, then run it through each of our 3 models. But we gather up the results, then output
   a .json file showing the results of every phoneme we have (from the labels.json). So the output would have each phoneme and show the top 5 results
  from each model and their scores, then another .json of a sum percentage of how many right per phoneme over total for mlp, then wav2vec2 ctc then
  wavlm ctc. This will then allow me to send you these files for you to create an .md file.

● I'll create a comprehensive testing script that evaluates all three models against your phoneme test data and generates detailed JSON reports.

● Update Todos
  ⎿  ☐ Create comprehensive model evaluation script
     ☐ Generate detailed per-phoneme results JSON with top 5 predictions
     ☐ Generate accuracy summary JSON with percentage correct per model
     ☐ Test script with all three models on phoneme dataset
  ⎿  Interrupted by user

> Oh, the recordings and workflows/*/dist/organized_recordings are the files to use - choose either the recordings or one of the organized_recordings.
  Perhaps recordings is the best since its not trimmed and will feel more like real recordings

● List(recordings)
  ⎿  Listed 712 paths (ctrl+r to expand)

● Perfect! I can see the structure. The recordings directory has:
  - Multiple speakers (callie, cassie, chloe, dan, sky)
  - Phoneme directories with actual phonemes in IPA notation (I_aɪ/, a_æ/, etc.)
  - Raw audio files with clear naming convention

  Let me check the labels.json to understand the phoneme mapping:

● Read(workflows/mlp_control_workflow/dist/phoneme_labels.json)
  ⎿  Read 39 lines (ctrl+r to expand)

● Now I'll create the comprehensive testing script:

● Write(evaluation/model_comparison_test.py)
  ⎿  Wrote 391 lines to evaluation/model_comparison_test.py
     #!/usr/bin/env python3
     """
     Comprehensive Model Evaluation Script for Epic 1 & Epic 2 Integration
     Tests all three models (MLP Control, Wav2Vec2 CTC, WavLM CTC) against real phoneme recordings
     and generates detailed JSON reports for analysis.
     Usage:
         poetry run python evaluation/model_comparison_test.py
     … +381 lines (ctrl+r to expand)

● Update Todos
  ⎿  ☒ Create comprehensive model evaluation script
     ☐ Generate detailed per-phoneme results JSON with top 5 predictions
     ☐ Generate accuracy summary JSON with percentage correct per model
     ☐ Test script with all three models on phoneme dataset

● Now let me test the script with a small sample first to ensure it works:

● Bash(poetry run python -c "
      from evaluation.model_comparison_test import ModelEvaluator…)
  ⎿  🧪 Testing single file: recordings/callie/a_æ/callie_a_æ_ep-æ_20250422_180744_1.wav
     ✅ Loaded MLP Control (two-stage):
        🧠 Stage 1: Wav2Vec2 feature extractor
     … +30 lines (ctrl+r to expand)

● Great! The script is working and all three models are running properly with temporal architecture. Now let me run the full evaluation:

● Bash(poetry run python evaluation/model_comparison_test.py)
  ⎿  Running in the background (down arrow to manage)

● Let me check the progress:

● BashOutput(Reading shell output)
  ⎿  🚀 Starting Model Comparison Evaluation
     🎯 Testing all three Epic 1 models on real recordings
     … +14 lines (ctrl+r to expand)
Evaluating:   0%|          | 0/5864 [00:00<?, ?it/s]

● The evaluation is running on 5864 audio files! This will take a while. Let me check progress in a few moments:

● BashOutput(Reading shell output)
  ⎿ /home/danie/Workspaces/fast-api-phoneme-python/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py:315: UserWarning: Passing `gr
    adient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using `model.gradient_checkpointing_enable()
    ` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`.
    … +2 lines (ctrl+r to expand)



● At 4.8 seconds per file, this would take ~8 hours for all files. Let me create a faster version that samples a subset for testing purposes:

● Kill Bash(Kill shell: bash_6)
  ⎿  Shell bash_6 killed
  ⎿  Interrupted by user

> Yes, the faster version, but then I still want that 8 hour version too.

● I'll create both versions - a fast sampling version for quick testing and the full comprehensive version for complete evaluation.

● Write(evaluation/model_comparison_fast.py)
  ⎿  Wrote 394 lines to evaluation/model_comparison_fast.py
     #!/usr/bin/env python3
     """
     Fast Model Evaluation Script - Sample-based testing
     Tests all three models on a representative sample of recordings (5 files per phoneme per speaker)
     for quick validation and analysis.
     Usage:
         poetry run python evaluation/model_comparison_fast.py
     … +384 lines (ctrl+r to expand)

● Now let me update the original comprehensive version to be more efficient and add better progress tracking:

● Update(evaluation/model_comparison_test.py)
  ⎿  Updated evaluation/model_comparison_test.py with 15 additions and 2 removals
       246
       247            # Process each audio file
       248            print("🔄 Processing audio files...")
       249 +          start_time = time.time()
       250 +
       251            for i, (file_path, expected_phoneme) in enumerate(tqdm(audio_files, desc="Evaluating")):
       252                file_result = self._evaluate_single_file(file_path, expected_phoneme)
       253                self.detailed_results['per_file_results'].append(file_result)
     ...
       259                            if model_result['correct_prediction']:
       260                                self.accuracy_summary['per_phoneme_accuracy'][expected_phoneme][model_id]['correct'] += 1
       261
       262 -              # Progress update every 50 files
       263 -              if (i + 1) % 50 == 0:
       262 +              # Progress update and time estimation every 100 files
       263 +              if (i + 1) % 100 == 0:
       264 +                  elapsed = time.time() - start_time
       265 +                  rate = (i + 1) / elapsed  # files per second
       266 +                  remaining = len(audio_files) - (i + 1)
       267 +                  eta_seconds = remaining / rate if rate > 0 else 0
       268 +                  eta_hours = eta_seconds / 3600
       269 +
       270                    print(f"   📊 Processed {i + 1}/{len(audio_files)} files")
       271 +                  print(f"   ⏱️  Rate: {rate:.1f} files/sec, ETA: {eta_hours:.1f} hours")
       272 +
       273 +                  # Save intermediate results every 500 files
       274 +                  if (i + 1) % 500 == 0:
       275 +                      self._save_intermediate_results(i + 1)
       276
       277            # Calculate final percentages
       278            self._calculate_accuracy_percentages()

● Now let me add the intermediate results saving method:

● Update(evaluation/model_comparison_test.py)
  ⎿  Updated evaluation/model_comparison_test.py with 25 additions
       320
       321            self.accuracy_summary['overall_accuracy'] = overall_stats
       322
       323 +      def _save_intermediate_results(self, files_processed: int) -> None:
       324 +          """Save intermediate results during long evaluation."""
       325 +          eval_dir = Path("evaluation")
       326 +          eval_dir.mkdir(exist_ok=True)
       327 +
       328 +          # Calculate current percentages
       329 +          self._calculate_accuracy_percentages()
       330 +
       331 +          # Save intermediate results with timestamp
       332 +          timestamp = time.strftime('%Y%m%d_%H%M%S')
       333 +          intermediate_path = eval_dir / f"intermediate_results_{files_processed}_{timestamp}.json"
       334 +
       335 +          intermediate_data = {
       336 +              'files_processed': files_processed,
       337 +              'total_files': self.detailed_results['metadata']['total_files'],
       338 +              'progress_percentage': (files_processed / self.detailed_results['metadata']['total_files']) * 100,
       339 +              'current_accuracy': self.accuracy_summary['overall_accuracy'],
       340 +              'timestamp': timestamp
       341 +          }
       342 +
       343 +          with open(intermediate_path, 'w') as f:
       344 +              json.dump(intermediate_data, f, indent=2)
       345 +
       346 +          print(f"   💾 Intermediate results saved: {intermediate_path}")
       347 +
       348        def _save_results(self) -> None:
       349            """Save evaluation results to JSON files."""
       350            # Create evaluation directory if it doesn't exist

● Now let me run the fast evaluation first:

● Bash(poetry run python evaluation/model_comparison_fast.py)
Evaluating:   0%|          | 0/180 [00:00<?,
     ?it/s]/home/danie/Workspaces/fast-api-phoneme-python/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py:315: UserWarning:
     Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 Transformers. Using
     `model.gradient_checkpointing_enable()` instead, or if you are using the `Trainer` API, pass `gradient_checkpointing=True` in your
     `TrainingArguments`.
       warnings.warn(
Evaluating:   2%|▏         | 2/180 [00:06<09:02,  3.05s/it]
Evaluating:   3%|▎         | 5/180 [00:11<05:24,  2.10s/it]
Evaluating:   4%|▍         | 7/180 [00:14<04:55,  1.71s/it]
Evaluating:   6%|▌         | 10/180 [00:19<04:34, 1.64s/it]
Evaluating:   7%|▋         | 12/180 [00:22<04:32,  1.62s/it]
Evaluating:   8%|▊         | 15/180 [00:26<04:58,  1.80s/it]
Evaluating:   9%|▉         | 17/180 [00:31<04:41,  1.80s/it]
Evaluating:  11%|█         | 19/180 [00:34<04:25,  1.65s/it]
Evaluating:  12%|█▏        | 22/180 [00:37<04:11,  1.58s/it]
Evaluating:  13%|█▎        | 24/180 [00:42<04:09,  1.59s/it]
Evaluating:  14%|█▍        | 26/180 [00:45<04:08,  1.62s/it]
Evaluating:  16%|█▌        | 29/180 [00:48<03:58,  1.57s/it]
Evaluating:  17%|█▋        | 31/180 [00:53<03:59,  1.59s/it]
Evaluating:  18%|█▊        | 33/180 [00:56<03:45,  1.53s/it]
Evaluating:  20%|██        | 36/180 [00:59<03:51,  1.60s/it]
Evaluating:  21%|██        | 38/180 [01:04<03:42,  1.58s/it]
Evaluating:  22%|██▏       | 40/180 [01:07<03:34,  1.53s/it]
Evaluating:  24%|██▍       | 43/180 [01:10<03:25,  1.49s/it]
Evaluating:  25%|██▌       | 45/180 [01:14<03:25,  1.52s/it]
Evaluating:  26%|██▌       | 47/180 [01:18<03:39,  1.65s/it]
Evaluating:  28%|██▊       | 50/180 [01:21<03:24,  1.56s/it]
Evaluating:  29%|██▉       | 52/180 [01:26<03:22,  1.55s/it]
Evaluating:  30%|███       | 54/180 [01:29<03:24,  1.62s/it]
Evaluating:  32%|███▏      | 57/180 [01:33<03:29,  1.69s/it]
Evaluating:  33%|███▎      | 59/180 [01:37<03:04,  1.60s/it]
Evaluating:  34%|███▍      | 61/180 [01:40<02:58,  1.50s/it]
Evaluating:  36%|███▌      | 64/180 [01:43<02:54,  1.49s/it]
Evaluating:  37%|███▋      | 66/180 [01:48<02:54,  1.56s/it]
Evaluating:  38%|███▊      | 68/180 [01:50<02:48,  1.50s/it]
Evaluating:  39%|███▉      | 71/180 [01:53<02:38,  1.44s/it]
Evaluating:  41%|████      | 73/180 [01:58<02:48,  1.58s/it]
     🚀 Starting FAST Model Comparison Evaluation
     �

     ... [49234 characters truncated] ...

     wavlm_temporal.onnx
        📁 CTC: workflows/ctc_wavlm_workflow/dist/phoneme_ctc.onnx