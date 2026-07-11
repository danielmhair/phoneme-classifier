# Research Sources — Children's Phoneme Recognition for UE5

**Compiled:** 2026-07-08
**Scope:** On-device children's phoneme recognition / pronunciation assessment, UE5 runtime integration, and related open-source models.

## Provenance note (read this first)

This is an **honest, deduplicated** list of the sources actually surfaced in this research, **not** a padded list of 330. Where a single paper appeared under multiple URLs (arXiv + PubMed + PMC + publisher mirror, etc.), it is listed **once** with mirrors noted. Two buckets are distinguished:

- **Directly retrieved** — appeared as a returned result in a search/fetch during this work.
- **Referenced within sources / deep-research pass** — key papers discussed in the analysis that came through other sources' reference lists or the initial deep-research report's citations. These are flagged so you know they weren't necessarily opened end-to-end.

**Total unique sources: 63** (55 directly retrieved + 8 referenced/deep-research).

If you want, I can go back and expand any single line into a full annotated summary, or pull the PDFs into the repo.

---

## 1. Children's speech & child ASR

1. Group-Aware Partial Model Merging for Children's ASR — arXiv 2511.23098 — https://arxiv.org/pdf/2511.23098
2. NTT — 18 papers at Interspeech 2025 (incl. SSL phoneme recognizer for child phonological error analysis) — https://group.ntt/en/topics/2025/08/07/interspeech2025.html
3. The Interspeech 2025 Speech Accessibility Project (SAP) Challenge — arXiv 2507.22047 — https://arxiv.org/pdf/2507.22047
4. ChildVox: Speech/Audio/LALM Benchmark across Childhood — arXiv 2605.29257 — https://arxiv.org/pdf/2605.29257
5. Transcribing Children's Speech: ASR Performance & Reliable Orthographic Transcriptions — arXiv 2605.28833 — https://arxiv.org/pdf/2605.28833
6. Closing the Child Speech Recognition Gap: Evidence, Limitations, and Paths Forward — The Learning Agency (Feb 2026) — https://the-learning-agency.com/guides-resources/closing-the-child-speech-recognition-gap-evidence-limitations-and-paths-forward/
7. SHNU Multilingual Conversational ASR System — Interspeech 2025 MLC-SLM Challenge — arXiv 2507.03343 — https://arxiv.org/pdf/2507.03343
8. Data augmentation using prosody and false starts to recognize non-native children's speech (Aalto, Interspeech 2020) — arXiv 2008.12914 — https://arxiv.org/pdf/2008.12914
9. Low Resource German ASR with Untranscribed Data Spoken by Non-native Children (Interspeech 2021 shared task) — arXiv 2106.09963 — https://arxiv.org/pdf/2106.09963

## 2. Pronunciation assessment, GOP & mispronunciation detection (MDD)

10. speechocean762: Open-Source Non-native English Speech Corpus for Pronunciation Assessment — arXiv 2104.01378 — https://arxiv.org/abs/2104.01378 *(mirrors: ar5iv, arxiv/pdf; OpenSLR #101; original Interspeech 2021 paper zhang21x also surfaced in the deep-research citations)*
11. Read to Hear: Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs — arXiv 2509.14187 — https://arxiv.org/pdf/2509.14187
12. PRiSM: Benchmarking Phone Realization in Speech Models — arXiv 2601.14046 — https://arxiv.org/pdf/2601.14046
13. Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling (VCTK-accent) — arXiv 2507.14346 — https://arxiv.org/pdf/2507.14346
14. Automated Pronunciation Evaluation for Korean Toddler Speech (diarization + SSL) — arXiv 2606.10213 — https://arxiv.org/pdf/2606.10213
15. Evaluating Logit-Based GOP Scores for Mispronunciation Detection (Interspeech 2025, parikh25b) — arXiv 2506.12067 — https://arxiv.org/html/2506.12067v2 *(mirror: isca-archive interspeech_2025/parikh25b)*
16. Phonological-Level wav2vec2-based Mispronunciation Detection and Diagnosis — arXiv 2311.07037 — https://arxiv.org/pdf/2311.07037
17. Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge (Interspeech 2025, parikh25) — arXiv 2506.02080 — https://arxiv.org/html/2506.02080v2 *(mirror: isca-archive interspeech_2025/parikh25)*
18. Light-weight Pronunciation Assessment via Discrete Speech Token Surprisal — arXiv 2606.19910 — https://arxiv.org/pdf/2606.19910
19. Phoneme-Level Mispronunciation Screening in Polish-Speaking Children with an Explainable Assistant — arXiv 2606.25181 — https://arxiv.org/pdf/2606.25181
20. Segmentation-free Goodness of Pronunciation — arXiv 2507.16838 — https://arxiv.org/pdf/2507.16838

## 3. Korean SSD validation study (the one we discussed in depth)

21. Kim, Jeong, Kang, Ahn, Hong, Im, Kim, Kim, Jang (2025) — *Usefulness of ASR Assessment of Children With Speech Sound Disorders: Validation Study*, J Med Internet Res 27:e60520 — https://www.jmir.org/2025/1/e60520/ *(mirrors: PMC11775490; PubMed 39576242; ScienceDirect S1438887125000627; ScienceOpen. Note: authors' ML affiliations MediaZen → Enuma / LG AI Research; dataset not public, available from corresponding author on request.)*
22. Ahn et al. — *ASR for the Diagnosis of Pronunciation of Speech Sound Disorders in Korean Children* (precursor preprint; wav2vec2 XLS-R, 137 children / 73 words) — arXiv 2403.08187 — https://arxiv.org/pdf/2403.08187

## 4. Datasets

23. MyST (My Science Tutor) — Pradhan, Cole, Ward (2024), LREC-COLING — corpus site: https://myst.cemantix.org (commercial license via Boulder Learning) *(also cited within ChildVox refs)*
24. IPAPack++ — 17,132 h normalized IPA transcriptions (curated for ZIPA; see #56) — via arXiv 2505.23170
25. SpeechOcean762 — free for commercial + non-commercial use, OpenSLR #101 — see #10

## 5. Unreal Engine / NNE / runtime integration

26. Speed Up Unreal Engine NNE Inference with NVIDIA TensorRT for RTX Runtime (NNERuntimeTRT, UE 5.7) — NVIDIA Technical Blog — https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime
27. NNERuntimeORT | Unreal Engine 5.8 Documentation — https://dev.epicgames.com/documentation/en-us/unreal-engine/API/PluginIndex/NNERuntimeORT
28. Neural Network Engine Overview | Unreal Engine 5.8 Documentation ("Beta … use caution when shipping") — https://dev.epicgames.com/documentation/unreal-engine/neural-network-engine-overview-in-unreal-engine
29. microsoft/OnnxRuntime-UnrealEngine (style-transfer sample; up to UE 5.2) — https://github.com/microsoft/OnnxRuntime-UnrealEngine
30. ONNX Runtime sample port for UE 5.5 using NNE RDG (GPU-only) — Epic Dev Community Forums — https://forums.unrealengine.com/t/onnx-runtime-sample-port-for-unreal-engine-5-5-using-nne-rdg-gpu-only-inference/2669899
31. Neural Network Engine (Unreal Engine) — Grokipedia (version history: experimental 5.2 → ORT default 5.5) — https://grokipedia.com/page/Neural_Network_Engine_Unreal_Engine
32. NNEngine – Neural Network Engine (marketplace plugin; separate ORT version) — https://www.unrealengine.com/marketplace/en-US/product/nnengine-neural-network-engine/questions
33. albertotrunk/UE5-ONNX (NNI plugin sample) — https://github.com/albertotrunk/UE5-ONNX
34. Course: Neural Network Engine (NNE) — Forums, p.10 (NPU/DirectML support, ORT upgrade cadence) — https://forums.unrealengine.com/t/course-neural-network-engine-nne/1162628?page=10

## 6. sherpa-onnx & on-device runtimes

35. sherpa-onnx pretrained models index (k2-fsa) — https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
36. sherpa-onnx documentation index — https://k2-fsa.github.io/sherpa/onnx/index.html
37. sherpa-onnx release: asr-models (incl. Cohere 14-lang, 2026) — https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
38. sherpa-onnx INT8 STT models (Kroko / Banafo Zipformer2 transducer) — https://huggingface.co/hudaiapa88/sherpa-stt-onnx
39. Sherpa-onnx — Grokipedia (project history, WASM 2025, NPU backends) — https://grokipedia.com/page/Sherpa-onnx
40. sherpa-onnx offline transducer models — https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
41. sherpa streaming Zipformer-CTC models (RST docs) — https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/online-ctc/zipformer-ctc-models.rst
42. k2-fsa/sherpa-onnx Releases (QNN / Qualcomm NPU support, 2026) — https://github.com/k2-fsa/sherpa-onnx/releases
43. ASR in Unreal Engine C++ · k2-fsa/sherpa-onnx Discussion #1960 (C API vs C++ API + UE GC) — https://github.com/k2-fsa/sherpa-onnx/discussions/1960 *(from the deep-research report's citations)*
44. VietSuperSpeech (uses sherpa-onnx / Zipformer-30M-RNNT pseudo-labeling) — arXiv 2603.01894 — https://arxiv.org/pdf/2603.01894

## 7. Moonshine & edge-STT surveys

45. moonshine-ai/moonshine (Moonshine Voice; ONNX/.ort, on-device) — https://github.com/moonshine-ai/moonshine
46. Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications — arXiv 2602.12241 — https://arxiv.org/abs/2602.12241
47. Moonshine vs Whisper ASR: Real-Time Benchmark (2026) — ModelsLab blog — https://modelslab.com/blog/audio-generation/moonshine-vs-whisper-asr-real-time-speech-2026
48. Best open-source speech-to-text models in 2026 — Gladia — https://www.gladia.io/blog/best-open-source-speech-to-text-models

## 8. Allosaurus & universal phone recognition

49. xinjli/allosaurus — universal phone recognizer, 2000+ languages (repo + README + releases/issues/packages) — https://github.com/xinjli/allosaurus
50. Universal Phone Recognition with a Multilingual Allophone System (ICASSP 2020; the Allosaurus paper) — arXiv 2002.11800 — https://arxiv.org/pdf/2002.11800
51. Fine-tuning Allosaurus — A Universal Phone Recognizer (tutorial) — Medium — https://medium.com/super-ai-engineer/fine-tuning-allosaurus-a-universal-phone-recognizer-41de0980b9e8
52. Phoneme Recognition through Fine-Tuning of Phonetic Representations: Luhya Language Varieties (fine-tunes Allosaurus w/ as few as 10 instances) — arXiv 2104.01624 — https://arxiv.org/pdf/2104.01624
53. CMULAB: Open-Source Framework for NLP Model Training/Deployment (Allosaurus-based phone API) — arXiv 2404.02408 — https://arxiv.org/pdf/2404.02408
54. Graham Neubig — Allosaurus pretrained models + fine-tuning code announcement — https://x.com/gneubig/status/1291337746108293120

## 9. ZIPA & newer phone-recognition models

55. ZIPA: A Family of Efficient Models for Multilingual Phone Recognition (ACL 2025; Zipformer CTC/transducer, benchmarks vs Allosaurus & wav2vec2-espeak) — arXiv 2505.23170 — https://arxiv.org/abs/2505.23170 *(mirror: ACL Anthology 2025.acl-long.961)*
56. BranchShine: Compact Raw-Audio-to-IPA Transcription with a RoPE E-Branchformer Encoder (surveys ZIPA/POWSM/PhoneticXEUS) — arXiv 2606.22824 — https://arxiv.org/pdf/2606.22824

## 10. Referenced within sources / from the deep-research pass

*(Discussed in the analysis; came via reference lists or the initial deep-research report rather than direct end-to-end retrieval.)*

57. Block Medin, Pellegrini, Gelin (2024) — *Self-supervised models for phoneme recognition: applications in children's speech for reading learning*, Interspeech 2024 — arXiv 2503.04710
58. Horii, Tawara, Ogawa, Araki (2025) — *Why is children's ASR so difficult? Analyzing children's phonological error patterns using SSL-based phoneme recognizers*, Interspeech 2025 — isca-archive interspeech_2025/horii25
59. Zhang, Yue, Patel, Scharenborg (2024) — *Improving child speech recognition with augmented child-like speech*, Interspeech 2024
60. Fan, Balaji Shankar, Alwan (2024) — *Benchmarking Children's ASR with Supervised and Self-supervised Speech Foundation Models*, Interspeech 2024
61. Shivakumar, Potamianos, Lee, Narayanan (2020) — *Transfer learning from adult to children for speech recognition*, Computer Speech & Language 63:101077
62. Getman et al. (2022) — *wav2vec2-based speech rating system for children with speech sound disorder*, Interspeech 2022
63. Dutta & Hansen (2025) — Whisper tiny.en fine-tuned on MyST, on Raspberry Pi 5 (RTF 0.23–0.41; COPPA/GDPR motivation) — arXiv 2507.14451

---

*Compiled honestly: 63 unique sources, deduplicated across mirrors. No fabricated entries.*