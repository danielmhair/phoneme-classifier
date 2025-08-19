📊 Architecture Analysis: Current System vs CTC Implementation

  🏗️ Current Architecture

  Pipeline Structure (13-step workflow):
  Audio Files → Wav2Vec2 Embedding → Mean Pooling → MLP → Classification
       ↓              ↓                    ↓         ↓
    recordings/   embeddings/        single vector  37 classes

  Key Components:
  - Feature Extraction: Wav2Vec2 (facebook/wav2vec2-base)
  - Aggregation: Mean pooling over time dimension
  - Classification: sklearn MLPClassifier (128, 64 hidden layers)
  - Output: Fixed 37 phoneme classes
  - Deployment: ONNX models for Unreal Engine

  🎯 CTC Implementation Gaps

  Critical Architecture Limitations

  1. Temporal Information Loss
  - Current: Mean pooling destroys time dimension
  - CTC Needs: Frame-level predictions for temporal alignment

  2. Sequence Modeling Absence
  - Current: Single embeddings → single predictions
  - CTC Needs: Variable-length sequences → variable-length outputs

  3. Loss Function Mismatch
  - Current: Cross-entropy loss (requires alignment)
  - CTC Needs: CTC loss (alignment-free)

  4. Fixed-Length Processing
  - Current: 1-second audio clips
  - CTC Needs: Variable-length streaming capability

  🔧 Minimal CTC Architecture (Non-Overengineered)

  Phase 1: Core CTC Implementation

  Modified Pipeline:
  Audio → Wav2Vec2 Sequence → LSTM → CTC Head → Phoneme Sequence
     ↓          ↓             ↓        ↓           ↓
  Variable   (T, 768)    (T, 128)  (T, 38)   Variable length

  Key Changes:

  1. s2_extract_embeddings.py
  # BEFORE: Mean pooling
  emb = outputs.last_hidden_state.mean(dim=1).squeeze()

  # AFTER: Keep temporal sequence
  emb = outputs.last_hidden_state.squeeze()  # (T, 768)

  2. s3_classifier_encoder.py → s3_ctc_classifier.py
  # Replace MLPClassifier with PyTorch CTC model
  class CTCClassifier(nn.Module):
      def __init__(self, input_dim=768, hidden_dim=128, num_classes=38):
          super().__init__()
          self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
          self.classifier = nn.Linear(hidden_dim, num_classes)  # +1 for CTC blank
          self.ctc_loss = nn.CTCLoss(blank=37)

  3. Inference Updates
  - Add CTC decoding (greedy or beam search)
  - Support variable-length outputs
  - Maintain ONNX export compatibility

  Phase 2: Live Processing (Future Epics)

  Streaming Architecture:
  - Sliding window processing
  - Real-time CTC decoding
  - Temporal buffering

  📋 Implementation Recommendations

  Immediate Steps (Epic 1)

  1. Data Compatibility ⚡
  - Keep existing single-phoneme recordings
  - Create sequence labels by phoneme repetition
  - Gradual transition to continuous speech

  2. Backward Compatibility 🔄
  - Maintain existing pipeline structure (s0-s12)
  - Add CTC as alternative to MLP
  - Keep ONNX export for Unreal integration