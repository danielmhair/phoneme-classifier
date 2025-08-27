🏗️ CTC Implementation Design Specification

  📋 System Requirements & Constraints

  Current System Analysis

  - Pipeline: 13-step modular workflow (s0-s12)
  - Performance: <100ms inference, ONNX export capability
  - Deployment: Cross-platform (WSL training → Windows deployment)
  - Integration: Unreal Engine compatibility required
  - Reliability: 99.9% uptime target, graceful degradation

  CTC-Specific Requirements

  - Temporal Modeling: Variable-length sequence processing
  - Alignment-Free: CTC loss without forced alignment
  - Live Streaming: Real-time phoneme recognition capability
  - Backward Compatibility: Maintain existing MLP functionality

  🔌 API Architecture Design

  Core API Interfaces

  1. CTC Training API
  class CTCTrainer:
      def train_ctc_model(self, audio_sequences: List[torch.Tensor],
                         phoneme_sequences: List[str],
                         config: CTCConfig) -> CTCModel

      def validate_model(self, validation_data: DataLoader) -> Dict[str, float]

      def export_model(self, model: CTCModel,
                      formats: List[str] = ['pytorch', 'onnx'],
                      target_path: str = 'dist/') -> List[str]

  2. CTC Inference API
  class CTCInference:
      def predict_sequence(self, audio_data: torch.Tensor,
                          beam_width: int = 1) -> Tuple[str, float]

      def predict_streaming(self, audio_stream: Iterator[torch.Tensor],
                           window_size: float = 2.0,
                           overlap: float = 0.5) -> Iterator[str]

      def get_confidence_scores(self, predictions: torch.Tensor) -> List[float]

  3. Unified Classifier Interface
  class UnifiedClassifier:
      def __init__(self, use_ctc: bool = True, fallback_to_mlp: bool = True)

      def predict(self, audio: torch.Tensor) -> Union[str, List[str]]

      def set_mode(self, mode: Literal['ctc', 'mlp', 'hybrid'])

  🔄 Data Flow Architecture

  Modified Processing Pipeline

  graph TD
      A[Audio Input] --> B[Wav2Vec2 Feature Extraction]
      B --> C{Preserve Temporal?}
      C -->|Yes - CTC| D[LSTM Sequence Modeling]
      C -->|No - MLP| E[Mean Pooling]
      D --> F[CTC Head + CTC Loss]
      E --> G[MLP Classifier]
      F --> H[CTC Decoding]
      G --> I[Direct Classification]
      H --> J[Phoneme Sequence]
      I --> J

  Component Specifications

  1. Feature Extraction Layer
  class TemporalFeatureExtractor:
      def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
          # Input: (batch, time) audio at 16kHz
          # Output: (batch, time_steps, 768) Wav2Vec2 features
          # NO mean pooling - preserve full temporal dimension

  2. Sequence Modeling Layer
  class SequenceModel(nn.Module):
      def __init__(self, input_dim=768, hidden_dim=128, num_layers=2):
          self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, bidirectional=True)
          self.layer_norm = nn.LayerNorm(hidden_dim * 2)

      def forward(self, x):
          # Input: (batch, time, 768)
          # Output: (batch, time, 256)  # bidirectional

  3. CTC Output Layer
  class CTCHead(nn.Module):
      def __init__(self, input_dim=256, num_classes=38):  # 37 phonemes + blank
          self.classifier = nn.Linear(input_dim, num_classes)
          self.ctc_loss = nn.CTCLoss(blank=37, zero_infinity=True)

      def forward(self, x, targets=None, input_lengths=None, target_lengths=None):
          # Output: (batch, time, num_classes) log probabilities

  🛡️ Reliability & Error Handling

  Error Handling Strategy

  1. Input Validation
  class AudioValidator:
      def validate_audio_input(self, audio_data: torch.Tensor) -> torch.Tensor:
          # Check sample rate, duration, amplitude
          # Handle silent audio, clipping, corrupted data
          # Return sanitized audio or raise AudioValidationError

      def handle_variable_lengths(self, audio_batch: List[torch.Tensor]) -> torch.Tensor:
          # Pad sequences for batch processing
          # Create attention masks for variable lengths

  2. Graceful Degradation
  class FaultTolerantClassifier:
      def predict_with_fallback(self, audio: torch.Tensor) -> str:
          try:
              return self.ctc_model.predict(audio)
          except CTCInferenceError:
              logger.warning("CTC inference failed, falling back to MLP")
              return self.mlp_model.predict(audio.mean(dim=1))  # Mean pooling for MLP
          except Exception as e:
              logger.error(f"All models failed: {e}")
              return ""  # Empty string with confidence=0

  3. Resource Management
  class ResourceManager:
      def __init__(self, memory_threshold=0.8, max_sequence_length=1000):
          self.memory_threshold = memory_threshold
          self.max_sequence_length = max_sequence_length

      def monitor_memory_usage(self) -> float:
          # Return current memory usage percentage

      def limit_sequence_length(self, audio: torch.Tensor) -> torch.Tensor:
          # Truncate sequences exceeding max_sequence_length

      def implement_batch_processing(self, audio_list: List[torch.Tensor],
                                    batch_size: int = 32) -> Iterator[torch.Tensor]:
          # Process in batches to manage memory

  📡 Streaming Architecture

  Real-Time Processing Pipeline

  1. Audio Buffer Management
  class AudioBuffer:
      def __init__(self, buffer_size: int = 16000):  # 1 second buffer
          self.ring_buffer = CircularBuffer(buffer_size)
          self.sample_rate = 16000

      def add_audio_chunk(self, chunk: np.ndarray) -> None:
          # Add new audio chunk to ring buffer

      def get_windowed_audio(self, window_size: float = 2.0,
                            overlap: float = 0.5) -> torch.Tensor:
          # Extract sliding window from buffer

      def handle_buffer_overflow(self) -> None:
          # Implement overflow strategy (drop oldest, compress, etc.)

  2. Sliding Window Processing
  class StreamingCTCProcessor:
      def __init__(self, window_size=2.0, overlap=0.5):
          self.window_size = window_size  # seconds
          self.overlap = overlap  # 50% overlap
          self.stride = window_size * (1 - overlap)  # 1 second stride

      def process_stream(self, audio_stream: Iterator[torch.Tensor]) -> Iterator[str]:
          for window in self.sliding_window(audio_stream):
              phoneme_sequence = self.ctc_model.predict(window)
              yield self.post_process_sequence(phoneme_sequence)

  Performance Targets:
  - Latency: <500ms end-to-end
  - Throughput: >100 audio clips/second
  - Memory: <2GB for batch processing
  - Reliability: 99.9% uptime

  🔄 Integration with Existing Pipeline

  Hybrid Pipeline Design

  Modified Workflow Steps:
  # 0_workflow.py modifications
  steps = [
      ("Cleanup previous runs", cleanup_dist),
      ("Prepare the dataset", prepare_wav_files_clean),
      ("Extract embeddings (temporal)", extract_embeddings_temporal),  # MODIFIED
      ("Train CTC classifier", train_ctc_classifier),  # NEW
      ("Train MLP classifier", classifier_encoder_clean),  # EXISTING
      ("Visualize Results", visualize_results_clean),
      ("Benchmark both models", benchmark_both_models),  # MODIFIED
      ("Export models to ONNX", export_models_onnx),  # MODIFIED
      ("Copy to Unreal Engine", overwrite_onnx_unreal)
  ]

  Configuration Management:
  class PipelineConfig:
      enable_ctc: bool = True
      enable_mlp: bool = True  # For backward compatibility
      ctc_model_config: CTCConfig = CTCConfig()
      fallback_to_mlp: bool = True
      export_both_models: bool = True

  📦 Deployment & ONNX Export

  ONNX Compatibility Design

  1. CTC Model Tracing
  class CTCModelForONNX(nn.Module):
      def __init__(self, ctc_model: CTCModel):
          super().__init__()
          self.feature_extractor = ctc_model.wav2vec2_wrapper
          self.sequence_model = ctc_model.lstm
          self.classifier = ctc_model.ctc_head.classifier

      def forward(self, input_values: torch.Tensor) -> torch.Tensor:
          features = self.feature_extractor(input_values)
          lstm_out, _ = self.sequence_model(features)
          logits = self.classifier(lstm_out)
          return F.log_softmax(logits, dim=-1)

  2. Unreal Engine Integration
  // Unreal Engine C++ interface (s12_overwrite_onnx_unreal.py output)
  UCLASS(BlueprintType)
  class PHONEMECLASSIFIER_API UCTCPhonemeClassifier : public UObject {
      GENERATED_BODY()

  public:
      UFUNCTION(BlueprintCallable, Category = "Phoneme Classification")
      TArray<FString> PredictPhonemeSequence(const TArray<float>& AudioData);

      UFUNCTION(BlueprintCallable, Category = "Phoneme Classification")
      void StartLiveRecognition();

      UFUNCTION(BlueprintCallable, Category = "Phoneme Classification")
      void StopLiveRecognition();
  };

  📋 Implementation Roadmap

  Phase 1: Core CTC Implementation (Weeks 1-2)

  New Files:
  - workflows/mlp_control_workflow/models/ctc_model.py - PyTorch CTC architecture
  - workflows/mlp_control_workflow/s3_ctc_classifier.py - CTC training script
  - workflows/mlp_control_workflow/utils/ctc_decoder.py - CTC decoding utilities
  - workflows/mlp_control_workflow/validations/classify_voice_ctc.py - CTC inference validation

  Modified Files:
  - s2_extract_embeddings_for_phonemes.py - Preserve temporal sequences
  - s9_trace_mlp_classifier.py - Add CTC model tracing
  - s10_onnx_export.py - Support CTC ONNX export
  - 0_workflow.py - Add CTC training step

  Phase 2: Streaming Implementation (Weeks 3-4)

  Features:
  - Real-time audio buffer management
  - Sliding window CTC inference
  - Live phoneme stream output
  - Performance optimization

  Performance Targets

  - Training Time: <30 minutes on GPU
  - Inference Latency: <200ms per 2-second audio
  - Memory Usage: <2GB during training
  - Model Size: <50MB for ONNX export
  - Accuracy: ≥90% phoneme recognition (match current MLP)

  This design provides a robust, reliable foundation for CTC implementation while maintaining backward compatibility and meeting performance requirements.