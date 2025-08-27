# Temporal Brain Algorithms Specification

## Overview

This document defines the specific algorithms and implementation details for the Temporal Brain system that stabilizes real-time phoneme detection by reducing flicker and improving reliability.

## Problem Statement

Raw CTC/MLP model outputs produce frame-by-frame phoneme probabilities that can fluctuate rapidly, causing:
- **Flicker**: Rapid switching between phoneme predictions
- **False Positives**: Brief high-confidence spikes for incorrect phonemes  
- **Instability**: Inconsistent results for the same speech input
- **Poor UX**: Unreliable feedback for interactive applications

## Algorithm Components

### 1. Smoothing Algorithm

#### Moving Average Smoothing
**Purpose**: Reduce high-frequency noise in probability sequences

```python
class MovingAverageSmoothing:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        self.history.append(probabilities)
        if len(self.history) < self.window_size:
            return probabilities  # Not enough history
        
        # Average across time window
        return np.mean(self.history, axis=0)
```

#### Exponential Smoothing
**Purpose**: Weight recent observations more heavily while maintaining history

```python
class ExponentialSmoothing:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha  # Smoothing factor (0=no smoothing, 1=no history)
        self.smoothed = None
    
    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        if self.smoothed is None:
            self.smoothed = probabilities.copy()
        else:
            self.smoothed = (self.alpha * probabilities + 
                           (1 - self.alpha) * self.smoothed)
        return self.smoothed
```

### 2. Hysteresis Control

#### Dual-Threshold Hysteresis
**Purpose**: Prevent rapid flip-flopping using different thresholds for lock/unlock

```python
class HysteresisControl:
    def __init__(self, 
                 lock_threshold: float = 0.7,    # Higher threshold to lock
                 unlock_threshold: float = 0.3,  # Lower threshold to unlock
                 min_lock_duration: int = 3):    # Minimum frames to stay locked
        self.lock_threshold = lock_threshold
        self.unlock_threshold = unlock_threshold
        self.min_lock_duration = min_lock_duration
        
        self.current_phoneme = None
        self.lock_count = 0
        self.is_locked = False
    
    def process(self, probabilities: np.ndarray, phoneme_labels: list) -> str:
        max_prob = np.max(probabilities)
        predicted_phoneme = phoneme_labels[np.argmax(probabilities)]
        
        if not self.is_locked:
            # Not locked - check if we should lock onto a phoneme
            if max_prob >= self.lock_threshold:
                self.current_phoneme = predicted_phoneme
                self.is_locked = True
                self.lock_count = 1
                
        else:
            # Currently locked - check if we should unlock
            current_prob = probabilities[phoneme_labels.index(self.current_phoneme)]
            
            if (current_prob < self.unlock_threshold and 
                self.lock_count >= self.min_lock_duration):
                self.is_locked = False
                self.lock_count = 0
                self.current_phoneme = None
            else:
                self.lock_count += 1
        
        return self.current_phoneme if self.is_locked else None
```

### 3. Confidence Gating

#### Adaptive Confidence Thresholds
**Purpose**: Per-phoneme confidence tuning with temporal persistence

```python
class ConfidenceGating:
    def __init__(self, 
                 default_threshold: float = 0.6,
                 persistence_frames: int = 3,
                 phoneme_thresholds: dict = None):
        self.default_threshold = default_threshold
        self.persistence_frames = persistence_frames
        self.phoneme_thresholds = phoneme_thresholds or {}
        
        self.candidate_phoneme = None
        self.candidate_count = 0
        self.candidate_confidence_sum = 0.0
    
    def get_threshold(self, phoneme: str) -> float:
        return self.phoneme_thresholds.get(phoneme, self.default_threshold)
    
    def process(self, probabilities: np.ndarray, phoneme_labels: list) -> tuple:
        max_idx = np.argmax(probabilities)
        predicted_phoneme = phoneme_labels[max_idx]
        confidence = probabilities[max_idx]
        threshold = self.get_threshold(predicted_phoneme)
        
        if confidence < threshold:
            # Below threshold - reset candidate
            self._reset_candidate()
            return None, 0.0
            
        if predicted_phoneme == self.candidate_phoneme:
            # Same candidate - accumulate
            self.candidate_count += 1
            self.candidate_confidence_sum += confidence
            
            if self.candidate_count >= self.persistence_frames:
                # Candidate has persisted long enough
                avg_confidence = self.candidate_confidence_sum / self.candidate_count
                self._reset_candidate()
                return predicted_phoneme, avg_confidence
                
        else:
            # New candidate - start tracking
            self.candidate_phoneme = predicted_phoneme
            self.candidate_count = 1
            self.candidate_confidence_sum = confidence
        
        return None, 0.0
    
    def _reset_candidate(self):
        self.candidate_phoneme = None
        self.candidate_count = 0
        self.candidate_confidence_sum = 0.0
```

## Integrated Temporal Brain

### Complete Temporal Processor
```python
class TemporalProcessor:
    def __init__(self, config: dict):
        self.smoother = self._create_smoother(config)
        self.hysteresis = HysteresisControl(
            lock_threshold=config['hysteresis']['lock_threshold'],
            unlock_threshold=config['hysteresis']['unlock_threshold'],
            min_lock_duration=config['hysteresis']['min_lock_duration']
        )
        self.confidence_gate = ConfidenceGating(
            default_threshold=config['confidence']['default_threshold'],
            persistence_frames=config['confidence']['persistence_frames'],
            phoneme_thresholds=config['confidence']['phoneme_thresholds']
        )
        
        self.phoneme_labels = config['phoneme_labels']
        self.flicker_tracker = FlickerTracker()
    
    def process_frame(self, raw_probabilities: np.ndarray) -> dict:
        # Step 1: Smooth probabilities
        smoothed_probs = self.smoother.smooth(raw_probabilities)
        
        # Step 2: Apply hysteresis control
        hysteresis_result = self.hysteresis.process(smoothed_probs, self.phoneme_labels)
        
        # Step 3: Apply confidence gating
        if hysteresis_result:
            gated_phoneme, confidence = self.confidence_gate.process(
                smoothed_probs, self.phoneme_labels
            )
            final_phoneme = gated_phoneme if gated_phoneme == hysteresis_result else None
        else:
            final_phoneme, confidence = None, 0.0
        
        # Step 4: Track flicker for metrics
        self.flicker_tracker.update(final_phoneme)
        
        return {
            'phoneme': final_phoneme,
            'confidence': confidence,
            'raw_probabilities': raw_probabilities,
            'smoothed_probabilities': smoothed_probs,
            'is_stable': final_phoneme is not None,
            'flicker_rate': self.flicker_tracker.get_flicker_rate(),
            'timestamp': time.time()
        }
    
    def _create_smoother(self, config):
        smoother_type = config['smoothing']['type']
        if smoother_type == 'moving_average':
            return MovingAverageSmoothing(config['smoothing']['window_size'])
        elif smoother_type == 'exponential':
            return ExponentialSmoothing(config['smoothing']['alpha'])
        else:
            raise ValueError(f"Unknown smoother type: {smoother_type}")
```

### Flicker Rate Measurement
```python
class FlickerTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.transitions = 0
        
    def update(self, phoneme: str):
        if len(self.history) > 0 and self.history[-1] != phoneme:
            self.transitions += 1
        
        self.history.append(phoneme)
        
        # Remove old transitions from count
        if len(self.history) == self.window_size:
            # Approximate transition removal (exact would require more state)
            self.transitions = max(0, self.transitions - 1)
    
    def get_flicker_rate(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.transitions / len(self.history)
```

## Configuration Schema

### Default Configuration
```json
{
    "smoothing": {
        "type": "moving_average",
        "window_size": 5,
        "alpha": 0.3
    },
    "hysteresis": {
        "lock_threshold": 0.7,
        "unlock_threshold": 0.3,
        "min_lock_duration": 3
    },
    "confidence": {
        "default_threshold": 0.6,
        "persistence_frames": 3,
        "phoneme_thresholds": {
            "AA": 0.7,
            "AE": 0.6,
            "AH": 0.65,
            "AO": 0.7,
            "AW": 0.6,
            "AY": 0.65,
            "B": 0.8,
            "CH": 0.75,
            "D": 0.8,
            "DH": 0.7,
            "EH": 0.6,
            "ER": 0.65,
            "EY": 0.6,
            "F": 0.75,
            "G": 0.8,
            "HH": 0.7,
            "IH": 0.6,
            "IY": 0.65,
            "JH": 0.8,
            "K": 0.8,
            "L": 0.7,
            "M": 0.75,
            "N": 0.7,
            "NG": 0.75,
            "OW": 0.65,
            "OY": 0.7,
            "P": 0.8,
            "R": 0.7,
            "S": 0.75,
            "SH": 0.75,
            "T": 0.8,
            "TH": 0.75,
            "UH": 0.65,
            "UW": 0.65,
            "V": 0.75,
            "W": 0.7,
            "Y": 0.7,
            "Z": 0.75,
            "ZH": 0.75
        }
    },
    "phoneme_labels": ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"],
    "metrics": {
        "flicker_window_size": 100,
        "target_flicker_rate": 0.15
    }
}
```

## Parameter Tuning Strategy

### Automated Optimization
```python
class ParameterTuner:
    def __init__(self, voice_samples: list, ground_truth: list):
        self.voice_samples = voice_samples
        self.ground_truth = ground_truth
    
    def tune_parameters(self) -> dict:
        # Grid search over parameter space
        best_config = None
        best_score = float('inf')
        
        for config in self._generate_config_grid():
            processor = TemporalProcessor(config)
            score = self._evaluate_config(processor)
            
            if score < best_score:
                best_score = score
                best_config = config
        
        return best_config
    
    def _evaluate_config(self, processor) -> float:
        total_flicker = 0
        total_accuracy = 0
        
        for sample, truth in zip(self.voice_samples, self.ground_truth):
            results = []
            for frame in sample:
                result = processor.process_frame(frame)
                results.append(result)
            
            flicker_rate = processor.flicker_tracker.get_flicker_rate()
            accuracy = self._calculate_accuracy(results, truth)
            
            # Combined score: minimize flicker while maintaining accuracy
            score = flicker_rate * 2.0 + (1.0 - accuracy)
            total_flicker += flicker_rate
            total_accuracy += accuracy
        
        return total_flicker / len(self.voice_samples)
```

## Performance Considerations

### Computational Complexity
- **Smoothing**: O(1) for exponential, O(W) for moving average
- **Hysteresis**: O(1) per frame
- **Confidence Gating**: O(1) per frame
- **Overall**: O(W) per frame where W is smoothing window size

### Memory Requirements
- **History Buffers**: ~5-10 frames × probability vector size
- **State Variables**: Minimal (current phoneme, counters)
- **Total**: <1MB for typical configuration

### Latency Impact
- **Processing Time**: <1ms per frame on modern CPU
- **Buffer Delay**: Window size × frame duration (e.g., 5 frames × 20ms = 100ms)
- **Total Added Latency**: ~100ms for stability vs. accuracy trade-off

## Platform Adaptations

### JavaScript Implementation
```javascript
class TemporalProcessor {
    constructor(config) {
        this.smoother = new MovingAverageSmoothing(config.smoothing.window_size);
        this.hysteresis = new HysteresisControl(config.hysteresis);
        this.confidenceGate = new ConfidenceGating(config.confidence);
    }
    
    processFrame(rawProbabilities) {
        const smoothed = this.smoother.smooth(rawProbabilities);
        const hysteresisResult = this.hysteresis.process(smoothed);
        const [phoneme, confidence] = this.confidenceGate.process(smoothed);
        
        return {
            phoneme: phoneme,
            confidence: confidence,
            isStable: phoneme !== null,
            timestamp: Date.now()
        };
    }
}
```

### C++ Implementation
```cpp
class TemporalProcessor {
private:
    std::unique_ptr<MovingAverageSmoothing> smoother_;
    std::unique_ptr<HysteresisControl> hysteresis_;
    std::unique_ptr<ConfidenceGating> confidence_gate_;
    
public:
    TemporalProcessor(const Config& config);
    
    PhonemeResult ProcessFrame(const std::vector<float>& raw_probabilities);
    
    float GetFlickerRate() const;
};
```

## Validation Methods

### Synthetic Testing
- Generate known phoneme sequences with noise
- Measure flicker reduction vs. accuracy preservation
- Validate parameter sensitivity analysis

### Real Voice Testing  
- Record family voice samples with ground truth labels
- Measure flicker rates across different speakers
- Tune per-phoneme thresholds for optimal performance

### Cross-Platform Validation
- Ensure identical behavior across Python/JS/C++
- Validate timing consistency and numerical precision
- Test with same audio samples on all platforms

This algorithmic foundation enables the Temporal Brain to provide stable, reliable phoneme detection suitable for interactive real-time applications.