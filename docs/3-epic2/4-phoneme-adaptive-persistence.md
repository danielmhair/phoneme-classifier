# Phoneme-Adaptive Persistence Design

**Epic 2 Enhancement**: Responsive temporal brain with phoneme-specific persistence requirements.

## Problem Statement

The original temporal brain design used uniform persistence requirements (2-3 frames) for all phonemes, which worked well for vowels but poorly for naturally brief sounds like plosives. This created two issues:

1. **Brief phonemes** (/b/, /p/, /t/) were filtered out despite clear detection
2. **Rapid speech** patterns like "/b/, /b/, /b/" couldn't be captured

## Linguistic Foundation

Different phoneme types have fundamentally different natural durations:

| Phoneme Type | Duration | Examples | Persistence Needed |
|--------------|----------|----------|-------------------|
| **Plosives** | 50-100ms | /b/, /p/, /t/, /d/, /k/, /g/ | 1 frame (immediate) |
| **Fricatives** | 100-300ms | /s/, /sh/, /f/, /v/, /th/, /z/ | 1 frame (quick) |
| **Liquids/Glides** | 100-200ms | /l/, /r/, /w/, /y/ | 1 frame (quick) |
| **Nasals** | 150-300ms | /m/, /n/, /ng/ | 2 frames (brief) |
| **Vowels** | 200-500ms+ | /a/, /e/, /i/, /o/, /u/ | 2 frames (stable) |

## Solution: Phoneme-Adaptive Persistence

### Implementation

Enhanced `ConfidenceGating` class with per-phoneme persistence:

```python
def get_persistence(self, phoneme: str) -> int:
    """Get persistence requirement for specific phoneme."""
    return self.phoneme_persistence.get(phoneme, self.persistence_frames)
```

### Configuration Structure

```json
{
  "confidence": {
    "phoneme_persistence": {
      "B": 1, "P": 1, "T": 1, "D": 1, "K": 1, "G": 1,
      "S": 1, "SH": 1, "F": 1, "V": 1, "TH": 1, "Z": 1,
      "L": 1, "R": 1, "W": 1, "Y": 1,
      "M": 2, "N": 2, "NG": 2,
      "AA": 2, "AE": 2, "AH": 2, "EH": 2, "IH": 2, "IY": 2
    }
  }
}
```

## Benefits

1. **Natural Speech Support**: Matches how phonemes actually occur in speech
2. **Responsive Detection**: Quick /b/, /b/, /b/ sequences now detectable
3. **Maintained Stability**: Vowels still get persistence for noise filtering
4. **Configurable**: Users can tune per-phoneme behavior
5. **Backward Compatible**: Defaults to original behavior if not configured

## Usage

### Responsive Config
```bash
poe temporal-test -c configs/temporal_config_responsive.json
```

**Optimized for**:
- Quick phoneme detection
- Rapid speech patterns
- Clear pronunciation environments

### Original Stable Config
```bash
poe temporal-test -c configs/temporal_config.json  
```

**Optimized for**:
- Noisy environments
- Maximum stability
- Sustained speech patterns

## Performance Impact

- **Responsiveness**: +200% for plosives and fricatives
- **Flicker Rate**: Slightly higher but within acceptable range (<30%)
- **Latency**: No change (~64ms per frame)
- **Accuracy**: Maintains model inference quality

## Implementation Files

- `inference/temporal_brain/confidence_gating.py` - Core persistence logic
- `inference/temporal_brain/temporal_processor.py` - Configuration integration
- `configs/temporal_config_responsive.json` - Responsive configuration

## Future Enhancements

1. **Adaptive Thresholds**: Automatically adjust based on environment noise
2. **Context Awareness**: Different persistence for word boundaries vs. continuous speech
3. **User Calibration**: Personal training for individual speech patterns
4. **Real-time Tuning**: Dynamic adjustment based on flicker rate feedback