#!/usr/bin/env python3
"""
Epic 2 Temporal Brain CLI Testing Tool

Interactive phoneme testing with temporal stabilization algorithms.
Supports all three Epic 1 models with model hot-swapping capability.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inference.temporal_brain import TemporalProcessor
from inference.cli.model_loader import ModelLoader
from inference.cli.audio_capture import AudioCapture
from inference.cli.audio_feature_extractor import AudioFeatureExtractor


class TemporalBrainTester:
    """Main CLI testing tool for temporal brain algorithms."""
    
    def __init__(self):
        """Initialize the temporal brain tester."""
        self.temporal_processor: Optional[TemporalProcessor] = None
        self.model_loader = ModelLoader()
        self.audio_capture = AudioCapture()
        self.feature_extractor = AudioFeatureExtractor()
        self.is_running = False
        
        # Statistics
        self.total_frames_processed = 0
        self.stable_frames = 0
        self.current_phoneme = None
        self.session_start_time: Optional[float] = None
    
    def load_temporal_config(self, config_path: str = "configs/temporal_config.json") -> bool:
        """Load temporal brain configuration.
        
        Args:
            config_path: Path to temporal configuration file
            
        Returns:
            True if loaded successfully
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                click.echo(f"❌ Config file not found: {config_path}")
                return False
            
            self.temporal_processor = TemporalProcessor.from_config_file(config_path)
            click.echo(f"✅ Loaded temporal brain config: {config_path}")
            return True
            
        except Exception as e:
            click.echo(f"❌ Failed to load config: {e}")
            return False
    
    def list_models(self):
        """Display available models."""
        models = self.model_loader.list_available_models()
        
        if not models:
            click.echo("❌ No models found in dist/ directory")
            click.echo("   Run Epic 1 workflows first: poe train-all")
            return
        
        click.echo("📦 Available Models:")
        for model in models:
            click.echo(f"   🎯 {model.model_id}: {model.name}")
            click.echo(f"      📝 {model.description}")
            click.echo(f"      🏷️  Type: {model.model_type.upper()}")
            click.echo("")
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if loaded successfully
        """
        return self.model_loader.load_model(model_id)
    
    def start_testing_session(self):
        """Start interactive phoneme testing session."""
        if not self.temporal_processor:
            click.echo("❌ Temporal brain not configured")
            return
        
        if not self.model_loader.current_session:
            click.echo("❌ No model loaded")
            return
        
        # Setup audio callback
        self.audio_capture.set_audio_callback(self._process_audio_frame)
        
        # Reset statistics
        self.total_frames_processed = 0
        self.stable_frames = 0
        self.session_start_time = time.time()
        self.is_running = True
        
        click.echo("🧠 Starting Temporal Brain Testing Session")
        click.echo("=" * 50)
        click.echo(f"Model: {self.model_loader.current_model_info.name}")
        click.echo(f"Config: Temporal brain enabled")
        click.echo(f"Target: <15% flicker rate, <150ms latency")
        click.echo("")
        click.echo("🎤 Speak phonemes clearly...")
        click.echo("Press Ctrl+C to stop")
        click.echo("")
        
        try:
            # Start audio capture
            self.audio_capture.start_capture()
            
            # Keep running until interrupted
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            click.echo("\n🛑 Stopping session...")
        finally:
            self.stop_testing_session()
    
    def stop_testing_session(self):
        """Stop the testing session."""
        self.is_running = False
        self.audio_capture.stop_capture()
        
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            self._display_session_summary(duration)
    
    def _process_audio_frame(self, audio_data: np.ndarray):
        """Process single audio frame through temporal brain pipeline with REAL model inference.
        
        Args:
            audio_data: Raw audio data from capture
        """
        try:
            # Extract features based on current model type
            feature_result = self.feature_extractor.extract_features_for_model(
                audio_data, 
                self.model_loader.current_model_info.model_type
            )
            
            if not feature_result['preprocessing_success']:
                if feature_result.get('is_silent', False):
                    # Silent audio - create zero probabilities
                    num_phonemes = len(self.model_loader.current_labels)
                    real_probs = np.ones(num_phonemes) / num_phonemes  # Uniform distribution
                else:
                    click.echo(f"⚠️  Feature extraction failed: {feature_result.get('error', 'Unknown error')}")
                    return
            else:
                # Run REAL model inference
                real_probs = self.model_loader.run_inference(feature_result['features'])
            
            # Process through temporal brain with REAL probabilities
            result = self.temporal_processor.process_frame(real_probs)
            
            # Update statistics
            self.total_frames_processed += 1
            if result['is_stable']:
                self.stable_frames += 1
            
            # Display real-time results
            self._display_realtime_result(result)
            
        except Exception as e:
            click.echo(f"⚠️  Processing error: {e}")
    
    
    def _display_realtime_result(self, result: dict):
        """Display real-time processing result.
        
        Args:
            result: Temporal brain processing result
        """
        # Only display when phoneme changes to avoid spam
        if result['phoneme'] != self.current_phoneme:
            self.current_phoneme = result['phoneme']
            
            if result['phoneme']:
                flicker_rate = result['flicker_rate']
                confidence = result['confidence']
                status = "🟢" if flicker_rate < 0.15 else "🟡"
                
                click.echo(f"{status} {result['phoneme']} "
                          f"(conf: {confidence:.2f}, "
                          f"flicker: {flicker_rate:.1%})")
            else:
                click.echo("🔇 [silence]")
    
    def _display_session_summary(self, duration: float):
        """Display session summary statistics.
        
        Args:
            duration: Session duration in seconds
        """
        if self.total_frames_processed == 0:
            return
        
        stability_rate = self.stable_frames / self.total_frames_processed
        flicker_rate = self.temporal_processor.get_metrics()['flicker_rate']
        
        click.echo("\n📊 Session Summary")
        click.echo("=" * 30)
        click.echo(f"Duration: {duration:.1f}s")
        click.echo(f"Frames processed: {self.total_frames_processed}")
        click.echo(f"Stable frames: {stability_rate:.1%}")
        click.echo(f"Final flicker rate: {flicker_rate:.1%}")
        
        # Performance assessment
        target_flicker = 0.15
        if flicker_rate <= target_flicker:
            click.echo(f"🎯 Target achieved! (<{target_flicker:.0%} flicker)")
        else:
            click.echo(f"⚠️  Above target ({flicker_rate:.1%} > {target_flicker:.0%})")


@click.command()
@click.option('--model', '-m', default='wavlm_ctc', 
              help='Model to use (mlp_control, wav2vec2_ctc, wavlm_ctc)')
@click.option('--config', '-c', default='configs/temporal_config.json',
              help='Temporal brain configuration file')
@click.option('--list-models', is_flag=True,
              help='List available models and exit')
@click.option('--list-devices', is_flag=True,
              help='List audio devices and exit')
def main(model: str, config: str, list_models: bool, list_devices: bool):
    """Epic 2 Temporal Brain CLI Testing Tool.
    
    Interactive phoneme testing with temporal stabilization algorithms.
    Supports hot-swapping between Epic 1 models.
    """
    tester = TemporalBrainTester()
    
    # Handle list-only commands
    if list_models:
        tester.list_models()
        return
    
    if list_devices:
        click.echo(tester.audio_capture.list_audio_devices())
        return
    
    # Setup temporal brain
    if not tester.load_temporal_config(config):
        sys.exit(1)
    
    # Load model
    if not tester.load_model(model):
        click.echo("Available models:")
        tester.list_models()
        sys.exit(1)
    
    # Start testing session
    tester.start_testing_session()


if __name__ == '__main__':
    main()