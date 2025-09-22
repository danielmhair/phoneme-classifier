#!/usr/bin/env python3
"""
Guided Test Data Recording CLI

Systematically records all Phase 1 and Phase 2 test sounds with automatic file saving
and progression through the planned phoneme sequence.

Usage:
    python tests/record_test_data.py
"""

import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the WSL2-compatible audio device selection
from inference.cli.audio_capture import AudioCaptureConfig

class GuidedRecorder:
    """Guided CLI for systematically recording test phonemes and sequences"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.recording_duration = 3.0  # seconds
        
        # Initialize audio capture config with WSL2-compatible device selection
        self.audio_config = AudioCaptureConfig(
            sample_rate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        
        # Create directories
        self.mlp_dir = Path("tests/test_recordings/mlp")
        self.ctc_dir = Path("tests/test_recordings/ctc")
        self.mlp_dir.mkdir(parents=True, exist_ok=True)
        self.ctc_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording session info
        self.session_start = datetime.now()
        self.recorded_files = []
        
        # Define the recording sequence (Phase 1 + Phase 2)
        self.recording_sequence = self._define_recording_sequence()
        
        print("🎤 Guided Test Data Recorder")
        print("=" * 60)
        print(f"📁 MLP files will be saved to: {self.mlp_dir}")
        print(f"📁 CTC files will be saved to: {self.ctc_dir}")
        print(f"🎯 Total recordings planned: {len(self.recording_sequence)}")
        print(f"🎛️  Audio device: {self.audio_config.device}")
        print("=" * 60)

    def _define_recording_sequence(self) -> List[Dict]:
        """Define the complete recording sequence for Phase 1 and 2"""
        
        # Phase 1: Core Foundation
        phase_1_mlp = [
            {"type": "mlp", "target": "B", "instruction": "Say 'buh' clearly (like the start of 'ball')", "priority": "HIGH"},
            {"type": "mlp", "target": "K", "instruction": "Say 'kuh' clearly (like the start of 'cat')", "priority": "HIGH"},
            {"type": "mlp", "target": "S", "instruction": "Say 'sss' clearly (like a snake sound)", "priority": "HIGH"},
            {"type": "mlp", "target": "A_Æ", "instruction": "Say 'aaa' clearly (like the 'a' in 'cat')", "priority": "HIGH"},
            {"type": "mlp", "target": "SH", "instruction": "Say 'shh' clearly (like telling someone to be quiet)", "priority": "HIGH"},
            {"type": "mlp", "target": "T", "instruction": "Say 'tuh' clearly (like the start of 'top')", "priority": "HIGH"},
            {"type": "mlp", "target": "M", "instruction": "Say 'mmm' clearly (like humming with mouth closed)", "priority": "HIGH"},
            {"type": "mlp", "target": "TH", "instruction": "Say 'thh' clearly (like the start of 'think')", "priority": "HIGH"},
        ]
        
        phase_1_ctc = [
            {"type": "ctc", "target": "BAT", "instruction": "Say 'bat' clearly (the animal or sports equipment)", "priority": "HIGH"},
            {"type": "ctc", "target": "CAT", "instruction": "Say 'cat' clearly (the animal)", "priority": "HIGH"},
            {"type": "ctc", "target": "SAT", "instruction": "Say 'sat' clearly (past tense of sit)", "priority": "HIGH"},
            {"type": "ctc", "target": "BATH", "instruction": "Say 'bath' clearly (B-A-TH sequence)", "priority": "HIGH"},
            {"type": "ctc", "target": "FISH", "instruction": "Say 'fish' clearly (the animal)", "priority": "HIGH"},
        ]
        
        # Phase 2: Extended Coverage
        phase_2_mlp = [
            {"type": "mlp", "target": "E", "instruction": "Say 'eh' clearly (like the 'e' in 'bed')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "I", "instruction": "Say 'ih' clearly (like the 'i' in 'bit')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "U", "instruction": "Say 'uh' clearly (like the 'u' in 'but')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "OO_UW", "instruction": "Say 'ooo' clearly (like the 'oo' in 'boot')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "F", "instruction": "Say 'fff' clearly (like air escaping)", "priority": "MEDIUM"},
            {"type": "mlp", "target": "V", "instruction": "Say 'vvv' clearly (like a vibrating sound)", "priority": "MEDIUM"},
            {"type": "mlp", "target": "D", "instruction": "Say 'duh' clearly (like the start of 'dog')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "G", "instruction": "Say 'guh' clearly (like the start of 'go')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "P", "instruction": "Say 'puh' clearly (like the start of 'pet')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "Z", "instruction": "Say 'zzz' clearly (like a buzzing sound)", "priority": "MEDIUM"},
            {"type": "mlp", "target": "CH", "instruction": "Say 'chh' clearly (like the start of 'cheese')", "priority": "MEDIUM"},
            {"type": "mlp", "target": "J_DƷ", "instruction": "Say 'jjj' clearly (like the start of 'jump')", "priority": "MEDIUM"},
        ]
        
        phase_2_ctc = [
            {"type": "ctc", "target": "MAT", "instruction": "Say 'mat' clearly (like a floor mat)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "SHIP", "instruction": "Say 'ship' clearly (the boat)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "CHIP", "instruction": "Say 'chip' clearly (like potato chip)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "ZIP", "instruction": "Say 'zip' clearly (like a zipper)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "BET", "instruction": "Say 'bet' clearly (like a wager)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "BIT", "instruction": "Say 'bit' clearly (small piece)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "BUT", "instruction": "Say 'but' clearly (however)", "priority": "MEDIUM"},
            {"type": "ctc", "target": "BEACH", "instruction": "Say 'beach' clearly (the sandy shore)", "priority": "MEDIUM"},
        ]
        
        # Combine phases in optimal order
        sequence = []
        
        # Start with Phase 1 (high priority sounds)
        sequence.extend(phase_1_mlp)
        sequence.extend(phase_1_ctc)
        
        # Add Phase 2 (extended coverage)
        sequence.extend(phase_2_mlp)
        sequence.extend(phase_2_ctc)
        
        return sequence

    def record_audio(self, duration: float = None) -> np.ndarray:
        """Record audio from microphone using WSL2-compatible device selection"""
        if duration is None:
            duration = self.recording_duration
            
        print(f"🔴 Recording for {duration} seconds... Speak now!")
        
        try:
            # Record audio using the properly selected device
            audio_data = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=self.audio_config.channels,
                dtype=self.audio_config.dtype,
                device=self.audio_config.device
            )
            sd.wait()  # Wait until recording is finished
            
            print("⏹️  Recording complete!")
            return audio_data.flatten()
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            print("🔧 Trying to list available devices...")
            
            try:
                devices = sd.query_devices()
                print("Available audio devices:")
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        print(f"  {i}: {device['name']}")
            except Exception as list_error:
                print(f"❌ Could not list devices: {list_error}")
            
            # Return empty audio on failure
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def save_recording(self, audio_data: np.ndarray, recording_info: Dict) -> str:
        """Save recording with proper naming convention"""
        
        if recording_info["type"] == "mlp":
            filename = f"{recording_info['target']}_test_001.wav"
            filepath = self.mlp_dir / filename
        else:  # ctc
            filename = f"{recording_info['target']}_sequence_001.wav"
            filepath = self.ctc_dir / filename
        
        # Check if file already exists and increment number
        counter = 1
        original_filepath = filepath
        while filepath.exists():
            counter += 1
            if recording_info["type"] == "mlp":
                filename = f"{recording_info['target']}_test_{counter:03d}.wav"
            else:
                filename = f"{recording_info['target']}_sequence_{counter:03d}.wav"
            filepath = filepath.parent / filename
        
        # Save the audio file
        sf.write(filepath, audio_data, self.sample_rate)
        
        # Record in session log
        self.recorded_files.append({
            "filepath": str(filepath),
            "target": recording_info["target"],
            "type": recording_info["type"],
            "timestamp": datetime.now().isoformat(),
            "priority": recording_info["priority"]
        })
        
        return str(filepath)

    def show_progress(self, current_index: int, total: int):
        """Show recording progress"""
        progress = (current_index / total) * 100
        print(f"\n📊 Progress: {current_index}/{total} ({progress:.1f}%)")
        
        phase_1_count = sum(1 for r in self.recording_sequence[:current_index] if r.get("priority") == "HIGH")
        phase_2_count = current_index - phase_1_count
        
        print(f"   Phase 1 (Core): {phase_1_count}/13 completed")
        print(f"   Phase 2 (Extended): {phase_2_count}/{len(self.recording_sequence)-13} completed")

    def record_single_item(self, recording_info: Dict, index: int, total: int) -> bool:
        """Record a single phoneme or sequence"""
        
        print("\n" + "="*60)
        self.show_progress(index, total)
        print("="*60)
        
        print(f"\n🎯 Recording #{index + 1}: {recording_info['type'].upper()} - {recording_info['target']}")
        print(f"📝 Instruction: {recording_info['instruction']}")
        print(f"⚡ Priority: {recording_info['priority']}")
        
        # Wait for user to be ready
        input(f"\n🎙️  Press Enter when ready to record '{recording_info['target']}'...")
        
        # Record audio
        audio_data = self.record_audio()
        
        # Save file
        filepath = self.save_recording(audio_data, recording_info)
        print(f"💾 Saved: {filepath}")
        
        # Ask for confirmation
        while True:
            response = input(f"\n✅ Recording saved. Continue to next sound? (y/n/r=retry): ").strip().lower()
            
            if response in ['y', 'yes', '']:
                return True
            elif response in ['n', 'no', 'q', 'quit']:
                return False
            elif response in ['r', 'retry']:
                print(f"🔄 Retrying recording for {recording_info['target']}...")
                audio_data = self.record_audio()
                filepath = self.save_recording(audio_data, recording_info)
                print(f"💾 Saved: {filepath}")
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'r' (retry)")

    def save_session_summary(self):
        """Save session summary to JSON"""
        summary = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_planned": len(self.recording_sequence),
            "total_recorded": len(self.recorded_files),
            "recorded_files": self.recorded_files,
            "completion_rate": len(self.recorded_files) / len(self.recording_sequence) * 100
        }
        
        summary_path = Path("tests/test_recordings/recording_session_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Session summary saved to: {summary_path}")

    def run_recording_session(self):
        """Run the complete guided recording session"""
        
        print(f"\n🚀 Starting guided recording session...")
        print(f"📝 Recording {len(self.recording_sequence)} sounds across Phase 1 & 2")
        print(f"⏱️  Estimated time: {len(self.recording_sequence) * 1.5:.0f} minutes")
        
        # Show overview
        mlp_count = sum(1 for r in self.recording_sequence if r["type"] == "mlp")
        ctc_count = sum(1 for r in self.recording_sequence if r["type"] == "ctc")
        print(f"📊 Breakdown: {mlp_count} MLP phonemes + {ctc_count} CTC sequences")
        
        input("\n🎬 Press Enter to begin recording session...")
        
        try:
            for index, recording_info in enumerate(self.recording_sequence):
                should_continue = self.record_single_item(recording_info, index, len(self.recording_sequence))
                
                if not should_continue:
                    print(f"\n⏹️  Recording session stopped by user at {index + 1}/{len(self.recording_sequence)}")
                    break
            else:
                print(f"\n🎉 Recording session complete! All {len(self.recording_sequence)} sounds recorded!")
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Recording session interrupted by user")
        
        # Save session summary
        self.save_session_summary()
        
        # Show final summary
        print(f"\n📊 Final Summary:")
        print(f"   Recorded: {len(self.recorded_files)}/{len(self.recording_sequence)} sounds")
        print(f"   Completion: {len(self.recorded_files)/len(self.recording_sequence)*100:.1f}%")
        
        if self.recorded_files:
            print(f"\n🧪 Ready to test! Run:")
            print(f"   poe test-models")
            print(f"   poe test-mlp-validation") 
            print(f"   poe test-ctc-validation")

def main():
    """Main function"""
    print("🎤 Guided Test Data Recording")
    print("This will walk you through recording all Phase 1 & 2 test sounds")
    print("systematically with automatic file saving.\n")
    
    recorder = GuidedRecorder()
    recorder.run_recording_session()

if __name__ == "__main__":
    main()