"""Neural Audio Synthesizer — waveform generation through nCPU.

Generate audio waveforms where every sample calculation is neural:
- Square wave: neural CMP for threshold
- Sawtooth: neural MOD for ramp
- Triangle: neural ABS via CMP+SUB
- Pulse width modulation: neural comparison for duty cycle
- Mixing: neural ADD for combining waves
- Envelope: neural MUL for amplitude control
- Output: WAV file header + PCM samples

Usage:
    python -m bridge.neural_audio demo
    python -m bridge.neural_audio generate <output.wav>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

SAMPLE_RATE = 8000  # Low for speed (neural ops per sample)
MAX_AMP = 127  # 8-bit audio


class NeuralOscillator:
    """Generate waveforms using neural arithmetic."""
    
    def __init__(self, bridge: NCPUBridge):
        self.bridge = bridge
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def _neural_mod(self, val: int, mod: int) -> int:
        q = self.bridge.div(val, mod)
        self._op()
        return self.bridge.sub(val, self.bridge.mul(q, mod))
    
    def square(self, phase: int, period: int) -> int:
        """Square wave: high if phase < period/2, else low."""
        half = self.bridge.div(period, 2)
        self._op()
        zf, sf = self.bridge.cmp(phase, half)
        self._op()
        return MAX_AMP if sf else self.bridge.sub(0, MAX_AMP)
    
    def sawtooth(self, phase: int, period: int) -> int:
        """Sawtooth: linear ramp from -MAX to +MAX."""
        # Scale phase to -127..127: (phase * 254 / period) - 127
        scaled = self.bridge.div(self.bridge.mul(phase, 254), period)
        self._op()
        return self.bridge.sub(scaled, MAX_AMP)
    
    def triangle(self, phase: int, period: int) -> int:
        """Triangle: up ramp then down ramp."""
        half = self.bridge.div(period, 2)
        self._op()
        zf, sf = self.bridge.cmp(phase, half)
        self._op()
        
        if sf:  # phase < half: rising
            return self.bridge.sub(self.bridge.div(self.bridge.mul(phase, 254), half), MAX_AMP)
        else:  # falling
            remaining = self.bridge.sub(period, phase)
            self._op()
            return self.bridge.sub(self.bridge.div(self.bridge.mul(remaining, 254), half), MAX_AMP)
    
    def pulse(self, phase: int, period: int, duty: int = 25) -> int:
        """Pulse wave with variable duty cycle (0-100%)."""
        threshold = self.bridge.div(self.bridge.mul(period, duty), 100)
        self._op()
        zf, sf = self.bridge.cmp(phase, threshold)
        self._op()
        return MAX_AMP if sf else self.bridge.sub(0, MAX_AMP)


class NeuralEnvelope:
    """ADSR envelope using neural arithmetic."""
    
    def __init__(self, bridge: NCPUBridge, attack: int, decay: int, sustain: int, release: int):
        self.bridge = bridge
        self.attack = attack    # samples
        self.decay = decay
        self.sustain = sustain  # level 0-100
        self.release = release
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def amplitude(self, sample_idx: int, total_samples: int) -> int:
        """Get envelope amplitude (0-100) at sample index."""
        release_start = self.bridge.sub(total_samples, self.release)
        self._op()
        
        # Attack phase
        zf, sf = self.bridge.cmp(sample_idx, self.attack)
        self._op()
        if sf:  # sample < attack
            return self.bridge.div(self.bridge.mul(sample_idx, 100), max(self.attack, 1))
        
        # Decay phase
        decay_end = self.bridge.add(self.attack, self.decay)
        self._op()
        zf, sf = self.bridge.cmp(sample_idx, decay_end)
        self._op()
        if sf:
            elapsed = self.bridge.sub(sample_idx, self.attack)
            self._op()
            drop = self.bridge.div(self.bridge.mul(elapsed, self.bridge.sub(100, self.sustain)), max(self.decay, 1))
            self._op()
            return self.bridge.sub(100, drop)
        
        # Release phase
        zf, sf = self.bridge.cmp(sample_idx, release_start)
        self._op()
        if not sf:  # sample >= release_start
            elapsed = self.bridge.sub(sample_idx, release_start)
            self._op()
            return self.bridge.sub(self.sustain, self.bridge.div(self.bridge.mul(elapsed, self.sustain), max(self.release, 1)))
        
        # Sustain
        return self.sustain


class NeuralAudioSynth:
    """Audio synthesizer combining oscillators + envelopes."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self.osc = NeuralOscillator(self.bridge)
        self._ops = 0
    
    def generate_tone(self, freq: int, duration_ms: int, wave: str = "square",
                      duty: int = 50) -> list[int]:
        """Generate a tone. Returns 8-bit unsigned PCM samples."""
        num_samples = self.bridge.div(self.bridge.mul(SAMPLE_RATE, duration_ms), 1000)
        period = self.bridge.div(SAMPLE_RATE, freq)
        
        # Envelope
        env = NeuralEnvelope(self.bridge, 
                           attack=num_samples // 10,
                           decay=num_samples // 10,
                           sustain=70,
                           release=num_samples // 5)
        
        samples = []
        for i in range(num_samples):
            phase = self.osc._neural_mod(i, period)
            
            if wave == "square":
                val = self.osc.square(phase, period)
            elif wave == "sawtooth":
                val = self.osc.sawtooth(phase, period)
            elif wave == "triangle":
                val = self.osc.triangle(phase, period)
            elif wave == "pulse":
                val = self.osc.pulse(phase, period, duty)
            else:
                val = self.osc.square(phase, period)
            
            # Apply envelope
            amp = env.amplitude(i, num_samples)
            val = self.bridge.div(self.bridge.mul(val, amp), 100)
            
            # Convert to unsigned 8-bit (0-255)
            sample = self.bridge.add(val, 128)
            # Clamp
            sample = max(0, min(255, sample))
            samples.append(sample)
        
        self._ops = self.osc._ops + env._ops
        return samples
    
    def mix(self, *tracks: list[int]) -> list[int]:
        """Mix multiple tracks using neural ADD + clamp."""
        if not tracks:
            return []
        
        max_len = max(len(t) for t in tracks)
        mixed = []
        
        for i in range(max_len):
            total = 0
            count = 0
            for track in tracks:
                if i < len(track):
                    total = self.bridge.add(total, self.bridge.sub(track[i], 128))
                    count += 1
            
            if count > 1:
                total = self.bridge.div(total, count)
            
            sample = self.bridge.add(total, 128)
            mixed.append(max(0, min(255, sample)))
        
        return mixed
    
    def to_wav(self, samples: list[int], path: str):
        """Write samples to WAV file."""
        num_samples = len(samples)
        data_size = num_samples
        file_size = 36 + data_size
        
        with open(path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", file_size))
            f.write(b"WAVE")
            
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<H", 1))   # PCM
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", SAMPLE_RATE))
            f.write(struct.pack("<I", SAMPLE_RATE))  # byte rate
            f.write(struct.pack("<H", 1))   # block align
            f.write(struct.pack("<H", 8))   # bits per sample
            
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(bytes(samples))


# ── CLI ──

def demo():
    synth = NeuralAudioSynth()
    
    print("Neural Audio Synthesizer")
    print("=" * 60)
    print("Waveform generation → neural arithmetic\n")
    
    # Generate short tones (keep small for speed)
    waves = ["square", "sawtooth", "triangle", "pulse"]
    
    for wave in waves:
        samples = synth.generate_tone(440, 10, wave=wave)  # 10ms A4 note
        
        # Show waveform ASCII
        width = 40
        step = max(1, len(samples) // width)
        viz = ""
        for i in range(0, min(len(samples), width * step), step):
            level = samples[i] // 32  # 0-7
            viz += "▁▂▃▄▅▆▇█"[min(level, 7)]
        
        print(f"  {wave:10s} 440Hz 50ms │{viz}│ {len(samples)} samples, {synth._ops} ops")
    
    print()
    
    # Generate a chord and save
    print("── Generate Chord ──")
    c = synth.generate_tone(262, 20, "triangle")   # C4
    e = synth.generate_tone(330, 20, "triangle")   # E4
    g = synth.generate_tone(392, 20, "triangle")   # G4
    
    chord = synth.mix(c, e, g)
    
    wav_path = "/Users/noc/projects/ncpu-bridge/neural_chord.wav"
    synth.to_wav(chord, wav_path)
    print(f"  C major chord (C4+E4+G4)")
    print(f"  Samples: {len(chord)}")
    print(f"  Saved: {wav_path}")
    print(f"  Duration: {len(chord)/SAMPLE_RATE*1000:.0f}ms")
    
    # Waveform visualization of chord
    print("\n── Chord Waveform ──")
    width = 60
    step = max(1, len(chord) // width)
    for row in range(7, -1, -1):
        line = "  │"
        for i in range(0, min(len(chord), width * step), step):
            level = chord[i] // 32
            line += "█" if level >= row else " "
        line += "│"
        print(line)
    print(f"  └{'─' * width}┘")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "generate" and len(sys.argv) > 2:
        synth = NeuralAudioSynth()
        samples = synth.generate_tone(440, 500, "square")
        synth.to_wav(samples, sys.argv[2])
        print(f"Generated {len(samples)} samples → {sys.argv[2]}")
    else:
        print("Usage: python -m bridge.neural_audio [demo|generate <output.wav>]")


if __name__ == "__main__":
    main()
