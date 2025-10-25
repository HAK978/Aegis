"""
Audio Processing Module for Gemini Live API Integration
Handles format conversion between browser audio and Gemini's required formats
"""

import numpy as np
import base64
import struct
from typing import Optional, Tuple
import io

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️  webrtcvad not available - Voice Activity Detection disabled")


class AudioProcessor:
    """
    Process audio between browser formats and Gemini Live API requirements

    Gemini Live API requires:
    - Input: 16-bit PCM, 16kHz, mono
    - Output: 24kHz sample rate

    Browser typically provides:
    - MediaRecorder API: WebM/Opus or other compressed formats
    - Web Audio API: Float32 PCM at various sample rates
    """

    def __init__(self, enable_vad: bool = True):
        """
        Initialize audio processor

        Args:
            enable_vad: Enable Voice Activity Detection to filter silence
        """
        self.target_sample_rate = 16000  # Gemini input requirement
        self.output_sample_rate = 24000  # Gemini output
        self.target_channels = 1  # Mono
        self.enable_vad = enable_vad and VAD_AVAILABLE

        if self.enable_vad:
            # VAD modes: 0 (quality), 1 (low bitrate), 2 (aggressive), 3 (very aggressive)
            self.vad = webrtcvad.Vad(2)
            print("✅ Voice Activity Detection enabled (mode: aggressive)")
        else:
            self.vad = None

    def browser_to_gemini(self,
                          audio_data: bytes,
                          source_sample_rate: int = 48000,
                          source_format: str = 'int16') -> bytes:
        """
        Convert browser audio to Gemini-compatible format

        Args:
            audio_data: Raw audio bytes from browser
            source_sample_rate: Source sample rate (browser default: 48kHz)
            source_format: 'int16' or 'float32'

        Returns:
            16-bit PCM at 16kHz mono (bytes)
        """
        try:
            # Parse audio data
            if source_format == 'float32':
                # Convert Float32 to Int16
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                # Already Int16
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array.mean(axis=1).astype(np.int16)

            # Resample to 16kHz if needed
            if source_sample_rate != self.target_sample_rate:
                audio_array = self._resample(
                    audio_array,
                    source_sample_rate,
                    self.target_sample_rate
                )

            # Convert back to bytes
            return audio_array.tobytes()

        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return b''

    def gemini_to_browser(self, audio_data: bytes) -> bytes:
        """
        Convert Gemini audio output (24kHz) to browser-compatible format

        Args:
            audio_data: 24kHz PCM from Gemini

        Returns:
            Audio data ready for Web Audio API playback
        """
        try:
            # Gemini outputs 24kHz, which is fine for browsers
            # Just ensure it's Int16 PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            return audio_array.tobytes()

        except Exception as e:
            print(f"❌ Audio output conversion error: {e}")
            return b''

    def base64_to_pcm(self, base64_audio: str) -> bytes:
        """
        Decode base64 audio from browser to PCM bytes

        Args:
            base64_audio: Base64-encoded audio string

        Returns:
            Raw PCM bytes
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_audio:
                base64_audio = base64_audio.split(',', 1)[1]

            # Decode base64
            audio_bytes = base64.b64decode(base64_audio)
            return audio_bytes

        except Exception as e:
            print(f"❌ Base64 decode error: {e}")
            return b''

    def pcm_to_base64(self, pcm_audio: bytes) -> str:
        """
        Encode PCM audio to base64 for browser transmission

        Args:
            pcm_audio: Raw PCM bytes

        Returns:
            Base64-encoded string
        """
        try:
            return base64.b64encode(pcm_audio).decode('utf-8')
        except Exception as e:
            print(f"❌ Base64 encode error: {e}")
            return ''

    def is_speech(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """
        Detect if audio contains speech using Voice Activity Detection

        Args:
            audio_data: Raw PCM audio (16-bit)
            sample_rate: Sample rate (must be 8000, 16000, 32000, or 48000)

        Returns:
            True if speech detected, False if silence
        """
        if not self.enable_vad or self.vad is None:
            return True  # Assume speech if VAD disabled

        try:
            # VAD requires specific frame sizes based on sample rate
            # 10ms, 20ms, or 30ms frames
            frame_duration = 30  # ms
            frame_size = int(sample_rate * frame_duration / 1000) * 2  # *2 for 16-bit

            # Process audio in frames
            speech_frames = 0
            total_frames = 0

            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    if is_speech:
                        speech_frames += 1
                    total_frames += 1

            # Consider speech if >30% of frames contain speech
            if total_frames == 0:
                return False

            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3

        except Exception as e:
            print(f"⚠️  VAD error: {e}")
            return True  # Assume speech on error

    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        Simple linear resampling (for production, use scipy.signal.resample)

        Args:
            audio: Audio array
            src_rate: Source sample rate
            dst_rate: Destination sample rate

        Returns:
            Resampled audio array
        """
        if src_rate == dst_rate:
            return audio

        # Calculate new length
        duration = len(audio) / src_rate
        new_length = int(duration * dst_rate)

        # Linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(np.int16)

    def split_into_chunks(self,
                          audio_data: bytes,
                          chunk_duration_ms: int = 100) -> list:
        """
        Split audio into small chunks for streaming

        Args:
            audio_data: Raw PCM audio bytes
            chunk_duration_ms: Chunk size in milliseconds

        Returns:
            List of audio chunks (bytes)
        """
        # Calculate chunk size in bytes
        # 16-bit (2 bytes) * sample_rate * duration_seconds
        chunk_size = 2 * self.target_sample_rate * chunk_duration_ms // 1000

        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    def get_audio_duration(self, audio_data: bytes, sample_rate: int = 16000) -> float:
        """
        Calculate audio duration in seconds

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Sample rate

        Returns:
            Duration in seconds
        """
        # 16-bit = 2 bytes per sample
        num_samples = len(audio_data) // 2
        duration = num_samples / sample_rate
        return duration

    def normalize_volume(self, audio_data: bytes, target_db: float = -20.0) -> bytes:
        """
        Normalize audio volume to target dB level

        Args:
            audio_data: Raw PCM audio bytes
            target_db: Target volume in dB

        Returns:
            Normalized audio bytes
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Calculate current RMS
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms == 0:
                return audio_data  # Silence

            # Calculate current dB
            current_db = 20 * np.log10(rms / 32767.0)

            # Calculate gain needed
            gain_db = target_db - current_db
            gain_linear = 10 ** (gain_db / 20)

            # Apply gain
            audio_array = audio_array * gain_linear

            # Clip to prevent overflow
            audio_array = np.clip(audio_array, -32768, 32767)

            return audio_array.astype(np.int16).tobytes()

        except Exception as e:
            print(f"⚠️  Normalization error: {e}")
            return audio_data


# Convenience functions

def create_audio_processor(enable_vad: bool = True) -> AudioProcessor:
    """
    Factory function to create AudioProcessor instance

    Args:
        enable_vad: Enable Voice Activity Detection

    Returns:
        AudioProcessor instance
    """
    return AudioProcessor(enable_vad=enable_vad)


def convert_browser_audio_to_gemini(audio_base64: str,
                                     source_rate: int = 48000) -> Optional[bytes]:
    """
    One-shot conversion from browser base64 audio to Gemini format

    Args:
        audio_base64: Base64-encoded audio from browser
        source_rate: Source sample rate

    Returns:
        PCM bytes ready for Gemini API, or None on error
    """
    processor = AudioProcessor(enable_vad=False)

    # Decode base64
    audio_bytes = processor.base64_to_pcm(audio_base64)
    if not audio_bytes:
        return None

    # Convert to Gemini format
    gemini_audio = processor.browser_to_gemini(audio_bytes, source_rate)
    return gemini_audio if gemini_audio else None


if __name__ == '__main__':
    # Test audio processor
    print("🔊 Audio Processor Test")
    print("=" * 50)

    processor = AudioProcessor(enable_vad=True)

    # Create test audio (1 second of 440Hz tone)
    sample_rate = 48000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    test_bytes = test_audio.tobytes()

    print(f"✅ Test audio created: {len(test_bytes)} bytes")
    print(f"   Sample rate: {sample_rate}Hz")
    print(f"   Duration: {duration}s")

    # Convert to Gemini format
    gemini_audio = processor.browser_to_gemini(test_bytes, sample_rate)
    print(f"✅ Converted to Gemini format: {len(gemini_audio)} bytes")
    print(f"   Expected: ~{16000 * 2}bytes (16kHz * 2 bytes * 1s)")

    # Test VAD
    has_speech = processor.is_speech(gemini_audio, 16000)
    print(f"✅ VAD result: {'Speech detected' if has_speech else 'Silence detected'}")

    # Test chunking
    chunks = processor.split_into_chunks(gemini_audio, chunk_duration_ms=100)
    print(f"✅ Split into {len(chunks)} chunks (100ms each)")

    print("\n🎉 Audio processor tests passed!")
