#!/usr/bin/env python3
"""
Lokaler KI-Sprachassistent für Raspberry Pi 5 (Wake-Word -> STT -> LLM -> TTS).

Installation (Raspberry Pi OS / Debian Bookworm, Python 3.11+ empfohlen):

    sudo apt update
    sudo apt install -y \
        python3 python3-venv python3-dev build-essential \
        portaudio19-dev libportaudiocpp0 libsndfile1 ffmpeg \
        espeak-ng alsa-utils

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools
    pip install \
        numpy pyaudio openwakeword ai-edge-litert onnxruntime \
        faster-whisper g4f piper-tts

Piper-Stimme herunterladen (Deutsch, empfohlen):

    mkdir -p voices
    python3 -m piper.download_voices --data-dir voices de_DE-thorsten-high

Alternative englische Stimme:

    python3 -m piper.download_voices --data-dir voices en_US-lessac-medium

Startbeispiele:

    python3 main.py
    python3 main.py --piper-model de_DE-thorsten-high --piper-data-dir ./voices
    python3 main.py --list-devices

Hinweise:
- Das Wake-Word ist standardmäßig "alexa" via openWakeWord.
- openWakeWord läuft standardmäßig im ONNX-Modus für bessere Kompatibilität auf dem Pi.
- faster-whisper läuft standardmäßig mit Modell "tiny", device="cpu", compute_type="int8".
- Piper wird lokal geladen und streamt Satz für Satz zur Audioausgabe.
- g4f nutzt mehrere Fallback-Provider, sofern im installierten g4f verfügbar.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
from piper import PiperVoice

try:
    from piper import SynthesisConfig
except ImportError:  # ältere piper-tts Versionen
    SynthesisConfig = None  # type: ignore[assignment]


@dataclass(slots=True)
class AssistantConfig:
    wake_word: str = "alexa"
    wake_threshold: float = 0.5
    openwakeword_vad_threshold: float = 0.0
    wakeword_inference_framework: str = "onnx"

    mic_rate: int = 16000
    engine_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunk_ms: int = 80
    mic_device_index: Optional[int] = None
    speaker_device_index: Optional[int] = None
    audio_queue_size: int = 256

    pre_roll_ms: int = 700
    silence_threshold: int = 450
    silence_multiplier: float = 2.5
    silence_seconds: float = 1.0
    speech_start_timeout: float = 4.0
    min_record_seconds: float = 0.35
    max_record_seconds: float = 15.0
    wake_cooldown_seconds: float = 1.25

    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = max(1, (os.cpu_count() or 4) // 2)
    stt_language: Optional[str] = None

    g4f_model: str = "gpt-4o-mini"
    g4f_providers: str = "auto,PollinationsAI,Blackbox,Copilot,OpenaiChat,You"
    llm_temperature: float = 0.3
    system_prompt: str = (
        "Du bist Alexa, ein lokaler Sprachassistent. "
        "Antworte kurz, direkt, natürlich und gut vorlesbar. "
        "Nutze maximal 2 kurze Sätze, außer der Nutzer verlangt mehr."
    )
    conversation_message_limit: int = 12

    piper_model: str = "de_DE-thorsten-high"
    piper_data_dir: str = "./voices"
    tts_output_rate: Optional[int] = None
    tts_backend: str = "auto"  # auto | pyaudio | aplay
    tts_volume: float = 1.0
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_noise_w_scale: float = 0.8
    tts_normalize_audio: bool = True

    debug: bool = False

    @property
    def frame_samples_engine(self) -> int:
        return int(self.engine_rate * self.chunk_ms / 1000)

    @property
    def frame_samples_mic(self) -> int:
        return int(self.mic_rate * self.chunk_ms / 1000)

    @property
    def frame_bytes_engine(self) -> int:
        return self.frame_samples_engine * self.sample_width * self.channels

    @property
    def pre_roll_frames(self) -> int:
        return max(1, int(self.pre_roll_ms / self.chunk_ms))


@dataclass(slots=True)
class TTSTask:
    text: str = ""
    barrier: Optional[threading.Event] = None
    stop: bool = False


def pcm_rms_int16(pcm_bytes: bytes, channels: int = 1) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples_f = samples.astype(np.float32)
    return float(np.sqrt(np.mean(samples_f * samples_f)))


def resample_int16_pcm(pcm_bytes: bytes, src_rate: int, dst_rate: int, channels: int = 1) -> bytes:
    if src_rate == dst_rate or not pcm_bytes:
        return pcm_bytes

    if channels < 1:
        raise ValueError("channels muss >= 1 sein")

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return pcm_bytes

    if samples.size % channels != 0:
        raise ValueError("PCM-Daten passen nicht zur Kanalzahl")

    frames = samples.reshape(-1, channels).astype(np.float32)
    src_len = frames.shape[0]
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))

    if src_len == 1:
        out = np.repeat(frames, dst_len, axis=0)
    else:
        x_src = np.arange(src_len, dtype=np.float32)
        x_dst = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
        out = np.empty((dst_len, channels), dtype=np.float32)
        for ch in range(channels):
            out[:, ch] = np.interp(x_dst, x_src, frames[:, ch])

    out = np.clip(np.rint(out), -32768, 32767).astype(np.int16)
    return out.tobytes()


class PCMResampler:
    def __init__(self, src_rate: int, dst_rate: int, sample_width: int = 2, channels: int = 1) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.sample_width = sample_width
        self.channels = channels

    def reset(self) -> None:
        return None

    def process(self, pcm_bytes: bytes) -> bytes:
        if self.sample_width != 2:
            raise ValueError("Dieses Skript unterstützt nur 16-bit PCM Audio")
        return resample_int16_pcm(pcm_bytes, self.src_rate, self.dst_rate, self.channels)


OPENWAKEWORD_RELEASE_BASE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
OPENWAKEWORD_MODEL_STEMS = {
    "alexa": "alexa_v0.1",
    "hey_mycroft": "hey_mycroft_v0.1",
    "hey_jarvis": "hey_jarvis_v0.1",
    "hey_rhasspy": "hey_rhasspy_v0.1",
    "timer": "timer_v0.1",
    "weather": "weather_v0.1",
}


class SentenceChunker:
    SENTENCE_END_RE = re.compile(r"(.+?[.!?…]+[\"'”»\])}]*)(?=\s+|$)|(.+?[.!?…]+)(?=\s+|$)", re.S)

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text_delta: str) -> list[str]:
        self.buffer += text_delta
        return self._pop_ready(force=False)

    def flush(self) -> list[str]:
        return self._pop_ready(force=True)

    def _pop_ready(self, force: bool) -> list[str]:
        ready: list[str] = []

        while True:
            match = self.SENTENCE_END_RE.search(self.buffer)
            if match and match.start() == 0:
                sentence = match.group(0).strip()
                self.buffer = self.buffer[match.end() :].lstrip()
                if sentence:
                    ready.append(sentence)
                continue

            if not force and len(self.buffer) > 160:
                split_candidates = [self.buffer.rfind(sep) for sep in (",", ";", ":", "\n")]
                split_at = max(split_candidates)
                if split_at >= 48:
                    sentence = self.buffer[: split_at + 1].strip()
                    self.buffer = self.buffer[split_at + 1 :].lstrip()
                    if sentence:
                        ready.append(sentence)
                    continue

            if force:
                tail = self.buffer.strip()
                self.buffer = ""
                if tail:
                    ready.append(tail)
            break

        return ready


def sanitize_for_tts(text: str) -> str:
    text = re.sub(r"```.+?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = text.replace("#", " ")
    text = text.replace("•", ", ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_leading_wake_word(text: str, wake_word: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:hey\s+)?{re.escape(wake_word)}(?:\s+bitte)?(?:[,.:;!?\-\s]+|$)",
        re.IGNORECASE,
    )
    return pattern.sub("", text).strip()


def write_temp_wav(samples: np.ndarray, sample_rate: int) -> str:
    fd, path = tempfile.mkstemp(prefix="alexa_", suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return path


def list_audio_devices() -> None:
    pa = pyaudio.PyAudio()
    try:
        print("\nVerfügbare Audio-Geräte:\n")
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            print(
                f"[{idx}] {info.get('name')} | "
                f"in={int(info.get('maxInputChannels', 0))} | "
                f"out={int(info.get('maxOutputChannels', 0))} | "
                f"default_rate={int(info.get('defaultSampleRate', 0))}"
            )
    finally:
        pa.terminate()


def resolve_piper_model_path(model_arg: str, data_dir: str) -> Path:
    candidate = Path(model_arg).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    if candidate.suffix == ".onnx":
        maybe = (Path(data_dir).expanduser() / candidate.name).resolve()
        if maybe.is_file():
            return maybe

    name = candidate.stem if candidate.suffix else candidate.name
    search_root = Path(data_dir).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(
            f"Piper-Datenverzeichnis nicht gefunden: {search_root}\n"
            f"Bitte Stimme laden, z. B.: python3 -m piper.download_voices --data-dir {search_root} de_DE-thorsten-high"
        )

    direct = search_root / f"{name}.onnx"
    if direct.is_file():
        return direct

    matches = sorted(search_root.rglob(f"{name}.onnx"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Konnte Piper-Modell '{model_arg}' unter '{search_root}' nicht finden.\n"
        f"Beispiel: python3 -m piper.download_voices --data-dir {search_root} de_DE-thorsten-high"
    )


def download_file_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    logging.info("Lade herunter: %s -> %s", url, destination)
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(destination)
    return destination


def prepare_openwakeword_assets(wake_word: str, inference_framework: str) -> dict[str, str]:
    candidate = Path(wake_word).expanduser()
    normalized = wake_word.strip().lower().replace(" ", "_")
    extension = ".onnx" if inference_framework == "onnx" else ".tflite"

    cache_dir = Path(".cache/openwakeword").resolve()
    model_dir = cache_dir / "models"

    melspec_name = f"melspectrogram{extension}"
    embedding_name = f"embedding_model{extension}"
    melspec_path = download_file_if_missing(
        f"{OPENWAKEWORD_RELEASE_BASE_URL}/{melspec_name}",
        model_dir / melspec_name,
    )
    embedding_path = download_file_if_missing(
        f"{OPENWAKEWORD_RELEASE_BASE_URL}/{embedding_name}",
        model_dir / embedding_name,
    )

    if candidate.is_file():
        wake_model_path = candidate.resolve()
    else:
        stem = OPENWAKEWORD_MODEL_STEMS.get(normalized)
        if not stem:
            available = ", ".join(sorted(OPENWAKEWORD_MODEL_STEMS.keys()))
            raise FileNotFoundError(
                f"Wake-Word-Modell '{wake_word}' wird von diesem Skript nicht unterstützt. "
                f"Verfügbare integrierte Modelle: {available}"
            )
        filename = f"{stem}{extension}"
        wake_model_path = download_file_if_missing(
            f"{OPENWAKEWORD_RELEASE_BASE_URL}/{filename}",
            model_dir / filename,
        )

    return {
        "wake_model": str(wake_model_path),
        "melspec_model": str(melspec_path),
        "embedding_model": str(embedding_path),
        "framework": inference_framework,
    }


def instantiate_openwakeword_model(assets: dict[str, str], vad_threshold: float = 0.0) -> WakeWordModel:
    attempts: list[tuple[str, bool, dict[str, object]]] = [
        (
            "positional+framework",
            False,
            {
                "vad_threshold": vad_threshold,
                "inference_framework": assets["framework"],
                "melspec_model_path": assets["melspec_model"],
                "embedding_model_path": assets["embedding_model"],
            },
        ),
        (
            "positional-no-framework",
            False,
            {
                "vad_threshold": vad_threshold,
                "melspec_model_path": assets["melspec_model"],
                "embedding_model_path": assets["embedding_model"],
            },
        ),
        (
            "keyword+framework",
            True,
            {
                "wakeword_models": [assets["wake_model"]],
                "vad_threshold": vad_threshold,
                "inference_framework": assets["framework"],
                "melspec_model_path": assets["melspec_model"],
                "embedding_model_path": assets["embedding_model"],
            },
        ),
        (
            "keyword-no-framework",
            True,
            {
                "wakeword_models": [assets["wake_model"]],
                "vad_threshold": vad_threshold,
                "melspec_model_path": assets["melspec_model"],
                "embedding_model_path": assets["embedding_model"],
            },
        ),
    ]

    last_error: Optional[Exception] = None
    for label, keyword_only, kwargs in attempts:
        try:
            if keyword_only:
                return WakeWordModel(**kwargs)
            return WakeWordModel([assets["wake_model"]], **kwargs)
        except Exception as exc:
            last_error = exc
            logging.debug("openWakeWord Init-Versuch '%s' fehlgeschlagen: %s", label, exc)

    raise RuntimeError(f"Konnte openWakeWord-Modell nicht initialisieren: {last_error}")


class MicrophoneReader(threading.Thread):
    def __init__(self, config: AssistantConfig, out_queue: queue.Queue[np.ndarray], stop_event: threading.Event) -> None:
        super().__init__(name="MicrophoneReader", daemon=True)
        self.config = config
        self.out_queue = out_queue
        self.stop_event = stop_event
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._leftover = b""
        self._resampler = PCMResampler(config.mic_rate, config.engine_rate, config.sample_width, config.channels)

    def run(self) -> None:
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.mic_rate,
                input=True,
                input_device_index=self.config.mic_device_index,
                frames_per_buffer=self.config.frame_samples_mic,
            )
            logging.info(
                "Mikrofon geöffnet: rate=%s, chunk=%s, device=%s",
                self.config.mic_rate,
                self.config.frame_samples_mic,
                self.config.mic_device_index,
            )

            while not self.stop_event.is_set():
                try:
                    data = self._stream.read(
                        self.config.frame_samples_mic,
                        exception_on_overflow=False,
                    )
                except OSError as exc:
                    logging.warning("Mikrofon-Lesefehler: %s", exc)
                    time.sleep(0.05)
                    continue

                data = self._resampler.process(data)
                self._leftover += data

                while len(self._leftover) >= self.config.frame_bytes_engine:
                    frame_bytes = self._leftover[: self.config.frame_bytes_engine]
                    self._leftover = self._leftover[self.config.frame_bytes_engine :]
                    frame = np.frombuffer(frame_bytes, dtype=np.int16).copy()
                    self._put_frame(frame)
        finally:
            self.close()

    def _put_frame(self, frame: np.ndarray) -> None:
        try:
            self.out_queue.put(frame, timeout=0.2)
        except queue.Full:
            try:
                _ = self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(frame)
            except queue.Full:
                logging.debug("Audio-Queue voll, Frame verworfen.")

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None


class AudioOutput:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._stream_signature: tuple[int, int, int] | None = None
        self._lock = threading.Lock()

    def play_piper_chunks(self, chunks: Iterable[object]) -> None:
        backend = self.config.tts_backend
        if backend not in {"auto", "pyaudio", "aplay"}:
            raise ValueError("tts_backend muss auto, pyaudio oder aplay sein")

        if backend == "aplay":
            self._play_with_aplay(chunks)
            return

        try:
            self._play_with_pyaudio(chunks)
        except Exception as exc:
            if backend == "pyaudio":
                raise
            logging.warning("PyAudio-Ausgabe fehlgeschlagen, wechsle zu aplay: %s", exc)
            self._play_with_aplay(chunks)

    def _play_with_pyaudio(self, chunks: Iterable[object]) -> None:
        for chunk in chunks:
            raw = getattr(chunk, "audio_int16_bytes")
            src_rate = int(getattr(chunk, "sample_rate", 22050))
            sample_width = int(getattr(chunk, "sample_width", 2))
            channels = int(getattr(chunk, "sample_channels", 1))
            out_rate = self.config.tts_output_rate or src_rate

            if sample_width != 2:
                raise RuntimeError(f"Nur 16-bit PCM wird unterstützt, bekam {sample_width * 8} bit")
            if out_rate != src_rate:
                raw = resample_int16_pcm(raw, src_rate, out_rate, channels)

            self._ensure_output_stream(out_rate, channels, sample_width)
            assert self._stream is not None
            self._stream.write(raw)

    def _play_with_aplay(self, chunks: Iterable[object]) -> None:
        proc: Optional[subprocess.Popen[bytes]] = None

        try:
            for chunk in chunks:
                raw = getattr(chunk, "audio_int16_bytes")
                src_rate = int(getattr(chunk, "sample_rate", 22050))
                sample_width = int(getattr(chunk, "sample_width", 2))
                channels = int(getattr(chunk, "sample_channels", 1))
                out_rate = self.config.tts_output_rate or src_rate

                if sample_width != 2:
                    raise RuntimeError(f"aplay-Fallback erwartet 16-bit PCM, bekam {sample_width * 8} bit")

                if out_rate != src_rate:
                    raw = resample_int16_pcm(raw, src_rate, out_rate, channels)

                if proc is None:
                    cmd = [
                        "aplay",
                        "-q",
                        "-f",
                        "S16_LE",
                        "-r",
                        str(out_rate),
                        "-c",
                        str(channels),
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

                assert proc.stdin is not None
                proc.stdin.write(raw)

            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
                proc.wait(timeout=10)
        finally:
            if proc is not None:
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                if proc.poll() is None:
                    proc.terminate()

    def _ensure_output_stream(self, rate: int, channels: int, sample_width: int) -> None:
        signature = (rate, channels, sample_width)
        with self._lock:
            if self._pa is None:
                self._pa = pyaudio.PyAudio()

            if self._stream is not None and self._stream_signature == signature:
                return

            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._stream_signature = None

            pa_format = self._pa.get_format_from_width(sample_width, unsigned=False)
            self._stream = self._pa.open(
                format=pa_format,
                channels=channels,
                rate=rate,
                output=True,
                output_device_index=self.config.speaker_device_index,
                frames_per_buffer=1024,
            )
            self._stream_signature = signature

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._stream_signature = None

            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None


class PiperTTSWorker(threading.Thread):
    def __init__(self, config: AssistantConfig) -> None:
        super().__init__(name="PiperTTSWorker", daemon=True)
        self.config = config
        self.model_path = resolve_piper_model_path(config.piper_model, config.piper_data_dir)
        self.voice = PiperVoice.load(str(self.model_path))
        self.audio_output = AudioOutput(config)
        self.queue: queue.Queue[TTSTask] = queue.Queue()
        self._syn_config = self._build_synthesis_config()

    def _build_synthesis_config(self):
        if SynthesisConfig is None:
            return None
        return SynthesisConfig(
            volume=self.config.tts_volume,
            length_scale=self.config.tts_length_scale,
            noise_scale=self.config.tts_noise_scale,
            noise_w_scale=self.config.tts_noise_w_scale,
            normalize_audio=self.config.tts_normalize_audio,
        )

    def speak(self, text: str) -> None:
        text = sanitize_for_tts(text)
        if text:
            self.queue.put(TTSTask(text=text))

    def barrier(self) -> threading.Event:
        event = threading.Event()
        self.queue.put(TTSTask(barrier=event))
        return event

    def stop(self) -> None:
        self.queue.put(TTSTask(stop=True))

    def run(self) -> None:
        logging.info("Piper geladen: %s", self.model_path)
        try:
            while True:
                task = self.queue.get()
                if task.stop:
                    break
                if task.barrier is not None:
                    task.barrier.set()
                    continue
                if task.text:
                    self._play_text(task.text)
        finally:
            self.audio_output.close()

    def _play_text(self, text: str) -> None:
        logging.info("TTS: %s", text)
        try:
            if self._syn_config is not None:
                chunks = self.voice.synthesize(text, syn_config=self._syn_config)
            else:
                chunks = self.voice.synthesize(text)
        except TypeError:
            chunks = self.voice.synthesize(text)
        self.audio_output.play_piper_chunks(chunks)


class SpeechRecognizer:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            cpu_threads=config.whisper_cpu_threads,
            num_workers=1,
        )
        logging.info(
            "Whisper geladen: model=%s, device=%s, compute_type=%s, threads=%s",
            config.whisper_model,
            config.whisper_device,
            config.whisper_compute_type,
            config.whisper_cpu_threads,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        wav_path = write_temp_wav(samples, self.config.engine_rate)
        kwargs = {
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
        if self.config.stt_language:
            kwargs["language"] = self.config.stt_language

        try:
            segments, info = self.model.transcribe(wav_path, **kwargs)
            texts = [segment.text.strip() for segment in segments if getattr(segment, "text", "").strip()]
            text = " ".join(texts).strip()
            if text:
                logging.info(
                    "STT: '%s' (sprache=%s, prob=%.2f)",
                    text,
                    getattr(info, "language", "?"),
                    float(getattr(info, "language_probability", 0.0)),
                )
            else:
                logging.info("STT: keine Sprache erkannt")
            return text
        finally:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass


class G4FResponder:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        from g4f.client import AsyncClient  # lazy import für klarere Fehlermeldungen

        self.AsyncClient = AsyncClient
        self.provider_module = __import__("g4f.Provider", fromlist=["dummy"])
        self.providers = self._resolve_providers(config.g4f_providers)
        logging.info("g4f Provider-Fallbacks: %s", ", ".join(self._provider_label(p) for p in self.providers))

    def _resolve_providers(self, provider_string: str) -> list[object | None]:
        providers: list[object | None] = []
        for raw_name in provider_string.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if name.lower() == "auto":
                providers.append(None)
                continue
            provider = getattr(self.provider_module, name, None)
            if provider is not None:
                providers.append(provider)
            else:
                logging.debug("g4f Provider nicht vorhanden, überspringe: %s", name)

        if not providers:
            providers.append(None)
        return providers

    @staticmethod
    def _provider_label(provider: object | None) -> str:
        return "auto" if provider is None else getattr(provider, "__name__", str(provider))

    @staticmethod
    def _extract_text(chunk: object) -> str:
        if isinstance(chunk, str):
            return chunk

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            content = getattr(chunk, "content", None)
            return str(content) if content else ""

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if content:
                return str(content)

        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content:
                return str(content)

        content = getattr(choice, "content", None)
        return str(content) if content else ""

    async def generate(self, messages: list[dict[str, str]], on_delta: Callable[[str], None]) -> str:
        last_error: Optional[Exception] = None

        for provider in self.providers:
            label = self._provider_label(provider)
            try:
                return await self._generate_with_provider(provider, label, messages, on_delta)
            except Exception as exc:
                last_error = exc
                logging.warning("g4f Provider '%s' fehlgeschlagen: %s", label, exc)

        raise RuntimeError(f"Alle g4f Provider fehlgeschlagen: {last_error}")

    async def _generate_with_provider(
        self,
        provider: object | None,
        label: str,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], None],
    ) -> str:
        client = self.AsyncClient(provider=provider) if provider is not None else self.AsyncClient()
        full_parts: list[str] = []

        try:
            stream_method = getattr(client.chat.completions, "stream", None)
            if callable(stream_method):
                stream = stream_method(
                    model=self.config.g4f_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    web_search=False,
                )
                async for chunk in stream:
                    text = self._extract_text(chunk)
                    if text:
                        full_parts.append(text)
                        on_delta(text)
            else:
                response = await client.chat.completions.create(
                    model=self.config.g4f_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    web_search=False,
                )
                text = self._extract_text(response)
                if text:
                    full_parts.append(text)
                    on_delta(text)
        except Exception:
            partial_text = "".join(full_parts).strip()
            if partial_text:
                logging.warning("Provider '%s' brach nach Teilantwort ab; verwende partielle Antwort.", label)
                return partial_text
            raise

        full_text = "".join(full_parts).strip()
        if not full_text:
            raise RuntimeError(f"Provider '{label}' lieferte keine Antwort")

        logging.info("LLM (%s): %s", label, full_text)
        return full_text


class VoiceAssistant:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.audio_frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.audio_queue_size)
        self.pre_roll: Deque[np.ndarray] = deque(maxlen=config.pre_roll_frames)
        self.history: Deque[dict[str, str]] = deque(maxlen=config.conversation_message_limit)
        self.history_lock = threading.Lock()
        self.busy_event = threading.Event()

        self.mic_reader = MicrophoneReader(config, self.audio_frames, self.stop_event)
        self.tts_worker = PiperTTSWorker(config)
        self.recognizer = SpeechRecognizer(config)
        self.responder = G4FResponder(config)
        self.wake_model = self._load_wake_model()

        self.state = "idle"
        self.recording_frames: list[np.ndarray] = []
        self.recording_started_at = 0.0
        self.last_voice_at = 0.0
        self.speech_detected = False
        self.last_wake_at = 0.0
        self.noise_floor = float(config.silence_threshold)
        self.current_recording_threshold = float(config.silence_threshold)

    def _load_wake_model(self) -> WakeWordModel:
        assets = prepare_openwakeword_assets(self.config.wake_word, "onnx")
        model = instantiate_openwakeword_model(assets, vad_threshold=0.0)
        logging.info(
            "openWakeWord geladen: wake_word=%s, framework=%s, model_path=%s",
            self.config.wake_word,
            assets["framework"],
            assets["wake_model"],
        )
        return model

    def run(self) -> None:
        logging.info("Starte Assistent. Wake-Word: '%s'", self.config.wake_word)
        self.tts_worker.start()
        self.mic_reader.start()

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.audio_frames.get(timeout=0.25)
                except queue.Empty:
                    if not self.mic_reader.is_alive() and not self.stop_event.is_set():
                        raise RuntimeError("Mikrofon-Thread wurde unerwartet beendet")
                    continue
                self._process_frame(frame)
        except KeyboardInterrupt:
            logging.info("Abbruch per Ctrl+C")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.stop_event.is_set():
            return

        logging.info("Fahre Assistent herunter...")
        self.stop_event.set()
        self.mic_reader.close()
        self.tts_worker.stop()

        if self.mic_reader.is_alive():
            self.mic_reader.join(timeout=2)
        if self.tts_worker.is_alive():
            self.tts_worker.join(timeout=5)

    def _process_frame(self, frame: np.ndarray) -> None:
        rms = pcm_rms_int16(frame.tobytes(), self.config.channels)
        self._update_noise_floor(rms)

        if self.state == "recording":
            self._handle_recording_frame(frame, rms)
            return

        self.pre_roll.append(frame.copy())

        if self.busy_event.is_set():
            return

        now = time.monotonic()
        if now - self.last_wake_at < self.config.wake_cooldown_seconds:
            return

        prediction = self.wake_model.predict(frame)
        scores = [float(v) for v in prediction.values()] if prediction else []
        score = max(scores) if scores else 0.0
        if score >= self.config.wake_threshold:
            self.last_wake_at = now
            self._start_recording(score)

    def _update_noise_floor(self, rms: float) -> None:
        if self.state == "recording" and rms >= self.current_recording_threshold:
            return
        alpha = 0.97
        self.noise_floor = alpha * self.noise_floor + (1.0 - alpha) * rms

    def _start_recording(self, score: float) -> None:
        self.state = "recording"
        self.recording_frames = [frame.copy() for frame in self.pre_roll]
        self.recording_started_at = time.monotonic()
        self.last_voice_at = self.recording_started_at
        self.speech_detected = False
        self.current_recording_threshold = max(
            float(self.config.silence_threshold),
            float(self.noise_floor) * self.config.silence_multiplier,
        )
        logging.info(
            "Wake-Word erkannt (score=%.3f). Aufnahme gestartet. silence_threshold=%.1f",
            score,
            self.current_recording_threshold,
        )

    def _handle_recording_frame(self, frame: np.ndarray, rms: float) -> None:
        now = time.monotonic()
        self.recording_frames.append(frame.copy())

        if rms >= self.current_recording_threshold:
            self.last_voice_at = now
            self.speech_detected = True

        recording_duration = now - self.recording_started_at
        silence_duration = now - self.last_voice_at

        if not self.speech_detected and recording_duration >= self.config.speech_start_timeout:
            logging.info("Keine Sprache nach Wake-Word erkannt, Aufnahme verworfen.")
            self._reset_recording_state()
            return

        if self.speech_detected:
            if recording_duration >= self.config.min_record_seconds and silence_duration >= self.config.silence_seconds:
                self._finalize_recording()
                return

        if recording_duration >= self.config.max_record_seconds:
            logging.info("Maximale Aufnahmedauer erreicht, stoppe Aufnahme.")
            self._finalize_recording()

    def _finalize_recording(self) -> None:
        samples = np.concatenate(self.recording_frames).astype(np.int16)
        self._reset_recording_state()

        if self.busy_event.is_set():
            logging.debug("Assistent ist bereits beschäftigt, Aufnahme verworfen.")
            return

        self.busy_event.set()
        worker = threading.Thread(
            target=self._handle_interaction,
            args=(samples,),
            name="InteractionWorker",
            daemon=True,
        )
        worker.start()

    def _reset_recording_state(self) -> None:
        self.state = "idle"
        self.recording_frames = []
        self.recording_started_at = 0.0
        self.last_voice_at = 0.0
        self.speech_detected = False

    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        with self.history_lock:
            history = list(self.history)
        return [{"role": "system", "content": self.config.system_prompt}, *history, {"role": "user", "content": user_text}]

    def _append_history(self, role: str, content: str) -> None:
        with self.history_lock:
            self.history.append({"role": role, "content": content})

    def _handle_interaction(self, samples: np.ndarray) -> None:
        try:
            text = self.recognizer.transcribe(samples)
            text = strip_leading_wake_word(text, self.config.wake_word)
            if not text:
                logging.info("Nur Wake-Word oder leere Eingabe erkannt; keine LLM-Anfrage.")
                return

            messages = self._build_messages(text)
            chunker = SentenceChunker()

            def on_delta(delta: str) -> None:
                for sentence in chunker.feed(delta):
                    self.tts_worker.speak(sentence)

            full_reply = asyncio.run(self.responder.generate(messages, on_delta))

            for sentence in chunker.flush():
                self.tts_worker.speak(sentence)

            full_reply = sanitize_for_tts(full_reply)
            if full_reply:
                self._append_history("user", text)
                self._append_history("assistant", full_reply)

            barrier = self.tts_worker.barrier()
            barrier.wait(timeout=30)
        except Exception as exc:
            logging.exception("Interaktion fehlgeschlagen: %s", exc)
            self.tts_worker.speak("Entschuldigung, ich kann gerade nicht antworten.")
            barrier = self.tts_worker.barrier()
            barrier.wait(timeout=10)
        finally:
            self.busy_event.clear()


def parse_args() -> AssistantConfig:
    parser = argparse.ArgumentParser(description="Lokaler Sprachassistent für Raspberry Pi 5")
    parser.add_argument("--list-devices", action="store_true", help="Audio-Geräte auflisten und beenden")
    parser.add_argument("--mic-device", type=int, default=None, help="PyAudio Input-Device-Index")
    parser.add_argument("--speaker-device", type=int, default=None, help="PyAudio Output-Device-Index")
    parser.add_argument("--mic-rate", type=int, default=16000, help="Mikrofon-Samplerate, z. B. 16000 oder 48000")
    parser.add_argument("--engine-rate", type=int, default=16000, help="Interne Samplerate für Wake-Word/STT")
    parser.add_argument("--wake-threshold", type=float, default=0.5, help="Wake-Word Schwellwert")
    parser.add_argument("--silence-threshold", type=int, default=450, help="Basis-RMS-Schwelle für Stille")
    parser.add_argument("--silence-seconds", type=float, default=1.0, help="Sekunden bis Aufnahmeende bei Stille")
    parser.add_argument("--stt-language", type=str, default=None, help="Optionale Whisper-Sprachvorgabe, z. B. de oder en")
    parser.add_argument("--piper-model", type=str, default="de_DE-thorsten-high", help="Piper-Modellname oder .onnx-Pfad")
    parser.add_argument("--piper-data-dir", type=str, default="./voices", help="Verzeichnis für Piper-Stimmen")
    parser.add_argument("--tts-output-rate", type=int, default=None, help="Optionale feste Ausgabe-Samplerate")
    parser.add_argument("--tts-backend", choices=["auto", "pyaudio", "aplay"], default="auto")
    parser.add_argument("--g4f-model", type=str, default="gpt-4o-mini", help="g4f Modellname")
    parser.add_argument(
        "--g4f-providers",
        type=str,
        default="auto,PollinationsAI,Blackbox,Copilot,OpenaiChat,You",
        help="Kommagetrennte Provider-Fallbacks, z. B. auto,PollinationsAI,Blackbox",
    )
    parser.add_argument("--debug", action="store_true", help="Ausführliche Logs aktivieren")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        raise SystemExit(0)

    return AssistantConfig(
        mic_rate=args.mic_rate,
        engine_rate=args.engine_rate,
        mic_device_index=args.mic_device,
        speaker_device_index=args.speaker_device,
        wake_threshold=args.wake_threshold,
        silence_threshold=args.silence_threshold,
        silence_seconds=args.silence_seconds,
        stt_language=args.stt_language,
        piper_model=args.piper_model,
        piper_data_dir=args.piper_data_dir,
        tts_output_rate=args.tts_output_rate,
        tts_backend=args.tts_backend,
        g4f_model=args.g4f_model,
        g4f_providers=args.g4f_providers,
        debug=args.debug,
    )


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )


def main() -> int:
    config = parse_args()
    configure_logging(config.debug)

    try:
        assistant = VoiceAssistant(config)
    except Exception as exc:
        logging.exception("Initialisierung fehlgeschlagen: %s", exc)
        return 1

    assistant.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
