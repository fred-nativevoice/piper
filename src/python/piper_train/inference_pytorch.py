#!/usr/bin/env python3
"""
This script performs text-to-speech synthesis using a VITS model checkpoint in PyTorch.
It takes text as input, converts it to phonemes using espeak-ng, and generates
a WAV audio file.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from vits.lightning import VitsModel

from scipy.io.wavfile import write as write_wav
from symbols import _bos, _eos, _pad, symbols

_LOGGER = logging.getLogger("piper_inference_pytorch")

# --- Global symbol mapping ---
# Create a dictionary to map each symbol to its unique ID.
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """
    Normalizes a float audio array to the int16 range for WAV file saving.

    Args:
        audio: A NumPy array containing the audio data in float format.
        max_wav_value: The maximum amplitude for the int16 range.

    Returns:
        A NumPy array with the audio data converted to int16.
    """
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


def get_phonemes(text: str) -> str:
    """
    Generates phonemes for a given text string using the espeak-ng command-line tool.

    Args:
        text: The input text to convert to phonemes.

    Returns:
        A string containing the generated phonemes, or None if an error occurs.
    """
    try:
        # Garante que a variável de ambiente para o idioma seja usada, se definida
        lang = os.environ.get('ESPEAK_LANG', 'en')
        command = ["espeak-ng", f"-v{lang}", "-q", "-x", text]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        _LOGGER.error(f"Erro ao gerar fonemas para '{text}': {e}")
        _LOGGER.error(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        _LOGGER.error(
            "Erro: 'espeak-ng' não encontrado. "
            "Certifique-se de que está instalado e no seu PATH."
        )
        return None
    except Exception as e:
        _LOGGER.error(f"Ocorreu um erro inesperado no espeak-ng: {e}")
        return None


def text_to_phoneme_ids(text: str) -> List[int]:
    """
    Converts an input text string into a list of phoneme IDs for the model.

    Args:
        text: The input text string.

    Returns:
        A list of integer phoneme IDs, or None if conversion fails.
    """
    phonemes = get_phonemes(text)
    if not phonemes:
        return None

    phoneme_ids_base = [_symbol_to_id[c] for c in list(phonemes) if c in _symbol_to_id]
    pad_id = _symbol_to_id[_pad]
    phoneme_ids = []

    for i, pid in enumerate(phoneme_ids_base):
        phoneme_ids.append(pid)
        if i < len(phoneme_ids_base) - 1:
            phoneme_ids.append(pad_id)

    phoneme_ids = [_symbol_to_id[_bos]] + [_symbol_to_id[_pad]]  + phoneme_ids + [_symbol_to_id[_pad]] + [_symbol_to_id[_eos]] 

    return phoneme_ids


def inference(
    model_path: str,
    output_dir: str,
    sample_rate: int,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    input_text: str,
    output_file: str,
    speaker_id: int = None
):
    """
    Loads the VITS model and runs inference to generate audio from text.

    Args:
        model_path: Path to the PyTorch model checkpoint (.ckpt).
        output_dir: Directory where the output WAV file will be saved.
        sample_rate: The audio sample rate the model was trained on.
        noise_scale: Stochastic noise scale for the model's output.
        noise_scale_w: Noise scale for the duration predictor.
        length_scale: Controls the speed of the speech (e.g., >1 is slower).
        input_text: The text to synthesize.
        output_filename: The name for the output WAV file.
        speaker_id: The ID of the speaker (for multi-speaker models).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Model ---
    _LOGGER.debug(f"Carregando modelo do checkpoint: {model_path}")
    # Use map_location='cpu' to ensure the model loads on CPU if no GPU is available.
    model = VitsModel.load_from_checkpoint(model_path, dataset=None, map_location='cpu')
    model.eval()

    # Apply weight normalization removal for faster inference.
    with torch.no_grad():
        try:
            model.model_g.dec.remove_weight_norm()
        except Exception as e:
            _LOGGER.warning(f"Não foi possível aplicar remove_weight_norm: {e}")

    _LOGGER.info(f"Modelo PyTorch carregado de {model_path}")

    # --- 2. Process Input Text ---
    phoneme_ids = text_to_phoneme_ids(input_text)
    if not phoneme_ids:
        _LOGGER.error("Falha ao converter texto em fonemas. Abortando.")
        return

    # --- 3. Prepare Tensors for the Model ---
    text = torch.LongTensor(phoneme_ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(phoneme_ids)])
    scales = [noise_scale, length_scale, noise_scale_w]
    sid = torch.LongTensor([speaker_id]) if speaker_id is not None else None

    # --- 4. Run Inference ---
    start_time = time.perf_counter()

    audio = model(text, text_lengths, scales, sid=sid).detach().numpy()

    end_time = time.perf_counter()

    # --- 5. Process and Save Audio ---
    audio = audio_float_to_int16(audio)

    # --- 6. Log Performance Metrics ---
    audio_duration_sec = audio.shape[-1] / sample_rate
    infer_sec = end_time - start_time
    real_time_factor = infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

    _LOGGER.info(f"Inference time: {infer_sec:.2f} seconds")
    _LOGGER.info(f"Audio duration: {audio_duration_sec:.2f} seconds")
    _LOGGER.info(f"Real-time factor: {real_time_factor:.2f}")

    # Ensure the output filename has a .wav extension.
    if not output_file.lower().endswith(".wav"):
        output_file += ".wav"
        
    output_path = output_dir / output_file
    write_wav(str(output_path), sample_rate, audio)
    print(f"\n✅ Audio saved to: {output_path}")


def main():
    """Main entry point for the script."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="inference_pytorch.py",
        description="Synthesize audio from text using a VITS PyTorch checkpoint.",
    )
    # Argumentos principais
    parser.add_argument("--model", required=True, help="Path to the model checkpoint file (.ckpt).")
    parser.add_argument("--output_dir", required=True, help="irectory to save the output audio file.")
    parser.add_argument("--input_text", required=True, help="The text to be synthesized.")
    parser.add_argument("--output_file", required=True, help="Name of the output audio file (default: 'output.wav').")

    # Argumentos de configuração do modelo
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate (must match the model's training rate).")
    parser.add_argument("--speaker_id", type=int, help="Speaker ID for multi-speaker models.")
    
    # Argumentos de escala (mesmos de infer.py)
    parser.add_argument("--noise_scale", type=float, default=0.667, help="Stochastic noise scale (default: 0.667).")
    parser.add_argument("--noise_scale_w", type=float, default=0.8, help="Noise scale for the duration predictor (default: 0.8).")
    parser.add_argument("--length_scale", type=float, default=1.0, help="Speech speed control (e.g., >1 is slower, <1 is faster).")
    
    args = parser.parse_args()

    inference(
        model_path=args.model,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        input_text=args.input_text,
        output_file=args.output_file,
        speaker_id=args.speaker_id
    )


if __name__ == "__main__":
    main()