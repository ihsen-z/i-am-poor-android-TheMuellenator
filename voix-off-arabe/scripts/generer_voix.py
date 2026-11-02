"""Génère le texte de test arabe avec plusieurs moteurs TTS open source (sherpa-onnx).

Moteurs couverts (tous hors-ligne, CPU, poids téléchargés depuis les releases
GitHub de sherpa-onnx) :
  - Piper / VITS      : voix ar_JO-kareem, ar_JO-SA_dii, ar_JO-SA_miro(_V2)
  - Kokoro / StyleTTS2: Nabra-82M (affiné sur l'arabe standard moderne)
  - Supertonic 3      : 10 voix (F1..F5, M1..M5), modèle multilingue 2026

Téléchargement des modèles (dans un dossier MODELS) :
  base=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models
  for n in vits-piper-ar_JO-kareem-medium vits-piper-ar_JO-SA_dii-high \
           vits-piper-ar_JO-SA_miro_V2-high nabra-82m-arabic-int8 \
           sherpa-onnx-supertonic-3-tts-int8-2026-05-11; do
    curl -L "$base/$n.tar.bz2" | tar -xj -C MODELS
  done

Usage :
  pip install "sherpa-onnx<2" soundfile numpy   # la 2.x n'embarque plus espeak
  python generer_voix.py --models MODELS --out ../audio/candidats

Chaque phrase est synthétisée séparément puis assemblée avec un silence,
comme un monteur le ferait pour une narration. Sortie en MP3 128 kb/s.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

ICI = Path(__file__).resolve().parent
TEXTES = {
    "tashkeel": ICI.parent / "texte" / "texte_tashkeel.txt",
    "brut": ICI.parent / "texte" / "texte_sans_tashkeel.txt",
}
PAUSE_S = 0.55
SUPERTONIC = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
# voice.bin est construit à partir des styles triés par nom : F1..F5 puis M1..M5
SUPERTONIC_VOIX = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


def piper(models: Path, nom: str) -> sherpa_onnx.OfflineTts:
    d = models / nom
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(next(d.glob("*.onnx"))),
                tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data"),
                noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
            ),
            num_threads=4, provider="cpu",
        ),
        max_num_sentences=1,
    ))


def nabra(models: Path) -> sherpa_onnx.OfflineTts:
    d = models / "nabra-82m-arabic-int8"
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(d / "model.int8.onnx"),
                voices=str(d / "voices.bin"),
                tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data"),
            ),
            num_threads=4, provider="cpu",
        ),
        max_num_sentences=1,
    ))


def supertonic(models: Path) -> sherpa_onnx.OfflineTts:
    d = models / SUPERTONIC
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=str(d / "duration_predictor.int8.onnx"),
                text_encoder=str(d / "text_encoder.int8.onnx"),
                vector_estimator=str(d / "vector_estimator.int8.onnx"),
                vocoder=str(d / "vocoder.int8.onnx"),
                tts_json=str(d / "tts.json"),
                unicode_indexer=str(d / "unicode_indexer.bin"),
                voice_style=str(d / "voice.bin"),
            ),
            num_threads=4, provider="cpu",
        ),
    ))


def synth(tts, phrases, sid=0, speed=1.0, supertonic_mode=False):
    morceaux, sr = [], None
    for phrase in phrases:
        if supertonic_mode:
            cfg = sherpa_onnx.GenerationConfig()
            cfg.sid = sid
            cfg.num_steps = 16  # plus d'étapes = meilleure qualité
            cfg.speed = speed
            cfg.extra["lang"] = "ar"
            audio = tts.generate(phrase, cfg)
        else:
            audio = tts.generate(phrase, sid=sid, speed=speed)
        sr = audio.sample_rate
        morceaux.append(np.asarray(audio.samples, dtype=np.float32))
        morceaux.append(np.zeros(int(PAUSE_S * sr), dtype=np.float32))
    signal = np.concatenate(morceaux)
    return 0.9 * signal / max(1e-6, float(np.abs(signal).max())), sr


def ecrire_mp3(signal, sr, mp3: Path):
    wav = mp3.with_suffix(".wav")
    sf.write(wav, signal, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-ar", "44100", "-ac", "1", "-b:a", "128k", str(mp3)], check=True)
    wav.unlink()
    print(f"ok {mp3.name}  ({len(signal) / sr:.1f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ICI.parent / "audio" / "candidats")
    ap.add_argument("--speed", type=float, default=0.95)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    phrases = {k: [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
               for k, p in TEXTES.items()}

    # Piper : le phonétiseur espeak a besoin du tashkeel -> texte vocalisé uniquement
    for nom in ["vits-piper-ar_JO-kareem-medium", "vits-piper-ar_JO-SA_dii-high",
                "vits-piper-ar_JO-SA_miro_V2-high", "vits-piper-ar_JO-SA_miro-high"]:
        if (args.models / nom).exists():
            tts = piper(args.models, nom)
            court = nom.replace("vits-piper-ar_JO-", "")
            ecrire_mp3(*synth(tts, phrases["tashkeel"], speed=args.speed),
                       args.out / f"piper_{court}__tashkeel.mp3")

    if (args.models / "nabra-82m-arabic-int8").exists():
        tts = nabra(args.models)
        for variante, ph in phrases.items():
            ecrire_mp3(*synth(tts, ph, speed=args.speed),
                       args.out / f"nabra82m__{variante}.mp3")

    if (args.models / SUPERTONIC).exists():
        tts = supertonic(args.models)
        for sid, voix in enumerate(SUPERTONIC_VOIX):
            for variante, ph in phrases.items():
                ecrire_mp3(*synth(tts, ph, sid=sid, speed=args.speed, supertonic_mode=True),
                           args.out / f"supertonic3_{voix}__{variante}.mp3")


if __name__ == "__main__":
    main()
