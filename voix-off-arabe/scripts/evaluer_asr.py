"""Mesure objective de l'intelligibilité : on fait transcrire chaque audio par un
modèle de reconnaissance vocale (Meta Omnilingual ASR 1B, via sherpa-onnx) et on
compare au texte de référence (CER = taux d'erreur par caractère, sans tashkeel).

Un CER bas = prononciation claire et correcte. Cela ne mesure PAS le naturel
(intonation, émotion) : pour ça, il faut écouter.

Modèle :
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05.tar.bz2

Usage :
  python evaluer_asr.py --asr DOSSIER_ASR --audio ../audio/candidats > ../resultats_asr.tsv
  python evaluer_asr.py --rescore ancien.tsv > ../resultats_asr.tsv   # recalcul sans ré-écouter
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx

ICI = Path(__file__).resolve().parent
REF = ICI.parent / "texte" / "texte_sans_tashkeel.txt"
# Omnilingual ASR écrit les nombres en chiffres (« ثلاثة » devient « 3 »), et les chiffres
# sont retirés à la normalisation. Les moteurs, eux, prononcent bien ces mots (vérifié :
# la phrase 1 dure 3 à 4 s de plus avec le nombre). On exclut donc ces mots de la
# référence pour ne pas pénaliser toutes les voix à tort.
IGNORER = ["ألف وثمانمئة وسبعة وأربعين", "ثلاثة"]


def normaliser(t: str) -> str:
    t = re.sub("[ً-ْٰـ]", "", t)  # tashkeel + tatweel
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^ء-ي ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def blocs_sur_silences(samples: np.ndarray, sr: int = 16000, max_s: float = 18.0,
                       pause_min_s: float = 0.3):
    """Découpe l'audio en blocs de 18 s au plus, en ne coupant que dans les vraies pauses.

    Une coupe à durée fixe, ou dans une micro-pause entre deux mots, peut tomber au
    milieu d'une phrase : le modèle perd alors quelques mots à la jonction, ce qui
    fausse le score. On ne coupe donc que dans les silences d'au moins 0,3 s.
    """
    import librosa
    brutes = librosa.effects.split(samples, top_db=35, frame_length=1024, hop_length=256)
    zones = []
    for a, b in brutes:
        if zones and (a - zones[-1][1]) / sr < pause_min_s:
            zones[-1][1] = b
        else:
            zones.append([a, b])
    blocs, debut, fin = [], None, None
    for a, b in zones:
        if debut is None:
            debut, fin = a, b
        elif (b - debut) / sr > max_s:
            blocs.append((debut, fin))
            debut, fin = a, b
        else:
            fin = b
    if debut is not None:
        blocs.append((debut, fin))
    marge = int(0.1 * sr)
    return [samples[max(0, a - marge):b + marge] for a, b in blocs]


def charger_audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", "16000",
                          "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def scores(ref: str, hyp: str):
    cer = 100 * distance(ref.replace(" ", ""), hyp.replace(" ", "")) / len(ref.replace(" ", ""))
    wer = 100 * distance(ref.split(), hyp.split()) / len(ref.split())
    return cer, wer


def sans_nombres(texte: str) -> str:
    for mot in IGNORER:
        texte = texte.replace(mot, " ")
    return texte


def reference() -> str:
    return normaliser(sans_nombres(REF.read_text(encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--rescore", type=Path, nargs="+",
                    help="recalcule les scores à partir de fichiers TSV déjà produits")
    args = ap.parse_args()
    ref = reference()
    print("fichier\tCER_%\tWER_%\tduree_s\ttranscription")

    if args.rescore:
        for tsv in args.rescore:
            for ligne in tsv.read_text(encoding="utf-8").splitlines()[1:]:
                nom, _, _, duree, hyp = ligne.split("\t")
                cer, wer = scores(ref, hyp)
                print(f"{nom}\t{cer:.1f}\t{wer:.1f}\t{duree}\t{hyp}")
        return

    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(args.asr / "model.int8.onnx"), tokens=str(args.asr / "tokens.txt"),
        num_threads=4)

    for mp3 in sorted(args.audio.glob("*.mp3")):
        samples = charger_audio(mp3)
        textes = []
        for bloc in blocs_sur_silences(samples):
            s = rec.create_stream()
            s.accept_waveform(16000, bloc)
            rec.decode_stream(s)
            textes.append(s.result.text)
        hyp = normaliser(" ".join(textes))
        cer, wer = scores(ref, hyp)
        print(f"{mp3.stem}\t{cer:.1f}\t{wer:.1f}\t{len(samples) / 16000:.1f}\t{hyp}", flush=True)


if __name__ == "__main__":
    main()
