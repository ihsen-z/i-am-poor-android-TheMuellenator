"""Améliore Supertonic 3 (voix M4, M1 et un hybride M1×M4) sans réentraîner le modèle.

Leviers :
1. Texte : on ajoute le tashkeel seulement sur les mots que le modèle rate (le tashkeel
   complet le fait trébucher), et on sépare la citation finale du récit.
2. Voix hybride : moyenne des deux vecteurs de style de M1 et de M4 (le premier règle le
   timbre et l'intonation, le second le rythme).
3. Plusieurs tirages : le modèle est stochastique, chaque génération est différente.
   On produit N tirages par segment et par variante de texte, puis on garde
   automatiquement le mieux prononcé (CER mesuré par reconnaissance vocale). Parmi les
   ex aequo, on prend le plus expressif, sans dépasser de plus d'un demi-ton la
   variation médiane des tirages, pour écarter les intonations aberrantes.
   Tous les tirages sont conservés : --reselection refait le choix sans régénérer.
4. Montage : silences de début et de fin retirés, pauses dosées selon le sens, vitesse
   adaptée à chaque segment, puis mastering voix off (passe-haut, de-esser, compresseur,
   normalisation à -16 LUFS en deux passes).

Usage :
  pip install "sherpa-onnx<2" soundfile numpy librosa
  python ameliorer_supertonic.py --models MODELS --asr DOSSIER_ASR --work /tmp/travail
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx
import soundfile as sf

from evaluer_asr import distance, normaliser, sans_nombres

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
SUPERTONIC = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
SID_ORIGINE = {"M1": 5, "M4": 8}  # voice.bin d'origine : F1..F5 (0-4) puis M1..M5 (5-9)
VOIX = ["M4", "M1", "hybride_M1xM4"]
NUM_STEPS = 24
DEBUT_S = 0.30
FIN_S = 0.60

# brut : texte tel quel ; cible : tashkeel uniquement sur les mots difficiles.
# vitesse : < 1 ralentit ; pause : silence après le segment, en secondes.
SEGMENTS = [
    dict(id="s1", vitesse=0.95, pause=0.70,
         brut="في ليلة شتوية قارسة من عام ألف وثمانمئة وسبعة وأربعين، اطمأن المستكشف العجوز إلى أن ضوضاء العاصفة قد تضاءلت أخيرا.",
         cible="في ليلة شتوية قارسة من عام ألف وثمانمئة وسبعة وأربعين، اطْمَأَنَّ المستكشف العجوز إلى أنّ ضوضاء العاصفة قد تَضَاءَلَتْ أخيرًا."),
    dict(id="s2", vitesse=0.97, pause=0.80,
         brut="تقدم بخطى وئيدة نحو المقبرة المنسية، وقلبه يخفق بين رجاء ملح وخوف مستتر، فيما كانت الظلال المتراقصة تتشبث بالجدران المتصدعة.",
         cible="تقدّم بِخُطًى وَئِيدَةٍ نحو المقبرة المنسية، وقلبه يخفق بين رجاءٍ مُلِحٍّ وخوفٍ مُسْتَتِرٍ، فيما كانت الظِّلالُ المتراقصة تتشبّث بالجدران المتصدّعة."),
    dict(id="s3", vitesse=0.92, pause=1.00,
         brut="أي سر دفين ظل مستعصيا على استقصاءات المؤرخين وتخميناتهم طوال ثلاثة آلاف عام؟",
         cible="أيُّ سرٍّ دفينٍ ظلَّ مستعصيًا على استقصاءاتِ المُؤَرِّخِينَ وتخميناتِهم طَوَالَ ثلاثةِ آلافِ عام؟"),
    dict(id="s4", vitesse=0.95, pause=0.75,
         brut="وحين أزاح الصخرة الضخمة بشق الأنفس، تصاعد غبار كثيف، وانبعثت رائحة عنبر عتيق امتزجت بعبق البخور المتحلل.",
         cible="وحين أزاح الصَّخْرَةَ الضَّخْمَةَ بِشِقِّ الأَنْفُسِ، تصاعد غبارٌ كثيف، وانبعثت رائحةُ عَنْبَرٍ عتيقٍ امتزجت بعبق البَخُّورِ المتحلِّل."),
    dict(id="s5a", vitesse=0.95, pause=0.40,
         brut="عندئذ لم يتمالك نفسه، فهمس بصوت مرتجف:",
         cible="عِنْدَئِذٍ لم يتمالكْ نفسَه، فهمس بصوتٍ مُرْتَجِف:"),
    dict(id="s5b", vitesse=0.90, pause=0.0,
         brut="يا للعجب! لقد صدقت الأساطير، وها هي ذي الحقيقة تتجلى أمام عيني.",
         cible="يا لَلْعَجَبِ! لقد صدقتِ الأساطير، وها هي ذي الحقيقةُ تتجلّى أمام عَيْنَيَّ."),
]


def ecrire_voice_bin(source: Path, cible: Path) -> None:
    """Crée un voice.bin à 3 voix : M4, M1 et leur moyenne."""
    brut = source.read_bytes()
    ttl_dims = np.frombuffer(brut[:24], dtype=np.int64)
    dp_dims = np.frombuffer(brut[24:48], dtype=np.int64)
    n_ttl, n_dp = int(np.prod(ttl_dims)), int(np.prod(dp_dims))
    ttl = np.frombuffer(brut[48:48 + 4 * n_ttl], dtype=np.float32).reshape(ttl_dims)
    dp = np.frombuffer(brut[48 + 4 * n_ttl:48 + 4 * (n_ttl + n_dp)], dtype=np.float32).reshape(dp_dims)
    m4, m1 = SID_ORIGINE["M4"], SID_ORIGINE["M1"]
    ttl_new = np.stack([ttl[m4], ttl[m1], (ttl[m1] + ttl[m4]) / 2])
    dp_new = np.stack([dp[m4], dp[m1], (dp[m1] + dp[m4]) / 2])
    with open(cible, "wb") as f:
        f.write(np.array(ttl_new.shape, dtype=np.int64).tobytes())
        f.write(np.array(dp_new.shape, dtype=np.int64).tobytes())
        f.write(ttl_new.astype(np.float32).ravel().tobytes())
        f.write(dp_new.astype(np.float32).ravel().tobytes())


def charger_tts(models: Path, voice_bin: Path) -> sherpa_onnx.OfflineTts:
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
                voice_style=str(voice_bin),
            ),
            num_threads=4, provider="cpu",
        ),
    ))


def cer(ref: str, audio16k: np.ndarray, rec) -> float:
    s = rec.create_stream()
    s.accept_waveform(16000, audio16k)
    rec.decode_stream(s)
    hyp = normaliser(s.result.text).replace(" ", "")
    r = normaliser(sans_nombres(ref)).replace(" ", "")
    return 100 * distance(r, hyp) / len(r)


def variation(audio16k: np.ndarray) -> float:
    f0, _, _ = librosa.pyin(audio16k, fmin=65, fmax=300, sr=16000)
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 10:
        return 0.0
    return float(np.std(12 * np.log2(f0 / np.median(f0))))


def rogner(x: np.ndarray, sr: int) -> np.ndarray:
    """Retire les silences de début et de fin, avec un fondu de 10 ms."""
    _, (a, b) = librosa.effects.trim(x, top_db=40)
    a, b = max(0, a - int(0.02 * sr)), min(len(x), b + int(0.04 * sr))
    y = x[a:b].copy()
    n = int(0.01 * sr)
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def masteriser(entree: Path, sortie: Path) -> None:
    chaine = ("highpass=f=70,deesser=i=0.35,"
              "acompressor=threshold=0.1:ratio=3:attack=5:release=150:makeup=2")
    mesure = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(entree), "-af",
         chaine + ",loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True).stderr
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", mesure).group(0))
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
          f":offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(entree), "-af",
                    chaine + "," + ln, "-ar", "44100", "-ac", "1", "-b:a", "128k", str(sortie)],
                   check=True)


def choisir(candidats: list) -> dict:
    """Prononciation d'abord ; parmi les ex aequo, le plus expressif sans être aberrant."""
    meilleur = min(c["cer"] for c in candidats)
    ex_aequo = [c for c in candidats if c["cer"] <= meilleur + 0.5]
    plafond = float(np.median([c["melodie"] for c in candidats])) + 0.5
    raisonnables = [c for c in ex_aequo if c["melodie"] <= plafond]
    if raisonnables:
        return max(raisonnables, key=lambda c: c["melodie"])
    return min(ex_aequo, key=lambda c: c["melodie"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--asr", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tirages", type=int, default=4)
    ap.add_argument("--reselection", action="store_true",
                    help="refait le choix et le montage à partir des tirages déjà générés")
    ap.add_argument("--out", type=Path, default=RACINE / "audio" / "ameliore")
    args = ap.parse_args()
    dossier_tirages = args.work / "tirages"
    dossier_tirages.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    journal = args.work / "tirages.json"

    if args.reselection:
        tous = json.loads(journal.read_text(encoding="utf-8"))
    else:
        voice_bin = args.work / "voice_M4_M1_hybride.bin"
        ecrire_voice_bin(args.models / SUPERTONIC / "voice.bin", voice_bin)
        tts = charger_tts(args.models, voice_bin)
        rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=str(args.asr / "model.int8.onnx"), tokens=str(args.asr / "tokens.txt"),
            num_threads=4)
        tous = {}
        for sid, voix in enumerate(VOIX):
            tous[voix] = {}
            for seg in SEGMENTS:
                candidats = []
                for variante in ("brut", "cible"):
                    for k in range(args.tirages):
                        cfg = sherpa_onnx.GenerationConfig()
                        cfg.sid, cfg.num_steps, cfg.speed = sid, NUM_STEPS, seg["vitesse"]
                        cfg.extra["lang"] = "ar"
                        a = tts.generate(seg[variante], cfg)
                        x = np.asarray(a.samples, dtype=np.float32)
                        wav = dossier_tirages / f"{voix}_{seg['id']}_{variante}_{k}.wav"
                        sf.write(wav, x, a.sample_rate)
                        x16 = librosa.resample(x, orig_sr=a.sample_rate, target_sr=16000)
                        candidats.append(dict(variante=variante, tirage=k, wav=str(wav),
                                              cer=round(cer(seg["brut"], x16, rec), 1),
                                              melodie=round(variation(x16), 2)))
                tous[voix][seg["id"]] = candidats
                print(f"{voix} {seg['id']}: CER des {len(candidats)} tirages : "
                      f"{sorted(c['cer'] for c in candidats)}", flush=True)
                journal.write_text(json.dumps(tous, ensure_ascii=False), encoding="utf-8")

    rapport = {}
    for voix in VOIX:
        morceaux, sr = [], None
        rapport[voix] = []
        for seg in SEGMENTS:
            candidats = tous[voix][seg["id"]]
            choix = choisir(candidats)
            x, sr = sf.read(choix["wav"], dtype="float32")
            print(f"{voix} {seg['id']}: retenu {choix['variante']} #{choix['tirage']} "
                  f"CER {choix['cer']} mélodie {choix['melodie']}", flush=True)
            rapport[voix].append(dict(
                segment=seg["id"], variante=choix["variante"], tirage=choix["tirage"],
                cer=choix["cer"], melodie=choix["melodie"],
                tirages=[{k: c[k] for k in ("variante", "tirage", "cer", "melodie")} for c in candidats]))
            morceaux.append(rogner(x, sr))
            if seg["pause"]:
                morceaux.append(np.zeros(int(seg["pause"] * sr), dtype=np.float32))
        signal = np.concatenate([np.zeros(int(DEBUT_S * sr), np.float32), *morceaux,
                                 np.zeros(int(FIN_S * sr), np.float32)])
        signal = 0.9 * signal / float(np.abs(signal).max())
        brut_wav = args.work / f"{voix}.wav"
        sf.write(brut_wav, signal, sr)
        masteriser(brut_wav, args.out / f"{voix}_ameliore.mp3")
        print(f"ok {voix}_ameliore.mp3 ({len(signal) / sr:.1f} s)", flush=True)

    (RACINE / "amelioration_rapport.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
