"""Mesure la hauteur (F0) et la variation mélodique de chaque prise.

F0 médiane : indique le registre (≈100-130 Hz voix masculine, ≈180-250 Hz voix féminine).
Variation : écart-type de la F0 en demi-tons. Autour de 2 = lecture plate ;
au-delà de 3 = intonation plus vivante.

Usage :
  pip install librosa
  python analyser_melodie.py ../audio/*.mp3
"""

import sys

import librosa
import numpy as np

for chemin in sys.argv[1:]:
    y, sr = librosa.load(chemin, sr=16000, duration=25)
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    demi_tons = 12 * np.log2(f0 / np.median(f0))
    print(f"{chemin}\tF0 {np.median(f0):.0f} Hz\tvariation ±{np.std(demi_tons):.1f} demi-tons")
