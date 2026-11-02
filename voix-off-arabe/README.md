# Casting voix off arabe (fusha) pour vidéos de narration

Objectif : trouver la voix de synthèse arabe **la plus proche d'un narrateur humain**,
gratuite, pour des vidéos de narration.

- `texte/` : le texte de test (5 phrases en arabe classique), avec et sans tashkeel
- `audio/` : les 8 prises retenues pour l'écoute (série 1)
- `ecoute.html` : la page d'écoute (lecteurs, écoute à l'aveugle, notes de 1 à 5)
- `resultats_asr.tsv` : le score de prononciation de chacun des 26 candidats générés
- `scripts/generer_voix.py` : génère tous les candidats (sherpa-onnx, CPU, hors ligne)
- `scripts/evaluer_asr.py` : mesure la clarté de prononciation par reconnaissance vocale
- `scripts/analyser_melodie.py` : mesure le registre (F0) et la variation mélodique de chaque prise

## Le texte de test

Cinq phrases d'un récit d'explorateur, choisies pour piéger les moteurs :

1. Un nombre en toutes lettres (ألف وثمانمئة وسبعة وأربعين), un hamzat al-wasl après une pause (اطْمَأَنَّ), et ض suivi d'une hamza (ضَوْضَاء، تَضَاءَلَتْ).
2. Une phrase longue à trois respirations, un tanwîn sur alif maqsûra (خُطًى), une hamza sur yâ' (وَئِيدَة), une shadda avec tanwîn (مُلِحٍّ) et le ظ emphatique.
3. Une intonation interrogative, une hamza sur wâw (المُؤَرِّخِين), le groupe consonantique de اسْتِقْصَاءَات et une madda (آلَاف).
4. Des emphatiques en série (الصَّخْرَة الضَّخْمَة), l'assimilation n→m (عَنْبَر), un tanwîn suivi d'un hamzat al-wasl (عَتِيقٍ امْتَزَجَتْ), ainsi que ع et غ.
5. Une exclamation au discours direct, le lâm d'appel (يَا لَلْعَجَب), la rencontre de deux sukûn (صَدَقَتِ الأَسَاطِير) et la shadda finale (عَيْنَيَّ).

## Méthode

Tous les modèles tournent **hors ligne sur CPU** avec [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
(version 1.13 : la 2.x n'embarque plus le phonétiseur espeak dont Piper et Kokoro ont besoin).
Chaque phrase est synthétisée séparément, puis les phrases sont assemblées avec 0,55 s de silence,
à une vitesse de 0,95. Les fichiers sont exportés en MP3 mono à 128 kb/s.

**Score objectif (CER)** : le modèle de reconnaissance vocale Meta Omnilingual ASR 1B retranscrit
chaque audio, et on compare sa transcription au texte d'origine, sans tashkeel. Un CER bas signifie
une prononciation claire et correcte. Ce score ne dit rien du naturel (intonation, émotion,
respiration) : pour cela, il faut écouter. Le modèle ASR fait lui-même des erreurs (il confond par
exemple ظ et ذ), donc le chiffre sert à comparer les voix entre elles, pas comme une valeur absolue.

## Résultats (série 1)

Les 8 prises retenues, classées par clarté de prononciation (CER) :

| Prise | Modèle | Voix | Genre | CER | Mélodie (±) | Débit | Licence |
|---|---|---|---|---|---|---|---|
| [01](audio/01_supertonic3_M4.mp3) | Supertonic 3 | voix M4 · texte brut | masculine | 2.8 % | 2.9 demi-tons | 90 mots/min | OpenRAIL-M |
| [02](audio/02_supertonic3_M1.mp3) | Supertonic 3 | voix M1 · texte brut | masculine | 3.5 % | 4.2 demi-tons | 85 mots/min | OpenRAIL-M |
| [03](audio/03_supertonic3_F2.mp3) | Supertonic 3 | voix F2 · texte brut | féminine | 4.0 % | 3.1 demi-tons | 90 mots/min | OpenRAIL-M |
| [04](audio/04_piper_kareem.mp3) | Piper (VITS) | kareem · texte vocalisé | masculine | 4.3 % | 1.9 demi-tons | 78 mots/min | licence à vérifier |
| [05](audio/05_supertonic3_F1.mp3) | Supertonic 3 | voix F1 · texte brut | féminine | 4.3 % | 3.7 demi-tons | 85 mots/min | OpenRAIL-M |
| [06](audio/06_piper_SA_dii.mp3) | Piper (VITS) | SA_dii · texte vocalisé | féminine | 7.3 % | 2.1 demi-tons | 99 mots/min | CC BY-NC-SA · non commercial |
| [07](audio/07_piper_SA_miro_V2.mp3) | Piper (VITS) | SA_miro V2 · texte vocalisé | masculine | 9.0 % | 2.0 demi-tons | 105 mots/min | CC BY-NC-SA · non commercial |
| [08](audio/08_nabra82m.mp3) | Nabra-82M (Kokoro) | af_msa · texte vocalisé | féminine | 13.5 % | 2.6 demi-tons | 109 mots/min | Apache-2.0 |

« Mélodie » est l'écart-type de la hauteur de la voix (F0), en demi-tons, sur les 25 premières secondes. Une voix plate et monotone tourne autour de 2 ; une lecture plus vivante dépasse 3.

Les 18 autres candidats (Supertonic F3, F4, F5, M2, M3, M5, les versions vocalisées de Supertonic, Nabra sur texte brut, Piper SA_miro v1) sont dans `audio/autres_candidats/`. Leurs scores sont dans `resultats_asr.tsv`.

## Constats

- **Supertonic 3 lit mieux le texte brut que le texte vocalisé** : son CER passe d'environ 9 % sans
  tashkeel à plus de 20 % avec. Le tashkeel le ralentit et le fait trébucher. C'est un avantage
  pratique : tu peux coller ton texte tel quel.
- **Piper et Nabra ont besoin du tashkeel.** Leur phonétiseur (espeak-ng) devine mal les voyelles
  d'un texte non vocalisé. En production, il faudrait ajouter un vocaliseur automatique
  (CATT, Mishkal…) avant la synthèse.
- Les voix Piper `SA_dii` et `SA_miro` sont entraînées sur des données synthétiques, sous licence
  **CC BY-NC-SA** (usage non commercial seulement).

## Série 2 (bloquée par le réseau de l'environnement cloud)

L'environnement cloud de cette session bloque `huggingface.co`, `speech.platform.bing.com`
(Edge TTS), `translate.google.com` et `api.elevenlabs.io`. Les candidats suivants, souvent plus
naturels, n'ont donc pas pu être testés :

| Modèle | Type | Licence | Remarque |
|---|---|---|---|
| Microsoft Edge Neural (`edge-tts`) | service gratuit sans clé | conditions Microsoft | voix saoudiennes, égyptiennes, émiraties… souvent la référence gratuite |
| Habibi-TTS (2026, F5-TTS) | open source, clonage | CC BY-NC-SA | arabe standard et dialectes |
| Chatterbox Multilingual (Resemble AI) | open source, clonage | MIT | voix expressive |
| XTTS-v2 (Coqui) | open source, clonage | non commerciale (CPML) | |
| Gemini TTS (Google AI Studio) | API, palier gratuit | conditions Google | style réglable par une consigne |
| ElevenLabs | API, offre gratuite limitée | conditions ElevenLabs | étalon haut de gamme |

Pour les tester, il faut autoriser ces domaines dans les réglages réseau de l'environnement
(<https://code.claude.com/docs/en/cloud-environments#network-access>), ou bien lancer les
scripts sur ton propre ordinateur.

## Reproduire

```bash
pip install "sherpa-onnx<2" soundfile numpy
```

```bash
mkdir -p models && base=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models
for n in vits-piper-ar_JO-kareem-medium vits-piper-ar_JO-SA_dii-high vits-piper-ar_JO-SA_miro_V2-high vits-piper-ar_JO-SA_miro-high nabra-82m-arabic-int8 sherpa-onnx-supertonic-3-tts-int8-2026-05-11; do curl -L "$base/$n.tar.bz2" | tar -xj -C models; done
```

```bash
python scripts/generer_voix.py --models models --out audio/candidats
```
