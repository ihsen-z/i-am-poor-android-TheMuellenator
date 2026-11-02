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

Deux précautions de mesure :
- L'audio est découpé en blocs de 18 s au plus, **uniquement dans les pauses d'au moins 0,3 s**.
  Une première version coupait toutes les 20 s, parfois au milieu d'un mot, ce qui faisait
  disparaître des mots de la transcription et faussait les scores dans les deux sens.
- Le modèle ASR écrit les nombres en chiffres (« ثلاثة » devient « 3 »). Les mots-nombres sont donc
  exclus de la référence : tous les moteurs les prononcent, il n'y a pas de raison de les pénaliser.

**Un seul tirage ne suffit pas pour classer les voix Supertonic.** Le modèle est stochastique :
sur une même phrase, le taux d'erreur varie de 0 à 8 % d'un tirage à l'autre. Les écarts de 1 à
2 points entre ses voix dans la série 1 relèvent donc en partie du hasard. Mieux vaut choisir la
voix à l'oreille (timbre, registre), puis laisser la série 2 corriger la prononciation.

## Résultats (série 1)

Les 8 prises retenues (CER recalculé après correction de la référence, voir `scripts/evaluer_asr.py`) :

| Prise | Modèle | Voix | Genre | CER | Mélodie (±) | Débit | Licence |
|---|---|---|---|---|---|---|---|
| [01](audio/01_supertonic3_M4.mp3) | Supertonic 3 | voix M4 · texte brut | masculine | 1.5 % | 2.9 demi-tons | 90 mots/min | OpenRAIL-M |
| [02](audio/02_supertonic3_M1.mp3) | Supertonic 3 | voix M1 · texte brut | masculine | 4.6 % | 4.2 demi-tons | 85 mots/min | OpenRAIL-M |
| [03](audio/03_supertonic3_F2.mp3) | Supertonic 3 | voix F2 · texte brut | féminine | 2.5 % | 3.1 demi-tons | 90 mots/min | OpenRAIL-M |
| [04](audio/04_piper_kareem.mp3) | Piper (VITS) | kareem · texte vocalisé | masculine | 4.1 % | 1.9 demi-tons | 78 mots/min | licence à vérifier |
| [05](audio/05_supertonic3_F1.mp3) | Supertonic 3 | voix F1 · texte brut | féminine | 2.3 % | 3.7 demi-tons | 85 mots/min | OpenRAIL-M |
| [06](audio/06_piper_SA_dii.mp3) | Piper (VITS) | SA_dii · texte vocalisé | féminine | 6.9 % | 2.1 demi-tons | 99 mots/min | CC BY-NC-SA · non commercial |
| [07](audio/07_piper_SA_miro_V2.mp3) | Piper (VITS) | SA_miro V2 · texte vocalisé | masculine | 7.9 % | 2.0 demi-tons | 105 mots/min | CC BY-NC-SA · non commercial |
| [08](audio/08_nabra82m.mp3) | Nabra-82M (Kokoro) | af_msa · texte vocalisé | féminine | 9.1 % | 2.6 demi-tons | 109 mots/min | Apache-2.0 |

« Mélodie » est l'écart-type de la hauteur de la voix (F0), en demi-tons, sur les 25 premières secondes. Une voix plate et monotone tourne autour de 2 ; une lecture plus vivante dépasse 3.

Les 18 autres candidats (Supertonic F3, F4, F5, M2, M3, M5, les versions vocalisées de Supertonic, Nabra sur texte brut, Piper SA_miro v1) sont dans `audio/autres_candidats/`. Leurs scores sont dans `resultats_asr.tsv`. Deux voix féminines y méritent une écoute : **F3** (0,5 %) et **F5** (1,8 %), ainsi que la voix masculine **M5** (1,0 %).

## Constats

- **Supertonic 3 lit mieux le texte brut que le texte vocalisé** : sur ses 10 voix, le CER moyen
  est de 2.0 % sans tashkeel et de 6.1 % avec. Le tashkeel le ralentit et le fait trébucher. C'est un avantage
  pratique : tu peux coller ton texte tel quel.
- **Piper et Nabra ont besoin du tashkeel.** Leur phonétiseur (espeak-ng) devine mal les voyelles
  d'un texte non vocalisé. En production, il faudrait ajouter un vocaliseur automatique
  (CATT, Mishkal…) avant la synthèse.
- Les voix Piper `SA_dii` et `SA_miro` sont entraînées sur des données synthétiques, sous licence
  **CC BY-NC-SA** (usage non commercial seulement).

## Série 2 : Supertonic 3 retravaillé (M4, M1 et une voix hybride)

Le modèle n'est pas réentraîné. On agit sur tout ce qui l'entoure
(`scripts/ameliorer_supertonic.py`) :

1. **Tashkeel ciblé.** Le tashkeel complet fait trébucher Supertonic, mais certains mots
   ambigus ont besoin d'aide (تَضَاءَلَتْ، بِخُطًى وَئِيدَةٍ، عِنْدَئِذٍ، الظِّلال…). On ne vocalise
   que ceux-là. Pour chaque segment, on essaie le texte brut et le texte ciblé.
2. **Plusieurs tirages et sélection automatique.** Chaque génération est différente : sur
   une même phrase, le taux d'erreur varie de 0 à 8 % d'un tirage à l'autre. On en produit
   donc 8 par segment (2 variantes de texte × 4 tirages) et on garde le mieux prononcé.
   Parmi les ex aequo, on prend le plus expressif, sans dépasser d'un demi-ton la variation
   médiane des tirages, pour écarter les intonations aberrantes. Tous les tirages sont
   conservés : `--reselection` refait le choix et le montage sans rien régénérer.
3. **Voix hybride M1×M4.** Le fichier `voice.bin` stocke deux vecteurs de style par voix :
   l'un règle le timbre et l'intonation, l'autre le rythme. La moyenne de M1 (intonation
   vivante) et de M4 (clarté) donne une voix intermédiaire.
4. **Montage de narrateur.** Silences parasites retirés, pauses dosées selon le sens
   (1 s de suspense après la question, 0,4 s avant la citation), citation finale générée
   à part et plus lentement, puis mastering voix off : passe-haut à 70 Hz, de-esser,
   compresseur 3:1 et normalisation à −16 LUFS en deux passes (norme des plateformes vidéo).

Résultats sur le texte entier (`resultats_asr_serie2.tsv`, détail par segment dans `amelioration_rapport.json`) :

| Voix | CER avant | CER après | Mélodie avant | Mélodie après | Durée après |
|---|---|---|---|---|---|
| [Supertonic 3 · M4](audio/ameliore/M4_ameliore.mp3) | 1.5 % | **0.8 %** | ±2.9 | ±3.1 | 57 s |
| [Supertonic 3 · M1](audio/ameliore/M1_ameliore.mp3) | 4.6 % | **1.0 %** | ±4.2 | ±4.2 | 60 s |
| [Hybride M1×M4](audio/ameliore/hybride_M1xM4_ameliore.mp3) | — | **0.8 %** | — | ±3.8 | 58 s |

**Ce qui a marché, et ce qui n'a pas marché** (sur les 144 tirages, détail dans
`amelioration_rapport.json`) :

- **La sélection parmi plusieurs tirages fait l'essentiel du gain.** Un tirage pris au hasard
  fait en moyenne 2,7 % d'erreurs par segment, contre 0,06 % pour les tirages retenus :
  17 segments sur 18 sont à 0 %.
- **Le tashkeel ciblé ne sert pas avec Supertonic.** Même partiel, il dégrade la lecture :
  4,1 % d'erreurs en moyenne, contre 1,3 % sur le texte brut. Il n'a été retenu que pour
  2 segments sur 18. Conclusion pratique : avec Supertonic, écris sans tashkeel.
- **La voix hybride tient ses promesses sur le papier** : aussi claire que M4 (0,8 %), avec une
  mélodie plus vivante (±3,8 contre ±3,1). Reste à vérifier à l'oreille qu'elle sonne naturelle.

### Aller plus loin

| Levier | Ce que ça corrige | Ce qu'il faut |
|---|---|---|
| Lexique de prononciation | Les mêmes mots ratés d'une vidéo à l'autre : on note leur forme vocalisée une fois, le script l'applique ensuite automatiquement | rien, faisable ici |
| Clonage d'un narrateur humain (Habibi-TTS, XTTS-v2, Chatterbox) | Le timbre et l'intonation : 10 à 30 s d'un vrai narrateur en fusha suffisent | accès à HuggingFace, et l'accord du narrateur si la voix est celle de quelqu'un |
| Affinage de Piper sur 1 à 2 h de narration humaine (corpus ClArTTS, par exemple) | Une voix entièrement à toi, sans licence restrictive si les données sont libres | une carte graphique pendant quelques heures, et HuggingFace |
| Affinage de Supertonic 3 | — | impossible : le code d'entraînement n'est pas publié (Supertone propose un service payant) |

## Modèles non testés (réseau bloqué)

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

Pour la série 2 (il faut aussi le modèle de reconnaissance vocale, ≈ 1 Go) :

```bash
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05.tar.bz2 | tar -xj
```

```bash
pip install librosa
```

```bash
cd scripts && python ameliorer_supertonic.py --models ../models --asr ../sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05 --work /tmp/travail
```
