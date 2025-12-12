# Emotion VAD Values Documentation

## 1. Introduction

This document describes the extraction of Valence-Arousal-Dominance (VAD) values for 32 emotion
categories used in the Emotion-aware LLM Scheduling system. These values are derived from the
NRC-VAD-Lexicon.

## 2. Data Source

The VAD values are extracted from the **NRC Valence, Arousal, and Dominance Lexicon (Version 2.1)**.

This lexicon provides real-valued scores of valence, arousal, and dominance for over 20,000 English
words. The scores were obtained through Best-Worst Scaling annotations from native English speakers.

### Dimensions

- **Valence**: The pleasantness of a stimulus (negative to positive)
- **Arousal**: The intensity of emotion (calm to excited)
- **Dominance**: The degree of control exerted (submissive to dominant)

### Value Range

All values in the lexicon are in the range **[-1, 1]**:
- -1: Lowest (most negative/calm/submissive)
- 0: Neutral
- +1: Highest (most positive/excited/dominant)

## 3. Methodology

### Extraction Process

1. Load the NRC-VAD-Lexicon TSV file
2. For each target emotion, look up the corresponding term in the lexicon
3. Extract valence, arousal, and dominance values
4. Classify each emotion into Russell's circumplex quadrant based on valence and arousal

### Russell Quadrant Classification

Based on the 2D circumplex model of affect (Russell, 1980), emotions are classified into four quadrants:

| Quadrant | Valence | Arousal | Description |
|----------|---------|---------|-------------|
| excited | ≥ 0 | ≥ 0 | High arousal, positive valence |
| calm | ≥ 0 | < 0 | Low arousal, positive valence |
| panic | < 0 | ≥ 0 | High arousal, negative valence |
| depression | < 0 | < 0 | Low arousal, negative valence |

### Substitutions

The following substitutions were made for terms not found in the lexicon:

| Target Emotion | Lexicon Term Used | Reason |
|---------------|-------------------|--------|
| anticipating | anticipation | Verb form not in lexicon; noun form used |

### Quadrant Distribution

| Quadrant | Count | Emotions |
|----------|-------|----------|
| excited | 9 | excited, joyful, proud, trusting, impressed, surprised, anticipating |
| calm | 8 | hopeful, faithful, grateful, confident, content, sentimental, prepared, caring |
| panic | 12 | terrified, afraid, anxious, angry, furious, annoyed, disgusted, jealous, embarrassed, apprehensive, guilty, ashamed, devastated |
| depression | 4 | nostalgic, sad, lonely, disappointed |

## 4. Reproducibility

To regenerate this data, run the extraction script:

```bash
cd /path/to/Emotion-aware-LLM-scheduling/tools
python extract_emotion_vad.py
```

## 5. References

Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance
for 20,000 English Words. In *Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pages 174-184, Melbourne, Australia.
Association for Computational Linguistics.

**BibTeX:**
```bibtex
@inproceedings{mohammad2018obtaining,
  title={Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words},
  author={Mohammad, Saif M.},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={174--184},
  year={2018},
  address={Melbourne, Australia},
  publisher={Association for Computational Linguistics}
}
```

**Lexicon Download:**
https://saifmohammad.com/WebPages/nrc-vad.html
