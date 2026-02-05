# Dataset Information

## Overview
Experiments use three standard Indonesian NER datasets: **IDNer2k**, **NER-UI**, and **NER-UGM**. This folder may contain samples or you can place the full datasets here for full reproduction (see paper for exact splits and entity counts).

## Datasets (paper references)

### IDNer2k
- **Reference:** Khairunnisa et al. [36] — "Towards a Standardized Dataset on Indonesian Named Entity Recognition" (AACL 2020)
- **Domain:** Online news
- **Splits:** train 1,464 / dev 367 / test 509 sentences
- **Entities:** BIO; PER, ORG, LOC
- **Files:** `idner2k/train_bio.txt`, `dev_bio.txt`, `test_bio.txt`

### NER-UI (IndoLEM)
- **Reference:** Koto et al. [8] — IndoLEM benchmark (COLING 2020)
- **Domain:** Indonesian Wikipedia
- **Splits:** train 1,530 / dev 170 / test 426 sentences
- **Entities:** BIO; PER, ORG, LOC
- **Files:** `nerui/train_bio.txt`, `dev_bio.txt`, `test_bio.txt`

### NER-UGM (IndoNLU)
- **Reference:** Wilie et al. [21] — IndoNLU benchmark (AACL 2020)
- **Domain:** Mixed (news, social media)
- **Splits:** train 1,687 / dev 187 / test 469 sentences
- **Entities:** BIO; PER, ORG, LOC
- **Files:** `nerugm/train_bio.txt`, `dev_bio.txt`, `test_bio.txt`

## Format
One token per line, tab-separated token and BIO label. Empty line = sentence boundary.

```
token1	TAG1
token2	TAG2

token1	TAG1
```

Example: `Ibnu\tB-PER`, `Jamil\tI-PER`, `Universitas\tB-ORG`, etc.

## Obtaining full datasets
- **IDNer2k:** See Khairunnisa et al. (2020); Indonesian NLP resources / ACL Anthology
- **NER-UI:** IndoLEM / indobenchmark
- **NER-UGM:** IndoNLU benchmark

Place the files in `data/idner2k/`, `data/nerui/`, and `data/nerugm/` with the names above so the training and evaluation scripts find them.
