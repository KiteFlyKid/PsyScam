# PsyScam

**A Benchmark for Psychological Techniques in Real-World Scams**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.15017)
[![Dataset](https://img.shields.io/badge/Dataset-Available%20on%20Request-blue.svg)](#dataset-access)

This repository contains the code and partial dataset accompanying our paper submitted to EMNLP 2025:

> **PsyScam: A Benchmark for Psychological Techniques in Real-World Scams**

## 🎯 Overview

Online scams exploit various psychological techniques (PTs) to manipulate victims. **PsyScam** provides a comprehensive benchmark to support the analysis and modeling of these techniques across three key NLP tasks:

- **🏷️ PT Classification**: Multi-label classification of psychological techniques in scam content
- **✍️ Scam Completion**: Generating realistic scam continuations given partial content
- **🔄 Scam Augmentation**: Creating variations of existing scam content while preserving psychological techniques

## 📁 Repository Structure

```
PsyScam/
├── crawlers/                # Web scrapers for collecting scam reports from public sources
├── data/
│   ├── D2.csv              # Evaluation subset used in our experiments (sample dataset)
│   └── PTs.csv             # Comprehensive list of psychological technique labels
├── LLMExtractor.py         # Human-LLM collaborative annotation using GPT-4
├── PTClassification.py     # Multi-label psychological technique classification
├── ScamCompletion.py       # Scam completion generation task implementation
├── ScamAugmentation.py     # Scam augmentation generation task implementation
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### API Configuration
Create an `api.key` file in the root directory with your OpenAI API key:

### Quick Start

1. **PT Classification**:
   ```bash
   python PTClassification.py --csv data/D2.csv
   ```

2. **Scam Completion**:
   ```bash
   python ScamCompletion.py --llm_model gpt41
   ```

3. **Scam Augmentation**:
   ```bash
   python ScamAugmentation.py --llm_model gpt41
   ```

## 📊 Dataset

Our benchmark includes carefully curated scam reports annotated with psychological techniques. Due to safety and ethical considerations, the complete dataset is available upon request for research purposes only.

We only include dataset (`D2.csv`) in this repo.

