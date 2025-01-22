# Artificial Intelligence Legal Interpreter

The **Artificial Intelligence Legal Interpreter** is a project aimed at fine-tuning a large language model (LLM) to predict judicial interpretations in response to legal questions derived from preliminary rulings by the Court of Justice of the European Union (CJEU). This tool is designed to assist legal professionals by providing predictive, context-aware insights into judicial reasoning, thereby contributing to legal research and decision-making.

---

## Motivation

Legal professionals face numerous challenges when navigating complex judicial rulings. This project seeks to address those challenges by:

- **Extracting question-answer pairs** from CJEU rulings.
- **Structuring data** into machine-readable formats enriched with metadata such as decision years and case references.
- **Fine-tuning an LLM** to ensure legal consistency and alignment with CJEU responses.
- **Providing an AI-driven tool** for legal research and decision-making.

### Key Features
- **Curated Dataset**: Data sourced from EURLEX and CURIA.
- **Metadata Enrichment**: Case details, decision years, and references for structured insights.
- **RAG Legal Document Access**: Proper citing of legal documents when answering.
- **Model Evaluation**: Benchmarked for alignment with CJEU rulings, citation accuracy, and interpretive correctness.

---

## Project Structure

The repository is organized as follows:

```
.
├── data
│   ├── celex_ids.json
│   ├── celex_mapping.json
│   ├── sample_celex_mapping.json
│   └── sample_docs.json
├── HTMLs
├── preprocessing
│   └── filter_docs.py
├── responses.json
├── scripts
│   ├── dataset
│   │   └── prepare_eurlex_dataset.py
│   ├── __init__.py
│   ├── preprocessing
│   │   └── __init__.py
│   └── scraping
│       ├── count_things.py
│       └── scrape_celex.py
├── setup.py
├── src
│   ├── custom_loss.py
│   ├── __init__.py
│   ├── rag
│   │   ├── data
│   │   ├── ingest_openai.py
│   │   ├── ingest.py
│   │   ├── model_openai.py
│   │   ├── model.py
│   │   └── vectorstore
│   └── train
│       └── config
└── tests
    ├── __init__.py
    └── test_loss.py
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Pipenv or virtualenv for dependency management

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lest161c/cjeu.ali.git
   cd cjeu-ali
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

### 1. Scraping Legal Data
Use the `scrape_celex.py` script to download raw data from CELEX:
```bash
python scripts/scraping/scrape_celex.py
```

### 2. Preprocessing Data
Filter and clean scraped documents using:
```bash
python preprocessing/filter_docs.py
```

### 3. Fine-Tuning the LLM
Prepare datasets and execute model training in the `src` directory.

### 4. Running Evaluations
Run tests to benchmark the model against evaluation metrics:
```bash
pytest tests
```

---

## Dataset Details

### Sources
- **EURLEX**: Legal acts and case law of the EU.
- **CURIA**: Court of Justice rulings and legal interpretations.

### Data Pipeline
1. **Scraping**: Collect rulings from CURIA.
2. **Filtering**: Clean and preprocess data using metadata filters.
3. **Enrichment**: Annotate data with structured metadata for better training outcomes.

---

## Challenges
- **Legal Language Complexity**: Adapting the model to nuanced legal phrasing.
- **Data Variability**: Handling diverse question formats and irregularities in rulings.
- **Citation Accuracy**: Ensuring references align with official case law.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with detailed changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Future Work
- Expand the dataset with rulings from additional jurisdictions.
- Enhance interpretive accuracy with human-in-the-loop training.
- Integrate with legal research tools for real-time predictions.

---

## Acknowledgments

Special thanks to the teams behind EURLEX, CURIA, and the open-source AI community for enabling this work.

---

## Contact

For questions, feedback, or collaboration opportunities, reach out via email at **leonard.starke@mailbox.tu-dresden.de**.
