# IG-06 · Balochistan Text-to-Image Benchmark
## Requirements

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
nbformat>=5.0.0
nbconvert>=7.0.0
ipykernel>=6.0.0
reportlab>=4.0.0
pypdf>=3.0.0
sentencepiece
tiktoken
```

## Quick Start

```bash
# Run full benchmark evaluation
python main.py

# Or run the notebook
cd notebooks && python generate_notebook.py
jupyter notebook IG06_Benchmark_Analysis.ipynb
```

## Project Structure

```
E:\LLM\
├── data/
│   ├── raw/
│   │   ├── prompts_without_balochi.csv   # 1000 English-only, unreviewed
│   │   └── prompts_with_balochi.csv      # 1000 English + Balochi, reviewed
│   └── processed/                        # Output scores & summaries
├── src/
│   ├── config.py              # Constants & paths
│   ├── data_loader.py         # Dataset loading & merging
│   ├── fidelity_evaluator.py  # Cultural fidelity scoring (4 dims)
│   ├── stereotype_rubric.py   # Stereotype attribute detector (8 attrs)
│   ├── evaluate.py            # Full evaluation pipeline
│   ├── visualise.py           # 8 benchmark figures
│   └── report_generator.py   # Text + PDF report
├── notebooks/
│   ├── generate_notebook.py   # Auto-generates the .ipynb
│   └── IG06_Benchmark_Analysis.ipynb
├── outputs/
│   ├── figures/               # 8 PNG charts
│   └── models/
├── report/
│   ├── ig06_benchmark_report.txt
│   └── ig06_benchmark_report.pdf
└── main.py                    # Entry point
```

## Benchmark Design (per IG-06 spec)

| Layer | Details |
|-------|---------|
| **Prompt** | 1,000 prompts × 4 categories; English + Balochi paired |
| **Fidelity rubric** | 4 dims: accuracy, specificity, avoidance, diversity |
| **Stereotype rubric** | 8 attributes; default-vs-presence distinction |
| **Headline metric** | Violence/weapons rate in neutral_control prompts |
| **Evaluator** | Text-proxy (calibrated); VLM-ready interface |
| **Paired analysis** | English vs Balochi fidelity diff (bias signal) |

## Label Categories

- `neutral_control` – Everyday Baloch life, no loaded attributes
- `gender_representation` – Tests male/female defaults
- `cultural_fidelity` – Named Balochi concepts, places, artefacts
- `stereotype_audit` – Directly probes known stereotypes

## Ethics

- Generated images **not** released (synthetic faces only)
- Findings framed as model properties, **not** Baloch people properties
- Results shared with open-model developers where contact channels exist
