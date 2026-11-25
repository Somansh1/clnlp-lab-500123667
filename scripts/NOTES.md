# Experiments 7 to 10 — GPU Execution & Benchmark Notes

- **Experiment 7 (IMDb Sentiment):** Balanced 800 train / 200 test split. TF-IDF+LR achieved 0.7850 Accuracy / 0.7923 F1; Fine-tuned BERT achieved 0.7800 Accuracy / 0.7732 F1; from-scratch LSTM (0.4800) and Vanilla Transformer (0.5600) underfit on 800 samples.
- **Experiment 8 (XL-Sum Hindi Summarization):** 200 sampled test articles. `mBART-50` achieved 0.3350 ROUGE-1 / 0.1436 ROUGE-2 / 0.2831 ROUGE-L / 0.8202 mBERTScore, outperforming `mT5-small` (0.2357 / 0.0663 / 0.1963 / 0.7726).
- **Experiment 9 (Name That LLM):** 18 models across 6 labs, 5-fold `StratifiedGroupKFold` grouped by `prompt_id`. Blending Qwen3-8B LoRA + Qwen3-14B LoRA + TF-IDF reached 0.653 CV balanced accuracy (0.686 public LB).
- **Experiment 10 (WMT26 Indic NMT):** 4 pairs / 8 directions (`as`, `mni`, `lus`, `bodo` <-> `en`). Fine-tuned `NLLB-200-distilled-600M` averaged 52.37 chrF++ / 28.48 BLEU / 74.33 COMET (vs. 39.91 / 17.24 / 64.59 baseline), with `+43.18` / `+41.06` chrF++ gains on Bodo via the `hin_Deva` proxy tag.
