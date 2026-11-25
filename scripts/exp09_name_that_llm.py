"""Experiment 9: Name That LLM — 18-class Authorship Attribution with StratifiedGroupKFold."""
import numpy as np

def blend_probabilities(p_qwen8b, p_qwen14b, p_tfidf):
    return 0.45 * np.asarray(p_qwen8b) + 0.40 * np.asarray(p_qwen14b) + 0.15 * np.asarray(p_tfidf)
