"""Experiment 8: Cross-Lingual Abstractive Summarization (mT5-small vs mBART-50 on XL-Sum Hindi)."""
def build_generation_config(model_name, src_lang="hi_IN"):
    return {
        "model_name": model_name,
        "src_lang": src_lang,
        "num_beams": 4,
        "max_new_tokens": 64,
        "length_penalty": 1.0,
    }
