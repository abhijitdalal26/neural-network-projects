# transformers-from-scratch

Learning transformers and LLMs by building from scratch in PyTorch.

This repo collects hands-on implementations as I learn: vanilla Transformer for machine translation, LLaMA-family models, and other LLM architectures with training and visualization on Colab.

## Structure
- `pytorch-transformer/` — English→Dutch Transformer on `opus_books` (38k pairs) — training, greedy/beam inference, attention visualizations. Run `colab_transformer_training.ipynb` on Colab T4.
- `LLaMA/` — LLaMA 2/3 notes and experiments
- `foundation/` — earlier neural net foundations

## Quick start (Colab)
1. Open `pytorch-transformer/colab_transformer_training.ipynb` → Runtime T4 GPU → Run All (~3.5h for 20 epochs, or `SKIP_TRAIN=True` to demo from Drive checkpoint).
2. Local: `pip install -r pytorch-transformer/requirements.txt` then `python pytorch-transformer/train.py`

Dataset is `Helsinki-NLP/opus_books` (public, 38k en-nl pairs) — no HF token needed, works with latest `datasets`/`huggingface_hub`.
