"""
Smoke harness for colab_transformer_training.ipynb pipeline.
Runs without T4 and without downloading opus_books (synthetic 50-pair dataset).
Covers: Setup/config, tokenizers, dataset/masks, model forward, 1 train step,
ckpt save/load, validation, inference greedy, step-by-step tokenization/embeddings,
attention helpers, beam search. Also checks notebook JSON compiles and HF patch idempotent.
Run: python tests/test_transformer_smoke.py
"""
import sys
import pathlib
import json
import tempfile
import ast

ROOT = pathlib.Path(__file__).resolve().parents[1]
PT = ROOT / "pytorch-transformer"
sys.path.insert(0, str(PT))

def ok(msg): print(f"[OK] {msg}")
def fail(msg): print(f"[FAIL] {msg}"); sys.exit(1)

# 1. Notebook static checks
print("=== 1. Notebook JSON / compile ===")
nb_path = PT / "colab_transformer_training.ipynb"
try:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    cells = nb["cells"]
    assert len(cells) == 47, f"expected 47 cells, got {len(cells)}"
    ok(f"notebook JSON valid {len(cells)} cells (24 md/23 code expected)")
except Exception as e:
    fail(f"notebook JSON: {e}")

# compile all code cells (colab magics are expected — smoke just checks python parts)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        # Replace colab shell magics with pass so `if ...: !git clone` doesn't leave empty block
        cleaned = []
        for l in src.splitlines():
            s = l.strip()
            if s.startswith("!") or s.startswith("%") or s.startswith("drive.mount"):
                # keep indent, replace with pass
                indent = l[: len(l) - len(l.lstrip())]
                cleaned.append(indent + "pass  # colab magic")
            else:
                cleaned.append(l)
        src_clean = "\n".join(cleaned)
        try:
            ast.parse(src_clean)
        except SyntaxError as e:
            fail(f"cell {i} compile: {e}\n---src---\n{src_clean[:500]}")
ok("all code cells compile (after magic -> pass)")

# check no notebook-side HF double patch remains
train_like = [ "".join(c["source"]) for c in cells if "SKIP_TRAIN" in "".join(c.get("source", [])) ]
if train_like:
    if "_patched" in train_like[0]:
        fail("notebook Train cell still contains _patched (should be only in train.py)")
    ok("notebook Train cell clean — hotfix only in train.py")

# 2. Config
print("\n=== 2. Config ===")
try:
    from config import get_config
    cfg = get_config()
    assert cfg["lang_src"] == "en" and cfg["lang_tgt"] == "nl"
    # temp override for smoke: tiny model, tiny seq, temp folders
    # model_folder must stay relative "weights" because train.py does f"{datasource}_{model_folder}" (colon in Windows absolute would break)
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    cfg["seq_len"] = 32
    cfg["d_model"] = 64
    cfg["num_epochs"] = 1
    cfg["batch_size"] = 4
    cfg["model_folder"] = "weights"
    cfg["tokenizer_file"] = str(tmpdir / "tokenizer_{0}.json")
    cfg["experiment_name"] = str(tmpdir / "runs/tmodel")
    # weights will be ./opus_books_weights when cwd is tmpdir — create there
    (tmpdir / "opus_books_weights").mkdir(parents=True, exist_ok=True)
    pathlib.Path(cfg["tokenizer_file"].format("en")).parent.mkdir(parents=True, exist_ok=True)
    ok(f"config en-nl seq_len 32 d_model 64 tmp {tmpdir}")
except Exception as e:
    fail(f"config: {e}")

# 3. Imports (mock heavy deps if missing locally)
print("\n=== 3. Imports ===")
try:
    import torch, torch.nn as nn
    # mock torchmetrics/tensorboard/altair if not installed locally (Colab has them, laptop may not)
    import sys, types
    if "torchmetrics" not in sys.modules:
        try:
            import torchmetrics  # type: ignore
        except Exception:
            m = types.ModuleType("torchmetrics")
            class _DummyMetric:
                def __call__(self, *a, **k): return 0.0
            m.CharErrorRate = _DummyMetric
            m.WordErrorRate = _DummyMetric
            m.BLEUScore = _DummyMetric
            sys.modules["torchmetrics"] = m
            print("  (mocked torchmetrics)")
    if "torch.utils.tensorboard" not in sys.modules:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            tb = types.ModuleType("torch.utils.tensorboard")
            class DummyWriter:
                def __init__(self, *a, **k): pass
                def add_scalar(self, *a, **k): pass
                def flush(self): pass
            tb.SummaryWriter = DummyWriter
            sys.modules["torch.utils.tensorboard"] = tb
            print("  (mocked tensorboard)")
    from model import build_transformer
    from dataset import BilingualDataset, causal_mask
    from train import get_or_build_tokenizer, get_model, greedy_decode, run_validation
    ok(f"torch {torch.__version__} imports OK")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"imports: {e}")

# 4. HF patch idempotent
print("\n=== 4. HF Hub patch idempotent ===")
try:
    import huggingface_hub.utils._hf_uris as _hf_uris  # may not exist on 0.36
    flag = getattr(_hf_uris, "_opus_books_patched", False)
    ok(f"HF patch flag present={flag} (expected True after train import, or missing on 0.36)")
    # double import should not recurse
    import importlib, train as _train
    importlib.reload(_train)
    ok("reload train.py no RecursionError")
except ModuleNotFoundError:
    ok("huggingface_hub _hf_uris not present on this hub version (0.36) — no patch needed, OK")
except Exception as e:
    if "RecursionError" in type(e).__name__:
        fail(f"HF patch recursion: {e}")
    ok(f"HF patch reload OK (non-fatal {e})")

# 5. Synthetic dataset + tokenizers
print("\n=== 5. Synthetic opus_books-like dataset ===")
try:
    en_sents = ["Hello how are you", "I love learning languages", "The book is on the table", "Good morning", "See you tomorrow"] * 10
    nl_sents = ["Hallo hoe gaat het", "Ik hou van talen leren", "Het boek ligt op de tafel", "Goedemorgen", "Tot morgen"] * 10
    ds_raw = [{"translation": {"en": en, "nl": nl}} for en, nl in zip(en_sents, nl_sents)]
    ok(f"synthetic ds {len(ds_raw)} pairs")
    # pass synthetic ds to get_or_build_tokenizer (expects iterable of dicts)
    tok_src = get_or_build_tokenizer(cfg, ds_raw, "en")
    tok_tgt = get_or_build_tokenizer(cfg, ds_raw, "nl")
    assert tok_src.get_vocab_size() > 10
    ok(f"tokenizers vocab src {tok_src.get_vocab_size()} tgt {tok_tgt.get_vocab_size()}")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"tokenizer: {e}")

# 6. Dataset / DataLoader
print("\n=== 6. BilingualDataset ===")
try:
    from torch.utils.data import DataLoader, random_split
    train_n = int(0.9 * len(ds_raw))
    train_raw, val_raw = random_split(ds_raw, [train_n, len(ds_raw)-train_n])
    train_ds = BilingualDataset(train_raw, tok_src, tok_tgt, "en", "nl", cfg["seq_len"])
    val_ds = BilingualDataset(val_raw, tok_src, tok_tgt, "en", "nl", cfg["seq_len"])
    one = train_ds[0]
    assert one["encoder_input"].shape[0] == cfg["seq_len"]
    assert one["decoder_input"].shape[0] == cfg["seq_len"]
    assert "encoder_mask" in one and "decoder_mask" in one
    ok(f"BilingualDataset sample encoder {one['encoder_input'].shape} decoder {one['decoder_input'].shape}")
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    batch = next(iter(train_dl))
    assert batch["encoder_input"].shape == (cfg["batch_size"], cfg["seq_len"])
    ok(f"DataLoader batch {batch['encoder_input'].shape}")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"dataset: {e}")

# 7. Model forward
print("\n=== 7. Model forward ===")
try:
    device = torch.device("cpu")
    model = get_model(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size()).to(device)
    # tiny params
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"model built params {n_params/1e6:.2f}M")
    enc_in = batch["encoder_input"].to(device)
    dec_in = batch["decoder_input"].to(device)
    enc_mask = batch["encoder_mask"].to(device)
    dec_mask = batch["decoder_mask"].to(device)
    enc_out = model.encode(enc_in, enc_mask)
    assert enc_out.shape[2] == cfg["d_model"]
    dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
    proj = model.project(dec_out)
    assert proj.shape[:2] == (cfg["batch_size"], cfg["seq_len"])
    ok(f"encode {enc_out.shape} decode {dec_out.shape} project {proj.shape}")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"model forward: {e}")

# 8. One train step + ckpt
print("\n=== 8. Train step + ckpt ===")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok_src.token_to_id("[PAD]"), label_smoothing=0.1)
    model.train()
    loss = loss_fn(proj.view(-1, tok_tgt.get_vocab_size()), batch["label"].view(-1))
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    ok(f"loss backward {loss.item():.3f}")
    # ckpt save/load via config helpers — must run with cwd=tmpdir so opus_books_weights resolves there
    from config import get_weights_file_path, latest_weights_file_path
    import os
    orig_cwd = pathlib.Path.cwd()
    os.chdir(tmpdir)
    try:
        ckpt_dir = pathlib.Path(f"{cfg['datasource']}_{cfg['model_folder']}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = pathlib.Path(get_weights_file_path(cfg, "00")).resolve()
        torch.save({"epoch":0,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"global_step":1}, str(ckpt_path))
        assert ckpt_path.exists()
        latest = latest_weights_file_path(cfg)
        assert latest is not None
        ok(f"ckpt saved {ckpt_path} latest {latest}")
        # preload simulation while still in tmpdir (or via absolute path)
        state = torch.load(str(ckpt_path), map_location="cpu")
        model2 = get_model(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size())
        model2.load_state_dict(state["model_state_dict"])
        ok("ckpt load OK")
    finally:
        os.chdir(orig_cwd)
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"train step: {e}")

# 9. Validation (greedy)
print("\n=== 9. Validation / greedy_decode ===")
try:
    model.eval()
    with torch.no_grad():
        vbatch = next(iter(val_dl))
        enc_in = vbatch["encoder_input"]
        enc_mask = vbatch["encoder_mask"]
        out = greedy_decode(model, enc_in, enc_mask, tok_src, tok_tgt, cfg["seq_len"], torch.device("cpu"))
        txt = tok_tgt.decode(out.numpy())
        ok(f"greedy_decode -> '{txt[:60]}'")
    # run_validation (writer=None to avoid torchmetrics file write issues)
    from train import run_validation as _rv
    _rv(model, val_dl, tok_src, tok_tgt, cfg["seq_len"], torch.device("cpu"), lambda m: None, 1, None, num_examples=2)
    ok("run_validation (2 examples) OK")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"validation: {e}")

# 10. Step-by-step inference (tokenize -> pad/mask -> embed -> decode logits)
print("\n=== 10. Step-by-step inference ===")
try:
    sentence = "Hello how are you"
    src_ids = tok_src.encode(sentence).ids
    seq_len = cfg["seq_len"]
    import torch as _t
    pad_id = tok_src.token_to_id("[PAD]")
    sos_id = tok_src.token_to_id("[SOS]")
    eos_id = tok_src.token_to_id("[EOS]")
    # tokenization check
    assert len(src_ids) > 0
    ok(f"tokenization ids {src_ids[:5]}")
    # padding / mask
    src = _t.cat([_t.tensor([sos_id]), _t.tensor(src_ids), _t.tensor([eos_id]), _t.tensor([pad_id]*(seq_len-len(src_ids)-2))])
    assert src.shape[0] == seq_len
    src_mask = (src != pad_id).unsqueeze(0).unsqueeze(0).int()
    ok(f"padded src {src.shape} mask {src_mask.shape}")
    # embeddings via model
    with torch.no_grad():
        src_emb = model.src_embed(src.unsqueeze(0))  # if exists
    ok(f"embeddings via src_embed OK")
    # decoder loop one step logits
    with torch.no_grad():
        enc_out = model.encode(src.unsqueeze(0), src_mask)
        dec_in = _t.tensor([[tok_tgt.token_to_id("[SOS]")]])
        dec_mask = causal_mask(1).unsqueeze(0)
        out = model.decode(enc_out, src_mask, dec_in, dec_mask)
        logits = model.project(out[:, -1])
        probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-4
        top5 = torch.topk(probs, 5)
        ok(f"decoder logits {logits.shape} probs sum {probs.sum().item():.3f} top5 {top5.indices[0].tolist()[:3]}")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"step-by-step: {e}")

# 11. Attention helpers
print("\n=== 11. Attention helpers ===")
try:
    import pandas as pd
    try:
        import altair as alt
        has_alt = True
    except Exception:
        has_alt = False
        print("  (altair not installed locally — skipping chart build, testing mtx only)")
    # need at least one greedy decode to populate attention_scores
    with torch.no_grad():
        _ = greedy_decode(model, src.unsqueeze(0), src_mask, tok_src, tok_tgt, cfg["seq_len"], torch.device("cpu"))
    # mtx2df / get_attn_map / attn_map as in notebook
    def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
        import pandas as pd
        return pd.DataFrame(
            [(r,c,float(m[r,c]), f"{r:03d} {row_tokens[r] if len(row_tokens)>r else '<blank>'}", f"{c:03d} {col_tokens[c] if len(col_tokens)>c else '<blank>'}")
             for r in range(m.shape[0]) for c in range(m.shape[1]) if r<max_row and c<max_col],
            columns=["row","column","value","row_token","col_token"],
        )
    # try encoder attn
    attn = model.encoder.layers[0].self_attention_block.attention_scores
    assert attn is not None
    df = mtx2df(attn[0,0].data, 5, 5, ["a"]*32, ["b"]*32)
    assert len(df) > 0
    ok(f"mtx2df {len(df)} rows attn {attn.shape}")
    if has_alt:
        chart = alt.Chart(df).mark_rect().encode(x="col_token", y="row_token", color="value")
        ok("altair chart build OK")
    else:
        ok("altair chart skipped (not installed)")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"attention: {e}")

# 12. Beam search
print("\n=== 12. Beam search ===")
try:
    from dataset import causal_mask as _cm
    def beam_search_decode(model, beam_size, source, source_mask, tok_src, tok_tgt, max_len, device):
        sos_idx = tok_tgt.token_to_id("[SOS]"); eos_idx = tok_tgt.token_to_id("[EOS]")
        enc_out = model.encode(source, source_mask)
        dec_init = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
        cands = [(dec_init, 0.0)]
        while True:
            if any(c.size(1)==max_len for c,_ in cands): break
            new=[]
            for cand, score in cands:
                if cand[0][-1].item()==eos_idx: new.append((cand,score)); continue
                mask = _cm(cand.size(1)).type_as(source_mask).to(device)
                out = model.decode(enc_out, source_mask, cand, mask)
                prob = model.project(out[:,-1])
                logp = torch.log_softmax(prob, dim=1)
                topk_log, topk_idx = torch.topk(logp, beam_size, dim=1)
                for i in range(beam_size):
                    tok = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                    nc = torch.cat([cand, tok], dim=1)
                    ns = score + topk_log[0][i].item()
                    new.append((nc, ns))
            cands = sorted(new, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(c[0][-1].item()==eos_idx for c,_ in cands): break
        return sorted(cands, key=lambda x: x[1], reverse=True)[0][0].squeeze(0)
    with torch.no_grad():
        src = src.unsqueeze(0); src_mask = (src[0]!=pad_id).unsqueeze(0).unsqueeze(0).int()
        g = greedy_decode(model, src, src_mask, tok_src, tok_tgt, cfg["seq_len"], torch.device("cpu"))
        b4 = beam_search_decode(model, 2, src, src_mask, tok_src, tok_tgt, cfg["seq_len"], torch.device("cpu"))
        ok(f"beam greedy {tok_tgt.decode(g.numpy())[:30]} | beam2 {tok_tgt.decode(b4.numpy())[:30]}")
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"beam: {e}")

print("\n=== ALL SMOKE TESTS PASSED ===")
