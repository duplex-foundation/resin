# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pyarrow>=15", "huggingface-hub>=0.35", "torch>=2.13", "transformers>=5.14.1"]
# ///
"""BF16 transformers greedy tokens and logits for Qwen/Qwen3-32B.

The prompt set hits prefill, decode, a cached continuation, and Resin's
256-token page boundaries. WikiText-2 windows and the pinned MMLU sample
are written beside the logits. Run this on CPU. The process binds itself
to NUMA node 1 and caps torch at 56 threads. It does not import JAX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
REPO = "Qwen/Qwen3-32B"
PAGE = 256
VOCAB = 151936
MMLU_SHA256 = "aea69f647253862af0c427c27d158f052e749f868d2241fe672ba26e9c0b7476"
WINDOW = 2048
WINDOWS = 4
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
THREADS = 56
ROOT = Path(__file__).resolve().parents[1]

# A few dozen new tokens. page_255 is longer so its continuation still has 32.
NEW = 40
PAGE_255_NEW = 48
DECODE_CROSS_NEW = 48
CONTINUATION_PREFIX = 16


def _prepare_env() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TPU_VISIBLE_CHIPS"] = ""
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _bind_numa() -> None:
    """Stay on NUMA node 1. Re-exec under numactl when the affinity is wider."""
    if os.environ.get("RESIN_NUMA_BOUND") == "1":
        return
    node = Path("/sys/devices/system/node/node1/cpulist").read_text()
    allowed = _parse_cpulist(node)
    if os.sched_getaffinity(0) <= allowed:
        os.environ["RESIN_NUMA_BOUND"] = "1"
        return
    os.environ["RESIN_NUMA_BOUND"] = "1"
    os.execvp(
        "numactl",
        ["numactl", "--cpunodebind=1", "--membind=1", sys.executable, *sys.argv],
    )


def _refuse_accelerators() -> None:
    loaded = [name for name in ("jax", "jaxlib", "libtpu") if name in sys.modules]
    if loaded:
        raise SystemExit(f"refusing to run with {loaded} imported")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name in ("config.json", "model.safetensors.index.json"):
        remote = Path(hf_hub_download(REPO, name, revision=REVISION))
        local = model_dir / name
        if _sha256(local) != _sha256(remote):
            raise SystemExit(f"{local} does not match {REPO}@{REVISION} {name}")


def _ids(value: object) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _wikitext_ids(tokenizer) -> list[int]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=WIKITEXT_REVISION,
    )
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    document = "\n".join(texts)
    encoded = tokenizer(document, add_special_tokens=False, truncation=False)
    return _ids(encoded["input_ids"])


def _cases(tokenizer, stream: list[int]) -> list[dict]:
    hand = [
        ("short", "The capital of France is", ("prefill", "decode")),
        (
            "code",
            'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n    ',
            ("prefill", "decode"),
        ),
        ("unicode", "请用一句话解释万有引力：", ("prefill", "decode")),
    ]
    cases = []
    for case_id, text, exercises in hand:
        prompt = _ids(tokenizer(text, add_special_tokens=False, truncation=False))
        cases.append(_case(case_id, prompt, NEW, exercises, text, token_start=None))
    chat_messages = [
        {
            "role": "user",
            "content": "What is 17 plus 25? Reply with the number only.",
        }
    ]
    chat_text = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    chat_ids = _ids(
        tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
    )
    cases.append(_case("chat", chat_ids, NEW, ("prefill", "decode"), chat_text, None))
    slices = [
        ("page_255", 255, PAGE_255_NEW, ("prefill", "decode", "page_boundary")),
        ("page_256", 256, NEW, ("prefill", "decode", "page_boundary")),
        ("page_257", 257, NEW, ("prefill", "decode", "page_boundary")),
        ("page_511", 511, NEW, ("prefill", "decode", "page_boundary")),
        ("page_512", 512, NEW, ("prefill", "decode", "page_boundary")),
        ("page_513", 513, NEW, ("prefill", "decode", "page_boundary")),
        ("decode_cross", 240, DECODE_CROSS_NEW, ("prefill", "decode", "page_boundary")),
    ]
    for case_id, length, new_tokens, exercises in slices:
        prompt = stream[:length]
        cases.append(
            _case(
                case_id,
                prompt,
                new_tokens,
                exercises,
                tokenizer.decode(prompt),
                token_start=0,
            )
        )
    cases.append(
        {
            "id": "continue_page_255",
            "parent": "page_255",
            "prefix_new_tokens": CONTINUATION_PREFIX,
            "new_tokens": PAGE_255_NEW - CONTINUATION_PREFIX,
            "exercises": ["continuation", "prefill", "page_boundary"],
            "token_start": None,
        }
    )
    return cases


def _case(case_id, prompt, new_tokens, exercises, text, token_start) -> dict:
    return {
        "id": case_id,
        "prompt_ids": prompt,
        "prompt_text": text,
        "prompt_len": len(prompt),
        "new_tokens": new_tokens,
        "exercises": list(exercises),
        "token_start": token_start,
        "parent": None,
    }


def _windows(tokenizer, stream: list[int]) -> list[dict]:
    need = WINDOW * WINDOWS
    if len(stream) < need:
        raise SystemExit(f"WikiText-2 test tokenized to {len(stream)} ids, need {need}")
    rows = []
    for index in range(WINDOWS):
        start = index * WINDOW
        token_ids = stream[start : start + WINDOW]
        rows.append(
            {
                "dataset": "Salesforce/wikitext",
                "revision": WIKITEXT_REVISION,
                "config": "wikitext-2-raw-v1",
                "split": "test",
                "join": "newline",
                "window": index,
                "token_start": start,
                "token_len": WINDOW,
                "token_ids": token_ids,
                "text": tokenizer.decode(token_ids),
            }
        )
    return rows


def _write_texts(out: Path, mmlu: Path, windows: list[dict]) -> None:
    if _sha256(mmlu) != MMLU_SHA256:
        raise SystemExit(f"{mmlu} sha256 is not {MMLU_SHA256}")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "mmlu-1000.jsonl"
    target.write_bytes(mmlu.read_bytes())
    lines = "".join(json.dumps(row) + "\n" for row in windows)
    (out / "wikitext-2-windows.jsonl").write_text(lines)


def _import_torch():
    _refuse_accelerators()
    import torch

    _refuse_accelerators()
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"interop threads already set: {exc}", flush=True)
    if torch.get_num_threads() != THREADS:
        raise SystemExit(f"torch threads are {torch.get_num_threads()}, want {THREADS}")
    return torch


def _load_model(torch, model_dir: Path):
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    return model, time.perf_counter() - started


def _greedy(model, torch, prompt_ids: list[int], new_tokens: int, *, log: bool):
    import numpy as np

    ids = torch.tensor([prompt_ids], dtype=torch.long)
    past = None
    generated: list[int] = []
    rows = []
    step_s: list[float] = []
    for step in range(new_tokens):
        started = time.perf_counter()
        with torch.inference_mode():
            if past is None:
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            else:
                out = model(
                    input_ids=ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
        past = out.past_key_values
        logits = out.logits[0, -1].to(torch.bfloat16).contiguous()
        del out
        nxt = int(torch.argmax(logits))
        bits = logits.view(torch.uint16).cpu().numpy().copy()
        floated = (bits.astype(np.uint32) << 16).view(np.float32)
        if int(floated.argmax()) != nxt or bits.shape != (VOCAB,):
            raise RuntimeError(f"step {step} logit row does not match argmax {nxt}")
        rows.append(bits)
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        step_s.append(time.perf_counter() - started)
        if log:
            kind = "prefill" if step == 0 else "decode"
            print(f"  {kind} {step} {step_s[-1]:.3f}s", flush=True)
    del past
    return generated, np.stack(rows), step_s


def _save_case(out: Path, case: dict, generated: list[int], logits) -> None:
    import numpy as np

    path = out / "logits" / f"{case['id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits_bf16=logits,
        generated_ids=np.asarray(generated, dtype=np.int32),
        prompt_ids=np.asarray(case["prompt_ids"], dtype=np.int32),
    )
    case["generated_ids"] = generated
    case["logits_file"] = f"logits/{case['id']}.npz"


def _run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    out = Path(args.out)
    _verify_checkpoint(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # The joined test split is longer than the tokenizer's warning threshold.
    tokenizer.model_max_length = 10**9
    stream = _wikitext_ids(tokenizer)
    cases = _cases(tokenizer, stream)
    windows = _windows(tokenizer, stream)
    for case in cases:
        if case.get("prompt_len") is not None:
            print(
                f"plan {case['id']} prompt={case['prompt_len']} new={case['new_tokens']}",
                flush=True,
            )
        else:
            print(
                f"plan {case['id']} parent={case['parent']} "
                f"prefix={case['prefix_new_tokens']} new={case['new_tokens']}",
                flush=True,
            )
    _refuse_accelerators()
    if args.plan:
        _write_texts(out, Path(args.mmlu), windows)
        print(f"wrote texts under {out}", flush=True)
        return

    torch = _import_torch()
    print(f"torch {torch.__version__} threads {torch.get_num_threads()}", flush=True)
    started = time.perf_counter()
    model, load_s = _load_model(torch, model_dir)
    print(f"loaded model in {load_s:.1f}s", flush=True)
    import transformers

    if args.time_one:
        short = next(case for case in cases if case["id"] == "short")
        print(f"timing {short['id']} prompt={short['prompt_len']}", flush=True)
        _generated, _logits, steps = _greedy(
            model, torch, short["prompt_ids"], 8, log=True
        )
        page = stream[:PAGE]
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            model(
                input_ids=torch.tensor([page], dtype=torch.long),
                use_cache=False,
                logits_to_keep=1,
            )
        prefill_256_s = time.perf_counter() - prefill_started
        print(f"prefill_256 {prefill_256_s:.3f}s", flush=True)
        payload = {
            "load_s": load_s,
            "short_prompt_len": short["prompt_len"],
            "short_step_s": steps,
            "prefill_256_s": prefill_256_s,
            "cases": [
                {
                    "id": case["id"],
                    "prompt_len": case.get("prompt_len"),
                    "new_tokens": case["new_tokens"],
                    "parent": case.get("parent"),
                    "prefix_new_tokens": case.get("prefix_new_tokens"),
                }
                for case in cases
            ],
        }
        print("TIMING " + json.dumps(payload), flush=True)
        return

    by_id = {}
    for case in cases:
        if case.get("parent"):
            parent = by_id[case["parent"]]
            prefix = case["prefix_new_tokens"]
            prompt = parent["prompt_ids"] + parent["generated_ids"][:prefix]
            case["prompt_ids"] = prompt
            case["prompt_len"] = len(prompt)
            case["prompt_text"] = tokenizer.decode(prompt)
            case["token_start"] = None
        print(
            f"run {case['id']} prompt={case['prompt_len']} new={case['new_tokens']}",
            flush=True,
        )
        generated, logits, steps = _greedy(
            model, torch, case["prompt_ids"], case["new_tokens"], log=False
        )
        if case.get("parent"):
            parent = by_id[case["parent"]]
            expected = parent["generated_ids"][case["prefix_new_tokens"] :]
            case["matches_parent"] = generated == expected
            print(f"  matches_parent {case['matches_parent']}", flush=True)
        _save_case(out, case, generated, logits)
        by_id[case["id"]] = case
        print(
            f"  done {sum(steps):.1f}s first={steps[0]:.3f}s "
            f"decode_mean={sum(steps[1:]) / max(len(steps) - 1, 1):.3f}s",
            flush=True,
        )
    _write_texts(out, Path(args.mmlu), windows)
    manifest = {
        "model": REPO,
        "revision": REVISION,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "page_size": PAGE,
        "vocab_size": VOCAB,
        "greedy": True,
        "eos": "not consulted; the loop always emits new_tokens",
        "logits": (
            "uint16 bfloat16 bits, one row per step, last-position raw lm_head "
            "output; step 0 is prefill and later steps are cached decode"
        ),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "threads": THREADS,
        "numa_node": 1,
        "load_s": load_s,
        "mmlu_sha256": MMLU_SHA256,
        "wikitext": {
            "dataset": "Salesforce/wikitext",
            "revision": WIKITEXT_REVISION,
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "join": "newline",
            "windows": WINDOWS,
            "window_tokens": WINDOW,
        },
        "wall_clock_s": time.perf_counter() - started,
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} in {manifest['wall_clock_s']:.1f}s", flush=True)


def main() -> None:
    _prepare_env()
    _bind_numa()
    _refuse_accelerators()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mmlu", required=True)
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests" / "fixtures" / "qwen3_32b"),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Check the checkpoint and write the MMLU sample and WikiText windows.",
    )
    parser.add_argument(
        "--time-one",
        action="store_true",
        help="Time the short prompt and one 256-token prefill, then exit.",
    )
    args = parser.parse_args()
    if args.plan and args.time_one:
        raise SystemExit("pass only one of --plan and --time-one")
    _run(args)


if __name__ == "__main__":
    main()
