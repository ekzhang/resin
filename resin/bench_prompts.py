"""Saved, chat-templated workloads shared by inference benchmarks.

Loading an artifact uses only the standard library. Preparation imports a
tokenizer lazily and runs entirely on CPU. Long prompts are explicitly packed
from distinct ShareGPT conversation turns; only the final real user message is
truncated, and the complete canonical chat template is applied afterwards.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import random
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path

FORMAT = "resin-chat-prompts-v1"
POLICY = "sharegpt-distinct-turn-packing-unseen-final-user-character-truncation-v2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


class PromptArtifact:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.digest = _sha(self.path.read_bytes())
        self._tokens = {}
        with zipfile.ZipFile(self.path) as archive:
            self._manifest = json.loads(archive.read("manifest.json"))
            if self._manifest.get("format") != FORMAT:
                raise ValueError("unsupported benchmark prompt artifact format")
            corpus = archive.read("corpus.json")
            if _sha(corpus) != self._manifest["corpus"]["normalized_sha256"]:
                raise ValueError("prompt corpus SHA256 mismatch")
            if (
                _sha(archive.read("chat_template.jinja"))
                != self._manifest["chat_template"]["sha256"]
            ):
                raise ValueError("chat template SHA256 mismatch")
            if (
                _sha(archive.read("tokenizer.json"))
                != self._manifest["tokenizer"]["sha256"]
            ):
                raise ValueError("tokenizer SHA256 mismatch")
            for context, entry in self._manifest["contexts"].items():
                size = int(context)
                count = entry["count"]
                if size <= 0 or type(count) is not int or count <= 0:
                    raise ValueError("invalid prompt artifact context/cohort size")
                data = archive.read(entry["file"])
                if _sha(data) != entry["sha256"] or len(data) != size * count * 4:
                    raise ValueError(
                        f"prompt token SHA256/length mismatch at context {context}"
                    )
                if len(entry["records"]) != count:
                    raise ValueError("prompt record count does not match its cohort")
                if (
                    len({record["token_sha256"] for record in entry["records"]})
                    != count
                ):
                    raise ValueError("prompt artifact cohort contains repeated prompts")
                tokens = array.array("I")
                tokens.frombytes(data)
                if sys.byteorder != "little":
                    tokens.byteswap()
                if (
                    len(tokens) != size * count
                    or max(tokens) >= self._manifest["tokenizer"]["vocab_size"]
                ):
                    raise ValueError("prompt token IDs exceed checkpoint vocabulary")
                for index, record in enumerate(entry["records"]):
                    row = tokens[index * size : (index + 1) * size]
                    if _sha(_token_bytes(row)) != record["token_sha256"]:
                        raise ValueError("individual prompt token SHA256 mismatch")
        self.metadata = {
            key: value for key, value in self._manifest.items() if key != "contexts"
        }
        self.metadata.update(
            artifact_sha256=self.digest,
            prompt_source_id=f"chat-templated-{self._manifest.get('source', 'sharegpt')}:"
            + self.digest,
            available_contexts={
                key: entry["count"] for key, entry in self._manifest["contexts"].items()
            },
        )

    def cohort(self, context_length: int, batch_size: int) -> list[list[int]]:
        if (
            type(context_length) is not int
            or type(batch_size) is not int
            or batch_size <= 0
        ):
            raise ValueError("context and batch must be positive integers")
        entry = self._manifest["contexts"].get(str(context_length))
        if entry is None or batch_size > entry["count"]:
            raise ValueError(
                f"prompt artifact has no complete cohort for context={context_length}, batch={batch_size}"
            )
        if context_length not in self._tokens:
            with zipfile.ZipFile(self.path) as archive:
                data = archive.read(entry["file"])
            if _sha(data) != entry["sha256"]:
                raise ValueError("prompt artifact changed after validation")
            tokens = array.array("I")
            tokens.frombytes(data)
            if sys.byteorder != "little":
                tokens.byteswap()
            self._tokens[context_length] = tokens
        tokens = self._tokens[context_length]
        return [
            tokens[index * context_length : (index + 1) * context_length].tolist()
            for index in range(batch_size)
        ]


def load_prompt_artifact(path: str | Path) -> PromptArtifact:
    return PromptArtifact(path)


def _token_bytes(tokens) -> bytes:
    packed = array.array("I", tokens)
    if packed.itemsize != 4:
        raise RuntimeError("prompt artifacts require a 32-bit unsigned array type")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _normalize(records, reserved):
    conversations, rejected = [], 0
    roles = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "chatgpt": "assistant",
    }
    for index, record in enumerate(records):
        turns = record.get("conversations", record.get("messages", []))
        messages = []
        for turn in turns:
            role = roles.get(turn.get("from", turn.get("role")))
            content = turn.get("value", turn.get("content"))
            if (
                role != ("user" if len(messages) % 2 == 0 else "assistant")
                or not isinstance(content, str)
                or not content.strip()
                or any(token in content for token in reserved)
            ):
                messages = []
                break
            messages.append({"role": role, "content": content})
        if len(messages) < 2:
            rejected += 1
            continue
        conversations.append(
            {
                "id": f"{record.get('id', index)}#{index}",
                "original_id": str(record.get("id", index)),
                "source_record_index": index,
                "messages": messages,
            }
        )
    if len(conversations) < 2:
        raise ValueError("ShareGPT corpus has too few valid alternating conversations")
    return conversations, rejected


def _fit_final_user(history, text, context, render, full_length):
    single_length = len(render([{"role": "user", "content": text}]))
    for full_history in (False, True):
        # A short-turn probe avoids repeatedly tokenizing a 65K history. Its
        # candidate is always checked by rendering the complete conversation;
        # templates with history-dependent behavior fall back to the full probe.
        offset = 0 if full_history else full_length - single_length
        target = context - offset

        def probe(cut):
            message = [{"role": "user", "content": text[:cut]}]
            return render(history + message if full_history else message)

        low, high = 0, len(text)
        attempted = set()
        while low <= high:
            cut = (low + high) // 2
            attempted.add(cut)
            ids = probe(cut)
            if len(ids) == target:
                complete = render(history + [{"role": "user", "content": text[:cut]}])
                if len(complete) == context:
                    return complete, cut
                break
            if len(ids) < target:
                low = cut + 1
            else:
                high = cut - 1
        for cut in range(max(0, high - 16), min(len(text), low + 16) + 1):
            if cut in attempted:
                continue
            ids = probe(cut)
            if len(ids) == target:
                complete = render(history + [{"role": "user", "content": text[:cut]}])
                if len(complete) == context:
                    return complete, cut
    return None


def prepare_prompt_artifact(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer,
    tokenizer_identity: dict,
    source_metadata: dict,
    contexts=(2048, 4096, 8192, 16384, 32768, 65536),
    count: int | Mapping[int, int] = 256,
    seed=41,
    template_kwargs=None,
) -> PromptArtifact:
    """Create a deterministic single-file workload; no GPU or random padding.

    Conversations are normalized and shuffled per request. Complete user/assistant
    pairs form the history. A real, sufficiently long user message is truncated
    at a character boundary until the *fully re-rendered* canonical prompt has
    exactly the requested token count. No template/control token is sliced.
    """
    counts = dict(count) if isinstance(count, Mapping) else {n: count for n in contexts}
    if (
        not contexts
        or any(type(n) is not int or n <= 0 for n in contexts)
        or any(type(counts.get(n)) is not int or counts[n] <= 0 for n in contexts)
    ):
        raise ValueError("positive context lengths and cohort size are required")
    template_kwargs = dict(template_kwargs or {})
    raw = json.loads(Path(corpus_path).read_text())
    records = raw if isinstance(raw, list) else raw["conversations"]
    conversations, rejected = _normalize(records, tokenizer.all_special_tokens)
    corpus_bytes = _json_bytes(conversations)
    template = tokenizer.get_chat_template().encode()
    tokenizer_bytes = tokenizer.backend_tokenizer.to_str().encode()

    def render(messages):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("chat tokenizer must return a flat list of token IDs")
        return list(ids)

    pairs, users = [], []
    for conversation in conversations:
        turns = conversation["messages"]
        for turn in range(0, len(turns), 2):
            user = turns[turn]
            length = len(tokenizer.encode(user["content"], add_special_tokens=False))
            users.append((length, conversation["id"], turn, user))
            if turn + 1 < len(turns):
                pair = turns[turn : turn + 2]
                # Only an estimate for packing. Every saved sequence below is
                # produced by the full canonical template, never concatenated IDs.
                estimate = len(
                    tokenizer.apply_chat_template(
                        pair,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=False,
                    )
                )
                pairs.append((estimate, conversation["id"], turn, pair))
    users.sort(key=lambda item: (-item[0], item[1], item[2]))
    print(
        f"[prompt-corpus] conversations={len(conversations)} pairs={len(pairs)} users={len(users)}",
        flush=True,
    )
    if not pairs or not users:
        raise ValueError("corpus has no complete ShareGPT conversation turns")
    max_user_tokens = users[0][0]
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "corpus": {
            **source_metadata,
            "normalized_sha256": _sha(corpus_bytes),
            "valid_conversations": len(conversations),
            "rejected_conversations": rejected,
        },
        "tokenizer": {
            **tokenizer_identity,
            "vocab_size": len(tokenizer),
            "sha256": _sha(tokenizer_bytes),
            "special_tokens_map": tokenizer.special_tokens_map,
        },
        "chat_template": {
            "sha256": _sha(template),
            "add_generation_prompt": True,
            "kwargs": template_kwargs,
        },
        "packing": {
            "policy": POLICY,
            "seed": seed,
            "description": "Distinct real ShareGPT user/assistant turns packed as an alternating chat history; final real user text truncated by characters and canonically re-templated to exact length. This is a packed context-length stress workload, not a naturally occurring 65K conversation.",
        },
        "contexts": {},
    }
    entries = {
        "corpus.json": corpus_bytes,
        "chat_template.jinja": template,
        "tokenizer.json": tokenizer_bytes,
    }
    for context in sorted(set(contexts)):
        cohort_count = counts[context]
        cohort_tokens = array.array("I")
        prompt_records = []
        for request_index in range(cohort_count):
            rng = random.Random(f"{seed}:{request_index}")
            order = list(range(len(pairs)))
            rng.shuffle(order)
            history, sources = [], []
            remaining = context - min(512, max(32, max_user_tokens // 3))
            for pair_index in order:
                estimate, conversation_id, turn, pair = pairs[pair_index]
                if estimate <= remaining:
                    history.extend(pair)
                    sources.append({"conversation_id": conversation_id, "turn": turn})
                    remaining -= estimate
            previous_user_texts = {
                message["content"].strip()
                for message in history
                if message["role"] == "user"
            }
            candidates = [
                user
                for user in users
                if user[0] >= max(64, remaining + 64)
                and user[3]["content"].strip() not in previous_user_texts
            ]
            candidates = (
                candidates[: min(64, len(candidates))]
                or [
                    user
                    for user in users
                    if user[3]["content"].strip() not in previous_user_texts
                ][:64]
            )
            if not candidates:
                raise ValueError("corpus has no unseen final user message")
            offset = rng.randrange(len(candidates))
            candidates = candidates[offset:] + candidates[:offset]
            selected = None
            for _, conversation_id, turn, user in candidates:
                text = user["content"].strip()
                while (
                    history
                    and len(render(history + [{"role": "user", "content": ""}]))
                    >= context
                ):
                    del history[-2:]
                    sources.pop()
                full = render(history + [{"role": "user", "content": text}])
                if len(full) < context:
                    continue
                fitted = _fit_final_user(history, text, context, render, len(full))
                if fitted is not None:
                    ids, cut = fitted
                    selected = ids, conversation_id, turn, cut, len(text)
                    break
            if selected is None:
                raise ValueError(
                    f"cannot canonically fit ShareGPT context={context}, request={request_index}; use a larger/more varied corpus"
                )
            ids, conversation_id, turn, kept, original = selected
            token_bytes = _token_bytes(ids)
            cohort_tokens.extend(ids)
            prompt_records.append(
                {
                    "request_index": request_index,
                    "token_sha256": _sha(token_bytes),
                    "history_pairs": sources,
                    "final_user": {
                        "conversation_id": conversation_id,
                        "turn": turn,
                        "kept_characters": kept,
                        "original_characters": original,
                    },
                    "canonical_template_applied_after_truncation": True,
                }
            )
            if (request_index + 1) % 32 == 0:
                print(
                    f"[prompt-progress] context={context} requests={request_index + 1}/{cohort_count}",
                    flush=True,
                )
        data = _token_bytes(cohort_tokens)
        filename = f"tokens-{context}.u32le"
        entries[filename] = data
        manifest["contexts"][str(context)] = {
            "file": filename,
            "count": cohort_count,
            "sha256": _sha(data),
            "records": prompt_records,
        }
        print(
            f"[prompt-artifact] context={context} requests={cohort_count} token_bytes={len(data)}",
            flush=True,
        )
    entries["manifest.json"] = _json_bytes(manifest)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for filename, data in sorted(entries.items()):
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=6)
    return load_prompt_artifact(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--source-metadata", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--contexts", default="2048,4096,8192,16384,32768,65536")
    prepare.add_argument("--count", type=int, default=256)
    prepare.add_argument(
        "--context-counts", help="optional exact cohorts as context:count,..."
    )
    prepare.add_argument("--seed", type=int, default=41)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--context", type=int)
    inspect.add_argument("--batch", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.revision, trust_remote_code=False
        )
        artifact = prepare_prompt_artifact(
            args.corpus,
            args.output,
            tokenizer=tokenizer,
            tokenizer_identity={
                "repo_id": args.tokenizer,
                "revision": args.revision,
                "checkpoint_revision": args.revision,
                "tokenizer_backend": tokenizer.__class__.__name__,
            },
            source_metadata=json.loads(args.source_metadata.read_text()),
            contexts=tuple(int(item) for item in args.contexts.split(",")),
            count=(
                {
                    int(pair.split(":")[0]): int(pair.split(":")[1])
                    for pair in args.context_counts.split(",")
                }
                if args.context_counts
                else args.count
            ),
            seed=args.seed,
        )
    else:
        artifact = load_prompt_artifact(args.path)
        if args.context is not None:
            artifact.cohort(args.context, args.batch)
    print(
        json.dumps(
            {"path": str(artifact.path), "metadata": artifact.metadata},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
