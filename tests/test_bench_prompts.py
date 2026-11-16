import array
import hashlib
import json
import zipfile

import pytest

from resin.bench_prompts import load_prompt_artifact


@pytest.mark.parametrize("source", [None, "published-tetris"])
def test_saved_prompt_preserves_literal_tokens_and_source_label(tmp_path, source):
    tokens = array.array("I", [1, 2, 3]).tobytes()
    entries = {
        "corpus.json": b'[{"role":"user","content":"literal prompt"}]',
        "chat_template.jinja": b"{{ messages }}",
        "tokenizer.json": b"{}",
        "tokens.u32le": tokens,
    }

    def digest(value):
        return hashlib.sha256(value).hexdigest()

    manifest = {
        "format": "resin-chat-prompts-v1",
        "corpus": {"normalized_sha256": digest(entries["corpus.json"])},
        "chat_template": {"sha256": digest(entries["chat_template.jinja"])},
        "tokenizer": {"sha256": digest(entries["tokenizer.json"]), "vocab_size": 10},
        "contexts": {
            "3": {
                "file": "tokens.u32le",
                "count": 1,
                "sha256": digest(tokens),
                "records": [{"token_sha256": digest(tokens)}],
            }
        },
    }
    if source is not None:
        manifest["source"] = source
    path = tmp_path / "prompts.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest))
    artifact = load_prompt_artifact(path)
    assert artifact.cohort(3, 1) == [[1, 2, 3]]
    assert artifact.metadata["prompt_source_id"].startswith(
        f"chat-templated-{source or 'sharegpt'}:"
    )
    with pytest.raises(ValueError, match="no complete cohort"):
        artifact.cohort(6, 1)
