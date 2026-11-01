"""Add property types, protocol bases, and imports absent from native stubs."""

import ast
from pathlib import Path

STUB_PATH = Path(__file__).resolve().parents[1] / "resin" / "_resin.pyi"
OUTPUT_COLUMNS = {
    "token_ids": "npt.NDArray[np.uint32]",
    "token_indptr": "npt.NDArray[np.uint32]",
    "logprobs": "npt.NDArray[np.float32] | None",
    "logprob_indptr": "npt.NDArray[np.uint32] | None",
    "top_indptr": "npt.NDArray[np.uint32] | None",
    "top_token_ids": "npt.NDArray[np.uint32] | None",
    "top_logprobs": "npt.NDArray[np.float32] | None",
    "prompt_logprobs": "npt.NDArray[np.float32] | None",
    "prompt_logprob_indptr": "npt.NDArray[np.uint32] | None",
    **{
        f"prompt_{kind}_{name}": f"npt.NDArray[np.{dtype}] | None"
        for kind in ("top", "target")
        for name, dtype in (
            ("indptr", "uint32"),
            ("token_ids", "uint32"),
            ("logprobs", "float32"),
        )
    },
}

PROPERTY_TYPES = {
    ("EngineMetrics", "requests"): "dict[str, Any]",
    ("EngineMetrics", "cache"): "dict[str, Any] | None",
    ("RunnerMetrics", "dp_attention_sync_wait_seconds"): "dict[str, Any] | None",
    ("SamplingParams", "stop_token_ids"): "tuple[int, ...]",
    ("DeepSeekV41Image", "types"): "tuple[int, ...]",
    ("DeepSeekV41Image", "patches"): "npt.NDArray[np.float32]",
    ("DeepSeekV41Media", "images"): "tuple[DeepSeekV41Image, ...]",
    **{
        (owner, name): "tuple[int, ...]"
        for owner, extra in (
            ("DeepSeekV41Media", ()),
            (
                "Gemma4Media",
                (
                    "image_positions",
                    "video_positions",
                    "block_ids",
                    "mm_token_type_ids",
                ),
            ),
            ("InklingMedia", ("image_positions",)),
            ("Qwen35Media", ("image_positions",)),
            ("Glm53FlashMedia", ("image_positions",)),
        )
        for name in ("token_ids", "cache_key", *extra)
    },
    **{
        (owner, "token_spans"): "tuple[MediaTokenSpan, ...]"
        for owner in (
            "DeepSeekV41Media",
            "Gemma4Media",
            "InklingMedia",
            "Qwen35Media",
            "Glm53FlashMedia",
        )
    },
    **{
        (owner, name): f"npt.NDArray[np.{dtype}]" + (" | None" if optional else "")
        for owner, name, dtype, optional in (
            ("InklingMedia", "pixel_values", "float32", True),
            ("Qwen35Media", "pixel_values", "float32", False),
            ("Glm53FlashMedia", "pixel_values", "float32", False),
            ("Glm53FlashMedia", "image_grid_thw", "int64", False),
            ("Qwen35Media", "image_grid_thw", "int64", False),
            ("Qwen35Media", "position_ids", "int64", False),
            ("Gemma4Media", "pixel_values", "float32", True),
            ("Gemma4Media", "image_position_ids", "int64", True),
            ("Gemma4Media", "pixel_values_videos", "float32", True),
            ("Gemma4Media", "video_position_ids", "int64", True),
        )
    },
    ("ParserEvent", "token_spans"): "tuple[TokenSpan, ...]",
    ("Message", "content"): "tuple[ContentBlock, ...]",
    ("RenderOptions", "reasoning_effort"): "str | int | float | None",
    ("RenderedPrompt", "media"): "MediaInput | None",
    ("RenderedPrompt", "token_ids"): "npt.NDArray[np.uint32]",
    ("NativeRecipe", "stop_token_ids"): "Sequence[int]",
    ("BatchPlan", "sequence_ids"): "npt.NDArray[np.uint64]",
    **{
        (owner, name): "npt.NDArray[np.uint32]"
        for owner, names in (
            (
                "BatchPlan",
                (
                    "context_lengths",
                    "prefill_lengths",
                    "token_ids",
                    "token_indptr",
                    "positions",
                    "prompt_score_indices",
                    "prompt_score_token_ids",
                    "prompt_score_indptr",
                    "prompt_score_top_counts",
                    "prompt_target_indptr",
                    "prompt_target_token_ids",
                ),
            ),
            (
                "CacheGroupPlan",
                (
                    "block_indptr",
                    "block_slab_ids",
                    "block_slotnums",
                    "block_table_start_indices",
                    "append_slab_ids",
                    "append_slotnums",
                    "append_block_offsets",
                ),
            ),
        )
        for name in names
    },
    ("BatchPlan", "sampling_mask"): "npt.NDArray[np.bool_]",
    ("BatchPlan", "cache_groups"): "tuple[CacheGroupPlan, ...]",
    ("BatchPlan", "sampling_params"): "tuple[SamplingParams, ...]",
    ("BatchPlan", "media"): "tuple[MediaInput | None, ...]",
    ("CacheLayout", "groups"): "tuple[CacheGroupLayout, ...]",
    **{
        ("ForwardOutput", name): annotation
        for name, annotation in OUTPUT_COLUMNS.items()
    },
}


BASES = {
    "NativeRecipe": "Recipe",
    "_NativeResponseParser": "FinalizingResponseParser",
}


class Normalizer(ast.NodeVisitor):
    def __init__(self, source: bytes) -> None:
        self.source = source
        self.line_offsets = [0]
        self.line_offsets.extend(
            index + 1 for index, byte in enumerate(source) if byte == ord("\n")
        )
        self.owner = ""
        self.edits: list[tuple[int, int, bytes]] = []
        self.found_returns: set[tuple[str, str]] = set()
        self.found_bases: set[str] = set()

    def replace(self, node: ast.expr, replacement: str) -> None:
        if (
            node.lineno is None
            or node.col_offset is None
            or node.end_lineno is None
            or node.end_col_offset is None
        ):
            raise ValueError("stub AST node has no source location")
        start = self.line_offsets[node.lineno - 1] + node.col_offset
        end = self.line_offsets[node.end_lineno - 1] + node.end_col_offset
        self.edits.append((start, end, replacement.encode()))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if base := BASES.get(node.name):
            if not node.bases and not node.keywords:
                start = self.line_offsets[node.lineno - 1] + node.col_offset
                header = f"class {node.name}".encode()
                if self.source[start : start + len(header)] != header:
                    raise ValueError(f"generated {node.name} header changed")
                position = start + len(header)
                self.edits.append((position, position, f"({base})".encode()))
            elif (
                len(node.bases) != 1
                or not isinstance(node.bases[0], ast.Name)
                or node.bases[0].id != base
                or node.keywords
            ):
                raise ValueError(f"generated {node.name} has unexpected bases")
            self.found_bases.add(node.name)
        previous, self.owner = self.owner, node.name
        self.generic_visit(node)
        self.owner = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        key = (self.owner, node.name)
        if replacement := PROPERTY_TYPES.get(key):
            if not any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in node.decorator_list
            ):
                raise ValueError(
                    f"generated stub is missing property {self.owner}.{node.name}"
                )
            if node.returns is None:
                raise ValueError(
                    f"generated stub is missing return type {self.owner}.{node.name}"
                )
            self.replace(node.returns, replacement)
            self.found_returns.add(key)

    def validate(self) -> None:
        missing_returns = PROPERTY_TYPES.keys() - self.found_returns
        missing_bases = BASES.keys() - self.found_bases
        if missing_returns or missing_bases:
            raise ValueError(
                "generated stub structure changed "
                f"(properties={sorted(missing_returns)}, "
                f"bases={sorted(missing_bases)})"
            )


def import_edits(
    tree: ast.Module, line_offsets: list[int]
) -> list[tuple[int, int, bytes]]:
    edits: list[tuple[int, int, bytes]] = []
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    incomplete_imports = [
        node
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module == "_typeshed"
    ]
    for node in incomplete_imports:
        if {alias.name for alias in node.names} != {"Incomplete"}:
            raise ValueError("cannot remove a mixed _typeshed import safely")
        assert node.end_lineno is not None
        edits.append(
            (line_offsets[node.lineno - 1], line_offsets[node.end_lineno], b"")
        )

    has_numpy = any(
        isinstance(node, ast.Import)
        and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in imports
    )
    has_numpy_typing = any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "numpy.typing" and alias.asname == "npt"
            for alias in node.names
        )
        for node in imports
    )
    protocol_names = {
        alias.name
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module == "resin.protocols"
        for alias in node.names
    }
    additions = []
    existing_imports = {
        (node.module, alias.name)
        for node in imports
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for module, name in (
        ("resin.cache_storage", "CacheStorage"),
        ("typing", "Literal"),
        ("resin.recipe", "Recipe"),
        ("resin.recipe", "ContentBlock"),
    ):
        if (module, name) not in existing_imports:
            additions.append(f"from {module} import {name}")
    if not has_numpy:
        additions.append("import numpy as np")
    if not has_numpy_typing:
        additions.append("import numpy.typing as npt")
    missing_protocols = {
        "FinalizingResponseParser",
        "MediaInput",
        "MediaTokenSpan",
        "TokenSpan",
        "Runner",
    } - protocol_names
    if missing_protocols:
        additions.append(
            f"from resin.protocols import {', '.join(sorted(missing_protocols))}"
        )
    if additions:
        remaining_imports = [node for node in imports if node not in incomplete_imports]
        if not remaining_imports:
            raise ValueError("generated stub has no import insertion point")
        ends = [node.end_lineno for node in remaining_imports]
        assert all(end is not None for end in ends)
        insert_at = line_offsets[max(end for end in ends if end is not None)]
        edits.append((insert_at, insert_at, ("\n".join(additions) + "\n").encode()))
    return edits


def apply_edits(source: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    previous_start = len(source) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous_start:
            raise ValueError("overlapping stub normalization edits")
        source = source[:start] + replacement + source[end:]
        previous_start = start
    return source


def main() -> None:
    source = STUB_PATH.read_bytes()
    tree = ast.parse(source)
    normalizer = Normalizer(source)
    normalizer.visit(tree)
    normalizer.validate()
    source = apply_edits(
        source,
        normalizer.edits + import_edits(tree, normalizer.line_offsets),
    )
    normalized = ast.parse(source)
    if any(
        isinstance(node, ast.Name) and node.id == "Incomplete"
        for node in ast.walk(normalized)
    ):
        raise ValueError("generated stub contains an unhandled incomplete type")
    STUB_PATH.write_bytes(source)


if __name__ == "__main__":
    main()
