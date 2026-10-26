Run `uv run pytest tests/recipe_fuzz -q` for 40 reproducible examples per style,
targeting roughly one second of fuzz execution per model. Interpreter imports
and one-time reference setup are separate; the complete cold suite takes about
ten seconds on the development Mac. Increase coverage with
`RESIN_FUZZ_EXAMPLES=10000 uv run pytest tests/recipe_fuzz -q`. Hypothesis shrinks
failures; retain minimized regressions as ordinary tests when fixing a bug.

Two styles share one driver: structured valid responses checked against format
oracles (plus prompt-rendering parity using those same generated values), and
arbitrary bytes compared with complete tokenizer decoding. Every case exercises
whole input, individual tokens, empty feeds, generated boundaries and idempotent
finalization. Truncated tools must either remain uncommitted or produce the
same partial events across chunk schedules, matching each format's contract.

| Format | Prompt oracle | Response oracle |
| --- | --- | --- |
| GPT-OSS | Harmony | Harmony; TML secondary |
| Inkling | pinned HF Jinja | generated schema values; TML v0 secondary |
| Qwen3.5/3.8 | pinned HF Jinja | generated schema values; TML secondary |
| Ornith | pinned HF Jinja | generated schema values; TML secondary |
| Gemma4 | pinned HF Jinja | HF argument macro plus semantic round trip |
| LFM2.5 | pinned HF Jinja | Python literal AST |

Gemma has no public reference output parser. Its test is a reference serializer
round trip plus independent chunk/decoder checks, not a claim of equivalence to
a nonexistent parser. Qwen's reference values are projected onto its XML grammar.
LFM's Python-literal rendering oracle uses Python 3.13 / Unicode 15.1.

Primary comparisons are exact. TML 0.2.0's intermediate JSON float conversion
can round numbers by one ULP (for example `3.777859190353473e-35`); only secondary
TML numeric comparisons allow that one-ULP difference. Primary comparisons and
native numeric regressions remain exact. Oracle exceptions fail tests.

Adapters project shared generated parts onto each model's legal ordering and
parameter-name grammar. Nested values, Unicode, marker-looking ordinary text,
malformed UTF-8 and tool truncation remain covered. Comparison preserves part
order, joins adjacent deltas, compares JSON values, and ignores generated tool
IDs and JSON spacing. Explicit unit tests cover malformed/out-of-domain grammar.

All reference libraries/templates are dev-only. See `references/README.md` and
`templates/README.md` for pinned sources and licenses. Larger model-specific
fixtures test template options without adding custom fuzz engines per model.
