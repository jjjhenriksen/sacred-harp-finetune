# Sacred Harp OpenClaw capability layer

This continues the trained 1B Sacred Harp LoRA instead of replacing it. The
new adapter rehearses the original corpus while learning a deliberately tiny
OpenClaw surface:

- `sacred_harp_search` for grounded corpus and Obsidian-crosslink retrieval;
- `discussion` for the ClickClack discussion bound to the current session;
- `message(action="send")` for explicit ClickClack delivery.

It is not trained as a general computer agent. The configured OpenClaw agent
denies shell, filesystem, web, scheduler, gateway, and subagent tools.

## Train and evaluate

```zsh
# Run from the repository root.
./openclaw_capability/train_openclaw_capability.sh
```

The source adapter is
`adapters/sacred_harp_1b_lora_vault`. The accepted,
ClickClack-corrected adapter is
`adapters/sacred_harp_1b_lora_openclaw_v2`. The current
interpretation-capable adapter continues from that checkpoint at
`adapters/sacred_harp_1b_lora_openclaw_v3_interpretation`.

Capability evaluation:

```zsh
python3 \
  openclaw_capability/evaluate_openclaw_capability.py \
  --adapter adapters/sacred_harp_1b_lora_openclaw_v2 \
  --harness-gating
```

The interpretation correction run is reproducible with:

```zsh
zsh openclaw_capability/train_interpretation_correction.sh
```

Meaning/theme turns receive bounded thematic cues rather than a copyable full
lyric. Exact lyric requests remain verbatim, and shared-text questions such as
“What tunes share the text from 47b?” are resolved from the structured text
family index.

Catastrophic-forgetting evaluation uses the untouched 396-example Sacred Harp
test split and compares its full loss before and after continued training.
See `OPENCLAW_EVALUATION_REPORT.md` for the raw-model/harness comparison and
failure attribution.

## OpenClaw integration

`sacred_harp_openclaw_server.py` is an OpenAI Chat Completions-compatible local
provider. It converts Llama 3.2's JSON function-call output into native
`tool_calls`, so OpenClaw—not a parallel agent loop—executes tools and returns
their results. The linked plugin exposes the existing Sacred Harp RAG as a
normal OpenClaw tool. Because ClickClack is the primary control surface, the
provider exposes only the one tool relevant to each turn, repairs broad
`message` arguments into the minimal ClickClack send contract, and suppresses
tools after retrieval. It removes ClickClack/OpenClaw transport context before
the 1B model sees the turn and bounds retrieval evidence to a small prompt.
Exact verse lookups and named-tune lyric requests are rendered verbatim from
the matching witness (preferring `sh2025`, then `sh1991`) instead of asking the
small model to copy a long multi-edition text without error.

## Run one grounded question

Use the OpenClaw agent for questions that must be grounded in the Sacred Harp
corpus. From the repository root:

```zsh
# Run from the repository root.
openclaw agent \
  --agent sacredharpbench \
  --session-key presentation-question \
  --message "What are the lyrics to Idumea? Use the Sacred Harp corpus."
```

This path requires the Sacred Harp provider and its local RAG service. If the
provider is not already running, start it in a separate terminal:

```zsh
python3 \
  openclaw_capability/sacred_harp_openclaw_server.py \
  --host 127.0.0.1 --port 18991
```

For the complete local preflight, which starts the provider when necessary and
checks model selection, tool use, configuration, and ClickClack account health:

```zsh
# Run from the repository root.
cd openclaw_capability
./verify_openclaw.sh
```

The preflight is not a substitute for a live ClickClack round-trip. Before a
presentation, send one fresh factual question in the target ClickClack
workspace and confirm that the visible answer contains corpus evidence.

## ClickClack presentation path

The Sacred Harp bot belongs in the `OpenClaw Web Scout` workspace. In its
`#general` channel, address the bot by the handle shown in that workspace
(currently `@sacredharpbench-scout`). The account is configured for both
`#general` and `#demonstration`, with mention-gating enabled; use `#general`
for testing and leave `#demonstration` unchanged unless a presentation
requires it.

Do not use `./evaluate_small.sh` for the presentation. That command runs only
the raw MLX model and bypasses retrieval, OpenClaw tool execution, ClickClack
routing, and grounded answer rendering.

After the accepted adapter exists:

```zsh
./openclaw_capability/install_openclaw.sh
```

That registers `sacred-harp-local/sacred-harp-1b-openclaw`, reuses the existing
`sacredharpbench` agent and ClickClack account, narrows its tool policy, and
adds an account-specific ClickClack binding. A Gateway restart is required for
the newly linked plugin. Bot-authored control messages must explicitly mention
the bot handle configured in the target workspace (currently
`@sacredharpbench-scout`); the installer preserves human access and allowlists the
local Blackbird controller ID. Override it with
`CLICKCLACK_SACREDHARP_CONTROLLER_USER_ID` on another installation. Then run
`verify_openclaw.sh`.

The verifier fails closed unless the Gateway reports the fine-tuned provider
and model, a successful `sacred_harp_search` call, matching host/sandbox tool
allowlists, and a running ClickClack account with the expected binding. This
catches the three integration failures that previously looked like a model
rejection: missing explicit ownership, intersecting tool policies that removed
every tool, and lean-mode meta-tool indirection.


Checkpoint selection stages a copy on the adapter's filesystem, verifies its
size and SHA-256, and atomically replaces `adapters.safetensors`. The selection
receipt directory is created as needed; the receipt is also staged and replaced
atomically, and records the adapter size and checksum. If receipt promotion
fails after adapter promotion, the complete new adapter and previous receipt
remain; rerun selection to publish the matching receipt.

Portable publication regressions run from the repository root without loading
MLX, training a model, or contacting a provider:

```sh
python3 -m unittest discover -s tests -v
```

### HTTP request limits

POST requests to `/sacred-harp/search` and `/v1/chat/completions` require one
nonnegative `Content-Length`, a UTF-8 JSON object, and at most 1 MiB of body
data. Reads use chunks of at most 64 KiB and a five-second total body deadline;
chunked transfer encoding is unsupported. Oversized bodies return 413; invalid
framing, incomplete uploads, malformed JSON, and invalid fields return 400
with `error.type = invalid_request_error`, before retrieval or generation.
Search requires a nonempty string `query` and integer `top_k` from 1 through 5.
Completions require nonempty message objects with supported roles; token limits
must be positive integers, temperature must be finite and between 0 and 2, and
`stream` must be a boolean. Valid tool history and streamed responses retain
their existing protocol. These checks are covered by loopback HTTP fixtures
with a fake runtime; they do not establish live model or RAG readiness.
