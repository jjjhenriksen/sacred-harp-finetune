# Sacred Harp 1B: ClickClack/OpenClaw evaluation

## Decision

Use `sacred_harp_1b_lora_openclaw_v2` behind the ClickClack-first harness.
The remaining raw-model failures are mostly tool-interface and routing problems,
not evidence that the Sacred Harp specialist needs a larger model.

## Held-out results

| System | Overall | Tool routes | Search | Discussion | Message | Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Sacred Harp adapter | 33/79 (41.8%) | 2/42 (4.8%) | 2/34 | 0/4 | 0/4 | 31/37 |
| OpenClaw v1 | 61/79 (77.2%) | 30/42 (71.4%) | 28/34 | 1/4 | 1/4 | 31/37 |
| ClickClack correction v2 | 63/79 (79.7%) | 32/42 (76.2%) | 28/34 | 2/4 | 2/4 | 31/37 |
| v2 plus production harness | 78/79 (98.7%) | 41/42 (97.6%) | 34/34 | 3/4 | 4/4 | 37/37 |

The strict evaluator compares requested argument tokens, not merely whether a
call is usable. The sole harness miss called `discussion(limit=30)` for a
"short recap" where the fixture expected 25. That is operationally valid.

The raw v2 model selected `discussion` on 4/4 and `message` on 4/4 unseen
ClickClack prompts. Their lower strict scores came from argument differences.
It selected `sacred_harp_search` on 31/34 factual prompts; 28/34 also met the
strict query-overlap threshold. It answered all 30 grounded-answer cases
without another tool call, but over-called tools on all six ordinary-chat
cases when all tools remained visible.

## Catastrophic-forgetting check

Lower loss is better on the untouched 396-example Sacred Harp test split.

| Adapter | Sacred Harp test loss | Perplexity |
| --- | ---: | ---: |
| Original Sacred Harp adapter | 3.574 | 35.673 |
| OpenClaw v1 | 3.516 | 33.665 |
| ClickClack correction v2 | 3.511 | 33.490 |

There is no measured catastrophic forgetting. Both continued adapters slightly
improved the original held-out loss.

## Failure attribution

### Training data

Training coverage caused part of the ClickClack weakness. The first adapter had
few discussion/send phrasings. A small correction set doubled strict unseen
discussion and message scores from 25% to 50% without changing model size or
hurting Sacred Harp performance. More varied interaction phrasing is useful,
but further repetitions alone have diminishing value now that tool choice is
already 8/8 on those unseen intents.

### Harness and tool-interface design

This is the dominant remaining cause. OpenClaw's native `message` tool is a
broad, flat schema containing fields this agent does not need. The 1B model
sometimes invented account, reply, thread, and destination values or put the
message in the wrong field. Keeping all tools visible also caused ordinary
conversation to become unnecessary search calls. Short follow-ups sometimes
lost the song subject even though the correct search tool was selected.

The production adapter therefore:

- classifies obvious ClickClack read, send, and Sacred Harp lookup intents;
- issues the single deterministic tool call instead of asking the model to
  rediscover routing and JSON formatting;
- reduces `message` to action, text, and an optional normalized ClickClack
  channel target;
- carries the previous Sacred Harp subject into pronoun/short follow-ups;
- clamps discussion/search limits and drops unsupported arguments;
- removes tools after a result so the model must write a grounded answer;
- leaves ordinary ClickClack replies as automatic visible replies.

This raised the same v2 checkpoint from 79.7% raw to 98.7% with no parameter
increase. It is the strongest evidence that routing and interface design, not
model scale, is the main lever.

### Model capability

The model remains weak at broad conversation, zero-context instruction
following, and unconstrained synthesis. It can still confidently answer from
memorized Sacred Harp patterns when retrieval is omitted. Those are genuine
1B limitations, but they are outside the intended role. Its production job is
to phrase a response from bounded evidence and maintain a small amount of
conversation state, not to act as a general planner.

## Recommended next harness changes

1. Replace the native broad `message` schema with ClickClack-owned wrappers such
   as `clickclack_reply(text)` and `clickclack_send(channel, text)`.
2. Inject a bounded current-discussion excerpt before the model turn when a
   ClickClack event already supplies it; do not make the model request context
   it can receive deterministically.
3. Return structured Sacred Harp fields separately from prose: book family,
   edition, number, tune, text family, verse, meter, key, composer, lyricist,
   and source paths. Render exact lyrics and exact verse joins without model
   rewriting.
4. Maintain an explicit per-session Sacred Harp subject record instead of
   relying only on transcript inference.
5. Validate or repair one tool call once, then fail closed with a short visible
   explanation. Never expose repeated schema-error walls to the 1B model.
6. Evaluate ClickClack workflows end to end: inbound account binding, bounded
   discussion context, retrieval, visible reply, explicit channel send, and a
   two-turn follow-up. Keep general OpenClaw tasks out of the acceptance gate.

## Current local integration state

The custom provider, local-service command, restricted `sacredharpbench` agent,
local retrieval plugin path, and account-specific
`clickclack:sacredharpbench` binding are present in OpenClaw configuration.
Direct provider checks passed for health, OpenAI-compatible tool calls, grounded
post-tool generation, and the SH 130 cross-witness verse lookup.

The OpenClaw rejection was confirmed as three harness defects rather than a
model failure:

- the embedded runner discarded the explicit owner while resolving inherited
  authentication state;
- the agent and sandbox tool allowlists were intersected, removing every tool;
- `localModelLean` hid the remaining narrow tools behind meta-tool indirection.

The local OpenClaw source now preserves the selected owner for inherited auth
and ownerless plugin inventory, while the installation recipe keeps identical
host/sandbox allowlists and disables lean indirection for this three-tool
specialist. A gateway-owned verification turn passed with provider
`sacred-harp-local`, model `sacred-harp-1b-openclaw`, one successful
`sacred_harp_search` call, zero tool failures, and the correct SH 130 / Boast Ye
Not cross-witness verse. The ClickClack account is configured, running, and
bound specifically to `sacredharpbench`.

The installed ClickClack npm beta is older than the 2026.8.1 gateway. It works
for the current account and binding, but its status adapter reports
`lifecycle: starting` even while `running: true`; the newer unpublished local
source fixes that status reporting. This version drift should not be confused
with the former ownership rejection.
