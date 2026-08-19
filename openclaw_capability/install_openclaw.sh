#!/bin/zsh
set -euo pipefail

HERE=${0:A:h}
ROOT=${HERE:h}
PROJECT=${ROOT:h}
PYTHON=${SACRED_HARP_PYTHON:-${PROJECT}/../sacred_harp_finetune_venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=${SACRED_HARP_PYTHON:-$(command -v python3)}
SERVER=${HERE}/sacred_harp_openclaw_server.py
PLUGIN=${HERE}/openclaw-plugin
MODEL_REF=sacred-harp-local/sacred-harp-1b-openclaw
ADAPTER=${SACRED_HARP_ADAPTER:-${ROOT}/adapters/sacred_harp_1b_lora_openclaw_v3_interpretation}
CLICKCLACK_CONTROLLER_ID=${CLICKCLACK_SACREDHARP_CONTROLLER_USER_ID:-usr_01kynetng1rgcetj04pfq3pzew}
OPENCLAW_CONFIG=${OPENCLAW_CONFIG_PATH:-${HOME}/.openclaw/openclaw.json}

[[ -x "$PYTHON" ]] || { print -ru2 "Missing Python runtime: $PYTHON"; exit 1; }
[[ -f "$SERVER" ]] || { print -ru2 "Missing provider server: $SERVER"; exit 1; }
[[ -f "${ADAPTER}/adapters.safetensors" ]] || {
  print -ru2 "The OpenClaw adapter has not been trained yet."
  print -ru2 "Expected: ${ADAPTER}/adapters.safetensors"
  exit 1
}

# OpenClaw 2026.8.1's global `plugins install` path lacks an agent selector, so
# register this trusted local plugin through the supported load-path config.
PLUGIN_ALLOW=$(jq -c --arg id "sacred-harp-openclaw" \
  '((.plugins.allow // []) + [$id]) | unique' "$OPENCLAW_CONFIG")
PLUGIN_PATHS=$(jq -c --arg path "$PLUGIN" \
  '((.plugins.load.paths // []) + [$path]) | unique' "$OPENCLAW_CONFIG")
openclaw config set 'plugins.allow' "$PLUGIN_ALLOW" --strict-json --replace
openclaw config set 'plugins.load.paths' "$PLUGIN_PATHS" --strict-json --replace

PROVIDER=$(jq -nc \
  --arg python "$PYTHON" \
  --arg server "$SERVER" \
  --arg cwd "$PROJECT" \
  '{
    baseUrl: "http://127.0.0.1:18991/v1",
    apiKey: "local-sacred-harp",
    api: "openai-completions",
    localService: {
      command: $python,
      args: [$server, "--host", "127.0.0.1", "--port", "18991"],
      cwd: $cwd,
      healthUrl: "http://127.0.0.1:18991/health",
      readyTimeoutMs: 120000,
      idleStopMs: 900000
    },
    models: [{
      id: "sacred-harp-1b-openclaw",
      name: "Sacred Harp 1B OpenClaw Specialist",
      reasoning: false,
      input: ["text"],
      cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0},
      contextWindow: 8192,
      contextTokens: 7168,
      maxTokens: 512,
      compat: {supportsTools: true, requiresToolResultName: true}
    }]
  }')

openclaw config set 'models.providers["sacred-harp-local"]' "$PROVIDER" --strict-json
openclaw config set 'plugins.entries["sacred-harp-openclaw"].enabled' 'true' --strict-json
openclaw config set 'plugins.entries["sacred-harp-openclaw"].config' \
  '{"baseUrl":"http://127.0.0.1:18991"}' --strict-json
openclaw config set 'agents.entries.sacredharpbench.model' "\"${MODEL_REF}\"" --strict-json
openclaw config set 'agents.entries.sacredharpbench.models' \
  "{\"${MODEL_REF}\":{}}" --strict-json --replace
# This agent already has a three-tool allowlist. Lean mode would hide those
# simple tools behind tool_search/tool_call indirection that a 1B model does
# not need and handles less reliably.
openclaw config set 'agents.entries.sacredharpbench.experimental.localModelLean' 'false' --strict-json
openclaw config set 'agents.entries.sacredharpbench.skills' '[]' --strict-json --replace
openclaw config set 'agents.entries.sacredharpbench.tools.allow' \
  '["sacred_harp_search","discussion","message"]' --strict-json --replace
# Sandbox tool policy is intersected with the agent allowlist. Keep the same
# narrow host-tool surface or OpenClaw will remove every permitted tool.
openclaw config set 'agents.entries.sacredharpbench.tools.sandbox.tools.allow' \
  '["sacred_harp_search","discussion","message"]' --strict-json --replace
openclaw config set 'agents.entries.sacredharpbench.tools.deny' \
  '["read","write","edit","apply_patch","exec","process","browser","canvas","cron","gateway","nodes","sessions_spawn","sessions_send","sessions_list","sessions_history","web_search","web_fetch","memory_search","memory_get","wiki_search","wiki_get","image_generate","music_generate","video_generate","skill_workshop","dashboard"]' \
  --strict-json --replace
openclaw agents bind --agent sacredharpbench --bind clickclack:sacredharpbench --json
openclaw config set 'channels.clickclack.accounts.sacredharpbench.allowBots' \
  '"mentions"' --strict-json
CLICKCLACK_ALLOW_FROM=$(jq -c --arg id "$CLICKCLACK_CONTROLLER_ID" \
  '((.channels.clickclack.accounts.sacredharpbench.allowFrom // ["*"]) + ["*", $id]) | unique' \
  "$OPENCLAW_CONFIG")
openclaw config set 'channels.clickclack.accounts.sacredharpbench.allowFrom' \
  "$CLICKCLACK_ALLOW_FROM" --strict-json --replace

print "Installed ${MODEL_REF} and bound ClickClack account sacredharpbench."
print "Bot-authored turns require @sacredharpbench and allow controller ${CLICKCLACK_CONTROLLER_ID}."
print "Restart the Gateway so the linked tool plugin is loaded, then run verify_openclaw.sh."
