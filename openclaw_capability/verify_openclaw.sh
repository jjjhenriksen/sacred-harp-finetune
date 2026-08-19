#!/bin/zsh
set -euo pipefail

HERE=${0:A:h}
ROOT=${HERE:h}
PROJECT=${ROOT:h}
PYTHON=${SACRED_HARP_PYTHON:-${PROJECT}/../sacred_harp_finetune_venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=${SACRED_HARP_PYTHON:-$(command -v python3)}
SERVER=${HERE}/sacred_harp_openclaw_server.py
REPORTS=${HERE}/reports
CLICKCLACK_CONTROLLER_ID=${CLICKCLACK_SACREDHARP_CONTROLLER_USER_ID:-usr_01kynetng1rgcetj04pfq3pzew}
OPENCLAW_CONFIG=${OPENCLAW_CONFIG_PATH:-${HOME}/.openclaw/openclaw.json}
mkdir -p "$REPORTS"

SERVER_PID=""
if ! curl --fail --silent http://127.0.0.1:18991/health >/dev/null 2>&1; then
  "$PYTHON" "$SERVER" --host 127.0.0.1 --port 18991 >"${REPORTS}/provider.log" 2>&1 &
  SERVER_PID=$!
  trap '[[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" >/dev/null 2>&1 || true' EXIT
  for _ in {1..120}; do
    curl --fail --silent http://127.0.0.1:18991/health >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl --fail --silent --show-error http://127.0.0.1:18991/health >"${REPORTS}/provider_health.json"

jq -nc '{query:"What is the third verse to SH 130, The Old Graveyard?",top_k:5}' |
  curl --fail --silent --show-error -H 'content-type: application/json' --data-binary @- \
    http://127.0.0.1:18991/sacred-harp/search >"${REPORTS}/provider_rag_smoke.json"

SESSION_KEY="agent:sacredharpbench:sacred-harp-verification-$(date +%s)"
openclaw agent \
  --agent sacredharpbench \
  --session-key "$SESSION_KEY" \
  --message "What is the third verse associated with SH 130, The Old Graveyard? Use your Sacred Harp corpus." \
  --json >"${REPORTS}/openclaw_agent_rag_smoke.json"

jq -e '
  .status == "ok" and
  .result.meta.agentMeta.provider == "sacred-harp-local" and
  .result.meta.agentMeta.model == "sacred-harp-1b-openclaw" and
  (.result.meta.agentMeta.terminalReceipt.successfulToolNames | index("sacred_harp_search") != null) and
  .result.meta.toolSummary.failures == 0
' "${REPORTS}/openclaw_agent_rag_smoke.json" >/dev/null

openclaw channels status --channel clickclack --probe --json >"${REPORTS}/clickclack_status.json"
jq -e --arg controller "$CLICKCLACK_CONTROLLER_ID" '
    .channelAccounts.clickclack[] |
    select(.accountId == "sacredharpbench") |
    .configured == true and .running == true and .connected == true and
    .lifecycle == "ready" and .lastError == null
  ' "${REPORTS}/clickclack_status.json" >/dev/null

jq -e --arg controller "$CLICKCLACK_CONTROLLER_ID" '
  .agents.entries.sacredharpbench.experimental.localModelLean == false and
  .agents.entries.sacredharpbench.tools.allow == ["sacred_harp_search", "discussion", "message"] and
  .agents.entries.sacredharpbench.tools.sandbox.tools.allow == ["sacred_harp_search", "discussion", "message"] and
  .channels.clickclack.accounts.sacredharpbench.allowBots == "mentions" and
  (.channels.clickclack.accounts.sacredharpbench.allowFrom | index("*") != null) and
  (.channels.clickclack.accounts.sacredharpbench.allowFrom | index($controller) != null) and
  any(.bindings[];
    .agentId == "sacredharpbench" and
    .match.channel == "clickclack" and
    .match.accountId == "sacredharpbench")
' "$OPENCLAW_CONFIG" >/dev/null

jq -n \
  --arg provider "$(jq -r '.result.meta.agentMeta.provider' "${REPORTS}/openclaw_agent_rag_smoke.json")" \
  --arg model "$(jq -r '.result.meta.agentMeta.model' "${REPORTS}/openclaw_agent_rag_smoke.json")" \
  --arg tool "$(jq -r '.result.meta.agentMeta.terminalReceipt.successfulToolNames[0]' "${REPORTS}/openclaw_agent_rag_smoke.json")" \
  '{verified:true, provider:$provider, model:$model, successfulTool:$tool, clickclackAccount:"sacredharpbench"}'
