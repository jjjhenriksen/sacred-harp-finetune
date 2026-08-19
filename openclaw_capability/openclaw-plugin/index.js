import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { jsonResult } from "openclaw/plugin-sdk/tool-results";

const PARAMETERS = {
  type: "object",
  properties: {
    query: {
      type: "string",
      description: "A self-contained Sacred Harp question or lookup query.",
    },
    top_k: {
      type: "integer",
      minimum: 1,
      maximum: 5,
      description: "Maximum corpus results; default 5.",
    },
  },
  required: ["query"],
  additionalProperties: false,
};

export default definePluginEntry({
  id: "sacred-harp-openclaw",
  name: "Sacred Harp OpenClaw",
  description: "Grounded retrieval from the local Sacred Harp corpus and Obsidian crosslinks.",
  register(api) {
    api.registerTool(
      {
        name: "sacred_harp_search",
        label: "Sacred Harp Search",
        description:
          "Search the local Sacred Harp corpus and Obsidian crosslinks. Use before answering any factual Sacred Harp question, including tune identity, lyrics, meter, key, composer, lyricist, edition, or song number.",
        parameters: PARAMETERS,
        async execute(_toolCallId, params, signal) {
          const pluginConfig = api.pluginConfig ?? {};
          const baseUrl = String(pluginConfig.baseUrl ?? "http://127.0.0.1:18991").replace(/\/$/, "");
          const response = await fetch(`${baseUrl}/sacred-harp/search`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ query: params.query, top_k: params.top_k ?? 5 }),
            signal,
          });
          if (!response.ok) {
            const detail = await response.text();
            throw new Error(`Sacred Harp retrieval failed (${response.status}): ${detail}`);
          }
          return jsonResult(await response.json());
        },
      },
      { name: "sacred_harp_search" },
    );
  },
});
