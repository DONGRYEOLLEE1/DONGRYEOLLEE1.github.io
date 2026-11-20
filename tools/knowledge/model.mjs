import { baseUrl, bodyExcerpt, excerpt, secret, send } from './http.mjs';

const OPENROUTER = 'https://openrouter.ai/api/v1';

const object = (properties) => ({
  type: 'object',
  properties,
  required: Object.keys(properties),
  additionalProperties: false,
});
const integer = { type: 'integer', minimum: 0 };
function responseSchema(input) {
  if (input.concepts) {
    return object({
      facts: {
        type: 'array',
        maxItems: 6,
        items: object({
          conceptId: { type: 'string', enum: input.concepts.map((c) => c[0]) },
          role: { type: 'string', enum: ['about', 'applies'] },
          block: { ...integer, maximum: input.blocks.length - 1 },
        }),
      },
    });
  }
  const possibilities = input.candidates.map((c) =>
    object({
      id: { type: 'string', enum: [c.id] },
      relation: { type: 'string', enum: c.relationOptions },
      reason: { type: 'string' },
      sourceBlock: { ...integer, maximum: c.sourceBlocks.length - 1 },
      targetBlock: { ...integer, maximum: c.targetBlocks.length - 1 },
    }),
  );
  return object({ recommendations: { type: 'array', maxItems: 4, items: { anyOf: possibilities } } });
}

// An OpenAI-compatible chat endpoint is sufficient; no tools or network access are granted to the model.
export function chatProvider(env = process.env, { fetch: request = fetch, requireKey = true } = {}) {
  const endpoint = baseUrl('RECOMMENDATIONS_API_BASE', env.RECOMMENDATIONS_API_BASE || OPENROUTER);
  const openrouter = endpoint === OPENROUTER;
  // The OpenRouter key is never sent to another host.
  const apiKey =
    secret('RECOMMENDATIONS_API_KEY', env.RECOMMENDATIONS_API_KEY) ||
    (openrouter ? secret('OPENROUTER_API_KEY', env.OPENROUTER_API_KEY) : undefined);
  if (requireKey && !apiKey && openrouter) throw new Error('Set OPENROUTER_API_KEY to sync recommendations.');
  const model = env.RECOMMENDATIONS_MODEL || 'openai/gpt-6-luna';
  // The default model rejects temperature, so determinism comes from seed and a fixed reasoning effort.
  // The output limit is left out: it decides whether a call fails, not what a successful call returns.
  const settings = openrouter
    ? { seed: 7, reasoning: { effort: env.RECOMMENDATIONS_REASONING_EFFORT || 'low' } }
    : { temperature: 0 };
  const maxTokens = Number(env.RECOMMENDATIONS_MAX_TOKENS) || 8000;
  const usage = { requests: 0, promptTokens: 0, completionTokens: 0 };
  return {
    model,
    settings,
    usage,
    async complete(instruction, input) {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers.Authorization = 'Bearer ' + apiKey;
      usage.requests++;
      const response = await send('Inference', request, endpoint + '/chat/completions', {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(240000),
        body: JSON.stringify({
          model,
          ...settings,
          max_tokens: maxTokens,
          // Without this OpenRouter may route to a host that silently drops seed or the schema.
          ...(openrouter ? { provider: { require_parameters: true } } : {}),
          response_format: {
            type: 'json_schema',
            json_schema: { name: 'judgment', strict: true, schema: responseSchema(input) },
          },
          messages: [
            { role: 'system', content: instruction },
            { role: 'user', content: JSON.stringify(input) },
          ],
        }),
      });
      if (!response.ok)
        throw failure(
          'Inference HTTP ' + response.status,
          response.status,
          await bodyExcerpt(response, [apiKey]),
        );
      let payload;
      try {
        payload = await response.json();
      } catch {
        throw new Error('Inference response is not JSON');
      }
      usage.promptTokens += payload?.usage?.prompt_tokens || 0;
      usage.completionTokens += payload?.usage?.completion_tokens || 0;
      const choice = payload?.choices?.[0];
      // OpenRouter can answer 200 with an error object, for the whole request or for one choice.
      const reported = payload?.error || choice?.error;
      if (reported)
        throw failure(
          'Inference error response',
          reported.code,
          excerpt(typeof reported === 'string' ? reported : (reported.message ?? JSON.stringify(reported)), [
            apiKey,
          ]),
        );
      if (choice?.finish_reason === 'length') throw new Error('Inference output truncated');
      if (typeof choice?.message?.content !== 'string') throw new Error('Inference returned no content');
      try {
        return JSON.parse(choice.message.content);
      } catch {
        throw new Error('Inference output is not valid JSON');
      }
    },
  };
}

// `rejected` marks an answer that says the request itself is unacceptable (key, model, schema,
// parameters): the next article would get the same answer. Rate limits and timeouts are not.
function failure(message, status, detail) {
  const error = new Error(message);
  if (Number.isInteger(status)) error.status = status;
  if (detail) error.detail = detail;
  error.rejected = status >= 400 && status < 500 && status !== 408 && status !== 429;
  return error;
}
