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
export function chatProvider(env = process.env) {
  const endpoint = (env.RECOMMENDATIONS_API_BASE || 'https://openrouter.ai/api/v1').replace(/\/+$/, '');
  const apiKey =
    env.RECOMMENDATIONS_API_KEY ||
    (endpoint === 'https://openrouter.ai/api/v1' ? env.OPENROUTER_API_KEY : undefined);
  if (!apiKey && endpoint === 'https://openrouter.ai/api/v1')
    throw new Error('Set OPENROUTER_API_KEY to sync recommendations.');
  const model = env.RECOMMENDATIONS_MODEL || 'qwen/qwen3.7-flash';
  return {
    model,
    async complete(instruction, input) {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers.Authorization = 'Bearer ' + apiKey;
      const response = await fetch(endpoint + '/chat/completions', {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(240000),
        body: JSON.stringify({
          model,
          temperature: 0,
          max_tokens: 2500,
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
      if (!response.ok) throw new Error('Inference HTTP ' + response.status);
      const payload = await response.json();
      return JSON.parse(payload.choices[0].message.content);
    },
  };
}
