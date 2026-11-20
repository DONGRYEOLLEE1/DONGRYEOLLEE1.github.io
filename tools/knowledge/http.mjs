// Request guards shared by the publishing-server tools. Error text is built here from names and
// codes only: a key or a URL with credentials must never reach a log line.

/** A header-safe secret, or an error that names the variable without printing its value. */
export function secret(name, value) {
  if (!value) return undefined;
  if (!/^[\x21-\x7E]+$/.test(value))
    throw new Error(name + ' contains whitespace, a line break or a non-ASCII character.');
  return value;
}

/** An http(s) base URL without credentials and without a trailing slash. */
export function baseUrl(name, value) {
  const url = URL.canParse(value) ? new URL(value) : null;
  if (!url || !/^https?:$/.test(url.protocol)) throw new Error(name + ' is not an http(s) URL.');
  if (url.username || url.password) throw new Error(name + ' must not contain a user name or password.');
  return value.replace(/\/+$/, '');
}

/** The first 200 characters of a response body, with the caller's secrets masked. */
export function excerpt(text, secrets = []) {
  let detail = String(text ?? '')
    .replace(/\s+/g, ' ')
    .trim();
  for (const value of secrets) if (value) detail = detail.replaceAll(value, '***');
  return detail.slice(0, 200);
}

export async function bodyExcerpt(response, secrets) {
  try {
    return excerpt(await response.text(), secrets);
  } catch {
    return '';
  }
}

/** `message: detail` for a log line. */
export const describe = (error) => error.message + (error.detail ? ': ' + error.detail : '');

/**
 * fetch that reports a transport failure as "<service> request failed (<code>)". The original
 * message is dropped on purpose: undici quotes an invalid header value or URL in full.
 */
export async function send(service, request, url, init) {
  try {
    return await request(url, init);
  } catch (error) {
    const timeout = ['TimeoutError', 'AbortError'].includes(error?.name);
    const code = error?.cause?.code || (timeout ? error.name : null);
    const failure = new Error(service + ' request failed' + (code ? ` (${code})` : ''));
    // A refused or unresolved connection fails every call alike; a timeout may be one slow request.
    failure.unreachable = !timeout;
    throw failure;
  }
}
