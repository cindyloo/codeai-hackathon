# Claude prompt endpoint

`POST /api/llm/generate` runs one prompt against one Claude model. It exists so
the pose and rigging teams can try prompts against the same API key the
deployed app uses, without each building their own harness.

It spends real money on a real Anthropic account, so it is closed by default and every
layer fails closed.

## Enabling it

Both are required; either one missing leaves the endpoint unusable.

```bash
LLM_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
CLAUDE_ALLOWED_MODELS="claude-opus-5-5,claude-sonnet-5-5,claude-haiku-4-5"
ANTHROPIC_API_KEY=sk-ant-...
```

`ANTHROPIC_API_KEY` is handed straight to the `anthropic` SDK; nothing else
touches it. Model IDs are the bare Anthropic IDs listed at
<https://docs.anthropic.com/en/docs/about-claude/models>.

On Heroku: `heroku config:set LLM_API_TOKEN=... CLAUDE_ALLOWED_MODELS=... ANTHROPIC_API_KEY=...`

## Using it

```bash
curl -s https://YOUR_APP/api/llm/generate \
  -H "Authorization: Bearer $LLM_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "model_id": "claude-sonnet-5-5",
        "prompt": "Describe a stick figure waving.",
        "system": "You reply in one sentence.",
        "max_tokens": 200
      }'
```

```json
{
  "model_id": "claude-sonnet-5-5",
  "text": "...",
  "stop_reason": "end_turn",
  "usage": { "input_tokens": 21, "output_tokens": 48, "total_tokens": 69 }
}
```

| Field | Required | Notes |
|---|---|---|
| `model_id` | yes | Must be in `CLAUDE_ALLOWED_MODELS` |
| `prompt` | yes | Up to `MAX_PROMPT_CHARS` (20k default) |
| `system` | no | System prompt |
| `max_tokens` | no | Default 1024, capped at `CLAUDE_MAX_TOKENS` |
| `temperature` | no | **Omitted unless you send it** — current Claude models reject sampling parameters, so a default would break them |

`GET /api/llm/models` returns the allowlist and current rate-limit
usage. Both routes take the same token.

## How it is protected

| Layer | Behaviour |
|---|---|
| No `LLM_API_TOKEN` set | Both routes return **404** |
| Missing or wrong token | **404**, identical to a route that doesn't exist |
| Token comparison | `secrets.compare_digest`, on every path including the missing-header one |
| Empty `CLAUDE_ALLOWED_MODELS` | Every model refused |
| Model not on the allowlist | Refused before any API call — exact match, not a prefix |
| Per-caller rate limit | `LLM_RATE_PER_MINUTE` (10), bucketed by token **and** client address |
| Whole-deployment cap | `LLM_RATE_PER_DAY` (500) — this is the one that bounds the bill |
| Prompt size | Rejected before any AWS call |
| Anthropic API errors | Translated to generic messages; the original goes to the server log |

**404 rather than 401** is deliberate: an unauthenticated caller cannot tell the
endpoint from a typo'd URL, so a scanner finds nothing to come back to. The
tradeoff is that a developer with a stale token also sees a 404 — the server log
distinguishes the two (`[llm] refused: bad or missing token from ...`).

**Unauthenticated requests consume no quota.** The token check runs before the
rate limiter, so nobody can exhaust a real caller's allowance without the token.

**Rate limiting is per-process and in memory.** It resets on restart and does not
coordinate across dynos. That is sound while the app runs on one dyno (which the
in-memory store already requires), and would need Redis if that ever changes.

## The token is not a browser secret

The frontend does not call this endpoint and must not. Anything shipped to the
browser is public — put this token in JavaScript and you have published an open
LLM proxy. Callers are developers with curl, or server-side code.

If a browser feature ever needs model output, the right shape is a
purpose-specific endpoint (like `/api/avatars/<id>/poses`) that takes a
constrained input and calls the model server-side, not a general prompt pipe.

## `stop_reason`

Passed through from the Messages API. `"refusal"` means a safety classifier
declined the prompt; `text` is then empty or partial. `"max_tokens"` means the
reply hit the cap.

## Rotating the token

```bash
heroku config:set LLM_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

The dyno restarts and every old token stops working immediately. There is no
revocation list because there is one token — if you need per-person revocation,
that is the point at which this should become real credentials rather than a
shared secret.

## Troubleshooting

| Symptom | Cause |
|---|---|
| 404 with a token you believe is right | Token mismatch or `LLM_API_TOKEN` unset — check the server log |
| 503 "No models are enabled" | `CLAUDE_ALLOWED_MODELS` is empty |
| 503 "isn't configured" | `ANTHROPIC_API_KEY` is unset |
| 400 "wasn't valid for this model" | Usually `temperature` sent to a model that rejects it, or `max_tokens` above the model's limit |
| 404 "That model isn't available" | Typo in the model ID, or a model this key's organization can't use |
| 503 "API key is invalid" | `ANTHROPIC_API_KEY` is wrong or revoked |
