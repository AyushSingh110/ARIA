# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Email **k2aserendipity@gmail.com** with:

1. A description of the vulnerability and its impact
2. Steps to reproduce
3. Any suggested fix

You will receive an acknowledgment within 72 hours. Once the issue is confirmed and fixed, you will be credited in the release notes (unless you prefer to remain anonymous).

## Scope notes

- ARIA executes LLM-generated tool calls (web search, calculator, **file read/write**). The file tools are sandboxed to the configured workspace directory — path traversal escapes from that sandbox are in scope and treated as high severity.
- The FastAPI runtime has no built-in authentication — it is designed for local/trusted-network use. Do not expose it to the public internet without a reverse proxy and auth layer.
- API keys are read from `.env` and never logged. If you find a code path that leaks them, report it.
