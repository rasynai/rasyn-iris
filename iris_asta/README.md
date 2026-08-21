# iris_asta

Corpus client library used by IRIS (see the repository root README).

It wraps the Asta / Semantic Scholar corpus tools with rate limiting,
retry handling, snapshot-date enforcement, and an optional MCP transport,
and provides the config loading (`iris_asta/config.py`) that the solver
stack depends on.

Note: the Asta corpus API and Semantic Scholar API have their own terms
of use; check them for your use case before deploying.
