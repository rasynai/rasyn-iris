"""Standalone Semantic Scholar (Asta tools) client for dev/run outside Inspect.

Benchmark facts served:

* All literature access goes through the S2 graph API
  (``https://api.semanticscholar.org/graph/v1/``, header ``x-api-key`` =
  ``cfg.asta_tool_key``) - the same corpus the ASTABench Asta tools expose, so
  standalone dev results transfer to in-harness runs (where solvers prefer the
  task-provided ``state.tools``).
* Date-restriction: ``paper_search`` supports ``publicationDateOrYear`` open
  ranges (``":YYYY-MM-DD"``) and ``snippet_search`` supports ``insertedBefore``
  - PFB/LitQA2 forbid using papers the benchmark's snapshot could not contain.
* The Asta key is limited to ~4 req/s → a token-bucket
  :class:`RateLimiter` at ``cfg.rate_limit_rps`` plus retry on HTTP
  429/504/529 (exponential backoff, 60 s cap, max 8 attempts). Other 4xx are
  real bugs and raise immediately.
* ``corpus_id`` is ALWAYS a str of digits - PFB's ``emit_pfb`` contract
  requires digits-only string paper ids.
* The current ASTA_TOOL_KEY (``ak-...``) is only valid against the Asta
  Scientific Corpus MCP endpoint, NOT ``api.semanticscholar.org`` - when MCP
  mode is active (``use_mcp=True``, or auto when ``cfg.asta_tool_key``
  starts with ``"ak-"``), every public method routes through
  :class:`iris_asta.asta_mcp.McpTransport` (tools/call over JSON-RPC) and
  adapts the results into the same :class:`Snippet`/:class:`Paper`
  dataclasses. The S2 HTTP path below stays untouched as the fallback.

Stdlib only. ``transport`` (test injection) is a callable
``transport(url: str) -> dict`` (parsed JSON body); it may raise
``urllib.error.HTTPError`` to simulate HTTP failures. ``mcp_transport``
(test injection) is anything with ``call_tool(name, arguments) -> dict``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .config import Config

BASE_URL = "https://api.semanticscholar.org/graph/v1/"
#: publicationTypes powers PFB's publication-type filter (gold sets carry
#: e.g. 'JournalArticle'); publicationDate powers date-window bisection of
#: saturated citer lists - both ride in ``Paper.extra``.
DEFAULT_FIELDS = (
    "title,abstract,corpusId,authors,year,venue,citationCount,"
    "publicationTypes,publicationDate"
)

#: Author-search fields (live-verified 2026-07-15): paperCount is required by
#: PFB's author disambiguation (exact-name match, tie-break by paperCount -
#: the server can rank an initialism alias record above the exact full-name
#: record, so top-hit selection is unsafe).
AUTHOR_FIELDS = "authorId,name,paperCount"

#: The relevance-search endpoint rejects limits above 100 (live-verified) -
#: clamp instead of erroring mid-eval.
_PAPER_SEARCH_MAX_LIMIT = 100

_RETRY_CODES = (429, 504, 529)
_MAX_ATTEMPTS = 8
_BACKOFF_CAP_S = 60.0


@dataclass
class Snippet:
    """One snippet/search hit; ``ref_mentions`` keeps only annotations that
    carry ``matchedPaperCorpusId`` (they power PFB's cited-paper attachment)."""

    corpus_id: str
    title: str
    text: str
    score: float
    section: str
    ref_mentions: list = field(default_factory=list)


@dataclass
class Paper:
    """One paper record; ``extra`` holds non-core fields (citationCount, ...)."""

    corpus_id: str
    title: str
    abstract: str
    year: int | None
    venue: str
    authors: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _backoff_delay(retry_index: int) -> float:
    """Exponential backoff delay for the given retry (0-based), capped at 60 s.

    Benchmark fact served: the Asta key throttles around 4 req/s and the S2
    API sheds load with 429/504/529; 1,2,4,...,60 s keeps long eval runs alive
    without hammering the shared endpoint.
    """
    return min(_BACKOFF_CAP_S, float(2 ** retry_index))


class RateLimiter:
    """Token bucket: ``rps`` tokens/second, burst capacity ``max(1, rps)``.

    Benchmark fact served: the ASTA_TOOL_KEY allows ~4 req/s; running at
    ``cfg.rate_limit_rps`` (3.5) below that ceiling avoids 429 storms that
    would burn the retry budget mid-eval. ``clock``/``sleep`` are injectable
    for offline tests.
    """

    def __init__(self, rps: float, clock=time.monotonic, sleep=time.sleep):
        self.rps = max(float(rps), 1e-3)
        self.capacity = max(1.0, self.rps)
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def acquire(self) -> None:
        """Block (sleep) until one request token is available, then take it."""
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rps)
        self._last = now
        if self.tokens < 1.0:
            delay = (1.0 - self.tokens) / self.rps
            self._sleep(delay)
            # Advance the clock past the sleep as well, otherwise the next
            # acquire() credits the slept interval a second time and the
            # effective rate settles at roughly twice ``rps``.
            self._last += delay
            self.tokens = 1.0
        self.tokens -= 1.0


def _within_date_range(paper: Paper, date_range: str) -> bool:
    """True iff the paper's publicationDate falls inside ``lo:hi`` (ISO dates).

    Benchmark fact served: the MCP tools filter ``publication_date_range``
    server-side and EXCLUDE papers whose publicationDate is null (verified
    live) - the S2 HTTP fallback path must mirror that semantics client-side
    so dev results transfer to in-harness runs. ISO date strings compare
    correctly as plain strings; open bounds (empty lo or hi) are supported.
    """
    pub = (paper.extra or {}).get("publicationDate")
    if not pub:
        return False
    pub = str(pub)[:10]
    lo, _, hi = date_range.partition(":")
    if lo and pub < lo:
        return False
    if hi and pub > hi:
        return False
    return True


def _paper_from_json(d: dict) -> Paper:
    """Parse an S2 paper JSON object into :class:`Paper`.

    Benchmark fact served: ``corpus_id`` is normalized to a str of digits
    (``emit_pfb`` rejects anything else); unconsumed fields (citationCount for
    the metadata channel's min-citations filter, etc.) survive in ``extra``.
    """
    cid = d.get("corpusId")
    year = d.get("year")
    return Paper(
        corpus_id="" if cid is None else str(cid),
        title=d.get("title") or "",
        abstract=d.get("abstract") or "",
        year=int(year) if year is not None else None,
        venue=d.get("venue") or "",
        authors=d.get("authors") or [],
        extra={
            k: v
            for k, v in d.items()
            if k not in ("corpusId", "title", "abstract", "year", "venue", "authors")
        },
    )


def _snippet_from_json(item: dict) -> Snippet:
    """Parse one snippet/search result item into :class:`Snippet`.

    Benchmark fact served: refMentions annotations are kept only when they
    carry ``matchedPaperCorpusId`` - those are the resolvable citation links
    PFB's semantic channel uses to attach cited papers as candidates.
    """
    sn = item.get("snippet") or {}
    paper = item.get("paper") or {}
    ann = sn.get("annotations") or {}
    refs = [
        r
        for r in (ann.get("refMentions") or [])
        if isinstance(r, dict) and r.get("matchedPaperCorpusId")
    ]
    cid = paper.get("corpusId")
    score = item.get("score")
    return Snippet(
        corpus_id="" if cid is None else str(cid),
        title=paper.get("title") or "",
        text=sn.get("text") or "",
        score=float(score) if score is not None else 0.0,
        section=sn.get("section") or sn.get("snippetKind") or "",
        ref_mentions=refs,
    )


# ------------------------------------------------- MCP result adaptation --


def _looks_like_not_found(exc: Exception) -> bool:
    """True when an MCP tool error means "id absent from the corpus".

    Benchmark fact served: on the S2 path a dead corpus id is an HTTP 404
    mapped to None so one dead id cannot kill a solve; the MCP server
    surfaces the same condition as a tool/JSON-RPC error whose message
    carries 404 / "not found" - map those (and only those) to None too.
    """
    text = str(exc).lower()
    return "404" in text or "not found" in text


def _mcp_items(resp) -> list:
    """Pull the result list out of an MCP tool payload, tolerantly.

    The Asta MCP tools wrap the same S2 corpus, but the envelope key is not
    contractual - accept a bare list or any of the observed/plausible list
    keys (``data`` is the S2-native one) so envelope drift cannot silently
    zero out retrieval mid-eval.
    """
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for key in ("data", "results", "matches", "papers", "snippets"):
            value = resp.get(key)
            if isinstance(value, list):
                return value
    return []


def _paper_from_any(d: dict) -> Paper:
    """Parse a paper dict from either the S2 shape or an MCP snake_case shape.

    Benchmark fact served: ``corpus_id`` must stay a digits-only str; MCP
    tool payloads may spell the id ``corpus_id`` or nest it under
    ``externalIds.CorpusId`` - normalize to the S2 ``corpusId`` key and
    reuse :func:`_paper_from_json`.
    """
    if d.get("corpusId") is None:
        cid = d.get("corpus_id")
        if cid is None:
            ext = d.get("externalIds") or {}
            cid = ext.get("CorpusId") if isinstance(ext, dict) else None
        if cid is not None:
            d = dict(d)
            d.pop("corpus_id", None)
            d["corpusId"] = cid
    return _paper_from_json(d)


def _snippet_from_any(item: dict) -> Snippet:
    """Parse a snippet result from either the S2 nested shape or a flat one.

    S2's ``snippet/search`` nests under ``snippet``/``paper`` keys - reuse
    :func:`_snippet_from_json` for that. MCP tool payloads may instead be
    flat records (``corpus_id``/``title``/``text``/``score``/``section``);
    map those tolerantly, keeping only resolvable ref mentions (the ones
    carrying ``matchedPaperCorpusId``) exactly like the S2 path.
    """
    if isinstance(item.get("snippet"), dict):
        return _snippet_from_json(item)
    cid = item.get("corpus_id", item.get("corpusId"))
    score = item.get("score")
    refs = [
        r
        for r in (item.get("ref_mentions") or item.get("refMentions") or [])
        if isinstance(r, dict) and r.get("matchedPaperCorpusId")
    ]
    return Snippet(
        corpus_id="" if cid is None else str(cid),
        title=item.get("title") or "",
        text=item.get("text") or item.get("snippet") or "",
        score=float(score) if score is not None else 0.0,
        section=item.get("section") or item.get("snippet_kind") or "",
        ref_mentions=refs,
    )


class AstaClient:
    """Semantic Scholar graph-API client (standalone dev/run path).

    Inside Inspect, solvers PREFER task-provided ``state.tools``; this client
    exists so the exact same solver code runs against the live API during
    development and as a fallback.

    ``use_mcp`` selects the wire path: ``None`` (default) auto-detects -
    MCP when ``cfg.asta_tool_key`` starts with ``"ak-"`` (those keys are
    only valid on the Asta MCP endpoint, not api.semanticscholar.org);
    ``True``/``False`` force it. ``mcp_transport`` injects a ready
    transport (tests); otherwise MCP mode lazily builds a
    :class:`iris_asta.asta_mcp.McpTransport`.
    """

    def __init__(self, cfg: Config, transport=None, use_mcp: bool | None = None,
                 mcp_transport=None, *, mcp_max_attempts: int | None = None):
        self.cfg = cfg
        self._transport = transport if transport is not None else self._http_transport
        self._headers = {"x-api-key": cfg.asta_tool_key} if cfg.asta_tool_key else {}
        self._limiter = RateLimiter(cfg.rate_limit_rps)
        self._sleep = time.sleep  # test hook: replace to avoid real backoff sleeps
        if use_mcp is None:
            use_mcp = cfg.asta_tool_key.startswith("ak-")
        self._mcp = None
        if use_mcp:
            if mcp_transport is not None:
                self._mcp = mcp_transport
            else:
                from .asta_mcp import McpTransport  # local import: no cycle

                self._mcp = McpTransport(cfg, max_attempts=mcp_max_attempts)

    # ---------------------------------------------------------------- HTTP --

    def _http_transport(self, url: str) -> dict:
        """Default transport: GET ``url`` with the ``x-api-key`` header.

        Benchmark fact served: stdlib-only constraint - raw urllib, no
        requests/httpx. Raises ``urllib.error.HTTPError`` on HTTP errors so
        ``_get`` can classify retryable status codes.
        """
        req = urllib.request.Request(url, headers=dict(self._headers), method="GET")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Rate-limited GET with retry on HTTP 429/504/529.

        Benchmark fact served: exp backoff capped at 60 s, max 8 attempts -
        long eval runs must survive S2 load-shedding; any other 4xx is a
        client bug and raises immediately (fail loud during dev, not silently
        empty results that would look like a bad solver).
        """
        query = urllib.parse.urlencode(params or {})
        url = BASE_URL + path + (("?" + query) if query else "")
        for attempt in range(_MAX_ATTEMPTS):
            if attempt:
                self._sleep(_backoff_delay(attempt - 1))
            self._limiter.acquire()
            try:
                return self._transport(url)
            except Exception as exc:
                code = getattr(exc, "code", None)
                if code in _RETRY_CODES and attempt < _MAX_ATTEMPTS - 1:
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exited without return/raise")

    # ------------------------------------------------------------- endpoints --

    def snippet_search(
        self,
        query: str,
        limit: int = 20,
        corpus_ids: list | None = None,
        inserted_before: str | None = None,
    ) -> list[Snippet]:
        """GET ``snippet/search``: full-text snippet retrieval.

        Benchmark facts served: ``inserted_before`` enforces the task's corpus
        snapshot date (papers indexed after it are invisible gold-side);
        ``corpus_ids`` restricts search to given papers via
        ``paperIds=CorpusId:<id>,...`` - ADT extracts each cell from snippets
        of exactly one paper this way. MCP path (live tools/list schema):
        tool ``snippet_search`` with ``query``/``limit``/``inserted_before``
        and ``paper_ids`` as a comma-joined STRING
        ``"CorpusId:<id>,CorpusId:<id>"``.
        """
        if self._mcp is not None:
            args: dict = {"query": query, "limit": limit}
            if inserted_before:
                args["inserted_before"] = inserted_before
            if corpus_ids:
                args["paper_ids"] = ",".join(f"CorpusId:{c}" for c in corpus_ids)
            resp = self._mcp.call_tool("snippet_search", args)
            return [_snippet_from_any(item) for item in _mcp_items(resp)
                    if isinstance(item, dict)]
        params: dict = {"query": query, "limit": limit}
        if inserted_before:
            params["insertedBefore"] = inserted_before
        if corpus_ids:
            params["paperIds"] = ",".join(f"CorpusId:{c}" for c in corpus_ids)
        resp = self._get("snippet/search", params)
        return [_snippet_from_json(item) for item in (resp.get("data") or [])]

    def paper_search(
        self,
        kquery: str,
        limit: int = 20,
        publication_date_before: str | None = None,
        venues: str | None = None,
    ) -> list[Paper]:
        """GET ``paper/search``: keyword paper search with default fields.

        Benchmark fact served: PFB metadata queries carry date constraints;
        ``publication_date_before`` maps to the open range
        ``publicationDateOrYear=":YYYY-MM-DD"`` so before-filters are executed
        server-side (deterministic, not LLM-judged). ``venues`` restricts to
        an exact long-form venue string (PFB venue-constrained plans with no
        other base pool enumerate a venue this way). MCP path (live
        tools/list schema, re-verified 2026-07-15): tool
        ``search_papers_by_relevance`` with ``keyword`` (NOT query) /
        ``fields`` / ``limit`` (hard max 100 -> clamped) / ``venues``; the
        date filter is ``publication_date_range`` with the same S2
        open-range syntax.
        """
        limit = min(int(limit), _PAPER_SEARCH_MAX_LIMIT)
        if self._mcp is not None:
            args: dict = {"keyword": kquery, "limit": limit, "fields": DEFAULT_FIELDS}
            if publication_date_before:
                args["publication_date_range"] = ":" + publication_date_before
            if venues:
                args["venues"] = venues
            resp = self._mcp.call_tool("search_papers_by_relevance", args)
            return [_paper_from_any(d) for d in _mcp_items(resp)
                    if isinstance(d, dict)]
        params: dict = {"query": kquery, "limit": limit, "fields": DEFAULT_FIELDS}
        if publication_date_before:
            params["publicationDateOrYear"] = ":" + publication_date_before
        if venues:
            params["venue"] = venues
        resp = self._get("paper/search", params)
        return [_paper_from_json(d) for d in (resp.get("data") or [])]

    def search_paper_by_title(self, title: str) -> Paper | None:
        """Resolve a known title to a single paper (top ``paper/search`` hit).

        Benchmark fact served: PFB "specific" route and must-cite anchors give
        paper TITLES; the executor needs their corpus ids to do set algebra
        over citations. Returns None when the corpus has no match. MCP path
        (live tools/list schema): dedicated tool ``search_paper_by_title``
        with ``title`` (NOT query - a live call with query failed
        validation) and ``fields``; tolerates both a list envelope and a
        bare paper object.
        """
        if self._mcp is not None:
            try:
                resp = self._mcp.call_tool(
                    "search_paper_by_title",
                    {"title": title, "fields": DEFAULT_FIELDS},
                )
            except RuntimeError as exc:
                if _looks_like_not_found(exc):
                    return None
                raise
            items = _mcp_items(resp)
            if items:
                return _paper_from_any(items[0]) if isinstance(items[0], dict) else None
            if isinstance(resp, dict) and (
                resp.get("corpusId") is not None
                or resp.get("corpus_id") is not None
                or resp.get("title")
            ):
                return _paper_from_any(resp)
            return None
        results = self.paper_search(title, limit=1)
        return results[0] if results else None

    def get_paper(self, corpus_id, fields: str | None = None) -> Paper | None:
        """GET ``paper/CorpusId:<id>``: fetch one paper by corpus id.

        Benchmark fact served: candidate hydration (title/year/venue/abstract)
        for evidence strings - PFB evidence is quoted as
        ``«Title» (year): ...``. Returns None on HTTP 404 (id absent from the
        snapshot) instead of raising, so one dead id cannot kill a solve.
        MCP path: tool ``get_paper`` with ``paper_id="CorpusId:<id>"`` and
        ``fields``; a not-found tool error maps to None the same way.
        """
        if self._mcp is not None:
            try:
                resp = self._mcp.call_tool(
                    "get_paper",
                    {"paper_id": f"CorpusId:{corpus_id}",
                     "fields": fields or DEFAULT_FIELDS},
                )
            except RuntimeError as exc:
                if _looks_like_not_found(exc):
                    return None
                raise
            return _paper_from_any(resp) if resp else None
        params = {"fields": fields or DEFAULT_FIELDS}
        try:
            resp = self._get(f"paper/CorpusId:{corpus_id}", params)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        return _paper_from_json(resp)

    def get_paper_batch(self, corpus_ids: list, fields: str | None = None) -> list[Paper]:
        """Fetch several papers by corpus id, skipping ids the corpus lacks.

        MCP path (live tools/list schema): tool ``get_paper_batch`` with
        ``ids`` as a JSON ARRAY ``["CorpusId:<id>", ...]`` (one call - the
        endpoint is rate-limited at 10 req/s). S2 fallback path: sequential
        :meth:`get_paper` calls (the batch endpoint is POST-only and the
        GET transport stays untouched); dead ids yield None there and are
        dropped, matching the MCP server's null batch entries.

        ``corpusId`` is force-included in custom ``fields``: callers re-align
        the returned papers to their input ids via ``Paper.corpus_id``, and a
        fields list without corpusId would leave every result unmatchable
        (corpus_id "") on both paths.
        """
        if fields and "corpusId" not in fields:
            fields = fields + ",corpusId"
        if self._mcp is not None:
            resp = self._mcp.call_tool(
                "get_paper_batch",
                {"ids": [f"CorpusId:{c}" for c in corpus_ids],
                 "fields": fields or DEFAULT_FIELDS},
            )
            return [_paper_from_any(d) for d in _mcp_items(resp)
                    if isinstance(d, dict)]
        papers = []
        for cid in corpus_ids:
            p = self.get_paper(cid, fields=fields)
            if p is not None:
                papers.append(p)
        return papers

    def get_citations(
        self,
        corpus_id,
        direction: str = "citations",
        limit: int = 100,
        publication_date_range: str | None = None,
        fields: str | None = None,
    ) -> list[Paper]:
        """GET ``paper/{id}/citations`` or ``.../references``.

        Benchmark fact served: PFB metadata plans include must-cite /
        must-not-cite constraints - answered by deterministic set algebra
        over citation edges.
        ``direction`` must be ``"citations"`` (papers citing this one; items
        under ``citingPaper``) or ``"references"`` (papers this one cites;
        items under ``citedPaper``). ``fields`` (default ``DEFAULT_FIELDS``)
        lets callers slim the per-edge payload - PFB's reference walk reads
        only corpusId/venue from up to 300 reference lists per predicate,
        and full-field pulls (abstracts included) pushed saturated windows
        past the SSE deadline. ``publication_date_range``
        (``"YYYY-MM-DD:YYYY-MM-DD"``) shards a citer list by citing-paper
        publication date - the ONLY enumeration mechanism past the tool's
        1000-row cap (live-verified: ``offset`` is silently ignored and the
        default order is recency-first, so an uncapped anchor returns only
        post-snapshot citers). MCP path (schema re-verified 2026-07-15):
        tool ``get_citations`` takes ``publication_date_range`` directly for
        the citations direction; the server whitelist has no references
        tool, so references ride on ``get_paper`` with nested
        ``references.<field>`` S2 graph fields. The S2 HTTP fallback has no
        server-side date filter on citation edges, so the range is applied
        client-side there (papers with null publicationDate excluded, the
        MCP semantics).
        """
        if direction not in ("citations", "references"):
            raise ValueError(f"direction must be 'citations' or 'references', got {direction!r}")
        key = "citingPaper" if direction == "citations" else "citedPaper"
        use_fields = fields or DEFAULT_FIELDS
        if self._mcp is not None:
            if direction == "citations":
                args = {"paper_id": f"CorpusId:{corpus_id}",
                        "fields": use_fields, "limit": limit}
                if publication_date_range:
                    args["publication_date_range"] = publication_date_range
                resp = self._mcp.call_tool("get_citations", args)
                papers = []
                for item in _mcp_items(resp):
                    if not isinstance(item, dict):
                        continue
                    sub = item.get(key) if key in item else item
                    if sub:
                        papers.append(_paper_from_any(sub))
                return papers
            ref_fields = ",".join(
                f"references.{f}" for f in use_fields.split(",")
            )
            resp = self._mcp.call_tool(
                "get_paper",
                {"paper_id": f"CorpusId:{corpus_id}", "fields": ref_fields},
            )
            refs = resp.get("references") or []
            papers = [_paper_from_any(r) for r in refs[:limit]
                      if isinstance(r, dict)]
            if publication_date_range:
                papers = [p for p in papers
                          if _within_date_range(p, publication_date_range)]
            return papers
        params = {"fields": use_fields, "limit": limit}
        resp = self._get(f"paper/CorpusId:{corpus_id}/{direction}", params)
        papers = []
        for item in resp.get("data") or []:
            sub = item.get(key)
            if sub:
                papers.append(_paper_from_json(sub))
        if publication_date_range:
            papers = [p for p in papers
                      if _within_date_range(p, publication_date_range)]
        return papers

    def search_authors_by_name(self, name: str) -> list[dict]:
        """GET ``author/search``: raw author records for a name query.

        Benchmark fact served: PFB metadata plans constrain by author; author
        disambiguation needs the raw records WITH ``paperCount``. The server
        can rank an initialism alias record (with a smaller paperCount) above
        the exact full-name record even for an exact-name query (live-verified
        2026-07-15), so callers pick by exact-name match plus paperCount,
        never the top hit. MCP path (schema re-verified): tool
        ``search_authors_by_name`` with ``name`` (NOT query) and ``fields``.
        """
        if self._mcp is not None:
            resp = self._mcp.call_tool(
                "search_authors_by_name",
                {"name": name, "fields": AUTHOR_FIELDS},
            )
            return list(_mcp_items(resp))
        resp = self._get("author/search", {"query": name, "fields": AUTHOR_FIELDS})
        return list(resp.get("data") or [])

    def get_author_papers(
        self,
        author_id,
        limit: int = 1000,
        publication_date_range: str | None = None,
    ) -> list[Paper]:
        """GET ``author/{id}/papers``: an author's papers with default fields.

        Benchmark fact served: author-constrained PFB queries intersect this
        list with venue/year/topic filters - deterministic set algebra, no LLM
        in the loop. Default limit is 1000, the tool max: a lower cap silently
        truncates the paper lists of prolific authors. ``publication_date_range``
        pre-filters server-side on the MCP path (schema re-verified
        2026-07-15) and client-side on the S2 fallback (null publicationDate
        excluded - the MCP semantics). MCP tool ``get_author_papers`` takes
        ``author_id`` and ``paper_fields`` (NOT fields).
        """
        if self._mcp is not None:
            args = {"author_id": str(author_id), "paper_fields": DEFAULT_FIELDS,
                    "limit": limit}
            if publication_date_range:
                args["publication_date_range"] = publication_date_range
            resp = self._mcp.call_tool("get_author_papers", args)
            return [_paper_from_any(d) for d in _mcp_items(resp)
                    if isinstance(d, dict)]
        params = {"fields": DEFAULT_FIELDS, "limit": limit}
        resp = self._get(f"author/{author_id}/papers", params)
        papers = [_paper_from_json(d) for d in (resp.get("data") or [])]
        if publication_date_range:
            papers = [p for p in papers
                      if _within_date_range(p, publication_date_range)]
        return papers
