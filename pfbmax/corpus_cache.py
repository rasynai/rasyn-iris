"""Read-through disk cache for corpus clients.

``CachedClient`` wraps ANY object that duck-types
``iris_asta.asta_client.AstaClient`` and caches the eight corpus-read
methods on disk (one JSON file per call signature) so repeated solver /
scoring iterations cost zero network calls. Stdlib only; the wrapped
client is never imported, only called.

Contract:

* Cache key: sha256 over ``(method_name, canonical-JSON of args/kwargs)``.
  Canonical means: dict keys sorted, tuples as lists, sets as sorted
  lists, compact separators — so semantically-equal kwargs always map to
  the same key. NOTE the key is built from the args as PASSED: calling
  ``get_paper("1")`` and ``get_paper(corpus_id="1")`` are different keys
  (both correct, just a duplicate fetch); pfbmax callers should pick one
  calling convention per site.
* Cached methods: ``snippet_search``, ``paper_search``,
  ``search_paper_by_title``, ``get_paper``, ``get_paper_batch``,
  ``get_citations``, ``search_authors_by_name``, ``get_author_papers``.
  Every other attribute (method or plain attr) delegates to the inner
  client uncached and uncounted.
* Serialization: results may contain iris_asta dataclasses
  (``Paper``/``Snippet``). Object nodes are converted to dicts via
  ``dataclasses.asdict`` (fallback ``vars()``) and tagged with a marker
  key; on load ONLY tagged nodes become ``types.SimpleNamespace`` —
  plain dicts (``Paper.extra``, author records, ``ref_mentions``) stay
  plain dicts, because consumers do ``extra.get("citationCount")`` /
  ``isinstance(record, dict)`` (see iris_asta/solvers/pfb.py). The
  round-trip preserves corpusId/corpus_id/title/abstract/text/score/
  year/venue/authors/citationCount under whichever attribute names the
  inner client produced.
* Both hits AND misses return the round-tripped (SimpleNamespace) form,
  so consumer behavior is identical on cold and warm cache.
* ``None`` results (dead corpus id, unresolvable title) are cached too —
  negative caching; file existence discriminates hit from miss.
* Counters: ``.calls`` (inner-client invocations made), ``.hits``,
  ``.misses``. For cached methods every invocation increments exactly
  one of hits/misses, and every miss increments calls (calls == misses
  unless you reset counters). Uncached delegation is not counted.
  Counters are per-instance and not thread-synchronized; the disk layer
  IS safe across processes (atomic ``os.replace`` writes).
* ``bypass=True`` disables cache READS (every call goes to the inner
  client and counts as a miss) but still WRITES results — use it to
  refresh entries while seeding the shared cache.
* Failure semantics: if the inner client raises, nothing is written and
  the exception propagates unchanged. Corrupt/unreadable cache files are
  treated as misses and rewritten. Cache-write failures (disk issues)
  are swallowed — caching must never break a solve.

Default cache directory: the literal default ``"pfbmax/cache"`` is
anchored at THIS file's directory (``<repo>/pfbmax/cache``) regardless
of the caller's cwd, so every stage shares one physical cache. Any
other explicit path is respected as given.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

__all__ = ["CachedClient", "CACHED_METHODS", "cache_key"]

#: Corpus-read methods served from disk; everything else delegates raw.
CACHED_METHODS = frozenset(
    {
        "snippet_search",
        "paper_search",
        "search_paper_by_title",
        "get_paper",
        "get_paper_batch",
        "get_citations",
        "search_authors_by_name",
        "get_author_papers",
    }
)

#: Default cache location; this exact value is anchored at the
#: pfbmax package directory so the shared cache location is cwd-independent.
_DEFAULT_CACHE_DIR = "pfbmax/cache"

#: Marker key tagging dict nodes that were attribute-objects (dataclasses /
#: vars()-able) at serialization time; only these become SimpleNamespace on
#: load. No corpus payload uses this key (S2/MCP JSON never dunders).
_OBJ_KEY = "__pfbmax_obj__"


# ------------------------------------------------------------------ keys --


def _canon(value):
    """Deterministic JSON-ready form of an args/kwargs value.

    tuples -> lists, sets -> sorted lists (sorted by their canonical JSON
    so mixed types cannot raise), dict keys stringified (json.dumps sorts
    them), exotic objects -> str. PYTHONHASHSEED can reorder set iteration
    between processes — sorting keeps keys stable across runs.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canon(v) for v in value),
            key=lambda v: json.dumps(v, sort_keys=True, default=str),
        )
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in value.items()}
    return str(value)


def cache_key(method: str, args=(), kwargs=None) -> str:
    """sha256 hex key for one call: (method_name, canonical args/kwargs)."""
    payload = json.dumps(
        {
            "method": str(method),
            "args": _canon(list(args)),
            "kwargs": _canon(dict(kwargs or {})),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------- serialization --


def _to_jsonable(value):
    """Normalize a result tree to pure JSON types.

    Object nodes (dataclasses via ``dataclasses.asdict``, other objects
    via ``vars()``) become dicts tagged with ``_OBJ_KEY``; plain dicts and
    lists pass through untagged so they round-trip as themselves.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.asdict(value)
    else:
        try:
            fields = vars(value)
        except TypeError:
            return str(value)  # opaque scalar-ish object: best-effort string
    out = {_OBJ_KEY: True}
    for k, v in fields.items():
        out[str(k)] = _to_jsonable(v)
    return out


def _from_jsonable(value):
    """Inverse of :func:`_to_jsonable`.

    Tagged dicts -> ``SimpleNamespace`` (attribute access for consumers'
    getattr-tolerant accessors); untagged dicts stay dicts (consumers do
    ``extra.get(...)`` / ``isinstance(record, dict)``); lists stay lists.
    """
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    if isinstance(value, dict):
        if value.get(_OBJ_KEY) is True:
            ns = SimpleNamespace()
            for k, v in value.items():
                if k != _OBJ_KEY:
                    ns.__dict__[k] = _from_jsonable(v)
            return ns
        return {k: _from_jsonable(v) for k, v in value.items()}
    return value


# ----------------------------------------------------------------- client --


class CachedClient:
    """Duck-typed AstaClient wrapper with a read-through disk cache.

    ``CachedClient(inner)`` is a drop-in replacement for ``inner``
    anywhere a corpus client is passed (router/solvers/semantic
    retrieval): cached methods are intercepted, everything else —
    including plain attributes like ``cfg`` — resolves on the inner
    client via ``__getattr__``.
    """

    def __init__(self, inner, cache_dir: str = _DEFAULT_CACHE_DIR, bypass: bool = False):
        self._inner = inner
        if cache_dir == _DEFAULT_CACHE_DIR:
            self.cache_dir = Path(__file__).resolve().parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bypass = bool(bypass)
        self.calls = 0  # inner-client invocations actually made
        self.hits = 0  # served from disk
        self.misses = 0  # went to the inner client

    # -------------------------------------------------------- delegation --

    def __getattr__(self, name):
        # Only reached for names not defined on CachedClient itself.
        if name == "_inner":  # guard: no recursion before __init__ ran
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if name in CACHED_METHODS and callable(attr):
            def cached_method(*args, _method=name, _fn=attr, **kwargs):
                return self._cached_call(_method, _fn, args, kwargs)

            cached_method.__name__ = name
            cached_method.__qualname__ = f"CachedClient.{name}"
            cached_method.__doc__ = getattr(attr, "__doc__", None)
            return cached_method
        return attr

    def __repr__(self):  # pragma: no cover - debugging aid
        return (
            f"CachedClient(inner={type(self._inner).__name__}, "
            f"dir={str(self.cache_dir)!r}, bypass={self.bypass}, "
            f"calls={self.calls}, hits={self.hits}, misses={self.misses})"
        )

    # ------------------------------------------------------------- cache --

    def cache_path(self, method: str, args=(), kwargs=None) -> Path:
        """Disk path that would serve this exact call (test/tooling aid)."""
        return self.cache_dir / (cache_key(method, args, kwargs) + ".json")

    def _cached_call(self, method, fn, args, kwargs):
        path = self.cache_path(method, args, kwargs)
        if not self.bypass:
            entry = self._read_entry(path)
            if entry is not None:
                self.hits += 1
                return _from_jsonable(entry["result"])
        self.misses += 1
        # PFBMAX_CACHE_ONLY=1: never call the remote on a miss -- return the
        # method's empty shape instead. For offline pool REBUILDS while the
        # corpus API is rate-dead: a missed channel costs a few candidates,
        # not a 2-minute retry ladder per call.
        if (os.environ.get("PFBMAX_CACHE_ONLY") or "").strip():
            return []
        self.calls += 1
        result = fn(*args, **kwargs)  # inner exception: nothing cached, propagates
        jsonable = _to_jsonable(result)
        self._write_entry(path, method, args, kwargs, jsonable)
        # Return the round-tripped form so hits and misses behave identically.
        return _from_jsonable(jsonable)

    @staticmethod
    def _read_entry(path: Path):
        """Load one cache entry; any unreadable/corrupt file is a miss."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or "result" not in data:
            return None
        return data

    def _write_entry(self, path: Path, method, args, kwargs, jsonable) -> None:
        """Atomically write one entry; write failures never break the call."""
        now = time.time()
        entry = {
            "method": method,
            "args": _canon(list(args)),
            "kwargs": _canon(dict(kwargs)),
            "timestamp": now,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "result": jsonable,
        }
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, path)  # atomic on POSIX and Windows (same volume)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
