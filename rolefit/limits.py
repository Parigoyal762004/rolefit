"""Rate limiting, counted in Postgres rather than process memory.

An in-memory counter is close to useless here. Every serverless invocation can
land on a fresh instance, so a per-process dict resets constantly and the real
ceiling becomes (limit x number of warm instances), which is not a number
anybody chose. The counter has to live somewhere shared.

Two ceilings, because they stop different things:

  per IP   one person hammering it
  global   a hundred people each staying politely under the per-IP limit and
           draining the API budget between them

The global cap is the one that protects the wallet. Each /ask is two to four
LLM calls, so an unbounded public endpoint is a standing invitation to spend
money on someone else's behalf.
"""

from dataclasses import dataclass

from . import supabase as sb

PER_IP_LIMIT = 20
PER_IP_WINDOW = "1 hour"

GLOBAL_LIMIT = 200
GLOBAL_WINDOW = "1 hour"


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


def _bump(bucket: str, window: str) -> int:
    return sb.rpc("rolefit_bump_rate", {"p_bucket": bucket, "p_window": window})


def check(ip: str) -> Verdict:
    """Count this request against both ceilings and say whether to serve it.

    Fails open. A broken counter should not take the whole demo down, and the
    counter lives in the same database the request needs anyway, so a failure
    here almost certainly means the request was going to fail regardless.
    """
    try:
        if _bump("global", GLOBAL_WINDOW) > GLOBAL_LIMIT:
            return Verdict(False, "This demo has hit its hourly cap across all "
                                  "visitors. Try again later.")
        if _bump(f"ip:{ip}", PER_IP_WINDOW) > PER_IP_LIMIT:
            return Verdict(False, f"Rate limit: {PER_IP_LIMIT} questions an "
                                  "hour. Try again later.")
        return Verdict(True)
    except Exception:
        return Verdict(True)


def client_ip(request) -> str:
    """Real client IP behind Vercel's proxy.

    request.client.host is the edge and is identical for everyone, so limiting
    on it would put the whole world in one bucket. Vercel sets x-forwarded-for
    with the client first. Only the first entry is trustworthy; anything after
    it is caller-supplied and spoofable.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return getattr(request.client, "host", "unknown")
