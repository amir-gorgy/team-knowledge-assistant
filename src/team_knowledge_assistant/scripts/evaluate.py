"""Score search modes on evals/questions.jsonl and compare them question by question.

    uv run python -m team_knowledge_assistant.scripts.evaluate                  # vector vs hybrid, top 5
    uv run python -m team_knowledge_assistant.scripts.evaluate --modes vector keyword hybrid -k 3
    uv run python -m team_knowledge_assistant.scripts.evaluate --keyword-weight 2 --rrf-k 30
    uv run python -m team_knowledge_assistant.scripts.evaluate --sweep          # tune hybrid's weights
    uv run python -m team_knowledge_assistant.scripts.evaluate --modes hybrid hybrid+rerank
    uv run python -m team_knowledge_assistant.scripts.evaluate --modes hybrid+rerank --reranker BAAI/bge-reranker-base

Each question lists its gold chunks (file + heading path) and how to match them:
  any   passes if any gold chunk is in the top k; its rank is the best gold chunk's rank
  all   passes only if every gold chunk is in the top k; its rank is the worst one's
  none  a "not found" question: no chunk should match. Retrieval always returns k hits, so
        these can't pass or fail here; the vector top-1 distance is shown against the 0.30
        cut-off the Copilot prompt uses.
Ranks are looked up to --depth (default 20), so a gold chunk at #12 still shows as 12 and not
just "missed"; that is what tells you whether a change moved it closer.

--sweep tries hybrid with several keyword weights and RRF constants. Picking settings on the same
questions you report on flatters them, so the answerable questions are split by id: odd ids
("tune") pick the best setting, even ids ("check") show how it does on questions it wasn't
picked on.

The +rerank modes re-order the first stage's top --depth hits with a cross-encoder, so a gold
chunk the first stage didn't return can't appear. Their top-1 score is also shown for the
not-found questions: unlike a distance or an RRF score, it may be usable as a "not found" signal.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from team_knowledge_assistant.Ingest.store import (
    RRF_K,
    connect,
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)

QUESTIONS = Path(__file__).parents[3] / "evals" / "questions.jsonl"
CUTOFF = 0.30  # .github/prompts/doc-impact.prompt.md: drop hits with distance above ~0.30
SWEEP_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 3.0]  # keyword list's weight; the vector list's is 1
SWEEP_RRF_K = [10, 30, 60]
CANDIDATES = 20  # per list, as hybrid_search's default

Search = Callable[[psycopg.Connection, str, list[float], int], list]
SEARCHES: dict[str, Search] = {
    "vector": lambda conn, q, emb, depth: vector_search(conn, emb, k=depth),
    "keyword": lambda conn, q, emb, depth: keyword_search(conn, q, k=depth),
    # The keyword half of hybrid on its own: any word may match, more matches rank higher.
    "keyword-any": lambda conn, q, emb, depth: keyword_search(conn, q, k=depth, match="any"),
    "hybrid": lambda conn, q, emb, depth: hybrid_search(conn, q, emb, k=depth),
}


def gold_rank(hits: Sequence, sources: list[dict], match: str) -> int | None:
    """Rank (1-based) at which the question counts as found, or None if it isn't in `hits`.

    A hit is a gold chunk when its file name and full heading path both match. "any" needs
    one gold chunk, so it returns the best rank; "all" needs every one, so it returns the worst.
    """
    ranks = []
    for s in sources:
        found = [
            i
            for i, h in enumerate(hits, 1)
            if Path(h.source).name == s["file"] and list(h.heading_path) == s["heading_path"]
        ]
        ranks.append(found[0] if found else None)
    if match == "any":
        return min((r for r in ranks if r is not None), default=None)
    if match == "all":
        return None if None in ranks else max(ranks)
    raise ValueError(f"can't rank match={match!r}")


def reciprocal_rank(rank: int | None) -> float:
    """1 for rank 1, 0.5 for rank 2, ... 0 if not found. Averaged, this is MRR."""
    return 1 / rank if rank else 0.0


def summarize(ranks: Sequence[int | None], k: int) -> tuple[int, float]:
    """(how many ranks are within the top k, MRR)."""
    return sum(1 for r in ranks if r is not None and r <= k), sum(map(reciprocal_rank, ranks)) / len(ranks)


def is_tune(qid: str) -> bool:
    """Odd question numbers tune, even ones check: q01 tunes, q02 checks."""
    return int(qid.lstrip("q")) % 2 == 1


def sweep(conn: psycopg.Connection, questions: list[dict], k: int, depth: int) -> None:
    """Score hybrid for every keyword weight x RRF constant, picking on tune and confirming on check.

    Each question's two candidate lists are fetched once; only the cheap fusion step is repeated.
    """
    from team_knowledge_assistant.Ingest.embedding import embed_query

    lists = []
    for q in questions:
        vector_hits = vector_search(conn, embed_query(q["question"]), k=CANDIDATES)
        keyword_hits = keyword_search(conn, q["question"], k=CANDIDATES, match="any")
        by_id = {h.id: h for h in [*vector_hits, *keyword_hits]}
        lists.append((q, [h.id for h in vector_hits], [h.id for h in keyword_hits], by_id))

    def ranks_for(weight: float | None, rrf_k: int) -> dict[str, int | None]:
        out = {}
        for q, vec, kw, by_id in lists:
            if weight is None:  # vector only, as the baseline row
                fused = vec
            else:
                fused = [i for i, _ in reciprocal_rank_fusion([vec, kw], k=rrf_k, weights=[1.0, weight])]
            out[q["id"]] = gold_rank([by_id[i] for i in fused[:depth]], q["sources"], q["match"])
        return out

    tune_ids = [q["id"] for q in questions if is_tune(q["id"])]
    check_ids = [q["id"] for q in questions if not is_tune(q["id"])]
    n_t, n_c, n = len(tune_ids), len(check_ids), len(questions)
    print(f"Hybrid sweep: pass@{k} and MRR on tune ({n_t} odd ids), check ({n_c} even ids) and all {n}\n")
    print(f"{'rrf_k':>6} {'kw weight':>9} | {'tune':^16} | {'check':^16} | {'all':^16}")

    rows = []
    settings = [(None, RRF_K)] + [(w, rk) for rk in SWEEP_RRF_K for w in SWEEP_WEIGHTS]
    for weight, rrf_k in settings:
        r = ranks_for(weight, rrf_k)
        t, c, a = (summarize([r[i] for i in ids], k) for ids in (tune_ids, check_ids, list(r)))
        rows.append((weight, rrf_k, t, c, a))
        label = f"{rrf_k:>6} {weight:>9}" if weight is not None else f"{'vector only':>16}"
        print(f"{label} | {t[0]:>2}/{n_t} {t[1]:>6.3f}   | {c[0]:>2}/{n_c} {c[1]:>6.3f}   | {a[0]:>2}/{n} {a[1]:>6.3f}")

    # Pick on tune only: best MRR, then best pass@k, then the smaller change from the default.
    weight, rrf_k, t, c, a = max(
        (row for row in rows if row[0] is not None),
        key=lambda row: (round(row[2][1], 3), row[2][0], -abs(row[0] - 1) - abs(row[1] - RRF_K) / 100),
    )
    print(f"\nBest on tune: keyword weight {weight}, rrf_k {rrf_k} "
          f"(tune {t[0]}/{n_t}, MRR {t[1]:.3f}). On check: {c[0]}/{n_c}, MRR {c[1]:.3f}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", nargs="+", choices=[*SEARCHES, "vector+rerank", "hybrid+rerank"], default=["vector", "hybrid"]
    )
    parser.add_argument("-k", type=int, default=5, help="a question passes if its gold is in the top k")
    parser.add_argument("--depth", type=int, default=20, help="how far down to look for the gold rank")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--keyword-weight", type=float, default=1.0, help="hybrid: keyword list's weight")
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help="hybrid: RRF constant")
    parser.add_argument("--sweep", action="store_true", help="tune hybrid's keyword weight and RRF constant")
    parser.add_argument("--reranker", help="cross-encoder for the +rerank modes (default: settings.reranker_model)")
    args = parser.parse_args()

    if args.sweep:
        questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
        with connect() as conn:
            sweep(conn, [q for q in questions if q["match"] != "none"], args.k, args.depth)
        return

    # Imported here so --modes keyword works without the embeddings server running.
    needs_embedding = any(not m.startswith("keyword") for m in args.modes)
    if needs_embedding:
        from team_knowledge_assistant.Ingest.embedding import embed_query

    if any(m.endswith("+rerank") for m in args.modes):
        from team_knowledge_assistant.Ingest.rerank import rerank

    def run(conn: psycopg.Connection, mode: str, question: str, emb: list[float]) -> list:
        first = mode.removesuffix("+rerank")
        if first == "hybrid":
            hits = hybrid_search(conn, question, emb, k=args.depth,
                                 keyword_weight=args.keyword_weight, rrf_k=args.rrf_k)
        else:
            hits = SEARCHES[first](conn, question, emb, args.depth)
        if mode.endswith("+rerank"):
            hits = rerank(question, hits, k=len(hits), model=args.reranker)
        return hits

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    ranks: dict[str, dict[str, int | None]] = {m: {} for m in args.modes}
    not_found: list[tuple[str, float, str]] = []
    rerank_modes = [m for m in args.modes if m.endswith("+rerank")]
    top_score: dict[str, dict[str, float]] = {m: {} for m in rerank_modes}  # top-1 rerank score per question
    seconds: dict[str, float] = {m: 0.0 for m in args.modes}

    with connect() as conn:
        for q in questions:
            emb = embed_query(q["question"]) if needs_embedding else []
            if q["match"] == "none":
                if needs_embedding:
                    top = vector_search(conn, emb, k=1)[0]
                    not_found.append((q["id"], top.distance, " > ".join(top.heading_path)))
                modes = rerank_modes  # only a reranker's score says anything about "not found"
            else:
                modes = args.modes
            for mode in modes:
                started = time.perf_counter()
                hits = run(conn, mode, q["question"], emb)
                seconds[mode] += time.perf_counter() - started
                if mode in top_score and hits:
                    top_score[mode][q["id"]] = hits[0].score
                if q["match"] != "none":
                    ranks[mode][q["id"]] = gold_rank(hits, q["sources"], q["match"])

    def cell(rank: int | None) -> str:
        if rank is None:
            return f">{args.depth}"
        return f"#{rank}" + (" ✓" if rank <= args.k else "")

    answerable = [q for q in questions if q["match"] != "none"]
    print(f"Gold rank per question (✓ = in the top {args.k})\n")
    print(f"{'id':<5}{'kind':<12}" + "".join(f"{m:>13}" for m in args.modes) + "   question")
    for q in answerable:
        row = "".join(f"{cell(ranks[m][q['id']]):>13}" for m in args.modes)
        print(f"{q['id']:<5}{q['kind']:<12}{row}   {q['question'][:60]}")

    print(f"\n{'':<17}" + "".join(f"{m:>13}" for m in args.modes))
    scores = {m: summarize(list(ranks[m].values()), args.k) for m in args.modes}
    print(f"{f'pass@{args.k}':<17}" + "".join(f"{f'{scores[m][0]}/{len(answerable)}':>13}" for m in args.modes))
    print(f"{'MRR':<17}" + "".join(f"{scores[m][1]:>13.3f}" for m in args.modes))
    for half, keep in (("MRR tune (odd)", True), ("MRR check (even)", False)):
        half_mrr = {m: summarize([r for qid, r in ranks[m].items() if is_tune(qid) == keep], args.k)[1]
                    for m in args.modes}
        print(f"{half:<17}" + "".join(f"{half_mrr[m]:>13.3f}" for m in args.modes))
    per_q = {m: 1000 * seconds[m] / len(answerable) for m in args.modes}
    print(f"{'ms / question':<17}" + "".join(f"{per_q[m]:>13.0f}" for m in args.modes))

    if len(args.modes) >= 2:
        a, b = args.modes[0], args.modes[1]
        big = args.depth + 1  # treat "not found" as just past the depth so it compares as worst
        wins = {a: [], b: [], "tie": []}
        for q in answerable:
            ra, rb = ranks[a][q["id"]] or big, ranks[b][q["id"]] or big
            wins[a if ra < rb else b if rb < ra else "tie"].append(q["id"])
        print(f"\n{a} better: {', '.join(wins[a]) or '-'}")
        print(f"{b} better: {', '.join(wins[b]) or '-'}")
        print(f"same rank: {', '.join(wins['tie']) or '-'}")

    if not_found:
        print(f"\nNot-found questions: vector top-1 distance (cut-off {CUTOFF}; above it = correctly 'not found')")
        for qid, dist, path in not_found:
            verdict = "ok" if dist > CUTOFF else "FALSE MATCH"
            print(f"{qid:<5}{dist:.3f}  {verdict:<12}{path}")

    for mode in rerank_modes:
        found = sorted(v for qid, v in top_score[mode].items() if qid in {q["id"] for q in answerable})
        missing = {qid: v for qid, v in top_score[mode].items() if qid not in {q["id"] for q in answerable}}
        if not missing:
            continue
        worst = max(missing.values())
        print(f"\n{mode}: top-1 score, answerable vs not-found (can a score threshold say 'not found'?)")
        print(f"  answerable ({len(found)}): min {found[0]:.2f}, 25% {found[len(found) // 4]:.2f}, "
              f"median {statistics.median(found):.2f}")
        print(f"  not-found: " + ", ".join(f"{qid} {v:.2f}" for qid, v in missing.items()))
        below = sum(1 for v in found if v <= worst)
        print(f"  a threshold above every not-found ({worst:.2f}) would also reject {below}/{len(found)} answerable")


if __name__ == "__main__":
    main()
