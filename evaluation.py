import csv
import json
import statistics
import time
from pathlib import Path

from job_matcher import retrieve_candidates


ROOT = Path(__file__).parent
JOBS_PATH = ROOT / "evaluation_data" / "job_descriptions.json"
LABELS_PATH = ROOT / "evaluation_data" / "relevance_labels.csv"


def load_data():
    jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    labels = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            labels.setdefault(row["job_id"], {})[row["candidate_id"]] = int(row["relevance"])
    return jobs, labels


def recall_at_k(relevances, k):
    relevant = {candidate for candidate, score in relevances.items() if score > 0}
    retrieved = set(list(relevances)[:k])
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(relevances, k):
    retrieved = list(relevances)[:k]
    return sum(relevances.get(candidate, 0) > 0 for candidate in retrieved) / k


def reciprocal_rank(relevances):
    for rank, score in enumerate(relevances.values(), start=1):
        if score > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances, k):
    import math
    ranked = list(relevances.values())[:k]
    dcg = sum((2 ** score - 1) / math.log2(rank + 2) for rank, score in enumerate(ranked))
    ideal = sorted(relevances.values(), reverse=True)[:k]
    idcg = sum((2 ** score - 1) / math.log2(rank + 2) for rank, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate():
    jobs, labels = load_data()
    metrics = []
    latencies = []
    for job in jobs:
        start = time.perf_counter()
        result = retrieve_candidates(job["description"], top_k=10)
        latencies.append((time.perf_counter() - start) * 1000)
        expected = labels.get(job["job_id"], {})
        ranked_ids = {}
        for match in result["top_matches"]:
            candidate_id = Path(match["resume_path"]).stem.lower()
            ranked_ids[candidate_id] = expected.get(candidate_id, 0)
        metrics.append({
            "job_id": job["job_id"],
            "recall@10": recall_at_k(ranked_ids, 10),
            "precision@10": precision_at_k(ranked_ids, 10),
            "mrr": reciprocal_rank(ranked_ids),
            "ndcg@10": ndcg_at_k(ranked_ids, 10),
        })
    return {
        "per_job": metrics,
        "mean_recall@10": statistics.mean(item["recall@10"] for item in metrics),
        "mean_precision@10": statistics.mean(item["precision@10"] for item in metrics),
        "mean_mrr": statistics.mean(item["mrr"] for item in metrics),
        "mean_ndcg@10": statistics.mean(item["ndcg@10"] for item in metrics),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        },
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
