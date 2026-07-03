"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    batch_id = payload["batch_id"]
    profile = ctx.tools.batch_profile(batch_id)
    if "error" in profile:
        return Verdict(alert=False, pillar="checks", reason=profile["error"])

    row_count = profile.get("row_count")
    null_rate = profile.get("null_rate", {}).get("customer_id")
    mean_amount = profile.get("mean_amount")
    staleness_min = profile.get("staleness_min")

    b = ctx.baseline

    # Check volume_spike / volume_drop
    if row_count < b["row_count_min"] or row_count > b["row_count_max"]:
        return Verdict(alert=True, pillar="checks", reason="row_count anomaly")

    # Check null_spike
    if null_rate > b["null_rate_max"]:
        return Verdict(alert=True, pillar="checks", reason="null_rate anomaly")

    # Check distribution_shift using a tighter 2.4 sigma limit
    mean_center = (b["mean_amount_min"] + b["mean_amount_max"]) / 2.0
    mean_half_width = (b["mean_amount_max"] - b["mean_amount_min"]) / 2.0
    sigma = mean_half_width / 3.0
    custom_max = mean_center + 2.4 * sigma
    custom_min = mean_center - 2.4 * sigma

    if mean_amount < custom_min or mean_amount > custom_max:
        return Verdict(alert=True, pillar="checks", reason="mean_amount anomaly")

    # Check freshness_lag
    if staleness_min > b["staleness_min_max"]:
        return Verdict(alert=True, pillar="checks", reason="staleness anomaly")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    contract_id = payload["contract_id"]
    checkpoint_batch_id = payload["checkpoint_batch_id"]
    diff = ctx.tools.contract_diff(contract_id, checkpoint_batch_id)
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts", reason=diff["error"])

    violations = diff.get("violations", [])
    freshness_delay_min = diff.get("freshness_delay_min")

    # Check schema_break and type_violation
    if len(violations) > 0:
        return Verdict(alert=True, pillar="contracts", reason=f"violations: {violations}")

    # Check SLA violations (freshness delay)
    b = ctx.baseline
    if freshness_delay_min > b["freshness_delay_max_min"]:
        return Verdict(alert=True, pillar="contracts", reason="freshness_delay anomaly")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    run_id = payload["run_id"]
    job = payload["job"]
    slice_data = ctx.tools.lineage_graph_slice(run_id)
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage", reason=slice_data["error"])

    duration_ms = slice_data.get("duration_ms")
    actual_upstream = slice_data.get("actual_upstream", [])
    actual_downstream_count = slice_data.get("actual_downstream_count")

    b = ctx.baseline

    # Check runtime_anomaly
    if duration_ms > b["lineage_duration_ms_max"]:
        return Verdict(alert=True, pillar="lineage", reason="duration anomaly")

    # Check orphan_output
    outputs = payload.get("outputs", [])
    if actual_downstream_count < len(outputs):
        return Verdict(alert=True, pillar="lineage", reason="orphan output anomaly")

    # Check missing_upstream using stateful set analysis
    actual_set = set(actual_upstream)
    job_upstreams = ctx.state.setdefault("job_upstreams", {})

    if job not in job_upstreams:
        job_upstreams[job] = actual_set
        return Verdict(alert=False, pillar="lineage")
    else:
        expected_set = job_upstreams[job]
        if actual_set.issubset(expected_set) and actual_set != expected_set:
            return Verdict(alert=True, pillar="lineage", reason="missing upstream anomaly")
        else:
            job_upstreams[job] = expected_set.union(actual_set)

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    feature_view = payload["feature_view"]
    batch_id = payload["batch_id"]
    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    mean_shift_sigma = drift.get("mean_shift_sigma")

    b = ctx.baseline
    # Use tighter 2.4 * baseline threshold
    if mean_shift_sigma > 2.4 * b["feature_mean_shift_sigma_max"]:
        return Verdict(alert=True, pillar="ai_infra", reason="feature_skew anomaly")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    corpus = payload["corpus"]
    chunk_batch_id = payload["chunk_batch_id"]
    drift = ctx.tools.embedding_drift(corpus, chunk_batch_id)
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"])

    centroid_shift = drift.get("centroid_shift")
    avg_doc_age_days = drift.get("avg_doc_age_days")

    b = ctx.baseline
    # Use 0.90 * baseline threshold for centroid shift
    if centroid_shift > 0.90 * b["embedding_centroid_shift_max"]:
        return Verdict(alert=True, pillar="ai_infra", reason="embedding drift anomaly")

    # Use 0.88 * baseline threshold for corpus staleness
    if avg_doc_age_days > 0.88 * b["corpus_avg_doc_age_days_max"]:
        return Verdict(alert=True, pillar="ai_infra", reason="corpus staleness anomaly")

    return Verdict(alert=False, pillar="ai_infra")
