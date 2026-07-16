from repro.aicrowd_whestbench import update_submission_tracker as tracker


def test_submission_tracker_preserves_submitted_artifact_provenance():
    registry = tracker._read_json(tracker.REGISTRY_PATH)
    snapshot = tracker._read_json(tracker.CACHE_PATH)
    rows = tracker._build_rows(registry["entries"], snapshot)
    by_id = {row["submission_id"]: row for row in rows}

    original = by_id[316625]
    repackaged = by_id[316628]

    assert original["artifact_sha256"] == (
        "dae36b30f44616518526eaee8e7aec639cf0b9b05e1187416b363b921c7fd11b"
    )
    assert original["artifact_current_sha256"] == (
        "309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778"
    )
    assert not original["artifact_hash_matches"]
    assert repackaged["artifact_hash_matches"]

    report = tracker._render_report(rows, snapshot)
    assert "### Artifact integrity notes" in report
    assert "#316625: submitted `dae36b30" in report
