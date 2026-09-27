from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import ack_event, get_events, post_records


def sample(sequence: int, at: str, turbidity: float = 1.0, device: str = "sensor-a"):
    return {
        "device_id": device,
        "sequence": sequence,
        "sampled_at": at,
        "turbidity": turbidity,
    }


def query_adherence(
    client: TestClient,
    device: str = "sensor-a",
    window_from: str = "2026-09-23T08:00:00Z",
    window_to: str = "2026-09-23T08:05:00Z",
    interval_seconds: int = 60,
    tolerance_seconds: int = 10,
) -> dict:
    response = client.post(
        f"/devices/{device}/sampling-adherence",
        json={
            "from": window_from,
            "to": window_to,
            "interval_seconds": interval_seconds,
            "tolerance_seconds": tolerance_seconds,
        },
    )
    assert response.status_code == 200
    return response.json()


def hits(body: dict) -> list[dict]:
    return [entry for entry in body["schedule"] if entry["hit"]]


def test_exact_on_schedule_samples_all_hit(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:00Z"),
            sample(3, "2026-09-23T08:02:00Z"),
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:03:00Z")
    assert body["scheduled_count"] == 3
    assert body["hit_count"] == 3
    assert [entry["sample"]["sequence"] for entry in body["schedule"]] == [1, 2, 3]
    assert [entry["deviation_seconds"] for entry in body["schedule"]] == [
        0.0,
        0.0,
        0.0,
    ]


def test_missed_instants_are_marked(client):
    post_records(client, [sample(1, "2026-09-23T08:01:05Z")])
    body = query_adherence(client, window_to="2026-09-23T08:03:00Z")
    assert body["scheduled_count"] == 3
    assert body["hit_count"] == 1
    first, second, third = body["schedule"]
    assert first == {
        "scheduled_at": "2026-09-23T08:00:00Z",
        "hit": False,
        "sample": None,
        "deviation_seconds": None,
    }
    assert third["hit"] is False
    assert second["hit"] is True
    assert second["sample"] == {
        "sequence": 1,
        "sampled_at": "2026-09-23T08:01:05Z",
        "turbidity": 1.0,
    }
    # Signed deviation: the sample is 5 seconds late.
    assert second["deviation_seconds"] == 5.0


def test_early_sample_has_negative_deviation(client):
    post_records(client, [sample(1, "2026-09-23T07:59:57Z")])
    body = query_adherence(client, window_to="2026-09-23T08:01:00Z")
    assert body["hit_count"] == 1
    assert body["schedule"][0]["deviation_seconds"] == -3.0


def test_tolerance_is_inclusive_on_both_sides(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T07:59:50Z"),  # exactly -10s
            sample(2, "2026-09-23T08:01:10Z"),  # exactly +10s
            sample(3, "2026-09-23T08:02:11Z"),  # 11s late: outside
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:03:00Z")
    assert [entry["hit"] for entry in body["schedule"]] == [True, True, False]
    assert body["schedule"][0]["deviation_seconds"] == -10.0
    assert body["schedule"][1]["deviation_seconds"] == 10.0


def test_samples_outside_window_can_still_match_edge_instants(client):
    # Samples before `from` / after the last instant still count when they
    # fall inside the tolerance of the boundary instants.
    post_records(
        client,
        [
            sample(1, "2026-09-23T07:59:55Z"),
            sample(2, "2026-09-23T08:02:09Z"),
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:03:00Z")
    assert body["hit_count"] == 2
    assert body["schedule"][0]["sample"]["sequence"] == 1
    assert body["schedule"][2]["sample"]["sequence"] == 2


def test_one_sample_cannot_cover_two_instants(client):
    # A single sample within tolerance of two instants matches only one;
    # the earliest instant wins the lexicographic tie-break.
    post_records(client, [sample(1, "2026-09-23T08:00:30Z")])
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", tolerance_seconds=30
    )
    assert body["hit_count"] == 1
    assert body["schedule"][0]["hit"] is True
    assert body["schedule"][1]["hit"] is False


def test_hit_count_is_maximized_not_greedy(client):
    # 08:00:00 can only be served by sample 1; a nearest-first greedy for
    # 08:01:00 would steal it (dev 5) and leave 08:00:00 unmatched.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:05Z"),
            sample(2, "2026-09-23T08:01:09Z"),
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:02:00Z")
    assert body["hit_count"] == 2
    assert body["schedule"][0]["sample"]["sequence"] == 1
    assert body["schedule"][1]["sample"]["sequence"] == 2
    assert body["schedule"][1]["deviation_seconds"] == 9.0


def test_total_absolute_deviation_is_minimized(client):
    # Both assignments hit both instants; (1->08:00, 2->08:01) totals
    # 4+4=8s while the crossed one totals 56+56=112s.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:04Z"),
            sample(2, "2026-09-23T08:00:56Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", tolerance_seconds=60
    )
    assert body["schedule"][0]["sample"]["sequence"] == 1
    assert body["schedule"][1]["sample"]["sequence"] == 2
    assert body["schedule"][0]["deviation_seconds"] == 4.0
    assert body["schedule"][1]["deviation_seconds"] == -4.0


def test_tie_break_prefers_earlier_sample_for_one_instant(client):
    # Both samples are exactly 5s from the instant; the earlier sampled_at
    # wins.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:05Z"),
            sample(2, "2026-09-23T07:59:55Z"),
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:01:00Z")
    assert body["schedule"][0]["sample"]["sequence"] == 2


def test_tie_break_prefers_lower_sequence_at_same_instant(client):
    post_records(
        client,
        [
            sample(7, "2026-09-23T08:00:05Z"),
            sample(3, "2026-09-23T08:00:05Z"),
        ],
    )
    body = query_adherence(client, window_to="2026-09-23T08:01:00Z")
    assert body["schedule"][0]["sample"]["sequence"] == 3


def test_tie_break_prefers_match_at_earlier_instant(client):
    # One sample within tolerance of two instants with equal deviation:
    # matching the earlier instant yields the smaller sorted list.
    post_records(client, [sample(1, "2026-09-23T08:00:30Z")])
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", tolerance_seconds=30
    )
    assert body["schedule"][0]["hit"] is True
    assert body["schedule"][1]["hit"] is False


def test_tie_break_earliest_instant_dominates_later_ones(client):
    # Two samples, two instants, all deviations 30s: both perfect
    # assignments tie on count and total deviation, so the earliest
    # instant takes the (sampled_at, sequence)-smaller sample.
    post_records(
        client,
        [
            sample(2, "2026-09-23T08:00:30Z"),
            sample(1, "2026-09-23T08:00:30Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", tolerance_seconds=30
    )
    assert body["schedule"][0]["sample"]["sequence"] == 1
    assert body["schedule"][1]["sample"]["sequence"] == 2


def test_instant_at_to_is_not_scheduled(client):
    body = query_adherence(client, window_to="2026-09-23T08:03:00Z")
    assert [entry["scheduled_at"] for entry in body["schedule"]] == [
        "2026-09-23T08:00:00Z",
        "2026-09-23T08:01:00Z",
        "2026-09-23T08:02:00Z",
    ]


def test_fractional_seconds_are_preserved(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00.250000Z")])
    body = query_adherence(
        client, window_to="2026-09-23T08:01:00Z", tolerance_seconds=1
    )
    assert body["hit_count"] == 1
    assert body["schedule"][0]["deviation_seconds"] == 0.25
    assert body["schedule"][0]["sample"]["sampled_at"] == (
        "2026-09-23T08:00:00.250000Z"
    )


def test_window_echo_is_utc_and_device_isolation(client):
    post_records(client, [sample(1, "2026-09-23T08:00:05Z", device="sensor-a")])
    post_records(client, [sample(1, "2026-09-23T08:00:03Z", device="sensor-b")])
    body = query_adherence(
        client,
        window_from="2026-09-23T16:00:00+08:00",
        window_to="2026-09-23T16:02:00+08:00",
    )
    assert body["device_id"] == "sensor-a"
    assert body["from"] == "2026-09-23T08:00:00Z"
    assert body["to"] == "2026-09-23T08:02:00Z"
    assert body["interval_seconds"] == 60
    assert body["tolerance_seconds"] == 10
    assert body["hit_count"] == 1
    assert body["schedule"][0]["sample"]["sequence"] == 1

    other = query_adherence(client, device="sensor-b")
    assert other["hit_count"] == 1
    assert other["schedule"][0]["deviation_seconds"] == 3.0


def test_unknown_device_reports_all_missed(client):
    body = query_adherence(client, device="sensor-unknown")
    assert body["scheduled_count"] == 5
    assert body["hit_count"] == 0
    assert all(not entry["hit"] for entry in body["schedule"])


def test_late_writes_are_reflected_in_the_next_query(client):
    post_records(client, [sample(1, "2026-09-23T08:00:02Z")])
    body = query_adherence(client, window_to="2026-09-23T08:02:00Z")
    assert body["hit_count"] == 1

    post_records(client, [sample(2, "2026-09-23T08:01:03Z")])
    body = query_adherence(client, window_to="2026-09-23T08:02:00Z")
    assert body["hit_count"] == 2
    assert body["schedule"][1]["sample"]["sequence"] == 2


def test_corrections_are_reflected_in_the_next_query(client):
    post_records(client, [sample(1, "2026-09-23T08:00:30Z")])
    body = query_adherence(client, window_to="2026-09-23T08:01:00Z")
    assert body["hit_count"] == 0

    response = client.post(
        "/records/correction",
        json={
            "device_id": "sensor-a",
            "sequence": 1,
            "old_sampled_at": "2026-09-23T08:00:30Z",
            "old_turbidity": 1.0,
            "new_sampled_at": "2026-09-23T08:00:04Z",
            "new_turbidity": 1.0,
        },
    )
    assert response.status_code == 200
    body = query_adherence(client, window_to="2026-09-23T08:01:00Z")
    assert body["hit_count"] == 1
    assert body["schedule"][0]["deviation_seconds"] == 4.0


def test_query_does_not_modify_events_acks_or_change_stream(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", turbidity=12.0),
            sample(2, "2026-09-23T08:01:00Z", turbidity=13.0),
        ],
    )
    events = get_events(client, "sensor-a")
    assert ack_event(client, events["events"][0]).status_code == 200
    events_before = get_events(client, "sensor-a")
    changes_before = client.get(
        "/event-changes", params={"after": 0, "limit": 1000}
    ).json()

    body = query_adherence(client, window_to="2026-09-23T08:02:00Z")
    assert body["hit_count"] == 2

    assert get_events(client, "sensor-a") == events_before
    assert (
        client.get("/event-changes", params={"after": 0, "limit": 1000}).json()
        == changes_before
    )


def test_more_than_200_instants_returns_422(client):
    response = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T08:03:21Z",  # 201 instants at 1s spacing
            "interval_seconds": 1,
            "tolerance_seconds": 0,
        },
    )
    assert response.status_code == 422


def test_exactly_200_instants_is_accepted(client):
    body = query_adherence(
        client,
        window_to="2026-09-23T08:03:20Z",
        interval_seconds=1,
        tolerance_seconds=0,
    )
    assert body["scheduled_count"] == 200
    assert body["hit_count"] == 0
    assert len(body["schedule"]) == 200


def test_invalid_window_returns_422(client):
    bad_windows = [
        ("2026-09-23T09:00:00Z", "2026-09-23T08:00:00Z"),  # from after to
        ("2026-09-23T08:00:00Z", "2026-09-23T08:00:00Z"),  # equal instants
        ("2026-09-23T10:00:00+02:00", "2026-09-23T08:00:00Z"),  # equal in UTC
        ("2026-09-23T08:00:00", "2026-09-23T09:00:00Z"),  # naive from
        ("2026-09-23T08:00:00Z", "2026-09-23T09:00:00"),  # naive to
    ]
    for window_from, window_to in bad_windows:
        response = client.post(
            "/devices/sensor-a/sampling-adherence",
            json={
                "from": window_from,
                "to": window_to,
                "interval_seconds": 60,
                "tolerance_seconds": 10,
            },
        )
        assert response.status_code == 422, (window_from, window_to)


def test_invalid_interval_and_tolerance_return_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T09:00:00Z",
    }
    bad = [
        {"interval_seconds": 0, "tolerance_seconds": 0},
        {"interval_seconds": -1, "tolerance_seconds": 0},
        {"interval_seconds": 1.5, "tolerance_seconds": 0},
        {"interval_seconds": "sixty", "tolerance_seconds": 0},
        {"interval_seconds": 60, "tolerance_seconds": -1},
        {"interval_seconds": 60, "tolerance_seconds": 61},  # exceeds interval
        {"interval_seconds": 60, "tolerance_seconds": 1.5},
    ]
    for extra in bad:
        response = client.post(
            "/devices/sensor-a/sampling-adherence", json={**base, **extra}
        )
        assert response.status_code == 422, extra


def test_missing_and_extra_fields_return_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T09:00:00Z",
        "interval_seconds": 60,
        "tolerance_seconds": 10,
    }
    for key in base:
        payload = {name: value for name, value in base.items() if name != key}
        response = client.post(
            "/devices/sensor-a/sampling-adherence", json=payload
        )
        assert response.status_code == 422, key
    response = client.post(
        "/devices/sensor-a/sampling-adherence", json={**base, "extra": 1}
    )
    assert response.status_code == 422


def test_zero_tolerance_requires_exact_timestamps(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:00.001000Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", tolerance_seconds=0
    )
    assert [entry["hit"] for entry in body["schedule"]] == [True, False]
    assert body["schedule"][0]["deviation_seconds"] == 0.0
