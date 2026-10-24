from scheduler import agenda, find_slots, has_conflict, to_utc, weekly

EXISTING = [("2026-06-10T00:00:00Z", "2026-06-10T01:00:00Z")]     # 10:00-11:00 Melbourne (AEST, +10)


def test_simple():
    assert has_conflict(EXISTING, "2026-06-10T10:30", "2026-06-10T11:30")
    assert not has_conflict(EXISTING, "2026-06-10T11:00", "2026-06-10T12:00")   # back to back


def test_to_utc_plain_day():
    assert to_utc("2026-06-10T09:00", "Australia/Melbourne").isoformat() == "2026-06-09T23:00:00+00:00"


def test_weekly_plain_week():
    assert weekly("2026-06-03T09:00", 2) == ["2026-06-02T23:00:00Z", "2026-06-09T23:00:00Z"]


def test_find_slots_plain_day():
    busy = [("2026-06-10T00:00:00Z", "2026-06-10T01:00:00Z")]      # 10:00-11:00 local
    assert find_slots(busy, "2026-06-10", 60, 3) == ["2026-06-09T14:00:00Z",
                                                      "2026-06-09T15:00:00Z",
                                                      "2026-06-09T16:00:00Z"]


def test_agenda_plain_day():
    busy = [("2026-06-09T23:30:00Z", "2026-06-10T00:15:00Z"),      # 09:30-10:15 local
            ("2026-06-10T02:00:00Z", "2026-06-10T03:00:00Z")]      # 12:00-13:00 local
    assert agenda(busy, "2026-06-10") == [
        "2026-06-10T09:30+10:00 2026-06-09T23:30:00Z 45 2026-06-10T10:15+10:00",
        "2026-06-10T12:00+10:00 2026-06-10T02:00:00Z 60 2026-06-10T13:00+10:00",
    ]


def test_find_slots_notice_plain_day():
    # 16:00Z is 02:00 local on 06-10, so the 1-hour slots start there
    assert find_slots([], "2026-06-10", 60, 2, now="2026-06-09T16:00:00Z") == ["2026-06-09T16:00:00Z",
                                                                              "2026-06-09T17:00:00Z"]
    assert find_slots([], "2026-06-10", 60, 2, now="2026-06-09T14:00:00Z", notice=120) == ["2026-06-09T16:00:00Z",
                                                                                          "2026-06-09T17:00:00Z"]
