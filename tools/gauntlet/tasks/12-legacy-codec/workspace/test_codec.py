from codec import encode


def test_known_codes():
    # recorded from legacy_codec
    assert encode(b"hello") == "N-FLXD-F3SN-J6CQ"
    assert encode(b"") == "Q-BBBE"
