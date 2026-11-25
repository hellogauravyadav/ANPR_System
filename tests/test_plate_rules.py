import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.plate_rules import clean, correct, pretty


def test_clean():
    assert clean("rj 14-ab 1234.") == "RJ14AB1234"


def test_valid_plate_untouched():
    r = correct("RJ14AB1234")
    assert r.valid and r.text == "RJ14AB1234" and r.fixes == 0 and r.state == "RJ"


def test_digit_letter_confusions_repaired():
    # O->0 in RTO, I->1, S->5 in number block, 0->O / 8->B in series
    assert correct("RJ1OAB1234").text == "RJ10AB1234"
    assert correct("KA41EN1O51").text == "KA41EN1051"
    assert correct("MH12A81234").text == "MH12AB1234"
    assert correct("GJ03LK056S").text == "GJ03LK0565"
    assert correct("RJ1AAB1234").text == "RJ14AB1234"   # 4 read as A, must not become a 1-digit RTO
    assert correct("R114AB1234").text == "RI14AB1234"   # repaired by type, then state check fails
    assert not correct("R114AB1234").valid


def test_unknown_state_is_invalid():
    assert not correct("ZZ14AB1234").valid


def test_too_short_or_long_invalid():
    assert not correct("AB123").valid
    assert not correct("RJ14AB12345678").valid


def test_single_series_letter_and_old_delhi():
    assert correct("DL1CAB1234").valid
    assert correct("KL01A5673").valid


def test_bharat_series():
    r = correct("22BH1234AA")
    assert r.valid and r.state == "BH"
    assert pretty(r.text, r.valid) == "22 BH 1234 AA"


def test_ind_prefix_stripped():
    assert correct("INDKA41EN1051").text == "KA41EN1051"


def test_pretty():
    assert pretty("RJ14AB1234", True) == "RJ 14 AB 1234"
    assert pretty("DL1CAB1234", True) == "DL 1 CAB 1234"
    assert pretty("XX", False) == "XX"
