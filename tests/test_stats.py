from ai_lens import stats


def test_consistent_drop_is_worse():
    result = stats.paired([-0.3, -0.25, -0.35, -0.2, -0.3])
    assert result["verdict"] == "worse"
    assert result["low"] <= result["delta"] <= result["high"] < 0


def test_noise_is_no_clear_change():
    assert stats.paired([0.2, -0.2, 0.1, -0.1, 0.05, -0.05])["verdict"] == "no clear change"


def test_small_samples_are_not_a_verdict():
    assert stats.paired([-0.5, -0.5])["verdict"] == "not enough data"
    assert stats.unpaired([0.9, 0.9], [0.1, 0.1])["verdict"] == "not enough data"


def test_compare_prefers_matched_inputs():
    before = {"a": 0.9, "b": 0.8, "c": 0.7}
    after = {"a": 0.6, "b": 0.5, "c": 0.4, "d": 1.0}
    result = stats.compare(before, after, list(before.values()), list(after.values()))
    assert result is not None
    assert result["paired"] is True
    assert result["n"] == 3


def test_results_are_reproducible():
    values = [0.1, -0.2, 0.3, 0.05]
    assert stats.paired(values) == stats.paired(values)


def test_inputs_needed_grows_with_noise():
    noisy = stats.paired([0.2, -0.2, 0.1, -0.1, 0.05, -0.05])
    assert stats.inputs_needed(noisy) > 0
    assert stats.inputs_needed(stats.paired([0.0, 0.0, 0.0])) == 0
    assert stats.inputs_needed(None) is None
