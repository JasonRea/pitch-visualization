from pitchviz.data.sequence import at_bat_sequences


def test_at_bat_count_and_ascending_order(outing_df):
    at_bats = at_bat_sequences(outing_df)

    assert len(at_bats) == 22
    numbers = [ab["at_bat_number"] for ab in at_bats]
    assert numbers == sorted(numbers)


def test_every_at_bat_starts_at_0_0_count(outing_df):
    at_bats = at_bat_sequences(outing_df)

    for ab in at_bats:
        first_pitch = ab["pitches"][0]
        assert first_pitch["balls"] == 0
        assert first_pitch["strikes"] == 0


def test_pitch_numbers_within_at_bat_are_sequential(outing_df):
    at_bats = at_bat_sequences(outing_df)

    for ab in at_bats:
        numbers = [p["pitch_number"] for p in ab["pitches"]]
        assert numbers == sorted(numbers)
        assert numbers == list(range(1, len(numbers) + 1))


def test_total_pitches_across_at_bats_matches_outing(outing_df):
    at_bats = at_bat_sequences(outing_df)

    assert sum(len(ab["pitches"]) for ab in at_bats) == len(outing_df)


def test_first_at_bat_golden_values(outing_df):
    at_bats = at_bat_sequences(outing_df)
    first = at_bats[0]

    assert first["at_bat_number"] == 1
    assert first["inning"] == 1
    assert first["inning_topbot"] == "Top"
    assert first["stand"] == "L"
    assert first["final_outcome"] == "Single"
    assert len(first["pitches"]) == 3
    assert first["pitches"][-1]["description"] == "hit_into_play"


def test_at_bat_sequences_on_empty_outing_returns_empty_list(empty_outing_df):
    assert at_bat_sequences(empty_outing_df) == []
