import pytest

from team_knowledge_assistant.Ingest.store import reciprocal_rank_fusion


def ids(fused):
    return [id_ for id_, _ in fused]


def test_single_list_keeps_its_order():
    assert ids(reciprocal_rank_fusion([[3, 1, 2]])) == [3, 1, 2]


def test_score_is_sum_of_one_over_k_plus_rank():
    fused = dict(reciprocal_rank_fusion([[7, 8], [8]], k=60))
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[8] == pytest.approx(1 / 62 + 1 / 61)


def test_found_by_both_beats_top_of_one():
    # 9 is only 5th in each list, but appearing twice outscores being 1st once.
    vector = [1, 2, 3, 4, 9]
    keyword = [5, 6, 7, 8, 9]
    assert ids(reciprocal_rank_fusion([vector, keyword]))[0] == 9


def test_same_rank_in_different_lists_ties():
    fused = dict(reciprocal_rank_fusion([[1], [2]]))
    assert fused[1] == fused[2]


def test_each_id_appears_once():
    fused = ids(reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]]))
    assert sorted(fused) == [1, 2, 3]


def test_smaller_k_favours_the_top_ranks_more():
    # 1 is 1st in one list; 2 is 4th in both. k=60: 1/61 < 2/64. k=1: 1/2 > 2/5.
    lists = [[1, 9, 7, 2], [8, 6, 5, 2]]
    assert ids(reciprocal_rank_fusion(lists, k=60))[0] == 2
    assert ids(reciprocal_rank_fusion(lists, k=1))[0] == 1


def test_empty_lists_give_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_default_weights_match_unweighted():
    lists = [[1, 2, 3], [3, 4]]
    assert reciprocal_rank_fusion(lists, weights=[1, 1]) == reciprocal_rank_fusion(lists)


def test_weight_scales_a_lists_votes():
    fused = dict(reciprocal_rank_fusion([[7], [8]], k=60, weights=[1, 2]))
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[8] == pytest.approx(2 / 61)


def test_heavier_list_wins_a_disagreement():
    # Each list puts a different id first; the heavier list's pick comes out on top.
    assert ids(reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[1, 1.5]))[0] == 2
    assert ids(reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[1.5, 1]))[0] == 1


def test_weights_must_match_lists():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1])
