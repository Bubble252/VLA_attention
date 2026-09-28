from scripts.inventory_libero_rlds import partition
import pytest


def test_realistic_single_task_split_preserves_all_episodes():
    rows=[{'episode_id':str(i)} for i in range(45)]
    parts=partition(rows,17)
    assert [len(parts[k]) for k in ['train','validation','offline_eval']]==[27,9,9]
    ids=[r['episode_id'] for p in parts.values() for r in p]
    assert len(set(ids))==len(ids)==45
    assert partition(list(reversed(rows)),17)==parts


def test_too_small_split_rejected():
    with pytest.raises(ValueError,match='too few'):
        partition([{'episode_id':'one'}],17)
