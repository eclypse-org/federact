from fedclypse.selection import at_most, select_all, uniform

IDS = [f"c{i}" for i in range(10)]


def test_select_all_returns_all_as_new_list():
    result = select_all(IDS)
    assert result == IDS
    assert result is not IDS


def test_uniform_selects_fraction_rounded():
    result = uniform(0.5, seed=1)(IDS)
    assert len(result) == 5
    assert set(result) <= set(IDS)
    assert len(set(result)) == 5


def test_uniform_is_deterministic_for_a_seed():
    assert uniform(0.3, seed=7)(IDS) == uniform(0.3, seed=7)(IDS)


def test_uniform_selects_at_least_one():
    assert len(uniform(0.01, seed=0)(IDS)) == 1


def test_uniform_on_empty_returns_empty():
    assert uniform(0.5, seed=0)([]) == []


def test_at_most_caps_at_k():
    result = at_most(3, seed=2)(IDS)
    assert len(result) == 3
    assert set(result) <= set(IDS)


def test_at_most_caps_at_population_size():
    assert len(at_most(100, seed=2)(IDS)) == 10
