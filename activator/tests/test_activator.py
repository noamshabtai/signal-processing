import copy

import activator.activator


def test_activator(kwargs_activator, mocker):
    kwargs = copy.deepcopy(kwargs_activator)
    system = mocker.Mock()

    with activator.activator.Activator(system_class=system, **kwargs["activator"]) as tested:
        assert tested.system is system.return_value
        mocker.spy(tested, "cleanup")
        assert not tested.completed
        tested.cleanup.assert_not_called()

    tested.cleanup.assert_called_once()
