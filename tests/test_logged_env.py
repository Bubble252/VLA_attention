import json
import io
import pytest
from scripts.eval_oft_b0_rollout import LoggedEnv


def test_environment_success_is_passed_through_and_logged():
    class Env:
        def step(self,action):return {'observation':1},1,True,{'native':True}
    output=io.StringIO();env=LoggedEnv(Env(),output)
    result=env.step([0]*7)
    assert result[2] is True
    assert json.loads(output.getvalue())['done'] is True


def test_environment_failure_is_not_hidden():
    class Env:
        def step(self,action):raise RuntimeError('renderer unavailable')
    env=LoggedEnv(Env(),io.StringIO())
    with pytest.raises(RuntimeError):env.step([0]*7)
    assert 'renderer unavailable' in env.error
