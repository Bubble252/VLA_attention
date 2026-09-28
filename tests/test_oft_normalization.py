import pytest
np=pytest.importorskip('numpy')
from vla_attention.benchmarks.oft_rlds import normalize


def test_bounds_gripper_and_constant_dimension():
    stats={'q01':[0.,0.,0.],'q99':[10.,1.,0.],'min':[0.,0.,0.],'max':[10.,1.,0.],
           'mask':[True,False,True]}
    x=np.array([[5.,1.,0.],[11.,0.,0.]])
    np.testing.assert_allclose(normalize(x,stats),[[0.,1.,0.],[1.,0.,0.]],atol=1e-6)


def test_matches_official_tf_normalization():
    tf=pytest.importorskip('tensorflow')
    pytest.importorskip('prismatic')
    from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio
    from prismatic.vla.constants import NormalizationType
    values=np.array([[1.,2.],[4.,2.]],np.float32)
    stats={'q01':[1.,2.],'q99':[3.,2.],'min':[1.,2.],'max':[4.,2.],'mask':[True,True]}
    converted={k:tf.constant(v) for k,v in stats.items()}
    result=normalize_action_and_proprio({'action':tf.constant(values),'observation':{'proprio':tf.constant(values)}},
                                       {'action':converted,'proprio':converted},NormalizationType.BOUNDS_Q99)
    np.testing.assert_allclose(result['action'].numpy(),normalize(values,stats),atol=1e-6)
