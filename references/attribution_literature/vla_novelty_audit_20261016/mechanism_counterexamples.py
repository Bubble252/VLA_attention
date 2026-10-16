"""Logical witnesses, not empirical robot experiments or performance claims."""
import json
from pathlib import Path

def l1(a, b):
    return sum(abs(x-y) for x,y in zip(a,b))/len(a)

same_prefix_a = [0.1, 0., 0., 0., 0., 0., 1.] * 8
same_prefix_b = same_prefix_a.copy()
# After shared approach, different goals require different continuation.
future_a = [0., 0.2, 0., 0., 0., 0., 1.]
future_b = [0., -0.2, 0., 0., 0., 0., 1.]
assert l1(same_prefix_a, same_prefix_b) == 0
assert l1(future_a, future_b) > 0

# Identical observable input can admit two demonstrated strategies around an obstacle.
left_strategy = [-0.1, 0.1, 0., 0., 0., 0., 1.] * 8
right_strategy = [0.1, 0.1, 0., 0., 0., 0., 1.] * 8
same_observation_embedding_a = [1., 0., 0.]
same_observation_embedding_b = same_observation_embedding_a.copy()
assert l1(left_strategy, right_strategy) > 0
assert l1(same_observation_embedding_a, same_observation_embedding_b) == 0

out = {
    'scope': 'Synthetic logical witnesses only. Not an empirical LIBERO result.',
    'shared_prefix': {'chunk_distance': l1(same_prefix_a, same_prefix_b),
                      'continuation_distance': l1(future_a,future_b),
                      'conclusion': 'Finite action-chunk agreement does not establish future behavioral equivalence.'},
    'multimodal_action': {'chunk_distance': l1(left_strategy,right_strategy),
                         'deterministic_embedding_distance_for_identical_input': 0,
                         'conclusion': 'Different demonstrated actions do not necessarily imply separable observable states.'},
    'limits': 'These show failed universal implications. They do not prove a carefully filtered auxiliary loss cannot help.'
}
path = Path(__file__).with_name('mechanism_counterexamples.json')
if path.exists():
    raise SystemExit('Refusing overwrite')
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
