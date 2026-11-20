"""Train a policy from rendered rewards, then fit two graph targets as a batch.

Run from a checkout: python -m examples.contextual_graph_learning
"""

from sdf_ui import Canvas, sdf
from sdf_ui.graph import GraphCase, GraphDatasetTrainer, GraphTrainer, RenderEnvironment
from sdf_ui.learning import (
    GaussianPolicy,
    Objective,
    ParameterStore,
    PolicyGradientTrainer,
)


def main():
    weights = ParameterStore()
    radius = weights.bind("radius", 1.0)
    field = sdf.circle((8, 8), radius)
    with Canvas((16, 16)) as canvas:
        with canvas.capture(sdf.circle((8, 8), 5)) as target:
            objective = Objective().mse("field", target)
            with canvas.compile({"field": field}, weights) as graph:
                environment = RenderEnvironment(graph, ["radius"], objective=objective)
                policy = _train_policy(environment)
                _train_differentiable_policy(canvas, policy, objective)
                _train_graph_batch(canvas, graph, objective, weights)


def _train_policy(environment):
    policy = GaussianPolicy(obs_dim=1, action_dim=1, initial_std=1.0)
    trainer = PolicyGradientTrainer(policy, seed=12)
    trainer.train(environment, lambda rng: [1.0], steps=40, batch_size=16)
    action = policy.deterministic_action([1.0])
    print("Policy radius:", float(action[0]))
    print("Policy reward:", environment([1.0], action))
    return policy


def _train_differentiable_policy(canvas, policy, objective):
    radius = policy.mean_expressions([1.0])[0].softplus()
    with canvas.compile({"field": sdf.circle((8, 8), radius)}) as policy_graph:
        with GraphTrainer(policy_graph, objective, lr=0.01) as direct:
            print("Direct policy loss:", direct.fit(10)[-1]["loss"])


def _train_graph_batch(canvas, graph, objective, weights):
    with canvas.capture(sdf.circle((8, 8), 3)) as second_target:
        cases = [
            GraphCase(graph, objective, "large"),
            GraphCase(graph, Objective().mse("field", second_target), "small"),
        ]
        with GraphDatasetTrainer(cases, lr=0.1) as batch:
            for _ in range(30):
                batch.step()
            print("Batch evaluation:", batch.evaluate())
            print("Shared radius:", weights["radius"].item())


if __name__ == "__main__":
    main()
