from bayes_opt import BayesianOptimization,UtilityFunction
import numpy as np

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1
optimizer = BayesianOptimization(
    f=None,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
    allow_duplicate_points=True
)
# next_point_to_probe = optimizer.suggest(utility)
# print("Next point to probe is:", next_point_to_probe)
# target = black_box_function(**next_point_to_probe)
# print("Found the target value to be:", target)
# optimizer.register(
#     params=next_point_to_probe,
#     target=target,
# )
# 随机生成一些点
candidates_xs = np.random.uniform(-2, 2, (100, 2))
print(candidates_xs)
xs = np.random.uniform(-2, 2, (10, 2))
ys = []
for _ in range(5):
    # next_point = optimizer.suggest(utility)
    # print(next_point)
    next_point = optimizer.suggest_in_candidates(candidates=candidates_xs, utility_function=utility)
    print("add point",next_point)
    
    target = black_box_function(**next_point)
    # 从candidates_xs中删除这个点
    candidates_xs = np.delete(candidates_xs, np.where((candidates_xs == [next_point['x'],next_point['y']]).all(axis=1)), axis=0)
    print("Found the target value to be:", target)
    optimizer.register(
        params=next_point,
        target=target,
    )
    print(optimizer.get_ucb(xs, utility))
    
print(optimizer.max)