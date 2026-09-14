"""
NSGA-II over binary feature masks. Objectives: maximize inner-CV balanced
accuracy, minimize number of selected features. Kept separate from
nested_cv.py so the optimizer can be swapped/audited independently of the
leakage-critical fitness code.
"""
import random
import numpy as np
from deap import base, creator, tools, algorithms

# DEAP's creator uses module-level globals; guard against re-registration
# if this module is imported more than once in one process (e.g. notebooks).
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # (maximize acc, minimize #features)
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def build_toolbox(n_features, eval_fn, seed=0):
    """eval_fn: mask(np.bool array) -> accuracy (float). Feature-count
    objective is computed here, not passed in, so eval_fn only ever needs
    to know about accuracy -- keeps nested_cv.py single-purpose."""
    rng = random.Random(seed)
    tb = base.Toolbox()

    def init_individual():
        # start sparse: each gene True with small prob, so initial masks
        # aren't ~50% of thousands of genes (which would be a slow, bad start
        # for p=7070 problems)
        p_on = min(0.05, 20.0 / n_features)
        return creator.Individual([1 if rng.random() < p_on else 0 for _ in range(n_features)])

    tb.register("individual", init_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)

    def evaluate(ind):
        mask = np.array(ind, dtype=bool)
        n_sel = int(mask.sum())
        if n_sel == 0:
            return (0.0, 0)
        acc = eval_fn(mask)
        return (acc, n_sel)

    tb.register("evaluate", evaluate)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
    tb.register("select", tools.selNSGA2)
    return tb


def run_nsga2(n_features, eval_fn, pop_size=60, n_gen=40, seed=0, verbose=False):
    """Returns the final Pareto front as a list of (mask: np.bool array, acc: float, n_sel: int)."""
    if pop_size % 4 != 0:
        raise ValueError(f"pop_size must be divisible by 4 for selTournamentDCD (got {pop_size})")
    tb = build_toolbox(n_features, eval_fn, seed=seed)
    random.seed(seed)
    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    pop = tb.select(pop, len(pop))
    for gen in range(n_gen):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [tb.clone(ind) for ind in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                tb.mate(c1, c2)
            tb.mutate(c1)
            tb.mutate(c2)
            del c1.fitness.values, c2.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = tb.evaluate(ind)
        pop = tb.select(pop + offspring, pop_size)
        if verbose and gen % 10 == 0:
            best_acc = max(ind.fitness.values[0] for ind in pop)
            print(f"  gen {gen:>3} best_acc={best_acc:.3f}")

    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    out = []
    seen = set()
    for ind in front:
        mask = tuple(ind)
        if mask in seen:
            continue
        seen.add(mask)
        acc, n_sel = ind.fitness.values
        out.append((np.array(ind, dtype=bool), acc, int(n_sel)))
    out.sort(key=lambda t: t[2])
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 60, 30
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

    import nested_cv as ncv
    cv = ncv.make_cv(n)
    eval_fn = lambda mask: ncv.eval_mask(X, y, mask, cv)

    front = run_nsga2(p, eval_fn, pop_size=40, n_gen=25, seed=0, verbose=True)
    print(f"\nfront size: {len(front)}")
    for mask, acc, n_sel in front[:8]:
        chosen = np.where(mask)[0].tolist()
        print(f"  n_sel={n_sel:>2} acc={acc:.3f}  features={chosen}")
