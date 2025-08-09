from collections.abc import Callable
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from time import perf_counter
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.managers import DictProxy
from _collections_abc import dict_keys, dict_values, Iterable

from tqdm import tqdm
from deap import creator, base, tools, algorithms  # type: ignore

from .locale import _

# ✅ Best Practice: Using DEAP's creator to define custom classes for genetic algorithm components
OUTPUT_FUNC = Callable[[str], None]
EVALUATE_FUNC = Callable[[dict], dict]
# ✅ Best Practice: Using DEAP's creator to define custom classes for genetic algorithm components
KEY_FUNC = Callable[[tuple], float]


# Create individual class used in genetic algorithm optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability


# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self) -> None:
        """"""
        self.params: dict[str, list] = {}
        # ⚠️ SAST Risk (Low): No validation on 'name' could lead to unexpected behavior if it contains special characters or is empty.
        self.target_name: str = ""

    # 🧠 ML Signal: Pattern of setting default values when optional parameters are not provided.
    def add_parameter(
        self,
        name: str,
        # ⚠️ SAST Risk (Low): No validation on 'start', 'end', and 'step' could lead to unexpected behavior if they are not numbers.
        start: float,
        end: float | None = None,
        step: float | None = None,
    ) -> tuple[bool, str]:
        """"""
        if end is None or step is None:
            self.params[name] = [start]
            # 🧠 ML Signal: Loop pattern for generating a sequence of numbers with a specific step.
            return True, _("固定参数添加成功")

        if start >= end:
            return False, _("参数优化起始点必须小于终止点")
        # 🧠 ML Signal: Pattern of storing generated sequences in a dictionary.
        # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters

        if step <= 0:
            # 🧠 ML Signal: Method that sets an attribute, indicating a setter pattern
            return False, _("参数优化步进必须大于0")
        # 🧠 ML Signal: Function signature with type hints can be used to infer expected input and output types

        value: float = start
        # 🧠 ML Signal: Usage of dictionary keys and values can indicate data structure patterns
        value_list: list[float] = []

        # 🧠 ML Signal: Use of itertools.product suggests combinatorial generation pattern
        while value <= end:
            value_list.append(value)
            value += step

        # ⚠️ SAST Risk (Low): Use of zip with strict=False can lead to unexpected behavior if lengths differ
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability.
        self.params[name] = value_list

        return True, _("范围参数添加成功，数量{}").format(len(value_list))

    def set_target(self, target_name: str) -> None:
        # ✅ Best Practice: Docstring is present but should describe the function's purpose and behavior.
        """"""
        self.target_name = target_name

    # 🧠 ML Signal: Checking if a method returns a truthy value is a common pattern.

    def generate_settings(self) -> list[dict]:
        # 🧠 ML Signal: Using a function to output messages is a common pattern.
        """"""
        keys: dict_keys = self.params.keys()
        values: dict_values = self.params.values()
        # 🧠 ML Signal: Checking if an attribute is set is a common pattern.
        products: list = list(product(*values))

        settings: list = []
        for p in products:
            setting: dict = dict(zip(keys, p, strict=False))
            settings.append(setting)

        return settings


# 🧠 ML Signal: Usage of a function to generate settings for optimization


# 🧠 ML Signal: Usage of an output function to log messages
def check_optimization_setting(
    optimization_setting: OptimizationSetting,
    # 🧠 ML Signal: Logging the size of the optimization space
    # 🧠 ML Signal: Measuring performance time
    output: OUTPUT_FUNC = print,
) -> bool:
    """"""
    if not optimization_setting.generate_settings():
        # ⚠️ SAST Risk (Low): Using 'spawn' context for multiprocessing, which is safer than 'fork' but still requires careful handling
        output(_("优化参数组合为空，请检查"))
        return False

    if not optimization_setting.target_name:
        output(_("优化目标未设置，请检查"))
        # 🧠 ML Signal: Usage of tqdm for progress tracking
        return False

    # ⚠️ SAST Risk (Low): Potential for high resource consumption with executor.map
    return True


# 🧠 ML Signal: Collecting and sorting results
# 🧠 ML Signal: Measuring end time for performance
# 🧠 ML Signal: Use of genetic algorithm for optimization
# ✅ Best Practice: Type hinting for function parameters and return type
# ✅ Best Practice: Sorting results with a key function for better organization
# 🧠 ML Signal: Calculating and logging the cost time
def run_bf_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    output: OUTPUT_FUNC = print,
) -> list[tuple]:
    """Run brutal force optimization"""
    settings: list[dict] = optimization_setting.generate_settings()
    # 🧠 ML Signal: Logging completion message with time cost

    output(_("开始执行穷举算法优化"))
    output(_("参数优化空间：{}").format(len(settings)))

    start: float = perf_counter()

    with ProcessPoolExecutor(
        max_workers,
        # ✅ Best Practice: Descriptive variable naming
        # ⚠️ SAST Risk (High): The function uses 'choice' without importing it, which can lead to NameError or unintended behavior if 'choice' is not defined.
        mp_context=get_context("spawn"),
        # ✅ Best Practice: The function lacks a docstring description, which reduces code readability and maintainability.
    ) as executor:
        # ✅ Best Practice: List comprehension for readability
        it: Iterable = tqdm(
            executor.map(evaluate_func, settings),
            # ⚠️ SAST Risk (High): 'choice' is used without being imported or defined, which can lead to security risks if 'choice' is not controlled.
            total=len(settings),
            # 🧠 ML Signal: The use of 'choice' indicates a random selection pattern, which can be a feature for ML models analyzing randomness in code.
        )
        # 🧠 ML Signal: Usage of a custom parameter generation function
        results: list[tuple] = list(it)
        results.sort(reverse=True, key=key_func)

        # 🧠 ML Signal: Random mutation based on probability
        end: float = perf_counter()
        cost: int = int(end - start)
        output(_("穷举算法优化完成，耗时{}秒").format(cost))

        # ⚠️ SAST Risk (Low): Potential for race conditions with multiprocessing context
        return results


# ⚠️ SAST Risk (Medium): Shared state across processes can lead to data corruption
def run_ga_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    # 🧠 ML Signal: Registration of genetic algorithm components
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    pop_size: int = 100,  # population size: number of individuals in each generation
    ngen: int = 30,  # number of generations: number of generations to evolve
    mu: (
        int | None
    ) = None,  # mu: number of individuals to select for the next generation
    lambda_: (
        int | None
    ) = None,  # lambda: number of children to produce at each generation
    cxpb: float = 0.95,  # crossover probability: probability that an offspring is produced by crossover
    mutpb: (
        float | None
    ) = None,  # mutation probability: probability that an offspring is produced by mutation
    indpb: float = 1.0,  # independent probability: probability for each gene to be mutated
    output: OUTPUT_FUNC = print,
) -> list[tuple]:
    """Run genetic algorithm optimization"""
    # Define functions for generate parameter randomly
    settings: list[dict] = optimization_setting.generate_settings()
    # ✅ Best Practice: Default values for parameters should be set explicitly
    parameter_tuples: list[list[tuple]] = [list(d.items()) for d in settings]

    def generate_parameter() -> list:
        """"""
        return choice(parameter_tuples)

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """"""
        # 🧠 ML Signal: Initialization of a population for genetic algorithms
        size: int = len(individual)
        paramlist: list = generate_parameter()
        # 🧠 ML Signal: Logging of genetic algorithm parameters
        for i in range(size):
            if random() < indpb:
                individual[i] = paramlist[i]
        return (individual,)

    # Set up multiprocessing Pool and Manager
    ctx: BaseContext = get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
        # Create shared dict for result cache
        cache: DictProxy[tuple, tuple] = manager.dict()

        # Set up toolbox
        # 🧠 ML Signal: Execution of a genetic algorithm
        toolbox: base.Toolbox = base.Toolbox()
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, generate_parameter
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=indpb)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("map", pool.map)
        toolbox.register(
            "evaluate",
            ga_evaluate,
            cache,
            evaluate_func,
            # 🧠 ML Signal: Sorting results based on a key function
            key_func,
        )

        # ✅ Best Practice: Convert list to tuple for immutability and use as a cache key
        # Set default values for DEAP parameters if not specified
        if mu is None:
            # ✅ Best Practice: Use cache to avoid redundant computations
            mu = int(pop_size * 0.8)

        if lambda_ is None:
            lambda_ = pop_size
        # ✅ Best Practice: Convert list to dict for named access in evaluate_func

        if mutpb is None:
            # ✅ Best Practice: Return a tuple for consistency with function signature
            # 🧠 ML Signal: Usage of evaluate_func suggests a pattern for function evaluation
            # ✅ Best Practice: Store result in cache to optimize future evaluations
            # 🧠 ML Signal: Usage of key_func suggests a pattern for extracting key metrics
            mutpb = 1.0 - cxpb

        total_size: int = len(parameter_tuples)
        pop: list = toolbox.population(pop_size)

        # Run ga optimization
        output(_("开始执行遗传算法优化"))
        output(_("参数优化空间：{}").format(total_size))
        output(_("每代族群总数：{}").format(pop_size))
        output(_("优良筛选个数：{}").format(mu))
        output(_("迭代次数：{}").format(ngen))
        output(_("交叉概率：{:.0%}").format(cxpb))
        output(_("突变概率：{:.0%}").format(mutpb))
        output(_("个体突变概率：{:.0%}").format(indpb))

        start: float = perf_counter()

        algorithms.eaMuPlusLambda(
            pop, toolbox, mu, lambda_, cxpb, mutpb, ngen, verbose=True
        )

        end: float = perf_counter()
        cost: int = int(end - start)

        output(_("遗传算法优化完成，耗时{}秒").format(cost))

        results: list = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results


def ga_evaluate(
    cache: dict, evaluate_func: Callable, key_func: Callable, parameters: list
) -> tuple[float,]:
    """
    Functions to be run in genetic algorithm optimization.
    """
    tp: tuple = tuple(parameters)
    if tp in cache:
        result: dict = cache[tp]
    else:
        setting: dict = dict(parameters)
        result = evaluate_func(setting)
        cache[tp] = result

    value: float = key_func(result)
    return (value,)
