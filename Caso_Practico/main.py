from grafo import SolutionGraph, Coeficients, Solution
from simulated_annealing import SimulatedAnnealing, print_solution_history
from data_processing import data_loader, separate_train_test
from modelo_sarima import SARIMAModel, CoeficientsTest


DATA_LOCATION = "./Caso_Practico/Electric_Production.csv"
INITIAL_TEMPERATURE = 10  # Del enunciado 10
COOLING_RATE = 0.95
MAX_ITERATIONS = 30  # Del enunciado 30
SEASONALITY_COEFICIENT = 12  # Del enunciado 12
LIMIT_COEFICIENT = 8
N_TEST_CASES = 50  # Del enunciado 50

if __name__ == "__main__":
    df = data_loader(DATA_LOCATION)
    df_train, df_test = separate_train_test(df, N_TEST_CASES)

    d, D = CoeficientsTest(df_train.IPG2211A2N).run_all_tests(SEASONALITY_COEFICIENT)

    limit_coeficients = Coeficients(
        LIMIT_COEFICIENT,
        d,
        LIMIT_COEFICIENT,
        LIMIT_COEFICIENT,
        D,
        LIMIT_COEFICIENT,
        SEASONALITY_COEFICIENT,
    )

    graph = SolutionGraph(limit_coeficients, SEASONALITY_COEFICIENT)

    initial_solution = Solution(
        Coeficients(1, 1, 1, 0, 1, 4, SEASONALITY_COEFICIENT), graph
    )

    sa = SimulatedAnnealing(
        initial_solution, INITIAL_TEMPERATURE, COOLING_RATE, MAX_ITERATIONS
    )

    def cost_function(solution: Solution) -> float:
        current_model = SARIMAModel(df_train.IPG2211A2N, solution)
        return current_model.get_test_cost(df_test.IPG2211A2N)

    best_solution = sa.optimize(cost_function)
    print("Best solution found:", best_solution)
    print_solution_history(sa.get_history())
