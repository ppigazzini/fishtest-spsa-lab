import nevergrad as ng
from pentamodel import PentaModel


class GameProvider:
    """
    Provides methods to generate pentanomials and scores using a given objective function.
    Fair model for how matches on fishtest work.
    """

    def __init__(self, objective_function, rounds=1000):
        """
        Initialize the GameProvider with an objective function and number of rounds.

        Args:
            objective_function (callable): Function to evaluate parameters.
            rounds (int, optional): Number of rounds for pentanomial generation. Defaults to 1000.
        """
        self.objective_function = objective_function
        self.rounds = rounds

    def next_pentanomial(self, x1, x2):
        """
        Generate the next pentanomial playing round games, based on the objective function.

        Args:
            x1 (float): First parameter.
            x2 (float): Second parameter.

        Returns:
            Pentanomial: The generated pentanomial for the given parameters.
        """
        ELo = self.objective_function(x1, x2)
        p = PentaModel(ELo)

        return p.return_match_pentanomial(self.rounds)

    def next_score(self, x1, x2):
        """
        Generate the next score playing rounds games, based on the objective function.

        Args:
            x1 (float): First parameter.
            x2 (float): Second parameter.

        Returns:
            float: Expected score for the given parameters.
        """
        ELo = self.objective_function(x1, x2)
        p = PentaModel(ELo)

        return p.pentonomial_to_score(p.return_match_pentanomial(self.rounds))


if __name__ == "__main__":
    """
    Main execution block for running optimization using Nevergrad.
    Defines an objective function, sets up the optimizer, and prints optimal parameters.
    """

    def objective_function(x1, x2):
        """
        Objective function to minimize, Elo advantage of the opponent based on two parameters.
        Mimics 5 Elo, minimum at (100, 200).
        Range of parameters +- 100.

        Args:
            x1 (float): First parameter.
            x2 (float): Second parameter.

        Returns:
            float: Value of the objective function.
        """

        value = 5 * (((x1 - 100) / 100) ** 2 + ((x2 - 200) / 100) ** 2)

        return value

    roundsPerBatch = 100
    totalBatches = 1000

    # Initialize GameProvider with the objective function
    games = GameProvider(objective_function, rounds=roundsPerBatch)

    # Define the nevergrad instrumentation for two bounded scalar parameters
    instrum = ng.p.Instrumentation(
        ng.p.Scalar(init=10).set_bounds(0, 200).set_mutation(sigma=50),
        ng.p.Scalar(init=110).set_bounds(100, 300).set_mutation(sigma=50),
    )

    # Define the optimizer
    optimizer = ng.optimizers.TBPSA(parametrization=instrum, budget=totalBatches)

    def negate_score(x1, x2):
        "We must maximize our score, with a minimizer"
        return -games.next_score(x1, x2)

    # Run optimization
    recommendation = optimizer.minimize(negate_score)

    x1, x2 = recommendation.value[0]
    print(f"After {totalBatches * roundsPerBatch * 2} games...")
    print("Optimal parameters:", x1, x2)
    print("Value of the objective function :", objective_function(x1, x2))
