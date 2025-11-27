import numpy as np
from src.core.states import QuantumState

Lamdas = np.ndarray[4]
LFunc = np.ndarray

class BellState(QuantumState):
    """
    Bell state representation.
    """
    def __init__(self, lambdas: Lamdas):
        """
        Initialize a Bell state with a given parameter w0.

        Parameters
        ----------
        w0 : float
            The Bell parameter, which must be a real number between 0 and 1.
        """
        self.lambdas: Lamdas = lambdas
        if isinstance(self.lambdas, list):
            if not all(np.isreal(l) and 0.0 <= l <= 1.0 for l in self.lambdas):
                raise InvalidBellParameterError(f"Invalid lambdas = {self.lambdas}")
        if not np.isclose(np.sum(lambdas), 1.0):
            raise TypeError(f"Invalid lambda parameters, sum of lambdas must be 1.0")

    def __repr__(self):
        return f"BellState(lambdas={self.lambdas})"

    def get_generation_sf(self, t_trunc) -> LFunc:
        """
        Generate the state quality function for the Bell state.

        Parameters
        ----------
        t_trunc : int
            The truncation time for the simulation.

        Returns
        -------
        LFunc
            An initial array of state quality values for each time step.
        """
        return np.array([self.lambdas] * t_trunc)



class InvalidBellParameterError(Exception):
    def __init__(self, message="Invalid Bell parameter w0. Must be a real number between 0 and 1."):
        super().__init__(message)


def polish_l_func(l_func: LFunc) -> LFunc:
    """
    Polish the state quality function by erasing unrealistic Bell parameters.
    This can happen when the probability is too small ~1.0e-20.
    """
    np.nan_to_num(l_func, copy=False, nan=1.0)
    np.clip(l_func, 0.0, 1.0, out=l_func)
    return l_func