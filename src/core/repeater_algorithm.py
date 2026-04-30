from copy import deepcopy
from collections.abc import Iterable
import logging

import numpy as np

from src.core.states import QuantumState
from src.core.werner.state import WFunc, WernerState, polish_w_func
from src.core.bell.state import BellState, LFunc, polish_l_func
from src.types.protocol_types import find_right_segment
from src.types.repeater_types import PMF, AsymProtocol, QuantumProtocol, SimParameters, SymProtocol, checkAsymProtocol, validate_heterogeneous_parameters
try:
    import cupy as cp # type: ignore
    _cupy_exist = True
except (ImportError, ModuleNotFoundError):
    _cupy_exist = False

from src.core.werner.protocol_units import werner_join
from src.core.werner.protocol_units_efficient import werner_join_efficient
from src.core.bell.protocol_units import bell_join
from src.core.bell.protocol_units_efficient import bell_join_efficient

__all__ = ["RepeaterChainEvaluation", "compute_unit", "werner_join", "repeater_sim"]


class HashableParameters():
    def __init__(self, parameters):
        self.parameters = parameters
    
    def __hash__(self):
        return hash(frozenset(self.parameters.items()))
    
    def __eq__(self, other):
        if isinstance(other, HashableParameters):
            return frozenset(self.parameters.items()) == frozenset(other.parameters.items())
        return False
    
    def set(self, key, value):
        self.parameters[key] = value

    def __repr__(self):
        return f"HashableParameters{self.parameters['protocol'] or '()'}"
    

class RepeaterChainEvaluation():
    def __init__(self, state_type = WernerState, use_fft = True, 
                 use_gpu = False, efficient = True, twirling = True):
        self.state_type = state_type
        self.use_fft = use_fft
        self.use_gpu = use_gpu
        self.gpu_threshold = 1000000
        self.efficient = efficient
        self.twirling = twirling
        self.zero_padding_size = None
        self._qutip = False

    def iterative_convolution(self,
            func, shift=0, first_func=None, p_swap=None):
        """
        Calculate the convolution iteratively:
        first_func * func * func * ... * func
        It returns the sum of all iterative convolution:
        first_func + first_func * func + first_func * func * func ...

        Parameters
        ----------
        func: array-like
            The function to be convolved in array form.
            It is always a probability distribution.
        shift: int, optional
            For each k the function will be shifted to the right. Using for
            time-out mt_cut.
        first_func: array-like, optional
            The first_function in the convolution. If not given, use func.
            It can be different, e.g., 
            first_func is `P_s` and func is `P_f`.
            It is upper bounded by 1.
            It can be a PMF, or a state function.
        p_swap: float, optimal
            Entanglement swap success probability.

        Returns
        -------
        sum_convolved: array-like
            The result of the sum of all convolutions.
        """
        if (
            first_func is None or 
            self.state_type == WernerState or
            self.state_type == BellState
        ):
            is_dm = False
        else:
            is_dm = True

        trunc = len(func)

        # determine the required number of convolution
        if shift != 0:
            # cut-off is added here.
            # because it is a constant, we only need size/mt_cut convolution.
            max_k = int(np.ceil((trunc/shift)))
        else:
            max_k = trunc

        if self.state_type == WernerState:
            coverage = np.sum(func)
        elif self.state_type == BellState:
            coverage = sum([lambdas[0] for lambdas in func])

        if p_swap is not None:
            assert not isinstance(p_swap, list), "p_swap should not be a list"
            pf = coverage * (1 - p_swap)
        else:
            pf = coverage
        
        with np.errstate(divide='ignore'):
            if pf <= 0.:  # pf ~ 0 and round-off error
                max_k = trunc
            else:
                max_k = min(max_k, (-52 - np.log(trunc))/ np.log(pf))
        max_k = int(max_k)
        
        # If first_func is not given, convolve func with itself
        if first_func is None: first_func = func

        if self.state_type == WernerState:
            if not is_dm: # shape (4,4,trunc) if density matrix
                first_func = first_func.reshape((trunc, 1, 1)) # shape (1,1,trunc) if Werner state
            first_func = np.transpose(first_func, (1, 2, 0))
        
        elif self.state_type == BellState:
            if not is_dm:
                first_func = first_func.reshape((trunc, 4, 1)) # shape (1, 4, trunc) if Bell state
            first_func = np.transpose(first_func, (1, 2, 0))
            func = func.reshape((trunc, 4, 1))
            func = np.transpose(func, (1, 2, 0))

        # Convolution
        result = np.empty(first_func.shape, first_func.dtype)
        for i in range(first_func.shape[0]):
            for j in range(first_func.shape[1]):
                if self.state_type == WernerState:
                    result[i][j] = self.iterative_convolution_helper(
                        func, first_func[i][j], trunc, shift, p_swap, max_k)
                elif self.state_type == BellState:
                    result[i][j] = self.iterative_convolution_helper(
                        func[i][j], first_func[i][j], trunc, shift, p_swap, max_k)
                
        # Permute the indices back
        result = np.transpose(result, (2, 0, 1))
        if not is_dm:
            if self.state_type == WernerState:
                result = result.reshape(trunc)
            elif self.state_type == BellState:
                result = result.reshape([trunc, 4])
        return result


    def iterative_convolution_helper(
            self, func, first_func, trunc, shift, p_swap, max_k):
        # initialize the result array
        sum_convolved = np.zeros(trunc, dtype=first_func.dtype)
        if p_swap is not None:
            sum_convolved[:len(first_func)] = p_swap * first_func
        else:
            sum_convolved[:len(first_func)] = first_func

        if shift <= trunc:
            zero_state = np.zeros(shift, dtype=func.dtype)
            func = np.concatenate([zero_state, func])[:trunc]

        # decide what convolution to use and prepare the data
        convolved = first_func
        if self.use_fft: # Use geometric sum in Fourier space
            shape = 2 * trunc - 1
            # TODO: shape = 2 * len(first_func) * len(func) - 1 , should be more accurate 
            # but impractical for large arrays
            
            # The following is from SciPy, they choose the size to be 2^n,
            # It increases the accuracy.
            if self.zero_padding_size is not None:
                shape = self.zero_padding_size
            else:
                shape = 2 ** np.ceil(np.log2(shape)).astype(int)
            if self.use_gpu and not _cupy_exist:
                logging.warning("CuPy not found, using CPU.")
                self.use_gpu = False
            if self.use_gpu and shape > self.gpu_threshold:
                # transfer the data to GPU
                sum_convolved = cp.asarray(sum_convolved)
                convolved = cp.asarray(convolved)
                func = cp.asarray(func)
            if self.use_gpu and shape > self.gpu_threshold:
                # use CuPy fft
                ifft = cp.fft.ifft
                fft = cp.fft.fft
                to_real = cp.real
            else:
                # use NumPy fft
                ifft = np.fft.ifft
                fft = np.fft.fft
                to_real = np.real

            convolved_fourier = fft(convolved, shape)
            func_fourier = fft(func, shape)

            if p_swap is not None:
                result= ifft(
                    p_swap*convolved_fourier / (1 - (1-p_swap) * func_fourier))
            else:
                result= ifft(convolved_fourier / (1 - func_fourier))

            # validity check
            last_term = abs(result[-1])
            if last_term > 10e-16:
                logging.warning(
                    f"The size of zero-padded array, shape={shape}, "
                    "for the Fourier transform is not big enough. "
                    "The resulting circular convolution might contaminate "
                    "the distribution."
                    f"The deviation is as least {float(last_term):.0e}.")

            result = to_real(result[:trunc])
            if self.use_gpu and shape > self.gpu_threshold:
                result = cp.asnumpy(result)

        else:  # Use exact convolution (not using FFT)
            zero_state = np.zeros(trunc - len(convolved), dtype=convolved.dtype)
            convolved = np.concatenate([convolved, zero_state])
            for k in range(1, max_k):
                convolved = np.convolve(convolved[:trunc], func[:trunc])
                if p_swap is not None:
                    coeff = p_swap*(1-p_swap)**(k)
                    sum_convolved += coeff * convolved[:trunc]
                else:
                    sum_convolved += convolved[:trunc]
            result = sum_convolved
        return result


    def swapping(self,
            parameters: SimParameters,
            pmf1: PMF, sf1, pmf2: PMF, sf2, p_swap,
            cutoff, t_coh, cut_type):
        """
        Calculate the waiting time and average Werner parameter with time-out
        for entanglement swap.

        Parameters
        ----------
        parameters: dict
            A dictionary contains the parameters of
            the repeater and the simulation.
        pmf1, pmf2: array-like 1-D
            The waiting time distribution of the two input links.
        sf1, sf2: array-like 1-D -- TODO: here, the type can be different
            The state quality (e.g., Werner parameter) as function of T of the two input links.
        p_swap: float
            The success probability of entanglement swap.
        cutoff: int or float
            The memory time cut-off, werner parameter cut-off, or
            run time cut-off.
        t_coh: int
            The coherence time.
        cut_type: str
            `memory_time`, `fidelity` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the entanglement swap.
        state_out: array-like 1-D
            The Werner parameter as function of T of the entanglement swap.
        """
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0

        if self.state_type == WernerState:
            if self.efficient and cut_type == "memory_time":
                join_links = werner_join_efficient
            else:
                join_links = werner_join

            # P'_f
            pf_cutoff = join_links(
                pmf1, pmf2, sf1, sf2, ycut=False,
                cutoff=cutoff, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
            # P'_s
            ps_cutoff = join_links(
                pmf1, pmf2, sf1, sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type, evaluate_func="1", t_coh=t_coh)
            # P_f or P_s (Differs only by a constant p_swap)
            pmf_cutoff = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=ps_cutoff)
            del ps_cutoff
            # Pr(Tout = t)
            pmf_swap = self.iterative_convolution(
                pmf_cutoff, shift=0, p_swap=p_swap)

            # Wsuc * P_s
            state_suc = join_links(
                pmf1, pmf2, w_func1=sf1, w_func2=sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                t_coh=t_coh, evaluate_func="w1w2")
            # Wprep * Pr(Tout = t)
            state_prep = self.iterative_convolution(
                pf_cutoff,
                shift=shift, first_func=state_suc)
            del pf_cutoff, state_suc
            # Wout * Pr(Tout = t)
            state_out = self.iterative_convolution(
                pmf_cutoff, shift=0,
                first_func=state_prep, p_swap=p_swap)
            del pmf_cutoff

            # TODO: check this part more carefully
            with np.errstate(divide='ignore', invalid='ignore'):
                if len(state_out.shape) == 1:
                    state_out[1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
                    state_out = np.where(np.isnan(state_out), 1., state_out)
                else:
                    state_out = np.transpose(state_out, (1, 2, 0))
                    state_out[:,:,1:] /= pmf_swap[1:]  # 0-th element has 0 pmf
                    state_out = np.transpose(state_out, (2, 1, 0))

        elif self.state_type == BellState:
            if self.efficient and cut_type == "memory_time":
                join_links = bell_join_efficient
            else:
                join_links = bell_join
            depolar_rate = parameters.get("depolarizing_rate", 0.)
            dephase_rate = parameters.get("dephasing_rate", 0.)

            # Tile pmf for computation for Bell states
            if self.state_type == BellState:
                pmf1 = np.tile(pmf1[:, np.newaxis], 4)
                pmf2 = np.tile(pmf2[:, np.newaxis], 4)
             
            # P'_f
            pf_cutoff = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2,
                ycut=False, cutoff=cutoff, cut_type=cut_type,
                evaluate_func="1", 
                depolar_rate=depolar_rate, dephase_rate=dephase_rate)
            # P'_s
            ps_cutoff = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2,
                ycut=True, cutoff=cutoff, cut_type=cut_type,
                evaluate_func="1",
                depolar_rate=depolar_rate, dephase_rate=dephase_rate)
            # P_f or P_s
            pmf_cutoff = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=ps_cutoff)
            del ps_cutoff
            # Pr(Tout = t)
            pmf_swap = self.iterative_convolution(
                pmf_cutoff, shift=0, p_swap=p_swap)
            # Wsuc * P_s
            state_suc = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2, 
                ycut=True, cutoff=cutoff, cut_type=cut_type,
                evaluate_func="f1f2", # TODO: bad naming
                depolar_rate=depolar_rate, dephase_rate=dephase_rate)
            # Wprep * Pr(Tout = t)
            state_prep = self.iterative_convolution(
                pf_cutoff,
                shift=shift, first_func=state_suc)
            del pf_cutoff, state_suc
            # Wout * Pr(Tout = t)
            state_out = self.iterative_convolution(
                pmf_cutoff, shift=0,
                first_func=state_prep, p_swap=p_swap)
            del pmf_cutoff

            # Checks whther state_out is 1D or 3D 
            # TODO: check this part more carefully
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(1, len(state_out)):
                    state_out[i] = state_out[i] / pmf_swap[i]
                    for j in range(0, state_out.shape[1]):
                        state_out[i][j] = np.where(np.isnan(state_out[i][j]), 1., state_out[i][j] ) #if nan, replace state_out with 1

            assert len(state_out.shape) == 2 and state_out.shape[1] == 4, "The state_out is not in the correct shape."

            # De-tile pmf for Bell states
            if self.state_type == BellState:
                pmf_swap = pmf_swap[:,0]
            
        return pmf_swap, state_out


    def distillation(self,
            parameters: SimParameters,
            pmf1, sf1, pmf2, sf2,
            cutoff, t_coh, cut_type):
        """
        Calculate the waiting time and average Werner parameter
        with time-out for the distillation.

        Parameters
        ----------
        parameters: dict
            A dictionary contains the parameters of
            the repeater and the simulation.
        pmf1, pmf2: array-like 1-D
            The waiting time distribution of the two input links.
        sf1, sf2: array-like 1-D -- TODO: here, the type can be different
            The state quality (e.g., Werner parameter) as function of T of the two input links.
        cutoff: int or float
            The memory time cut-off, werner parameter cut-off, or 
            run time cut-off.
        t_coh: int
            The coherence time.
        cut_type: str
            `memory_time`, `fidelity` or `run_time`.

        Returns
        -------
        t_pmf: array-like 1-D
            The waiting time distribution of the distillation.
        w_func: array-like 1-D
            The Werner parameter as function of T of the distillation.
        """
        if cut_type == "memory_time":
            shift = cutoff
        else:
            shift = 0

        if self.state_type == WernerState:
            if self.efficient and cut_type == "memory_time":
                join_links = werner_join_efficient
            else:
                join_links = werner_join

            # P'_f  cutoff attempt when cutoff fails
            pf_cutoff = join_links(
                pmf1, pmf2, sf1, sf2, ycut=False,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="1", t_coh=t_coh)
            # P'_ss  cutoff attempt when cutoff and dist succeed
            pss_cutoff = join_links(
                pmf1, pmf2, sf1, sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="0.5+0.5w1w2", t_coh=t_coh)
            # P_s  dist attempt when dist succeeds
            ps_dist = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=pss_cutoff)
            del pss_cutoff
            # P'_sf  cutoff attempt when cutoff succeeds but dist fails
            psf_cutoff = join_links(
                pmf1, pmf2, sf1, sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="0.5-0.5w1w2", t_coh=t_coh)
            # P_f  dist attempt when dist fails
            pf_dist = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=psf_cutoff)
            del psf_cutoff
            # Pr(Tout = t)
            pmf_dist = self.iterative_convolution(
                pf_dist, shift=0,
                first_func=ps_dist)
            del ps_dist

            # Wsuc * P'_ss
            state_suc = join_links(
                pmf1, pmf2, sf1, sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="w1+w2+4w1w2", t_coh=t_coh)
            # Wprep * P_s
            state_prep = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=state_suc)
            del pf_cutoff, state_suc
            # Wout * Pr(Tout = t)
            state_out = self.iterative_convolution(
                pf_dist, shift=0,
                first_func=state_prep)
            del pf_dist, state_prep

            with np.errstate(divide='ignore', invalid='ignore'):
                state_out[1:] /= pmf_dist[1:]
                state_out = np.where(np.isnan(state_out), 1., state_out)
        
        elif self.state_type == BellState:
            if self.efficient and cut_type == "memory_time":
                join_links = bell_join_efficient
            else:
                join_links = bell_join
            depolar_rate = parameters.get("depolarizing_rate", 0.)
            dephase_rate = parameters.get("dephasing_rate", 0.)

            # Tile pmf for computation for Bell states
            if self.state_type == BellState:
                pmf1 = np.tile(pmf1[:, np.newaxis], 4)
                pmf2 = np.tile(pmf2[:, np.newaxis], 4)

            # P'_f  cutoff attempt when cutoff fails
            pf_cutoff = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2, ycut=False,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="1", 
                depolar_rate=depolar_rate, dephase_rate=dephase_rate, twirling=self.twirling)
            # P'_ss  cutoff attempt when cutoff and dist succeed
            pss_cutoff = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="0.5+0.5f1f2",  depolar_rate=depolar_rate, dephase_rate=dephase_rate, twirling=self.twirling)
            # P_s  dist attempt when dist succeeds
            ps_dist = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=pss_cutoff)
            del pss_cutoff
            # P'_sf  cutoff attempt when cutoff succeeds but dist fails
            psf_cutoff = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="0.5-0.5f1f2",
                depolar_rate=depolar_rate, dephase_rate=dephase_rate, twirling=self.twirling)
            # P_f  dist attempt when dist fails
            pf_dist = self.iterative_convolution(
                pf_cutoff, shift=shift,
                first_func=psf_cutoff)
            del psf_cutoff
            # Pr(Tout = t)
            pmf_dist = self.iterative_convolution(
                pf_dist, shift=0,
                first_func=ps_dist)
            del ps_dist

            # Wsuc * P'_ss
            state_suc = join_links(
                pmf1, pmf2, lambda_func1=sf1, lambda_func2=sf2, ycut=True,
                cutoff=cutoff, cut_type=cut_type,
                evaluate_func="f1+f2+4f1f2", 
                depolar_rate=depolar_rate, dephase_rate=dephase_rate, twirling=self.twirling)
            assert len(state_suc.shape) == 2 and state_suc.shape[1] == 4

            # Wprep * P_s
            state_prep = self.iterative_convolution(
                func=pf_cutoff, shift=shift,
                first_func=state_suc)
            del pf_cutoff, state_suc
            assert len(state_prep.shape) == 2 and state_prep.shape[1] == 4

            # Wout * Pr(Tout = t)
            state_out = self.iterative_convolution(
                func=pf_dist, shift=0,
                first_func=state_prep)
            del pf_dist, state_prep
            assert len(state_out.shape) == 2 and state_out.shape[1] == 4

            with np.errstate(divide='ignore', invalid='ignore'):
                state_out[1:] /= pmf_dist[1:]
                state_out = np.where(np.isnan(state_out), 1., state_out)

            # De-tile pmf for Bell states
            if self.state_type == BellState:
                pmf_dist = pmf_dist[:,0]
            
            assert len(state_out.shape) == 2 and state_out.shape[1] == 4, "The state_out is not in the correct shape."
            pmf_dist[0] = 0.
            state_out[0] = np.zeros(4, dtype=state_out.dtype)
            
        return pmf_dist, state_out


    def compute_unit(self,
            parameters, pmf1: PMF, sf1, pmf2: PMF = None, sf2 = None,
            unit_kind="swap", step_size=1):
        """
        Calculate the the waiting time distribution and
        the Werner parameter of a protocol unit swap or distillation.
        Cut-off is built in swap or distillation.

        Parameters
        ----------
        parameters: dict
            A dictionary contains the parameters of
            the repeater and the simulation.
        pmf1, pmf2: array-like 1-D
            The waiting time distribution of the two input links.
        sf1, sf2: array-like 1-D -- TODO: here, the type can be different
            The state quality (e.g., Werner parameter) as function of T of the two input links.
        unit_kind: str
            "swap" or "dist"

        Returns
        -------
        t_pmf, sf: array-like 1-D -- TODO: here, the type can be different
            The output waiting time and state quality (e.g., Werner parameters)
        """
        # If only one link is given, assume the operation is done on two identical links
        if pmf2 is None:
            pmf2 = pmf1
        if sf2 is None:
            sf2 = sf1
        
        p_gen = parameters["p_gen"]
        p_swap = parameters["p_swap"]
        t_coh = parameters.get("t_coh", np.inf)

        cut_type = parameters.get("cut_type", "memory_time")
        if "cutoff" in parameters.keys():
            cutoff = parameters["cutoff"]
        elif cut_type == "memory_time":
            cutoff = parameters.get("mt_cut", np.iinfo(int).max)
        elif cut_type == "fidelity":
            cutoff = parameters.get("w_cut", 1.0e-16)  # shouldn't be zero
            if cutoff == 0.:
                cutoff = 1.0e-16
        elif cut_type == "run_time":
            cutoff = parameters.get("rt_cut", np.iinfo(int).max)
        else:
            cutoff = np.iinfo(int).max

        # TODO: refactor to type check in the appropriate place
        # type check (allow for list of p_gen)
        if isinstance(p_gen, Iterable):
            if not all(np.isreal(p) for p in p_gen):
                raise TypeError("p_gen must be a float number.")
        elif not np.isreal(p_gen):
            raise TypeError("p_gen must be a float number.")
        if isinstance(t_coh, Iterable):
            if not all(np.isreal(t) for t in t_coh):
                raise TypeError("The coherence time must be a real number.")
        elif not np.isreal(t_coh):
            raise TypeError(
                f"The coherence time must be a real number, not {t_coh}")
        if not np.isreal(p_swap):
            raise TypeError("p_swap must be a float number.")
        if cut_type in ("memory_time", "run_time") and not np.issubdtype(type(cutoff), np.integer):
            raise TypeError(f"Time cut-off must be an integer. not {cutoff}")
        if cut_type == "fidelity" and not (cutoff >= 0. or cutoff < 1.):
            raise TypeError(f"Fidelity cut-off must be a real number between 0 and 1.")

        # Perform swap or distillation
        if unit_kind == "swap":
            pmf, sf = self.swapping(
                parameters,
                pmf1, sf1, pmf2, sf2, p_swap,
                cutoff=cutoff, t_coh=t_coh, cut_type=cut_type)
        elif unit_kind == "dist":
            pmf, sf = self.distillation(
                parameters,
                pmf1, sf1, pmf2, sf2,
                cutoff=cutoff, t_coh=t_coh, cut_type=cut_type)

        # Polish the state quality function from non-sensical values
        if isinstance(sf, WFunc):
            sf = polish_w_func(sf)
        elif isinstance(sf, LFunc):
            sf = polish_l_func(sf)

        # Check probability coverage
        coverage = np.sum(pmf)
        if coverage < 0.99:
            logging.warning(
                "The truncation time only covers {:.2f}% of the distribution, "
                "please increase t_trunc.\n".format(
                    coverage*100))
        
        return pmf, sf


    def nested_protocol(self, parameters, all_level=False):
        """
        Compute the waiting time and the Werner parameter of a symmetric
        repeater protocol.

        Parameters
        ----------
        parameters: dict
            A dictionary contains the parameters of
            the repeater and the simulation.
        all_level: bool
            If true, Return a list of the result of all the levels.
            [(t_pmf0, sf0), (t_pmf1, sf1), ...]

        Returns
        -------
        t_pmf, sf: array-like 1-D -- TODO: here, the type can be different
            The output waiting time and state quality (e.g., Werner parameters)
        
        TODO: rename it as doubling_protocol, remove all_level
        """
        parameters = deepcopy(parameters)
        protocol: SymProtocol = parameters["protocol"]

        # Preliminary check
        if isinstance(protocol, int):
            # In case of protocol with only one operation, ensure protocol is treated as a tuple
            if protocol == 0 or protocol == 1: 
                protocol = (protocol,)
            else:
                raise ValueError("The protocol must be a tuple of 0 and 1, "
                                 "or a singleton 0 or 1.")

        p_gen: int = parameters["p_gen"]
        if isinstance(p_gen, Iterable):
            raise NotImplementedError("Heterogeneous nested protocols are not supported, "
                                      "Please specify the protocol as a binary tree.")

        if "tau" in parameters:  # backward compatibility
            parameters["mt_cut"] = parameters.pop("tau")
        if "cutoff_dict" in parameters.keys():
            cutoff_dict = parameters["cutoff_dict"]
            mt_cut = cutoff_dict.get("memory_time", np.iinfo(int).max)
            w_cut = cutoff_dict.get("fidelity", 1.e-8)
            rt_cut = cutoff_dict.get("run_time", np.iinfo(int).max)
        else:
            mt_cut = parameters.get("mt_cut", np.iinfo(int).max)
            w_cut = parameters.get("w_cut", 1.e-8)
            rt_cut = parameters.get("rt_cut", np.iinfo(int).max)
        if "cutoff" in parameters:
            cutoff = parameters["cutoff"]
        if not isinstance(mt_cut, Iterable):
            mt_cut = (mt_cut,) * len(protocol)
        else:
            mt_cut = tuple(mt_cut)
        if not isinstance(w_cut, Iterable):
            w_cut = (w_cut,) * len(protocol)
        else:
            w_cut = tuple(w_cut)
        if not isinstance(rt_cut, Iterable):
            rt_cut = (rt_cut,) * len(protocol)
        else:
            rt_cut = tuple(rt_cut)

        # Truncation time for the protocol
        t_trunc = parameters["t_trunc"]

        # GEN: Elementary link generation
        t_list: np.ndarray = np.arange(1, t_trunc)
        pmf: PMF = p_gen * (1 - p_gen)**(t_list - 1)
        pmf: PMF = np.concatenate((np.array([0.]), pmf))

        if self.state_type == WernerState:
            sf: WFunc = WernerState(parameters["w0"]).get_generation_sf(t_trunc)
        elif self.state_type == BellState:
            sf: LFunc = BellState(parameters["lambdas"]).get_generation_sf(t_trunc)

        if all_level:
            full_result = [(pmf, sf)]
        
        total_step_size = 1

        # Compute protocol units
        i = 0
        while i < len(protocol):
            operation = protocol[i]
            if "cutoff" in parameters and isinstance(cutoff, Iterable):
                parameters["cutoff"] = cutoff[i]
            parameters["mt_cut"] = mt_cut[i]
            parameters["w_cut"] = w_cut[i]
            parameters["rt_cut"] = rt_cut[i]

            if operation == 0:
                pmf, sf = self.compute_unit(
                    parameters, pmf, sf, unit_kind="swap", step_size=total_step_size)
            elif operation == 1:
                pmf, sf = self.compute_unit(
                    parameters, pmf, sf, unit_kind="dist", step_size=total_step_size)
            
            if all_level:
                full_result.append((pmf, sf))
            i += 1

        if all_level:
            return full_result
        else:
            return pmf, sf
    

    def asymmetric_homogeneous_protocol(self, parameters, number_of_segments):
        """
        Compute the waiting time and the Werner parameter of an asymmetric homogeneous protocol.
        Parameters
        ----------
        parameters: dict
            A dictionary contains the (homogeneous) parameters of
            the repeater and the simulation.
        
        number_of_segments: int
            The number of segments in the protocol.

        Returns
        -------
        t_pmf, w_func: array-like 1-D
            The output waiting time and Werner parameters
        """
        S = number_of_segments
        parameters = deepcopy(parameters)

        protocol: AsymProtocol = parameters["protocol"]

        p_gen = parameters["p_gen"]
        t_trunc = parameters["t_trunc"]
        
        # In case of 1-level protocol, ensure protocol is treated as a tuple
        if isinstance(protocol, str): 
            protocol = (protocol,)

        # Each segment will keep a distribution for waiting time and Werner parameter
        segments = []

        # GEN: Elementary link generation
        t_list: np.ndarray = np.arange(1, t_trunc)
        pmf: PMF = p_gen * (1 - p_gen)**(t_list - 1)
        pmf: PMF = np.concatenate((np.array([0.]), pmf))

        if self.state_type == WernerState:
            sf: WFunc = WernerState(parameters["w0"]).get_generation_sf(t_trunc)
        elif self.state_type == BellState:
            sf: LFunc = BellState(parameters["lambdas"]).get_generation_sf(t_trunc)
    
        # For each segment, generate its distribution
        for _ in range(S):
            segments.append((pmf, sf))

        # Compute step by step the whole protocol
        # Given idx as the index of the segment (or left segment in case of swap)
        for i in range(len(protocol)):
            step: str = protocol[i]
            operation = step[0]
            idx = int(step[1:]) 
            curr_segment = segments[idx] 
            
            if operation == 's':
                next_idx = find_right_segment(segments, idx)
                next_segment = segments[next_idx]
                pmf, w_func = self.compute_unit(
                    parameters, *curr_segment, *next_segment, unit_kind="swap", step_size=1)
                segments[idx] = None
                segments[next_idx] = (pmf, w_func)

            elif operation == 'd':
                pmf, w_func = self.compute_unit(
                    parameters, *curr_segment, unit_kind="dist", step_size=1)
                segments[idx] = (pmf, w_func)

        return next((segm for segm in segments if segm is not None), (None, None))


    def asymmetric_heterogeneous_protocol(self, parameters, number_of_segments):
        """
        Compute the waiting time and the Werner parameter of an asymmetric protocol.
        Parameters
        ----------
        parameters: dict
            A dictionary contains the (heterogeneous) parameters of
            the repeater and the simulation.

        number_of_segments: int
            The number of segments in the protocol.

        Returns
        -------
        t_pmf, w_func: array-like 1-D
            The output waiting time and Werner parameters
        """
        # Preliminary check
        validate_heterogeneous_parameters(parameters, number_of_segments, self.state_type)
        
        S = number_of_segments
        parameters = deepcopy(parameters)

        protocol = parameters["protocol"]

        p_gens = parameters["p_gen"]
        t_trunc = parameters["t_trunc"]

        if self.state_type == WernerState:
            t_cohs = parameters.get("t_coh", [np.inf]*(S+1))
        else:
            depolarizing_rates = parameters.get("depolarizing_rate", [0.]*(S+1))
            dephasing_rates = parameters.get("dephasing_rate", [0.]*(S+1))

        # In case of 1-level protocol, ensure protocol is treated as a tuple
        if isinstance(protocol, str):
            protocol = (protocol,)

        # Each segment will keep
        # - an integer for the segment length
        # - a distribution for waiting time and Werner parameter
        segments = []

        # GEN: Elementary link generation
        # For each segment, generate its distribution
        t_list = np.arange(1, t_trunc)
        for i in range(S):
            pmf = p_gens[i] * (1 - p_gens[i])**(t_list - 1)
            pmf = np.concatenate((np.array([0.]), pmf))
            if self.state_type == WernerState:
                # TODO: refactor this as below
                sf: WFunc = WernerState(parameters["w0"]).get_generation_sf(t_trunc, i)
            elif self.state_type == BellState:
                sf: LFunc = BellState(parameters["lambdas"][i]).get_generation_sf(t_trunc)

            # Keep track of segment endpoints
            segments.append((pmf, sf, i, i+1))

        # Compute step by step the whole protocol
        # Given idx as the index of the segment (or left segment in case of swap)
        for i in range(len(protocol)):
            step: str = protocol[i]
            operation = step[0]
            idx = int(step[1:])
            curr_segment = segments[idx]

            if operation == 's':
                next_idx = find_right_segment(segments, idx)
                next_segment = segments[next_idx]
                assert curr_segment[3] == next_segment[2], f"Segments {curr_segment[2]} and {next_segment[3]} are not compatible."

                if self.state_type == WernerState:
                    parameters["t_coh"] = [t_cohs[curr_segment[2]], t_cohs[next_segment[2]], t_cohs[next_segment[3]]] 
                elif self.state_type == BellState:
                    parameters["depolarizing_rate"] = [depolarizing_rates[curr_segment[2]], depolarizing_rates[next_segment[2]], depolarizing_rates[next_segment[3]]]
                    parameters["dephasing_rate"] = [dephasing_rates[curr_segment[2]], dephasing_rates[next_segment[2]], dephasing_rates[next_segment[3]]]

                pmf, w_func = self.compute_unit(
                    parameters, curr_segment[0], curr_segment[1], next_segment[0], next_segment[1], 
                    unit_kind="swap", step_size=1)
                segments[idx] = None
                segments[next_idx] = (pmf, w_func, curr_segment[2], next_segment[3])
            
            elif operation == 'd':
                if self.state_type == WernerState:
                    parameters["t_coh"] = [t_cohs[curr_segment[2]], t_cohs[curr_segment[3]]]
                elif self.state_type == BellState:
                    parameters["depolarizing_rate"] = [depolarizing_rates[curr_segment[2]], depolarizing_rates[curr_segment[3]]]
                    parameters["dephasing_rate"] = [dephasing_rates[curr_segment[2]], dephasing_rates[curr_segment[3]]]

                pmf, w_func = self.compute_unit(
                    parameters, curr_segment[0], curr_segment[1], unit_kind="dist", step_size=1)
                segments[idx] = (pmf, w_func, curr_segment[2], curr_segment[3])

        final_segment = next((segm for segm in segments if segm is not None), (None, None))
        return (final_segment[0], final_segment[1])


def repeater_sim(parameters, all_level=False, state_type=WernerState):
    """
    Functional wrapper for nested (Li et al. 2021) or asymmetric (La Corte et al. 2025) protocol evaluation.
    A first typecheck on the protocol is done to identify the evaluation to run, i.e.
        - If the protocol is a tuple of integers, run the nested protocol
        - Otherwise, the tuples should be of strings, so run the asymmetric protocol
    Notice that cut-offs and 'all_level' are not implemented yet for asymmetric protocols.

    Parameters
    ----------
    parameters: dict
        A dictionary contains the parameters of
        the repeater and the simulation.
        
    all_level: bool
        If true, Return a list of the result of all the levels.
        [(t_pmf0, w_func0), (t_pmf1, w_func1) ...]

    Returns
    -------
    t_pmf, w_func: array-like 1-D
        The output waiting time and Werner parameters
    TODO: implement typing for simulation parameters (check repeater types)
            and  implement type for protocol types: symmetric (doubling), asymmetric (heuristic)
    --> TODO: refactor asymmetric protocol in input to be a binary tree
    TODO: remove all_level
    """
    simulator = RepeaterChainEvaluation(state_type)

    # Redirect to symmetric (nested) or asymmetric protocol
    if isinstance(parameters["protocol"], Iterable) and all(isinstance(i, int) for i in parameters["protocol"]):
        return simulator.nested_protocol(parameters=parameters, all_level=all_level)
    
    elif isinstance(parameters["protocol"], Iterable) and all(isinstance(i, str) for i in parameters["protocol"]):
        # Preliminary check for asymmetric protocols
        number_of_segments = checkAsymProtocol(parameters["protocol"])
        if "cutoff" in parameters:
            raise NotImplementedError("Cut-offs are not implemented for heterogeneous protocols.")
        if "all_level" in parameters:
            raise NotImplementedError("All levels are not implemented for heterogeneous protocols.")
        
        # Redirect to homogeneous or heterogeneous protocol
        if isinstance(parameters["p_gen"], Iterable):
            return simulator.asymmetric_heterogeneous_protocol(parameters, number_of_segments)
        else:
            return simulator.asymmetric_homogeneous_protocol(parameters, number_of_segments)
    else:
        raise ValueError("The protocol must be a tuple of integers or strings.")
