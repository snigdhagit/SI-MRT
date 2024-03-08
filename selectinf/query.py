from typing import NamedTuple
import numpy as np

from .exact_reference import exact_grid_inference

class QuerySpec(NamedTuple):
    # the mean and covariance of o|S,u
    cond_mean: np.ndarray
    cond_cov: np.ndarray

    # how S enters into E[o|S,u]
    opt_linear: np.ndarray

    # constraints
    linear_part: np.ndarray
    offset: np.ndarray

    # score / randomization relationship
    M1: np.ndarray
    M2: np.ndarray
    M3: np.ndarray

    # observed values
    observed_opt_state: np.ndarray
    observed_score_state: np.ndarray
    observed_subgrad: np.ndarray
    observed_soln: np.ndarray
    observed_score: np.ndarray


class gaussian_query(object):
    r"""
    This class is the base of randomized selective inference
    based on convex programs.
    The main mechanism is to take an initial penalized program
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B)
    and add a randomization and small ridge term yielding
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B) -
        \langle \omega, B \rangle + \frac{\epsilon}{2} \|B\|^2_2
    """

    def __init__(self, randomization, perturb=None):

        """
        Parameters
        ----------
        randomization : `selection.randomized.randomization.randomization`
            Instance of a randomization scheme.
            Describes the law of $\omega$.
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """
        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    @property
    def specification(self):
        return QuerySpec(cond_mean=self.cond_mean,
                         cond_cov=self.cond_cov,
                         opt_linear=self.opt_linear,
                         linear_part=self.linear_part, # linear_part o < offset
                         offset=self.offset,
                         M1=self.M1,
                         M2=self.M2,
                         M3=self.M3,
                         observed_opt_state=self.observed_opt_state, # o
                         observed_score_state=self.observed_score_state, # S
                         observed_subgrad=self.observed_subgrad, # c
                         observed_soln=self.observed_opt_state, # o
                         observed_score=self.observed_score_state + self.observed_subgrad) # S + c

    # Methods reused by subclasses

    def randomize(self, perturb=None):
        """
        The actual randomization step.
        Parameters
        ----------
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """

        if not self._randomized:
            (self.randomized_loss,
             self._initial_omega) = self.randomization.randomize(self.loss,
                                                                 self.epsilon,
                                                                 perturb=perturb)
        self._randomized = True

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler, doc='Sampler of optimization (augmented) variables.')


    # Private methods
    def _setup_sampler(self,
                       linear_part,
                       offset,
                       opt_linear,
                       observed_subgrad,
                       dispersion=1):

        A, b = linear_part, offset

        if not np.all(A.dot(self.observed_opt_state[:self._active.sum()]) - b <= 0):
            raise ValueError('constraints not satisfied')

        (cond_mean,
         cond_cov,
         cond_precision,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov
        self.opt_linear = opt_linear
        self.linear_part = linear_part
        self.offset = offset
        self.observed_subgrad = observed_subgrad
        self.observed_score = self.observed_score_state + self.observed_subgrad


    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        cov_rand, prec = self.randomizer.cov_prec
        # omega, omega_inverse

        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...
        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        # M2 = M1.dot(self._unscaled_cov_score)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                M1,
                M2,
                M3)

    def inference(self,
                  target_spec,
                  level=0.90):

        query_spec = self.specification

        G = exact_grid_inference(query_spec,
                                 target_spec)

        return G.summary(alternatives=target_spec.alternatives,
                         level=level)