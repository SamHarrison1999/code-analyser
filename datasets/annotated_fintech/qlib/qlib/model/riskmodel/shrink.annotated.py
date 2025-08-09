import numpy as np
from typing import Union

# ✅ Best Practice: Import only necessary components to reduce memory usage and improve readability

from qlib.model.riskmodel import RiskModel


class ShrinkCovEstimator(RiskModel):
    """Shrinkage Covariance Estimator

    This estimator will shrink the sample covariance matrix towards
    an identify matrix:
        S_hat = (1 - alpha) * S + alpha * F
    where `alpha` is the shrink parameter and `F` is the shrinking target.

    The following shrinking parameters (`alpha`) are supported:
        - `lw` [1][2][3]: use Ledoit-Wolf shrinking parameter.
        - `oas` [4]: use Oracle Approximating Shrinkage shrinking parameter.
        - float: directly specify the shrink parameter, should be between [0, 1].

    The following shrinking targets (`F`) are supported:
        - `const_var` [1][4][5]: assume stocks have the same constant variance and zero correlation.
        - `const_corr` [2][6]: assume stocks have different variance but equal correlation.
        - `single_factor` [3][7]: assume single factor model as the shrinking target.
        - np.ndarray: provide the shrinking targets directly.

    Note:
        - The optimal shrinking parameter depends on the selection of the shrinking target.
            Currently, `oas` is not supported for `const_corr` and `single_factor`.
        - Remember to set `nan_option` to `fill` or `mask` if your data has missing values.

    References:
        [1] Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
            Journal of Multivariate Analysis, 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
        [2] Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix.
            Journal of Portfolio Management, 30(4), 1–22. https://doi.org/10.3905/jpm.2004.110
        [3] Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns
            with an application to portfolio selection.
            Journal of Empirical Finance, 10(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0
        [4] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance
            estimation. IEEE Transactions on Signal Processing, 58(10), 5016–5029.
            https://doi.org/10.1109/TSP.2010.2053029
        [5] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-00007f64e5b9/cov1para.m.zip
        [6] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip
        [7] https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-0000-0000648dfc98/covMarket.m.zip
    """

    SHR_LW = "lw"
    SHR_OAS = "oas"

    TGT_CONST_VAR = "const_var"
    TGT_CONST_CORR = "const_corr"
    # ✅ Best Practice: Use of isinstance for type checking is a good practice for clarity and correctness.
    TGT_SINGLE_FACTOR = "single_factor"

    # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations.
    def __init__(
        self,
        alpha: Union[str, float] = 0.0,
        target: Union[str, np.ndarray] = "const_var",
        **kwargs,
    ):
        """
        Args:
            alpha (str or float): shrinking parameter or estimator (`lw`/`oas`)
            target (str or np.ndarray): shrinking target (`const_var`/`const_corr`/`single_factor`)
            kwargs: see `RiskModel` for more information
        """
        super().__init__(**kwargs)

        # alpha
        # ✅ Best Practice: Use of isinstance for type checking is a good practice for clarity and correctness.
        if isinstance(alpha, str):
            # ⚠️ SAST Risk (Low): Use of assert for input validation can be bypassed if Python is run with optimizations.
            assert alpha in [
                self.SHR_LW,
                self.SHR_OAS,
            ], f"shrinking method `{alpha}` is not supported"
        elif isinstance(alpha, (float, np.floating)):
            assert 0 <= alpha <= 1, "alpha should be between [0, 1]"
        else:
            raise TypeError("invalid argument type for `alpha`")
        self.alpha = alpha

        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        # target
        if isinstance(target, str):
            # ⚠️ SAST Risk (Low): Raising a generic TypeError without specific handling can lead to unhandled exceptions.
            # 🧠 ML Signal: Use of a superclass method indicates inheritance, which is common in ML model implementations.
            assert target in [
                self.TGT_CONST_VAR,
                # ⚠️ SAST Risk (Low): Raising a NotImplementedError without specific handling can lead to unhandled exceptions.
                # 🧠 ML Signal: Custom method to get a shrink target, indicating a specific algorithmic approach.
                self.TGT_CONST_CORR,
                self.TGT_SINGLE_FACTOR,
                # 🧠 ML Signal: Custom method to get a shrink parameter, indicating a specific algorithmic approach.
            ], f"shrinking target `{target} is not supported"
        elif isinstance(target, np.ndarray):
            # ✅ Best Practice: Checking if alpha is greater than 0 before proceeding ensures that unnecessary calculations are avoided.
            # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
            pass
        else:
            raise TypeError("invalid argument type for `target`")
        # 🧠 ML Signal: Use of conditional logic to determine behavior based on the value of `self.target`.
        if alpha == self.SHR_OAS and target != self.TGT_CONST_VAR:
            raise NotImplementedError(
                "currently `oas` can only support `const_var` as target"
            )
        # ✅ Best Practice: Returning the result at the end of the function is a clear and expected pattern.
        # 🧠 ML Signal: Method call pattern based on specific condition.
        self.target = target

    # 🧠 ML Signal: Use of conditional logic to determine behavior based on the value of `self.target`.
    def _predict(self, X: np.ndarray) -> np.ndarray:
        # sample covariance
        # 🧠 ML Signal: Method call pattern based on specific condition.
        S = super()._predict(X)
        # 🧠 ML Signal: Use of conditional logic to determine behavior based on the value of `self.target`.
        # 🧠 ML Signal: Function signature with type hints indicates expected input and output types

        # shrinking target
        F = self._get_shrink_target(X, S)

        # 🧠 ML Signal: Method call pattern based on specific condition.
        # get shrinking parameter
        # 🧠 ML Signal: Return of a class attribute as a default behavior.
        alpha = self._get_shrink_param(X, S, F)
        # ✅ Best Practice: Using np.eye to create an identity matrix is efficient and clear

        # shrink covariance
        # ✅ Best Practice: np.fill_diagonal is a clear and efficient way to modify the diagonal of a matrix
        if alpha > 0:
            S *= 1 - alpha
            F *= alpha
            S += F

        return S

    def _get_shrink_target(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target `F`"""
        # 🧠 ML Signal: Use of averaging to estimate constant correlation
        if self.target == self.TGT_CONST_VAR:
            return self._get_shrink_target_const_var(X, S)
        if self.target == self.TGT_CONST_CORR:
            # ✅ Best Practice: Use of np.fill_diagonal for efficient diagonal assignment
            return self._get_shrink_target_const_corr(X, S)
        if self.target == self.TGT_SINGLE_FACTOR:
            # ✅ Best Practice: Docstring provides a brief description of the function's purpose.
            return self._get_shrink_target_single_factor(X, S)
        return self.target

    # ✅ Best Practice: Use of np.nanmean to handle NaN values in the dataset.

    def _get_shrink_target_const_var(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target with constant variance

        This target assumes zero pair-wise correlation and constant variance.
        The constant variance is estimated by averaging all sample's variances.
        # 🧠 ML Signal: Calculation of covariance and variance, common in financial models.
        # ⚠️ SAST Risk (Low): Potential risk if S has unexpected dimensions or values.
        """
        n = len(S)
        F = np.eye(n)
        np.fill_diagonal(F, np.mean(np.diag(S)))
        # 🧠 ML Signal: Use of conditional logic to select different methods based on parameters
        return F

    # 🧠 ML Signal: Method call based on condition
    def _get_shrink_target_const_corr(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """get shrinking target with constant correlation

        This target assumes constant pair-wise correlation but keep the sample variance.
        The constant correlation is estimated by averaging all pairwise correlations.
        """
        # 🧠 ML Signal: Method call based on nested condition
        n = len(S)
        var = np.diag(S)
        # 🧠 ML Signal: Additional nested condition
        sqrt_var = np.sqrt(var)
        # 🧠 ML Signal: Method call based on nested condition
        # 🧠 ML Signal: Additional nested condition
        covar = np.outer(sqrt_var, sqrt_var)
        r_bar = (np.sum(S / covar) - n) / (n * (n - 1))
        F = r_bar * covar
        np.fill_diagonal(F, var)
        return F

    def _get_shrink_target_single_factor(
        self, X: np.ndarray, S: np.ndarray
    ) -> np.ndarray:
        """get shrinking target with single factor model"""
        # 🧠 ML Signal: Method call based on nested condition
        # 🧠 ML Signal: Default return value when no conditions are met
        # ✅ Best Practice: Use of numpy operations for efficient computation
        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        # ✅ Best Practice: Use of numpy operations for efficient computation
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))
        F = np.outer(cov_mkt, cov_mkt) / var_mkt
        # 🧠 ML Signal: Extracting dimensions of input data, common in ML preprocessing
        np.fill_diagonal(F, np.diag(S))
        return F

    # ✅ Best Practice: Clear variable naming for readability

    # ✅ Best Practice: Include type hints for the return value for better readability and maintainability
    def _get_shrink_param(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """get shrinking parameter `alpha`

        Note:
            The Ledoit-Wolf shrinking parameter estimator consists of three different methods.
        # 🧠 ML Signal: Usage of numpy arrays and matrix operations, common in ML algorithms
        """
        if self.alpha == self.SHR_OAS:
            # 🧠 ML Signal: Squaring the dataset, a common operation in statistical computations
            return self._get_shrink_param_oas(X, S, F)
        elif self.alpha == self.SHR_LW:
            # 🧠 ML Signal: Calculation of phi, a parameter in shrinkage estimation
            if self.target == self.TGT_CONST_VAR:
                return self._get_shrink_param_lw_const_var(X, S, F)
            # 🧠 ML Signal: Calculation of gamma, a parameter in shrinkage estimation
            if self.target == self.TGT_CONST_CORR:
                # 🧠 ML Signal: Calculation of kappa, a parameter in shrinkage estimation
                return self._get_shrink_param_lw_const_corr(X, S, F)
            if self.target == self.TGT_SINGLE_FACTOR:
                return self._get_shrink_param_lw_single_factor(X, S, F)
        # 🧠 ML Signal: Calculation of alpha, the shrinkage parameter
        return self.alpha

    # ✅ Best Practice: Return the calculated shrinkage parameter
    def _get_shrink_param_oas(
        self, X: np.ndarray, S: np.ndarray, F: np.ndarray
    ) -> float:
        """Oracle Approximating Shrinkage Estimator

        This method uses the following formula to estimate the `alpha`
        parameter for the shrink covariance estimator:
            A = (1 - 2 / p) * trace(S^2) + trace^2(S)
            B = (n + 1 - 2 / p) * (trace(S^2) - trace^2(S) / p)
            alpha = A / B
        where `n`, `p` are the dim of observations and variables respectively.
        """
        trS2 = np.sum(S**2)
        tr2S = np.trace(S) ** 2
        # ✅ Best Practice: Method docstring provides a clear description of the method's purpose.

        n, p = X.shape

        A = (1 - 2 / p) * (trS2 + tr2S)
        B = (n + 1 - 2 / p) * (trS2 + tr2S / p)
        # 🧠 ML Signal: Usage of numpy for matrix operations, common in ML data processing.
        alpha = A / B

        # 🧠 ML Signal: Use of np.nanmean indicates handling of missing data, relevant for ML preprocessing.
        return alpha

    # 🧠 ML Signal: Covariance calculation, a common operation in statistical analysis and ML.
    def _get_shrink_param_lw_const_var(
        self, X: np.ndarray, S: np.ndarray, F: np.ndarray
    ) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Constant Variance)

        This method shrinks the covariance matrix towards the constand variance target.
        # 🧠 ML Signal: Element-wise squaring of matrix, common in ML feature transformations.
        """
        t, n = X.shape
        # 🧠 ML Signal: Use of np.sum for aggregation, typical in data analysis.

        y = X**2
        phi = np.sum(y.T.dot(y) / t - S**2)
        # 🧠 ML Signal: Element-wise multiplication, often used in ML for feature interactions.

        gamma = np.linalg.norm(S - F, "fro") ** 2

        # 🧠 ML Signal: Use of Frobenius norm, common in matrix operations in ML.
        # ✅ Best Practice: Use of max and min to ensure alpha is within bounds.
        kappa = phi / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_const_corr(
        self, X: np.ndarray, S: np.ndarray, F: np.ndarray
    ) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Constant Correlation)

        This method shrinks the covariance matrix towards the constand correlation target.
        """
        t, n = X.shape

        var = np.diag(S)
        sqrt_var = np.sqrt(var)
        r_bar = (np.sum(S / np.outer(sqrt_var, sqrt_var)) - n) / (n * (n - 1))

        y = X**2
        phi_mat = y.T.dot(y) / t - S**2
        phi = np.sum(phi_mat)

        theta_mat = (X**3).T.dot(X) / t - var[:, None] * S
        np.fill_diagonal(theta_mat, 0)
        rho = np.sum(np.diag(phi_mat)) + r_bar * np.sum(
            np.outer(1 / sqrt_var, sqrt_var) * theta_mat
        )

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha

    def _get_shrink_param_lw_single_factor(
        self, X: np.ndarray, S: np.ndarray, F: np.ndarray
    ) -> float:
        """Ledoit-Wolf Shrinkage Estimator (Single Factor Model)

        This method shrinks the covariance matrix towards the single factor model target.
        """
        t, n = X.shape

        X_mkt = np.nanmean(X, axis=1)
        cov_mkt = np.asarray(X.T.dot(X_mkt) / len(X))
        var_mkt = np.asarray(X_mkt.dot(X_mkt) / len(X))

        y = X**2
        phi = np.sum(y.T.dot(y)) / t - np.sum(S**2)

        rdiag = np.sum(y**2) / t - np.sum(np.diag(S) ** 2)
        z = X * X_mkt[:, None]
        v1 = y.T.dot(z) / t - cov_mkt[:, None] * S
        roff1 = (
            np.sum(v1 * cov_mkt[:, None].T) / var_mkt
            - np.sum(np.diag(v1) * cov_mkt) / var_mkt
        )
        v3 = z.T.dot(z) / t - var_mkt * S
        roff3 = (
            np.sum(v3 * np.outer(cov_mkt, cov_mkt)) / var_mkt**2
            - np.sum(np.diag(v3) * cov_mkt**2) / var_mkt**2
        )
        roff = 2 * roff1 - roff3
        rho = rdiag + roff

        gamma = np.linalg.norm(S - F, "fro") ** 2

        kappa = (phi - rho) / gamma
        alpha = max(0, min(1, kappa / t))

        return alpha
