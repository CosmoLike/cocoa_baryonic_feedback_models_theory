"""
Baryon Feedback Suppression Theory Block for Cobaya/Cocoa

Provides baryon feedback effects on the matter power spectrum using external
baryon feedback models (pyspk, BCEmu, or Flamingo). This theory block computes
suppression factors S(k,z) that are applied to the nonlinear power spectrum.

Physics Background:
    Baryon feedback processes (e.g., AGN heating, stellar feedback) suppress
    the power spectrum on small scales (k > 0.1 h/Mpc) through ejection of
    baryonic matter from overdense regions. The suppression factor S(k,z) < 1
    quantifies this effect: P_nl(k,z) = S(k,z) * P_DMO(k,z)

    References:
    - pyspk: Salcido+ 2015 (arXiv:2305.09710)
    - BCEmu: Giri+ 2021 (arxiv:2108.08863)
    - Flamingo: Schaller+ 2025 (arxiv:2410.17109)

Author: Nihar Dalal, Kunhao Zhong, João Rebouças, CoCoA Developers
Date: May 2026
"""

import numpy as np
import logging
import pyspk as spk
import BCemu
import FlamingoBaryonResponseEmulator as fre
import baccoemu
from scipy.interpolate import interp1d, RectBivariateSpline
from astropy.cosmology import FlatLambdaCDM
from cobaya.theory import Theory
from cobaya.log import LoggedError

AVAILABLE_BARYON_MODELS = {
    1: "SP(k)",
    2: "BCEmu",
    3: "FlamingoBaryonResponseEmulator",
    4: "BACCOemu",
    5: "BCemu2025",
}

class bfmt(Theory):
    """
    Theory block for baryon suppression using external models.

    Attributes:
        params (dict): Declares sampled parameters alpha_spk, beta_spk, gamma_spk
                       (for pyspk model; extensible for BCEmu, PCA)
        requested_z (array): Redshifts at which to compute suppression
        requested_k (array): Wavenumbers (h/Mpc) at which to compute suppression
        baryon_model (int): Model selector (1=pyspk, 2=BCEmu, 3=PCA)
        log (logger): Cobaya logger for warnings/errors
    """

    # Define configuration defaults
    baryon_model: int = 1  # 1=pyspk, 2=bcemu, 3=flamingo, 4=BACCOemu
    # Internal computation grid: S(k,z) is evaluated on nz x nk points and
    # 2D-splined onto the (much larger) grid the likelihood requests, in the
    # same spirit as Cobaya's matter-power-spectrum interpolator
    nz: int = 20
    nk: int = 100
    # S(k,z) for z above the selected model's calibration range:
    #   "unity": S = 1, no feedback (default)
    #   "constant": S clamped to its value at the model's zmax
    above_zmax: str = "unity"
    # BCemu2025 only: q2 interpolation coordinate (native grid: 0.5, 0.7, 1.0)
    q2_bcemu25: float = 0.70

    def initialize(self):
        """
        Initialize the theory block.

        Sets up:
        - Empty arrays for requested z and k scales
        - Calibration ranges for pyspk
        - Parameter validation bounds (3-sigma from priors)
        """
        # Define the parameters this theory class needs to evaluate.
        if self.baryon_model == 1: # SP(k)
            self.params = {
                "alpha_spk": None,  # Alpha parameter for pyspk model
                "beta_spk": None,   # Beta parameter for pyspk model
                "gamma_spk": None,  # Gamma parameter for pyspk model
            }
        elif self.baryon_model == 2: # BCemu
            self.params = {
                "log10Mc_bcemu": None,
                "mu_bcemu": None,
                "thej_bcemu": None,
                "gamma_bcemu": None,
                "delta_bcemu": None,
                "eta_bcemu": None,
                "deta_bcemu": None,
            }
            # Load the emulator files once; the baryon fraction fb = Ob/Om
            # (the only cosmology dependence) is updated at every sample via
            # update_cosmology() in _calculate_bcemu
            self.bcemulator = BCemu.BCM_7param(verbose=False)
        elif self.baryon_model == 3: # FlamingoEmulator
            self.params = {
                "fgas_sigma_flamingo": None,
                "mstar_sigma_flamingo": None,
                "jet_frac_flamingo": None,
            }
        elif self.baryon_model == 4: # BACCOemu
            self.params = {
                "M_c_baccoemu": None,
                "eta_baccoemu": None,
                "beta_baccoemu": None,
                "M1_z0_cen_baccoemu": None,
                "theta_inn_baccoemu": None,
            }
            self.baccoemulator = baccoemu.Matter_powerspectrum(baryonic_model_name='Burger2025')
        elif self.baryon_model == 5: # BCemu2025
            self.params = {
                "Theta_co_bcemu25": None,
                "log10Mc_bcemu25": None,
                "mu_bcemu25": None,
                "delta_bcemu25": None,
                "eta_bcemu25": None,
                "deta_bcemu25": None,
                "Nstar_bcemu25": None,
            }
            # Load the emulator files once; fb = Ob/Om (the only cosmology
            # dependence) enters the parameter dict at every sample
            self.bcemulator25 = BCemu.BCemu2025(backend='numpy')
            q2lo = float(self.bcemulator25.q2_grid.min())
            q2hi = float(self.bcemulator25.q2_grid.max())
            if not (q2lo <= self.q2_bcemu25 <= q2hi):
                raise LoggedError(
                    self.log,
                    f"Invalid `q2_bcemu25`={self.q2_bcemu25}; the BCemu2025 "
                    f"grid covers q2 in [{q2lo}, {q2hi}]",
                )
        else:
            raise LoggedError(self.log, f"Invalid choice of `baryon_model`. Available options are 1 (SP(k)), 2 (BCEmu), 3 (FlamingoEmulator), 4 (BACCOemu), or 5 (BCemu2025)")
        
        if self.above_zmax not in ("unity", "constant"):
            raise LoggedError(
                self.log,
                f"Invalid `above_zmax`='{self.above_zmax}'. Available options: "
                "'unity' (S=1; default) or 'constant' (clamp S to its value at zmax)",
            )

        self.requested_z = np.array([])
        self.requested_k = np.array([])

        # pyspk Calibration ranges (from Kunhao's testing/tuning)
        self.z_min_calib = 0.125  # SP(k) calibration floor; below this the
                                  # suppression is clamped to its z=0.125 value
        self.z_max_calib = 3.0  # Above this, pyspk not well-calibrated: unity
        self.k_min_calib = 8.73e-3  # h/Mpc; below this, outside calibration

        # Parameter validation bounds (3-sigma conservative from YAML priors)
        # Expected ranges: alpha ~4.18±0.12, beta ~1.26±0.08, gamma ~0.42±0.10
        self.alpha_min, self.alpha_max = 3.8, 4.6
        self.beta_min, self.beta_max = 1.0, 1.6
        self.gamma_min_spk, self.gamma_max_spk = 0.1, 0.75

        # Parameter validation bounds for BCEmu (based on Giri+ 2021 and reasonable extensions)
        self.log10Mc_min, self.log10Mc_max = 11.0, 15.0
        self.mu_min, self.mu_max = 0.0, 2.0
        self.thej_min, self.thej_max = 2.0, 8.0
        self.gamma_min_bcemu, self.gamma_max_bcemu = 1.0, 4.0
        self.delta_min, self.delta_max = 3.0, 11.0
        self.eta_min, self.eta_max = 0.05, 4.0
        self.deta_min, self.deta_max = 0.05, 4.0

        # Parameter validation bounds for Flamingo (based on Schaller+ 2025 and reasonable extensions)
        self.fgas_sigma_min, self.fgas_sigma_max = -10.0, 4.0
        self.mstar_sigma_min, self.mstar_sigma_max = -3.0, 2.0
        self.jet_frac_min, self.jet_frac_max = 0.0, 1.0

        # Parameter validation bounds for BACCOemu (based on Burger+ 2025 and reasonable extensions)
        self.M_c_baccoemu_min, self.M_c_baccoemu_max = 10, 16
        self.eta_baccoemu_min, self.eta_baccoemu_max = -0.7, 0.2
        self.beta_baccoemu_min, self.beta_baccoemu_max = -1, 0.7
        self.M1_z0_cen_baccoemu_min, self.M1_z0_cen_baccoemu_max = 9, 13
        self.theta_inn_baccoemu_min, self.theta_inn_baccoemu_max = -2, 0

        # Parameter validation bounds for BCemu2025 (training LHC ranges,
        # recovered from the emulator's input StandardScaler: mean +- sqrt(3)*std)
        self.Theta_co_bcemu25_min, self.Theta_co_bcemu25_max = 0.0, 0.8
        self.log10Mc_bcemu25_min, self.log10Mc_bcemu25_max = 11.0, 15.0
        self.mu_bcemu25_min, self.mu_bcemu25_max = 0.0, 3.0
        self.delta_bcemu25_min, self.delta_bcemu25_max = 2.0, 12.0
        self.eta_bcemu25_min, self.eta_bcemu25_max = -0.2, 0.2
        self.deta_bcemu25_min, self.deta_bcemu25_max = 0.0, 0.4
        self.Nstar_bcemu25_min, self.Nstar_bcemu25_max = 0.0, 0.05
        self.fb_bcemu25_min, self.fb_bcemu25_max = 0.10, 0.20

        self.log.debug(
            "BaryonSuppression: Initialized with baryon_model=%d, "
            "z_calibration=[%.3f, %.3f], k_min_calib=%.3e h/Mpc",
            self.baryon_model,
            self.z_min_calib,
            self.z_max_calib,
            self.k_min_calib,
        )

    def get_requirements(self):
        """
        Declare dependencies on other theory components.

        Returns:
            list: Required products from other theory blocks
                - "H0": Hubble constant (for cosmology)
                - "omegam": Matter density (for cosmology)
                - "omegab": Baryon density (for BCEmu)
        """
        return ["H0", "omegam", "omegab"]

    def must_provide(self, **requirements):
        """
        Parse product requests from the Cosmolike likelihood.

        Stores the z and k grids at which the likelihood needs suppression factors.
        This allows the theory block to compute only what's needed, improving
        efficiency when multiple likelihoods have different k/z requirements.

        Important note: the Cosmolike likelihood provides the k-array in 1/Mpc so we must convert to h/Mpc ourselves if needed
        This is in line with what they do with Boltzmann solvers

        Args:
            **requirements (dict): Map of product names to their specifications.
                Expected key: "baryon_suppression" with value
                {
                    "z": array of redshifts,
                    "k": array of wavenumbers *(1/Mpc)*
                }

        Raises:
            LoggedError: If baryon_suppression is requested but z or k is missing.
        """
        if "baryon_suppression" in requirements:
            req = requirements["baryon_suppression"]

            # Extract and validate z array
            requested_z = req.get("z", None)
            if requested_z is None:
                raise LoggedError(
                    self.log, "baryon_suppression requires 'z' array in requirements"
                )
            self.requested_z = np.atleast_1d(requested_z)

            # Extract and validate k array
            requested_k = req.get("k", None)
            if requested_k is None:
                raise LoggedError(
                    self.log, "baryon_suppression requires 'k' array in requirements"
                )
            self.requested_k = np.atleast_1d(requested_k)

            n_z = len(self.requested_z)
            n_k = len(self.requested_k)
            self.log.info(
                "BaryonSuppression.must_provide: baryon_suppression requested; "
                "n_z=%d [%.3f, %.3f], n_k=%d [%.3e, %.3e]",
                n_z,
                self.requested_z.min(),
                self.requested_z.max(),
                n_k,
                self.requested_k.min(),
                self.requested_k.max(),
            )

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Compute baryon suppression factors and store in state.

        Fetches cosmological parameters from the provider, retrieves baryon
        model parameters from the sampler, and computes S(k,z) for all
        requested (k, z) pairs. Handles multiple baryon models (selector via
        self.baryon_model) and implements comprehensive error handling with
        graceful degradation (returns unity suppression on error).

        Args:
            state (dict): Cobaya state dictionary where results are stored.
                         On output, state["baryon_suppression"] = {z: S(k,z) array}
            want_derived (bool): Whether to compute derived parameters (unused).
            **params_values_dict (dict): Map of parameter names to current values.
                Expected keys: "alpha_spk", "beta_spk", "gamma_spk"

        Returns:
            None: Results stored in state["baryon_suppression"]

        Notes:
            - Returns unity suppression (graceful degradation) on any error
            - Applies calibration masking: z outside [0.125, 3.0] or k < 8.73e-3 h/Mpc
            - Validates parameters against 3-sigma bounds before calling pyspk
            - Uses log-space interpolation to avoid numerical issues
        """

        # The likelihood requests S(k,z) on its full interpolation grid
        # (e.g. 140 z x 1500 k). Evaluating the feedback models there is
        # wasteful: compute on the small internal nz x nk grid and 2D-spline
        # onto the requested grid instead (the strategy Cobaya uses for the
        # matter power spectrum interpolator).
        z_out = self.requested_z
        k_out = self.requested_k

        if len(z_out) == 0 or len(k_out) == 0:
            state["baryon_suppression"] = {}
            return

        # Highest z each model can evaluate; above it, suppression is unity
        zmax_table = {1: self.z_max_calib, 2: 2.0, 3: 3.0, 4: 3.0}
        if self.baryon_model == 5:
            zmax_table[5] = float(self.bcemulator25.z_grid.max())
        zmax_model = zmax_table.get(self.baryon_model, np.inf)
        z_top = min(z_out.max(), zmax_model)

        if z_top <= z_out.min():
            # every requested redshift is above the model range
            if self.above_zmax == "unity":
                state["baryon_suppression"] = {
                    z_val: np.ones_like(k_out) for z_val in z_out
                }
                return
            # "constant": still evaluate the model in a narrow band ending at
            # its zmax, so the clamped S(k, zmax) row exists below
            z_lo = max(0.0, z_top - 0.25)
        else:
            z_lo = z_out.min()

        nz = max(4, min(int(self.nz), len(z_out)))
        nk = max(4, min(int(self.nk), len(k_out)))
        z_int = np.linspace(z_lo, z_top, nz)
        k_int = np.logspace(np.log10(k_out.min()), np.log10(k_out.max()), nk)

        # Route to appropriate baryon model (computed on the internal grid)
        if self.baryon_model == 1:
            suppression_dict = self._calculate_pyspk(params_values_dict, z_int, k_int)
        elif self.baryon_model == 2:
            suppression_dict = self._calculate_bcemu(params_values_dict, z_int, k_int)
        elif self.baryon_model == 3:
            suppression_dict = self._calculate_flamingo(params_values_dict, z_int, k_int)
        elif self.baryon_model == 4:
            suppression_dict = self._calculate_baccoemu(params_values_dict, z_int, k_int)
        elif self.baryon_model == 5:
            suppression_dict = self._calculate_bcemu2025(params_values_dict, z_int, k_int)
        else:
            self.log.error(
                "baryon_model=%d is invalid; must be 1 (pyspk), 2 (bcemu), 3 (flamingo), 4 (BACCOemu), or 5 (BCemu2025); "
                "returning unity suppression",
                self.baryon_model,
            )
            suppression_dict = self._unity_suppression(z_int, k_int)

        # None signals a rejected sample (message already logged by
        # _reject_sample); returning False makes cobaya assign -inf to this
        # point and continue the run
        if suppression_dict is None:
            return False

        # 2D spline of ln(S) over (z, log10 k), evaluated on the requested grid
        lnS = np.log(np.vstack([suppression_dict[z_val] for z_val in z_int]))
        spline = RectBivariateSpline(
            z_int,
            np.log10(k_int),
            lnS,
            kx=min(3, nz - 1),
            ky=min(3, nk - 1),
        )
        log10k_out = np.log10(k_out)
        result = {}
        row_above_zmax = None
        for z_val in z_out:
            if z_val > z_top:
                # `above_zmax` option: "unity" (default) -> S = 1;
                # "constant" -> S clamped to its value at the model's zmax
                if row_above_zmax is None:
                    if self.above_zmax == "constant":
                        row_above_zmax = np.exp(spline(z_top, log10k_out)[0])
                    else:
                        row_above_zmax = np.ones_like(k_out)
                result[z_val] = row_above_zmax.copy()
            else:
                result[z_val] = np.exp(spline(z_val, log10k_out)[0])

        self.log.debug(
            "baryon suppression: computed on internal %d x %d (z, k) grid, "
            "splined onto the requested %d x %d grid",
            nz, nk, len(z_out), len(k_out),
        )

        # Store result in state
        state["baryon_suppression"] = result

    def _reject_sample(self, msg):
        """
        Log a parameter/box violation loudly and signal sample rejection.

        Returns None, which calculate() converts into `return False` — the
        cobaya mechanism for rejecting the point (-inf posterior) without
        aborting the run. NOTE: do NOT raise LoggedError for per-sample
        rejections — LoggedError is in cobaya's always_stop_exceptions and
        aborts the whole run regardless of stop_at_error.
        """
        self.log.warning("%s; rejecting this sample", msg)
        return None

    def _calculate_pyspk(self, params_values_dict, z_arr, k_arr):
        """
        Compute suppression factors using pyspk model.

        Implements the SPk baryon feedback model with comprehensive validation
        and calibration masking.

        Args:
            params_values_dict (dict): {"alpha_spk": float, "beta_spk": float, "gamma_spk": float}

        Returns:
            dict: {z_val: suppression_array} for each requested redshift.
                  Returns unity suppression dict on any error.
        """
        try:
            # 1. Fetch sampled baryon parameters
            alpha = params_values_dict.get("alpha_spk", 4.18)
            beta = params_values_dict.get("beta_spk", 1.26)
            gamma = params_values_dict.get("gamma_spk", 0.42)

            self.log.debug(
                "SPk baryon suppression: alpha=%.4f, beta=%.4f, gamma=%.4f",
                alpha,
                beta,
                gamma,
            )

            # 2. Validate parameters are within acceptable ranges (3-sigma bounds)
            # Reject invalid samples via _reject_sample (calculate() then returns False -> -inf)
            if not (self.alpha_min < alpha < self.alpha_max):
                return self._reject_sample(
                    f"SPk parameter alpha_spk={alpha:.4f} outside valid range"
                    f"[{self.alpha_min:.4f}, {self.alpha_max:.4f}]",
                )

            if not (self.beta_min < beta < self.beta_max):
                return self._reject_sample(
                    f"SPk parameter beta_spk={beta:.4f} outside valid range "
                    f"[{self.beta_min:.4f}, {self.beta_max:.4f}]",
                )

            if not (self.gamma_min_spk < gamma < self.gamma_max_spk):
                return self._reject_sample(
                    f"SPk parameter gamma_spk={gamma:.4f} outside valid range "
                    f"[{self.gamma_min_spk:.4f}, {self.gamma_max_spk:.4f}]",
                )

            # 3. Fetch cosmological parameters from provider (e.g., CAMB/CLASS)
            H0 = self.provider.get_param("H0")
            h = H0/100
            omegam = self.provider.get_param("omegam")
            cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

            self.log.debug("SPk cosmology: H0=%.3f, omegam=%.4f", H0, omegam)

            # 4. Compute suppression for each requested redshift
            suppression_dict = {}

            for i_z, z_val in enumerate(z_arr):
                # Above the calibration range, return unity (no feedback)
                if z_val > self.z_max_calib:
                    self.log.debug(
                        "SPk z=%.3f above calibration range [%.3f, %.3f]; "
                        "using unity suppression for this redshift",
                        z_val,
                        self.z_min_calib,
                        self.z_max_calib,
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                # Below the calibration floor, clamp: evaluate SP(k) at
                # z_min_calib and reuse that suppression for all lower z
                z_eval = max(z_val, self.z_min_calib)

                try:
                    # Call pyspk to compute suppression on its native k-grid
                    k_spk, sup_spk = spk.sup_model(
                        SO=500,  # Spherical overdensity radius
                        z=z_eval,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        cosmo=cosmo,
                        verbose=False,
                    )

                except Exception as e:
                    self.log.error(
                        "SPk model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                # Defensive check: verify pyspk output is valid (no NaN/Inf)
                if not np.all(np.isfinite(k_spk)) or not np.all(np.isfinite(sup_spk)):
                    n_bad_k = np.sum(~np.isfinite(k_spk))
                    n_bad_sup = np.sum(~np.isfinite(sup_spk))
                    self.log.error(
                        "SPk returned non-finite values at z=%.3f: "
                        "%d bad k-values, %d bad suppression values; "
                        "using unity suppression for this redshift",
                        z_val,
                        n_bad_k,
                        n_bad_sup,
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                # Log suppression range for diagnostics
                self.log.debug(
                    "SPk at z=%.3f: k_range=[%.3e, %.3e], "
                    "suppression_range=[%.6f, %.6f]",
                    z_val,
                    k_spk.min(),
                    k_spk.max(),
                    sup_spk.min(),
                    sup_spk.max(),
                )

                # 5. Interpolate pyspk suppression onto likelihood's k-grid
                # Use log-space interpolation for better numerical stability
                # IMPORTANT: Use boundary value clamping for extrapolation to avoid
                # non-physical values outside pyspk's calibration range
                try:
                    # Use fill_value with boundary clamping instead of linear extrapolation
                    # This preserves suppression values at k boundaries (no unphysical extrapolation)
                    log_k_min = np.log10(k_spk.min())
                    log_k_max = np.log10(k_spk.max())
                    log_sup_min = np.log(sup_spk[0])  # Suppression at minimum k
                    log_sup_max = np.log(sup_spk[-1])  # Suppression at maximum k

                    interp_spk = interp1d(
                        np.log10(k_spk),
                        np.log(sup_spk),
                        kind="linear",
                        fill_value=(
                            log_sup_min,
                            log_sup_max,
                        ),  # Clamp to boundary values
                        bounds_error=False,
                        assume_sorted=True,
                    )
                    # JVR NOTE: k_arr is in 1/Mpc but SP(k) assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    sup_interp = np.exp(interp_spk(np.log10(k_arr) - np.log10(h)))

                except Exception as e:
                    self.log.error(
                        "Interpolation failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                # 6. Apply calibration masking: suppress effect outside calibration ranges
                # For k < 8.73e-3 h/Mpc, set suppression to 1 (outside calib; Kunhao's choice)
                # For k > pyspk's max k, use boundary value (no extrapolation needed now)
                # JVR NOTE: also keep the units in mind here
                sup_interp[k_arr < h*self.k_min_calib] = 1.0

                # Defensive check: verify interpolated suppression is physically reasonable
                if not np.all(np.isfinite(sup_interp)):
                    n_bad = np.sum(~np.isfinite(sup_interp))
                    self.log.error(
                        "Interpolation produced %d non-finite values at z=%.3f; "
                        "using unity suppression for this redshift",
                        n_bad,
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                # Sanity check: warn if suppression is wildly unphysical
                # (e.g., S < 0 or S > 2, which would indicate serious problems)
                # Values > 1.0 at high k are expected from interpolation; values slightly > 1
                # indicate numerical precision or model behavior at calibration boundaries
                n_unphysical_low  = np.sum(sup_interp < 0.0)
                n_unphysical_high = np.sum(sup_interp > 2.0)

                if n_unphysical_low > 0 or n_unphysical_high > 0:
                    self.log.warning(
                        "SPk at z=%.3f produced %d values < 0.0 and %d values > 2.0 "
                        "(k_range_pyspk=[%.3e, %.3e], k_range_requested=[%.3e, %.3e]); "
                        "these are unphysical and may indicate issues with parameters or interpolation",
                        z_val,
                        n_unphysical_low,
                        n_unphysical_high,
                        k_spk.min(),
                        k_spk.max(),
                        k_arr.min(),
                        k_arr.max(),
                    )

                suppression_dict[z_val] = sup_interp

            self.log.info(
                "SPk suppression computed for %d redshifts, %d k-values",
                len(suppression_dict),
                len(k_arr),
            )
            return suppression_dict

        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in SPk calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression(z_arr, k_arr)

    def _calculate_bcemu(self, params_values_dict, z_arr, k_arr):
        try:
            log10Mc_bcemu = params_values_dict.get("log10Mc_bcemu", 13.0)
            mu_bcemu = params_values_dict.get("mu_bcemu", 1.0)
            thej_bcemu = params_values_dict.get("thej_bcemu", 5.0)
            gamma_bcemu = params_values_dict.get("gamma_bcemu", 2.5)
            delta_bcemu = params_values_dict.get("delta_bcemu", 7.0)
            eta_bcemu = params_values_dict.get("eta_bcemu", 2.0)
            deta_bcemu = params_values_dict.get("deta_bcemu", 2.0)
            H0 = self.provider.get_param("H0")
            h = H0/100

            self.log.debug(
                "BCEmu baryon suppression: log10Mc=%.4f, mu=%.4f, thej=%.4f, gamma=%.4f, delta=%.4f, eta=%.4f, deta=%.4f",
                log10Mc_bcemu,
                mu_bcemu,
                thej_bcemu,
                gamma_bcemu,
                delta_bcemu,
                eta_bcemu,
                deta_bcemu,
            )

            if not (self.log10Mc_min < log10Mc_bcemu < self.log10Mc_max):
                return self._reject_sample(
                    f"BCEmu parameter log10Mc_bcemu={log10Mc_bcemu:.4f} outside valid range "
                    f"[{self.log10Mc_min:.4f}, {self.log10Mc_max:.4f}]",
                )
            if not (self.mu_min < mu_bcemu < self.mu_max):
                return self._reject_sample(
                    f"BCEmu parameter mu_bcemu={mu_bcemu:.4f} outside valid range "
                    f"[{self.mu_min:.4f}, {self.mu_max:.4f}]",
                )
            if not (self.thej_min < thej_bcemu < self.thej_max):
                return self._reject_sample(
                    f"BCEmu parameter thej_bcemu={thej_bcemu:.4f} outside valid range "
                    f"[{self.thej_min:.4f}, {self.thej_max:.4f}]",
                )
            if not (self.gamma_min_bcemu < gamma_bcemu < self.gamma_max_bcemu):
                return self._reject_sample(
                    f"BCEmu parameter gamma_bcemu={gamma_bcemu:.4f} outside valid range "
                    f"[{self.gamma_min_bcemu:.4f}, {self.gamma_max_bcemu:.4f}]",
                )
            if not (self.delta_min < delta_bcemu < self.delta_max):
                return self._reject_sample(
                    f"BCEmu parameter delta_bcemu={delta_bcemu:.4f} outside valid range "
                    f"[{self.delta_min:.4f}, {self.delta_max:.4f}]",
                )
            if not (self.eta_min < eta_bcemu < self.eta_max):
                return self._reject_sample(
                    f"BCEmu parameter eta_bcemu={eta_bcemu:.4f} outside valid range "
                    f"[{self.eta_min:.4f}, {self.eta_max:.4f}]",
                )
            if not (self.deta_min < deta_bcemu < self.deta_max):
                return self._reject_sample(
                    f"BCEmu parameter deta_bcemu={deta_bcemu:.4f} outside valid range "
                    f"[{self.deta_min:.4f}, {self.deta_max:.4f}]",
                )

            bcmdict = {
                "log10Mc": log10Mc_bcemu,
                "mu": mu_bcemu,
                "thej": thej_bcemu,
                "gamma": gamma_bcemu,
                "delta": delta_bcemu,
                "eta": eta_bcemu,
                "deta": deta_bcemu,
            }

            # The emulator files were loaded once in initialize(); the only
            # cosmology dependence is the baryon fraction fb = Ob/Om, which
            # BCemu exposes as a per-call update
            bfcemu = self.bcemulator
            bfcemu.update_cosmology(
                self.provider.get_param("omegab"),
                self.provider.get_param("omegam"),
            )
            # Training-set k range (h/Mpc), read from the emulator itself so
            # BCEmu is never asked to extrapolate outside its training set
            kmin_bcemu = float(np.min(bfcemu.ks0))
            kmax_bcemu = float(np.max(bfcemu.ks0))
            logkmin_bcemu = np.log10(kmin_bcemu)
            logkmax_bcemu = np.log10(kmax_bcemu)
            suppression_dict = {}
            for i_z, z_val in enumerate(z_arr):
                # Check if redshift is within BCEmu calibration range
                # (BCEmu is calibrated for z=0-2; outside this, we return unity)
                if z_val < 0.0 or z_val > 2.0:
                    self.log.debug(
                        "BCEmu z=%.3f outside calibration range [0.0, 2.0]; "
                        "using unity suppression for this redshift",
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                try:
                    k_bcemu = 10 ** np.linspace(
                        logkmin_bcemu, logkmax_bcemu, 100
                    )  # h/Mpc
                    # pin endpoints exactly: 10**log10(k) can land a float
                    # epsilon outside [ks0.min(), ks0.max()], which makes
                    # BCEmu print an extrapolation warning per call
                    k_bcemu[0] = kmin_bcemu
                    k_bcemu[-1] = kmax_bcemu
                    sup_bcemu = bfcemu.get_boost(z_val, bcmdict, k_bcemu)

                    # Interpolate BCEmu suppression onto requested k-grid
                    interp_bcemu = interp1d(
                        np.log10(k_bcemu),
                        np.log(sup_bcemu),
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                        assume_sorted=True,
                    )
                    # JVR NOTE: k_arr is in 1/Mpc but BCEmu assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    sup_interp = np.exp(interp_bcemu(np.log10(k_arr) - np.log10(h)))

                    # No feedback on scales below the training range:
                    # S = 1 for k < kmin (k_arr in 1/Mpc, kmin in h/Mpc)
                    sup_interp[k_arr < h * kmin_bcemu] = 1.0

                    suppression_dict[z_val] = sup_interp

                except Exception as e:
                    self.log.error(
                        "BCEmu model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in BCEmu calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression(z_arr, k_arr)

        return suppression_dict

    def _calculate_flamingo(self, params_values_dict, z_arr, k_arr):
        try:
            fgas_sigma_flamingo = params_values_dict.get("fgas_sigma_flamingo", 0)
            mstar_sigma_flamingo = params_values_dict.get("mstar_sigma_flamingo", 0)
            jet_frac_flamingo = params_values_dict.get("jet_frac_flamingo", 0)

            self.log.debug(
                "Flamingo baryon suppression: fgas_sigma=%.4f, mstar_sigma=%.4f, jet_frac=%.4f",
                fgas_sigma_flamingo,
                mstar_sigma_flamingo,
                jet_frac_flamingo,
            )

            if not (self.fgas_sigma_min < fgas_sigma_flamingo < self.fgas_sigma_max):
                return self._reject_sample(
                    f"Flamingo parameter fgas_sigma_flamingo={fgas_sigma_flamingo:.4f} outside valid range "
                    f"[{self.fgas_sigma_min:.4f}, {self.fgas_sigma_max:.4f}]",
                )
            if not (self.mstar_sigma_min < mstar_sigma_flamingo < self.mstar_sigma_max):
                return self._reject_sample(
                    f"Flamingo parameter mstar_sigma_flamingo={mstar_sigma_flamingo:.4f} outside valid range "
                    f"[{self.mstar_sigma_min:.4f}, {self.mstar_sigma_max:.4f}]",
                )
            if not (self.jet_frac_min <= jet_frac_flamingo <= self.jet_frac_max):
                return self._reject_sample(
                    f"Flamingo parameter jet_frac_flamingo={jet_frac_flamingo} outside valid range "
                    f"[{self.jet_frac_min:.4e}, {self.jet_frac_max:.4e}]",
                )

            myemu = fre.FlamingoBaryonResponseEmulator()
            logkmin_flamingo = -1.5
            logkmax_flamingo = 1.5
            suppression_dict = {}
            for i_z, z_val in enumerate(z_arr):
                # Check if redshift is within Flamingo calibration range
                # (Flamingo is calibrated for z=0-3; outside this, we return unity)
                if z_val < 0.0 or z_val > 3.0:
                    self.log.debug(
                        "Flamingo z=%.3f outside calibration range [0.0, 3.0]; "
                        "using unity suppression for this redshift",
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue

                try:
                    k_flamingo = 10 ** np.linspace(
                        logkmin_flamingo, logkmax_flamingo, 100
                    )  # h/Mpc
                    sup_flamingo = myemu.predict(
                        k_flamingo,
                        z_val,
                        fgas_sigma_flamingo,
                        mstar_sigma_flamingo,
                        jet_frac_flamingo,
                    )

                    # Interpolate Flamingo suppression onto requested k-grid
                    interp_flamingo = interp1d(
                        np.log10(k_flamingo),
                        np.log(sup_flamingo),
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                        assume_sorted=True,
                    )
                    
                    # JVR NOTE: k_arr is in 1/Mpc but SP(k) assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    H0 = self.provider.get_param("H0")
                    h = H0/100
                    sup_interp = np.exp(interp_flamingo(np.log10(k_arr) - np.log10(h)))

                    suppression_dict[z_val] = sup_interp

                except Exception as e:
                    self.log.error(
                        "Flamingo model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(k_arr)
                    continue
        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in Flamingo calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression(z_arr, k_arr)

        return suppression_dict

    def _calculate_baccoemu(self, params_values_dict, z_arr, k_arr):
        """
        Compute suppression factors using the BACCOemu (Burger2025) baryonic boost.

        Sampled baryon parameters are validated up front and the emulator's own
        training-box assertions (which also cover the cosmology: omega_cold,
        omega_baryon, sigma8_cold, ...) are converted into clean sample
        rejections, so MCMC treats out-of-box points as rejected instead of
        the theory block dying with a raw AssertionError.

        Args:
            params_values_dict (dict): sampled *_baccoemu baryon parameters.

        Returns:
            dict: {z_val: suppression_array} for each requested redshift.
        """
        suppression_dict = {}

        # 1. Fetch and validate sampled baryon parameters; reject invalid samples via _reject_sample instead of raising (MCMC will treat this point as rejected)
        M_c = params_values_dict.get("M_c_baccoemu")
        eta = params_values_dict.get("eta_baccoemu")
        beta = params_values_dict.get("beta_baccoemu")
        M1_z0_cen = params_values_dict.get("M1_z0_cen_baccoemu")
        theta_inn = params_values_dict.get("theta_inn_baccoemu")

        for name, value, vmin, vmax in (
            ("M_c_baccoemu", M_c,
             self.M_c_baccoemu_min, self.M_c_baccoemu_max),
            ("eta_baccoemu", eta,
             self.eta_baccoemu_min, self.eta_baccoemu_max),
            ("beta_baccoemu", beta,
             self.beta_baccoemu_min, self.beta_baccoemu_max),
            ("M1_z0_cen_baccoemu", M1_z0_cen,
             self.M1_z0_cen_baccoemu_min, self.M1_z0_cen_baccoemu_max),
            ("theta_inn_baccoemu", theta_inn,
             self.theta_inn_baccoemu_min, self.theta_inn_baccoemu_max),
        ):
            if not (vmin < value < vmax):
                return self._reject_sample(
                    f"BACCOemu parameter {name}={value:.4f} outside valid range "
                    f"[{vmin:.4f}, {vmax:.4f}]",
                )

        # 2. Fetch cosmological parameters from the provider
        A_s = self.provider.get_param("As")
        ns = self.provider.get_param("ns")
        omegab = self.provider.get_param("omegab")
        mnu = self.provider.get_param("mnu")
        h = self.provider.get_param("H0")/100
        omegacb = self.provider.get_param("omegam") - mnu/94.13/(h*h)
        try:
            w0 = self.provider.get_param("w")
        except (KeyError, AttributeError):
            w0 = -1
        try:
            wa = self.provider.get_param("wa")
        except (KeyError, AttributeError):
            wa = 0

        common_params = {
            'omega_cold'    :  omegacb,
            'A_s'           :  A_s,
            'omega_baryon'  :  omegab,
            'ns'            :  ns,
            'hubble'        :  h,
            'neutrino_mass' :  mnu,
            'w0'            :  w0,
            'wa'            :  wa,
            'M_c'           :  params_values_dict.get("M_c_baccoemu"),
            'eta'           :  params_values_dict.get("eta_baccoemu"),
            'beta'          :  params_values_dict.get("beta_baccoemu") ,
            'M1_z0_cen'     :  params_values_dict.get("M1_z0_cen_baccoemu") ,
            'theta_inn'     :  params_values_dict.get("theta_inn_baccoemu") ,
            # 'expfactor'     :  1, # This will be set at each redshift
        }
        # 3. Compute suppression for each requested redshift. The emulator
        #    asserts its full training box (cosmology included, e.g.
        #    omega_baryon in [0.04, 0.06], omega_cold in [0.23, 0.40],
        #    sigma8_cold in [0.73, 0.90]); convert violations into clean
        #    sample rejections
        for i_z, z_val in enumerate(z_arr):
            a = 1/(1+z_val)
            if a < 0.25:
                self.log.debug(
                    "BACCOemu z=%.3f outside calibration range a \\in [0.25, 1.0]; "
                    "using unity suppression for this redshift",
                    z_val,
                )
                suppression_dict[z_val] = np.ones_like(k_arr)
                continue
            common_params.update({"expfactor": a})
            try:
                k_bacco, S = self.baccoemulator.get_baryonic_boost(**common_params)
            except AssertionError as e:
                return self._reject_sample(
                    f"BACCOemu training-box violation at z={z_val:.3f}: {e}",
                )
            interp_bacco = interp1d(
                np.log10(k_bacco),
                np.log(S),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
                assume_sorted=True,
            )
            sup_interp = np.exp(interp_bacco(np.log10(k_arr) - np.log10(h)))
            suppression_dict[z_val] = sup_interp

        return suppression_dict

    def _calculate_bcemu2025(self, params_values_dict, z_arr, k_arr):
        """
        Compute suppression factors using the BCemu2025 emulator.

        The emulator was loaded once in initialize(); the baryon fraction
        fb = Ob/Om (the only cosmology dependence) enters the parameter dict
        at every sample. The q2 coordinate is the yaml option `q2_bcemu25`.

        Args:
            params_values_dict (dict): sampled *_bcemu25 parameters.

        Returns:
            dict: {z_val: suppression_array} for each requested redshift.
        """
        # 1. Fetch and validate sampled parameters; reject invalid samples
        vals = {}
        for name in ("Theta_co", "log10Mc", "mu", "delta", "eta", "deta", "Nstar"):
            v = params_values_dict.get(f"{name}_bcemu25")
            vmin = getattr(self, f"{name}_bcemu25_min")
            vmax = getattr(self, f"{name}_bcemu25_max")
            if not (vmin < v < vmax):
                return self._reject_sample(
                    f"BCemu2025 parameter {name}_bcemu25={v:.4f} outside valid "
                    f"range [{vmin:.4f}, {vmax:.4f}]"
                )
            vals[name] = v

        # 2. Baryon fraction from the provider; part of the training box
        h = self.provider.get_param("H0") / 100
        fb = self.provider.get_param("omegab") / self.provider.get_param("omegam")
        if not (self.fb_bcemu25_min < fb < self.fb_bcemu25_max):
            return self._reject_sample(
                f"BCemu2025 baryon fraction fb=Ob/Om={fb:.4f} outside the "
                f"training range [{self.fb_bcemu25_min:.4f}, {self.fb_bcemu25_max:.4f}]"
            )
        vals["fb"] = fb

        # 3. Compute suppression at each internal redshift on the emulator's
        #    native k grid, then interpolate onto the requested grid
        emu = self.bcemulator25
        kmin_emu = float(np.min(emu.k))  # h/Mpc
        suppression_dict = {}
        for i_z, z_val in enumerate(z_arr):
            try:
                k_emu, sup_emu = emu.get_boost(vals, z_val, q2=self.q2_bcemu25)
            except Exception as e:
                self.log.error(
                    "BCemu2025 failed at z=%.3f: %s; "
                    "using unity suppression for this redshift",
                    z_val,
                    str(e),
                )
                suppression_dict[z_val] = np.ones_like(k_arr)
                continue

            interp_emu = interp1d(
                np.log10(k_emu),
                np.log(sup_emu),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
                assume_sorted=True,
            )
            # k_arr is in 1/Mpc but BCemu2025 assumes h/Mpc units
            sup_interp = np.exp(interp_emu(np.log10(k_arr) - np.log10(h)))

            # No feedback on scales below the training range
            sup_interp[k_arr < h * kmin_emu] = 1.0

            suppression_dict[z_val] = sup_interp

        return suppression_dict

    def _unity_suppression(self, z_arr, k_arr):
        """
        Return unity suppression factors (no baryon feedback).

        Used as a graceful fallback when baryon model computation fails.

        Returns:
            dict: {z_val: ones_array} for each requested redshift
        """
        return {z_val: np.ones_like(k_arr) for z_val in z_arr}

    def get_baryon_suppression(self):
        """
        Accessor method for the likelihood to fetch suppression factors.

        Returns:
            dict: {z_val: suppression_array} from the current state.
                  Each suppression_array has shape (n_k,).

        Raises:
            AttributeError: If calculate() has not been called yet.

        ============================================================================
        NOTE ON SUPPRESSION VALUES > 1.0 AT HIGH K:
        ============================================================================

        It is physically reasonable and expected that log-space interpolation of
        pyspk output can produce S(k,z) > 1.0 at high wavenumbers (k > ~8 h/Mpc),
        particularly near the edges of pyspk's calibration grid.

        WHY THIS HAPPENS:
        - Boundary behavior in log-space: pyspk outputs on [k_min, k_max]; at the
          upper boundary, linear interpolation in log-log space can produce values
          slightly > 1.0 due to the shape of the log-suppression curve
        - Physical: at very high k, the suppression effect weakens; marginal values
          > 1 (e.g., 1.001-1.1) indicate interpolation near calibration boundaries
        - Numerical precision: floating-point precision in exponentiation can cause
          tiny excursions above 1.0

        WHAT WE ACCEPT:
        - Values in range [0, 2] are physically acceptable
        - Warnings are issued only for wildly unphysical values (S < 0 or S > 2.0)
        - Values > 1.0 are NOT automatically clipped; they are preserved as computed

        WHAT WOULD BE PROBLEMATIC:
        - S < 0: indicates negative suppression (power enhancement), unphysical
        - S > 2: extreme suppression or strong power enhancement, likely indicates
          parameter issues or extrapolation far outside calibration range

        For concerns about specific values, check log output:
          "k_range_pyspk=[...], k_range_requested=[...]"
        ============================================================================
        """
        return self.current_state["baryon_suppression"]
