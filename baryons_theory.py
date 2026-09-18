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
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from cobaya.theory import Theory
from cobaya.log import LoggedError

AVAILABLE_BARYON_MODELS = {
    1: "SP(k)",
    2: "BCEmu",
    3: "FlamingoBaryonResponseEmulator",
    4: "BACCOemu",
}

class BaryonSuppression(Theory):
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
    baryon_model: int = 1  # 1=pyspk, 2=bcemu, 3=flamingo

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
        else:
            raise LoggedError(self.log, f"Invalid choice of `baryon_model`. Available options are 1 (SP(k), 2 (BCEmu), or 3 (FlamingoEmulator))")
        
        self.requested_z = np.array([])
        self.requested_k = np.array([])

        # pyspk Calibration ranges (from Kunhao's testing/tuning)
        self.z_min_calib = 0    # JVR NOTE: apparently SP(k) is not calibrated below z=0.125 but we are extending to z = 0
        self.z_max_calib = 3.0  # Above this, pyspk not well-calibrated
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

        # Initialize result dictionary
        suppression_dict = {}

        # Route to appropriate baryon model
        if self.baryon_model == 1:
            suppression_dict = self._calculate_pyspk(params_values_dict)
        elif self.baryon_model == 2:
            suppression_dict = self._calculate_bcemu(params_values_dict)
        elif self.baryon_model == 3:
            suppression_dict = self._calculate_flamingo(params_values_dict)
        elif self.baryon_model == 4:
            suppression_dict = self._calculate_baccoemu(params_values_dict)
        else:
            self.log.error(
                "baryon_model=%d is invalid; must be 1 (pyspk), 2 (bcemu), 3 (flamingo), or 4 (BACCOemu); "
                "returning unity suppression",
                self.baryon_model,
            )
            suppression_dict = self._unity_suppression()

        # Store result in state
        state["baryon_suppression"] = suppression_dict

    def _calculate_pyspk(self, params_values_dict):
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
            # Reject invalid parameters by raising exception; MCMC will treat as rejected sample
            if not (self.alpha_min < alpha < self.alpha_max):
                raise LoggedError(
                    self.log,
                    f"SPk parameter alpha_spk={alpha:.4f} outside valid range"
                    f"[{self.alpha_min:.4f}, {self.alpha_max:.4f}]",
                )

            if not (self.beta_min < beta < self.beta_max):
                raise LoggedError(
                    self.log,
                    f"SPk parameter beta_spk={beta:.4f} outside valid range "
                    f"[{self.beta_min:.4f}, {self.beta_max:.4f}]",
                )

            if not (self.gamma_min_spk < gamma < self.gamma_max_spk):
                raise LoggedError(
                    self.log,
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

            for i_z, z_val in enumerate(self.requested_z):
                # Check if redshift is within pyspk calibration range
                # Outside this range, we return unity (no suppression)
                if z_val < self.z_min_calib or z_val > self.z_max_calib:
                    self.log.debug(
                        "SPk z=%.3f outside calibration range [%.3f, %.3f]; "
                        "using unity suppression for this redshift",
                        z_val,
                        self.z_min_calib,
                        self.z_max_calib,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                try:
                    # Call pyspk to compute suppression on its native k-grid
                    k_spk, sup_spk = spk.sup_model(
                        SO=500,  # Spherical overdensity radius
                        z=z_val,
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
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
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
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
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
                    # JVR NOTE: self.requested_k is in 1/Mpc but SP(k) assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    sup_interp = np.exp(interp_spk(np.log10(self.requested_k) - np.log10(h)))

                except Exception as e:
                    self.log.error(
                        "Interpolation failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                # 6. Apply calibration masking: suppress effect outside calibration ranges
                # For k < 8.73e-3 h/Mpc, set suppression to 1 (outside calib; Kunhao's choice)
                # For k > pyspk's max k, use boundary value (no extrapolation needed now)
                # JVR NOTE: also keep the units in mind here
                sup_interp[self.requested_k < h*self.k_min_calib] = 1.0

                # Defensive check: verify interpolated suppression is physically reasonable
                if not np.all(np.isfinite(sup_interp)):
                    n_bad = np.sum(~np.isfinite(sup_interp))
                    self.log.error(
                        "Interpolation produced %d non-finite values at z=%.3f; "
                        "using unity suppression for this redshift",
                        n_bad,
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
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
                        self.requested_k.min(),
                        self.requested_k.max(),
                    )

                suppression_dict[z_val] = sup_interp

            self.log.info(
                "SPk suppression computed for %d redshifts, %d k-values",
                len(suppression_dict),
                len(self.requested_k),
            )
            return suppression_dict

        except LoggedError:
            # Parameter validation rejection—let it propagate to MCMC
            raise
        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in SPk calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression()

    def _calculate_bcemu(self, params_values_dict):
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
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter log10Mc_bcemu={log10Mc_bcemu:.4f} outside valid range "
                    f"[{self.log10Mc_min:.4f}, {self.log10Mc_max:.4f}]",
                )
            if not (self.mu_min < mu_bcemu < self.mu_max):
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter mu_bcemu={mu_bcemu:.4f} outside valid range "
                    f"[{self.mu_min:.4f}, {self.mu_max:.4f}]",
                )
            if not (self.thej_min < thej_bcemu < self.thej_max):
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter thej_bcemu={thej_bcemu:.4f} outside valid range "
                    f"[{self.thej_min:.4f}, {self.thej_max:.4f}]",
                )
            if not (self.gamma_min_bcemu < gamma_bcemu < self.gamma_max_bcemu):
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter gamma_bcemu={gamma_bcemu:.4f} outside valid range "
                    f"[{self.gamma_min_bcemu:.4f}, {self.gamma_max_bcemu:.4f}]",
                )
            if not (self.delta_min < delta_bcemu < self.delta_max):
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter delta_bcemu={delta_bcemu:.4f} outside valid range "
                    f"[{self.delta_min:.4f}, {self.delta_max:.4f}]",
                )
            if not (self.eta_min < eta_bcemu < self.eta_max):
                raise LoggedError(
                    self.log,
                    f"BCEmu parameter eta_bcemu={eta_bcemu:.4f} outside valid range "
                    f"[{self.eta_min:.4f}, {self.eta_max:.4f}]",
                )
            if not (self.deta_min < deta_bcemu < self.deta_max):
                raise LoggedError(
                    self.log,
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

            bfcemu = BCemu.BCM_7param(
                Ob=self.provider.get_param("omegab"),
                Om=self.provider.get_param("omegam"),
            )
            kmin_bcemu = 0.034  # from BCemu
            kmax_bcemu = 12.517  # from BCemu
            logkmin_bcemu = np.log10(kmin_bcemu)
            logkmax_bcemu = np.log10(kmax_bcemu)
            suppression_dict = {}
            for i_z, z_val in enumerate(self.requested_z):
                # Check if redshift is within BCEmu calibration range
                # (BCEmu is calibrated for z=0-2; outside this, we return unity)
                if z_val < 0.0 or z_val > 2.0:
                    self.log.debug(
                        "BCEmu z=%.3f outside calibration range [0.0, 2.0]; "
                        "using unity suppression for this redshift",
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

                try:
                    k_bcemu = 10 ** np.linspace(
                        logkmin_bcemu, logkmax_bcemu, 100
                    )  # h/Mpc
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
                    # JVR NOTE: self.requested_k is in 1/Mpc but BCEmu assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    sup_interp = np.exp(interp_bcemu(np.log10(self.requested_k) - np.log10(h)))

                    suppression_dict[z_val] = sup_interp

                except Exception as e:
                    self.log.error(
                        "BCEmu model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue

        except LoggedError:
            # Parameter validation rejection—let it propagate to MCMC
            raise
        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in BCEmu calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression()

        return suppression_dict

    def _calculate_flamingo(self, params_values_dict):
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
                raise LoggedError(
                    self.log,
                    f"Flamingo parameter fgas_sigma_flamingo={fgas_sigma_flamingo:.4f} outside valid range "
                    f"[{self.fgas_sigma_min:.4f}, {self.fgas_sigma_max:.4f}]",
                )
            if not (self.mstar_sigma_min < mstar_sigma_flamingo < self.mstar_sigma_max):
                raise LoggedError(
                    self.log,
                    f"Flamingo parameter mstar_sigma_flamingo={mstar_sigma_flamingo:.4f} outside valid range "
                    f"[{self.mstar_sigma_min:.4f}, {self.mstar_sigma_max:.4f}]",
                )
            if not (self.jet_frac_min <= jet_frac_flamingo <= self.jet_frac_max):
                raise LoggedError(
                    self.log,
                    f"Flamingo parameter jet_frac_flamingo={jet_frac_flamingo} outside valid range "
                    f"[{self.jet_frac_min:.4e}, {self.jet_frac_max:.4e}]",
                )

            myemu = fre.FlamingoBaryonResponseEmulator()
            logkmin_flamingo = -1.5
            logkmax_flamingo = 1.5
            suppression_dict = {}
            for i_z, z_val in enumerate(self.requested_z):
                # Check if redshift is within Flamingo calibration range
                # (Flamingo is calibrated for z=0-3; outside this, we return unity)
                if z_val < 0.0 or z_val > 3.0:
                    self.log.debug(
                        "Flamingo z=%.3f outside calibration range [0.0, 3.0]; "
                        "using unity suppression for this redshift",
                        z_val,
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
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
                    
                    # JVR NOTE: self.requested_k is in 1/Mpc but SP(k) assumes h/Mpc units so here we must convert requested_k to h/Mpc
                    H0 = self.provider.get_param("H0")
                    h = H0/100
                    sup_interp = np.exp(interp_flamingo(np.log10(self.requested_k) - np.log10(h)))

                    suppression_dict[z_val] = sup_interp

                except Exception as e:
                    self.log.error(
                        "Flamingo model failed at z=%.3f: %s; "
                        "using unity suppression for this redshift",
                        z_val,
                        str(e),
                    )
                    suppression_dict[z_val] = np.ones_like(self.requested_k)
                    continue
        except LoggedError:
            # Parameter validation rejection—let it propagate to MCMC
            raise
        except Exception as e:
            # Catch-all: any other uncaught exception → graceful degradation
            self.log.error(
                "Uncaught exception in Flamingo calculation: %s; "
                "returning unity suppression",
                str(e),
            )
            return self._unity_suppression()

        return suppression_dict

    def _calculate_baccoemu(self, params_values_dict):
        suppression_dict = {}

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
        for i_z, z_val in enumerate(self.requested_z):
            a = 1/(1+z_val)
            if a < 0.25: 
                self.log.debug(
                    "BACCOemu z=%.3f outside calibration range a \\in [0.25, 1.0]; "
                    "using unity suppression for this redshift",
                    z_val,
                )
                suppression_dict[z_val] = np.ones_like(self.requested_k)
                continue
            common_params.update({"expfactor": a})
            k_bacco, S = self.baccoemulator.get_baryonic_boost(**common_params)
            interp_bacco = interp1d(
                np.log10(k_bacco),
                np.log(S),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
                assume_sorted=True,
            )
            sup_interp = np.exp(interp_bacco(np.log10(self.requested_k) - np.log10(h)))
            suppression_dict[z_val] = sup_interp

        return suppression_dict

    def _unity_suppression(self):
        """
        Return unity suppression factors (no baryon feedback).

        Used as a graceful fallback when baryon model computation fails.

        Returns:
            dict: {z_val: ones_array} for each requested redshift
        """
        return {z_val: np.ones_like(self.requested_k) for z_val in self.requested_z}

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
