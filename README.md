# External Baryonic Feedback Models

This repository implements external baryonic feedback suppression models as the `Cobaya`
theory block `bfmt`, used alongside Cosmolike. The theory block provides the product
`baryon_suppression`: the ratio $S(k,z)$ that the likelihood applies to the nonlinear
matter power spectrum, $P_\mathrm{nl}(k,z) = S(k,z) \, P_\mathrm{DMO}(k,z)$. The
implemented models are

- SP(k), Salcido et al 2023 https://arxiv.org/abs/2305.09710
- BCEmu, Giri & Schneider 2021 https://arxiv.org/abs/2108.08863
- FlamingoBaryonResponseEmulator, Schaller et al 2024 https://arxiv.org/abs/2410.17109
- BACCOemu, Burger et al 2025 https://arxiv.org/abs/2506.18974

See the corresponding papers for the parameterizations.

## Installation

Cocoa installs these models via environmental keys on `set_installation_options.sh`:
this repository (linked into Cobaya as the theory `bfmt`) and the four emulator codes,
at pinned commits, installed on Cocoa's `.local` together with their Python
dependencies (including `smt==1.0.0`, which BCEmu requires — other versions are
incompatible with the emulator). Users must ensure the following lines are commented
out in `set_installation_options.sh` before running `setup_cocoa.sh` and
`compile_cocoa.sh`. *By default, these lines should be commented out, but it is worth
checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      # insert the # symbol (i.e., unset these environmental keys on `set_installation_options.sh`)
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block (this repository)

      (...)

      export BFMT_THEORY_URL="https://github.com/CosmoLike/cocoa_baryonic_feedback_models_theory.git"
      export BFMT_NAME="baryon_suppression"
      #export BFMT_GIT_TAG="v1.0"

      (...)

      export PYSPK_URL="https://github.com/jemme07/pyspk.git"
      export PYSPK_GIT_COMMIT="50737f9295fee75ef2fe97e4bdd284134ed0d474"
      export PYSPK_NAME="pyspk"

      export BCEMU_URL="https://github.com/sambit-giri/BCemu.git"
      export BCEMU_GIT_COMMIT="c32577654e1b48b4bdf079c01a26b16f1473598f"
      export BCEMU_NAME="bcemu"

      export FBRE_URL="https://github.com/FLAMINGOSIM/FlamingoBaryonResponseEmulator.git"
      export FBRE_GIT_COMMIT="76b259cd44d28d9ab24b5f08ffaf536a9d4c41d7"
      export FBRE_NAME="fbre"

      export BACCOEMU_URL="https://bitbucket.org/rangulo/baccoemu.git"
      export BACCOEMU_GIT_COMMIT="2dfea6e3960fe0239c1d56480f952c3da7e9843b"
      export BACCOEMU_NAME="baccoemu"

> [!Warning]
> Do not `pip install` the emulators directly. Cocoa pins their commits and seeds their
> Python dependencies with guarded versions; a direct pip install can upgrade
> numpy/scipy on `.local` and break the environment.

## Usage

Baryonic feedback requires three additions to the YAML file.

**Step :one:**: add the theory block and select the model:

```yaml
theory:
  bfmt:
    baryon_model: 1 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu
    nz: 20  # internal (z, k) computation grid; the result is 2D-splined
    nk: 100 # onto the grid the likelihood requests
    above_zmax: unity # S(k,z) above the model range: unity (default) or constant
```

The values above are the defaults; `baryon_model` is the only required key.

**Step :two:**: enable the correction on the Cosmolike likelihood:

```yaml
likelihood:
  roman_real.cosmic_shear:
    external_baryon_suppression: True
```

**Step :three:**: add the sampled parameters of the selected model to the `params` block:

| `baryon_model` | Model    | Sampled parameters |
| -------------- | -------- | ------------------ |
| 1 | SP(k)    | `alpha_spk`, `beta_spk`, `gamma_spk` |
| 2 | BCEmu    | `log10Mc_bcemu`, `mu_bcemu`, `thej_bcemu`, `gamma_bcemu`, `delta_bcemu`, `eta_bcemu`, `deta_bcemu` |
| 3 | Flamingo | `fgas_sigma_flamingo`, `mstar_sigma_flamingo`, `jet_frac_flamingo` |
| 4 | BACCOemu | `M_c_baccoemu`, `eta_baccoemu`, `beta_baccoemu`, `M1_z0_cen_baccoemu`, `theta_inn_baccoemu` |

`projects/roman_real/EXAMPLE_EVALUATE1.yaml` is a working example with every `bfmt` key
set explicitly. For a dark-matter-only comparison, set
`external_baryon_suppression: False`.

> [!Warning]
> Behavior outside the models' validity ranges.
>
> 1. Calibration redshift ranges: SP(k) $z \in [0.125, 3]$; BCEmu $z \in [0, 2]$;
>    Flamingo $z \in [0, 3]$; BACCOemu $z \lesssim 3$ ($a \geq 0.25$). Above these
>    ranges the theory block returns unity suppression, $S = 1$ (no feedback); the key
>    `above_zmax: constant` instead clamps $S$ to its value at the model's zmax. Below
>    the SP(k) floor, the suppression is clamped to its value at $z = 0.125$.
> 2. Sampled parameters outside the validation bounds reject the sample ($-\infty$
>    posterior) and print a warning naming the parameter; the run continues.
> 3. BACCOemu additionally enforces its cosmology training box
>    (`omega_baryon` $\in [0.04, 0.06]$, `omega_cold` $\in [0.23, 0.40]$,
>    `sigma8_cold` $\in [0.73, 0.90]$); violations reject the sample the same way.
>    Keep the cosmology priors inside the box when sampling with `baryon_model: 4`.

## Design

All communication between Cosmolike and the external models goes through Cobaya; no
changes to the core Cosmolike code. The likelihood requests `baryon_suppression` on its
$(k, z)$ interpolation grid ($k$ in $1/\mathrm{Mpc}$; the theory block converts to
$h/\mathrm{Mpc}$ internally) and, in `_cosmolike_prototype_base.py`, multiplies the
nonlinear power spectrum by $S(k,z)$ after it is computed in `set_cosmo_related()`.

The theory block does not evaluate the feedback model on the full requested grid.
It computes $S$ on the internal `nz` $\times$ `nk` grid and 2D-splines
$\ln S$ over $(z, \log_{10} k)$ onto the requested grid — the same strategy Cobaya
uses for its matter-power-spectrum interpolator. Emulators are never asked to
extrapolate outside their training ranges in $k$: below the training minimum the
suppression is set to $S = 1$ (no feedback on large scales).
