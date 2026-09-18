# External Baryonic Feedback Models

This repository implements external baryonic feedback suppresion models as a `Cobaya` theory block that can be used alongside Cosmolike. The implemented models are
- SP(k), Salcido et al 2023 https://arxiv.org/abs/2305.09710
- BCEmu, Giri & Schneider 2021 https://arxiv.org/abs/2108.08863
- FlamingoBaryonResponseEmulator, Schaller et al 2024 https://arxiv.org/abs/2410.17109
- BACCOemu, Burger et al 2025 https://arxiv.org/abs/2506.18974

See the corresponding papers for more details on the parameterizations.

## Setup

First, navigate to the `Cocoa/projects/` folder and clone the repository as `roman_real`:

```
    git clone https://github.com/nihardalal/cocoa_roman_real_baryons.git roman_real
```

Go back to the `Cocoa/` folder, activate the Cocoa environments (NOTE: you must do `source start_cocoa.sh` AFTER cloning) and install the Cosmolike interface as usual with the command:

```
    source projects/roman_real/installation_scripts/compile_roman_real.sh
```

You'll also need to install [SP(k)](https://github.com/jemme07/pyspk), [BCEmu](https://github.com/sambit-giri/BCemu), [FlamingoBaryonResponseEmulator](https://github.com/FLAMINGOSIM/FlamingoBaryonResponseEmulator), and [BACCOemu](https://bitbucket.org/rangulo/baccoemu/src/master/) in your `.local` environment (i.e. after `source start_cocoa.sh`). BCEmu further needs `smt==1.0.0` -- other versions are incompatible with the emulator and may break the environment. You can install all codes with:

```
    pip install pyspk BCemu smt==1.0.0 FlamingoBaryonResponseEmulator baccoemu
```

## Examples

The yaml files `EXAMPLE_SPK.yaml`, `EXAMPLE_BCEMU.yaml`, `EXAMPLE_FLAMINGO.yaml`, `EXAMPLE_BACCOEMU.yaml`, and `EXAMPLE_DMO.yaml` compute the model vector for the same cosmology but different baryon suppression modelling. The user should be able to run all 4 yaml files with `cobaya-run projects/roman-real/EXAMPLE_<CASE>.yaml`, where `<CASE>` is one of the four cases above. Each run saves the corresponding model vector as `<CASE>.modelvector`.

After running all four files, the notebook `plot_dvs.ipynb` plots the resulting data vectors from each example, showing the effect of baryonic feedback suppresion at data vector level.

## Changes compared to the original version

We introduce a new `Theory` block that takes cosmological parameters as well as model-specific variables as input, and returns a `baryon_suppression`, a 2D array that contains the baryonic feedback suppression ratio at a grid of wavenumbers and redshifts.

All communication between Cosmolike and the external baryonic feedback models is done through Cobaya, so no changes are necessary to the core Cosmolike code. A single variable, `external_baryon_suppression`, is added to the Cosmolike likelihood to trigger the inclusion of baryonic feedback suppression correction.

The only major change in the `_cosmolike_prototype_base.py` code is that, after the nonlinear matter power spectrum is computed (i.e. in the `set_cosmo_related()` function), if `external_baryon_suppression` is set, the baryon suppression gets added to the results. 