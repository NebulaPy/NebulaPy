# Spectral-energy distribution examples

These scripts rebin ATLAS, PoWR, or blackbody spectra into their configured
radiation energy groups. Generated plots and PION-formatted tables are written
to `output/` by default.

Install the NebulaPy database and set its location before running an example:

```bash
nebulapy database install --destination "$HOME/.nebulapy"
export NEBULAPY_DB="$HOME/.nebulapy/nebulapy-db-1.0.0"
```

Run the examples from the repository root:

```bash
python examples/SEDs/atlas_sed.py
python examples/SEDs/powr_sed.py
python examples/SEDs/blackbody_sed.py
```

Choose another output directory with `--output`:

```bash
python examples/SEDs/atlas_sed.py --output "/path/to/output"
```

Model parameters can also be selected from the command line:

```bash
python examples/SEDs/atlas_sed.py --metallicity 0.0 --gravity 4.5
python examples/SEDs/powr_sed.py \
  --metallicity SMC --composition WNL-H20 --mdot -5.0
```

Use `--help` with any script to display all available options.
