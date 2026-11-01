# Isochoric cooling example

Run from the repository root with NebulaPy installed:

```sh
python examples/Cooling/isochoric_cooling.py
```

The example uses `cooling(ion).generate_cooling_rate_map(temperature, ne)` to
calculate radiative losses at constant density. H, He, C, N, O, Ne, Si, S,
and Fe have normalized mass abundances and equal, fixed fractions across all
charge states: an element with atomic number `Z` has fraction `1/(Z+1)` in
each state, from neutral to fully stripped. These fractions are prescribed,
not equilibrium values, and do not evolve. No heating is included. Electron density is 100 cm^-3 and gamma is 5/3. Missing tables
are skipped without changing the particle densities.

The volumetric cooling rate is `L(T) = ne * sum(ni * Lambda_i)`. Total particle
density includes electrons and all nuclei, including neutral atoms. The
temperature history follows radiative energy loss at fixed mass density:

```text
t(T) = n_tot*k_B/(gamma - 1) * integral from T to 1e9 K of dT'/L(T')
```

Cooling time is integrated in log temperature with a refined Simpson grid
until cumulative times agree to a relative tolerance of `1e-8`. The script
writes 50 temperature checkpoints from 1e9 to 1e4 K to `isochoric_cooling.txt`
beside the script, with elapsed time in seconds and temperature in kelvin.
The output header contains column names, units, and package/database versions.
The physical description is kept here rather than repeated in the script or output.

The default database is the bundled `nebulapy-db-1.0.0`; set `NEBULAPY_DB` to
use another installation. This example uses NebulaPy's linear table
interpolation and the tabulated free-free, line, and two-photon cooling.
