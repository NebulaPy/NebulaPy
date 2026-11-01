#!/usr/bin/env python3

"""Isochoric cooling example; see README.md for the physical setup and usage."""


def main():
    import os
    from pathlib import Path

    import numpy as np
    from scipy.integrate import cumulative_simpson
    from NebulaPy import __version__
    from NebulaPy.src import Constants as const
    from NebulaPy.src.Cooling import cooling

    folder = Path(__file__).resolve().parent
    database = Path(os.environ.get('NEBULAPY_DB', folder.parents[1] / 'nebulapy-db-1.0.0'))
    os.environ['NEBULAPY_DB'] = str(database)
    abundances = dict(H=0.7381, He=0.2495736, C=0.002368, N=0.000693,
                      O=0.005739, Ne=0.0012579, Si=0.0006656,
                      S=0.0003096, Fe=0.0012933)
    total = sum(abundances.values())
    abundances = {element: value / total for element, value in abundances.items()}
    ne, gamma = 100.0, 5.0 / 3.0
    rho = ne / sum(value * const.ATOMIC_NUMBER[element] / 2 / const.ATOMIC_MASS[element]
                   for element, value in abundances.items())
    nuclei = {element: rho * value / const.ATOMIC_MASS[element]
              for element, value in abundances.items()}
    n_tot = ne + sum(nuclei.values())

    # Evaluate each ion at the table knots using the public cooling API.
    knots = 10.0 ** np.array(const.COOLING_LOG_TEMPERATURE_GRID)
    loss = np.zeros_like(knots)
    skipped = []
    for element in abundances:
        z = const.ATOMIC_NUMBER[element]
        for charge in range(z + 1):
            ion = element if charge == 0 else f'{element}{charge}+'
            if not (database / 'chianti_cooling_rates' / f'{element.lower()}_{charge + 1}.txt').is_file():
                skipped.append(ion)
                continue
            coefficients = cooling(ion).generate_cooling_rate_map(knots, np.full_like(knots, ne))
            loss += ne * nuclei[element] / (z + 1) * coefficients
    if not np.all(np.isfinite(loss)) or np.any(loss <= 0):
        raise ValueError('Total cooling must be finite and positive')

    # Integrate dt = -n_tot*k_B/(gamma-1) * dT/L(T), refining until converged.
    temperature = np.geomspace(1e9, 1e4, 50)
    previous = None
    for refinement in range(10):
        log_t = np.unique(np.r_[np.linspace(np.log(1e4), np.log(1e9), 4097 * 2**refinement),
                                np.log(temperature), np.log(knots[knots >= 1e4])])[::-1]
        grid = np.exp(log_t)
        integrand = n_tot * const.BOLTZMANN_CONSTANT / (gamma - 1) * grid / np.interp(grid, knots, loss)
        time = cumulative_simpson(integrand, x=-log_t, initial=0)
        elapsed = np.interp(-np.log(temperature), -log_t, time)
        if previous is not None and np.allclose(elapsed[1:], previous[1:], rtol=1e-8, atol=0):
            break
        previous = elapsed
    else:
        raise ValueError('Cooling times did not converge')
    if not np.all(np.isfinite(elapsed)) or np.any(np.diff(elapsed) <= 0):
        raise ValueError('Invalid cooling times')

    version = database / 'VERSION'
    database_version = version.read_text().strip() if version.is_file() else 'unknown'
    header = (f'Data generated using NebulaPy {__version__}, database version {database_version}\n\n'
              + 'elapsed_time [s]          temperature [K]')
    output = folder / 'isochoric_cooling.txt'
    np.savetxt(output, np.column_stack((elapsed, temperature)), fmt='%24.17e', header=header)
    print(f'Wrote {output}')
    if skipped:
        print('Skipped missing tables: ' + ', '.join(skipped))


if __name__ == '__main__':
    main()
