"""Plot intrinsic line luminosity per cylindrical cell (not surface brightness)."""

from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Macbook -> Set up paths and filenames
OutputDir = '/Users/tony/Desktop/XrayBowshock/post-processing/LuminosityMaps'
SiloDir = '/Users/tony/Desktop/Bowshock-2026/multi-ion-bowshock/high-res-silo-200kyr'
Filebase = 'Ostar_mhd-nemo-dep_d2n0384l3'
start_time = 200  # kyr, as specified by time_unit
finish_time = None
time_unit = 'kyr'
out_frequency = None
SimulationName = 'Bowshock'

# Edit the ion and CHIANTI wavelengths (Angstrom) as needed.
pion_ion = 'C2+'
lines = [977.02]

# Colour limits for log10(cell luminosity / (erg/s)).
# Leave either limit as None to determine it from each map.
vmin = 23  # e.g. 25
vmax = 25  # e.g. 30


def main():
    import NebulaPy.src as nebula

    batched_silos = nebula.Silo.batch(
        SiloDir, Filebase, start_time=start_time, finish_time=finish_time,
        time_unit=time_unit, out_frequency=out_frequency,
    )
    if not batched_silos:
        raise RuntimeError('No snapshots matched the requested time selection.')

    pion = nebula.pion(batched_silos, progress=True)
    pion.load_geometry(scale='cm')  # All luminosity calculations use CGS.

    nemo = nebula.NEMO(pion)
    nemo.load_chemistry()
    emission = nebula.line_emission(pion_ion)
    cell_volume = pion.get_grid_volumes_2D()  # cm^3; static nested grid
    output_dir = Path(OutputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step, silo_instant in enumerate(batched_silos):
        sim_time = pion.get_simulation_time(silo_instant, time_unit=time_unit)

        #maps = emission.line_luminosity_map_2D(
        #    lines=lines,
        # #   temperature=pion.get_parameter('Temperature', silo_instant),
        #    ne=nemo.get_ne(silo_instant),
        #    species_density=nemo.get_ion_number_density(pion_ion, silo_instant),
        #    cell_volume=cell_volume,
        #    grid_mask=pion.geometry_container['mask'],
        #)


        print(emission.line_luminosity_2D(lines=lines,
            temperature=pion.get_parameter('Temperature', silo_instant),
            ne=nemo.get_ne(silo_instant),
            species_density=nemo.get_ion_number_density(pion_ion, silo_instant),
            cell_volume=cell_volume,
            grid_mask=pion.geometry_container['mask'],)
        )

        '''
        for label, level_maps in maps.items():
            # Maps already exclude masked cells; sum linear luminosities in erg/s.
            total_luminosity = sum(np.sum(values, dtype=np.float64) for values in level_maps)
            print(f'{label} | {sim_time.value:.3f} {time_unit} | '
                  f'Total luminosity = {total_luminosity:.6e} erg/s')
            tag = label.replace(' ', '_')
            filename = output_dir / (
                f'{Filebase}_luminosity_{tag}_{step:04d}_'
                f'{sim_time.value:.6f}{time_unit}.png'
            )
            positive = [a[np.isfinite(a) & (a > 0)] for a in level_maps]
            positive = [a for a in positive if a.size]
            if not positive:
                print(f'Skipping {label}: no positive finite luminosities.')
                continue
            plot_vmin = np.log10(min(a.min() for a in positive)) if vmin is None else vmin
            plot_vmax = np.log10(max(a.max() for a in positive)) if vmax is None else vmax
            if plot_vmin == plot_vmax and vmin is None and vmax is None:
                plot_vmin, plot_vmax = plot_vmin - 1, plot_vmax + 1
            if not (-np.inf < plot_vmin < plot_vmax < np.inf):
                raise ValueError('Log10 colour limits must be finite with vmin < vmax.')
            edges_min = [edge.to_value(u.pc) for edge in pion.geometry_container['edges_min']]
            edges_max = [edge.to_value(u.pc) for edge in pion.geometry_container['edges_max']]

            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            ax.set_facecolor('0.9')
            for level, values in enumerate(level_maps):
                lo, hi = edges_min[level], edges_max[level]
                # Transparent excluded/zero cells let other valid grid levels show.
                data = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
                data = np.ma.log10(data)
                image = ax.imshow(
                    data, origin='lower', interpolation='nearest', cmap='inferno',
                    vmin=plot_vmin, vmax=plot_vmax,
                    extent=[lo[0], hi[0], lo[1], hi[1]], aspect='equal',
                )
            ax.set_xlim(edges_min[0][0], edges_max[0][0])
            ax.set_ylim(edges_min[0][1], edges_max[0][1])
            ax.set_xlabel('z (pc)')
            ax.set_ylabel('R (pc)')
            ax.set_title(
                f'{SimulationName} | {label} Å | {sim_time.value:.3f} {time_unit}\n'
                f'Total luminosity = {total_luminosity:.3e} erg/s'
            )
            fig.colorbar(image, ax=ax, label=r'$\log_{10}[L_{\mathrm{cell}} / (\mathrm{erg\,s^{-1}})]$')
            fig.savefig(filename, dpi=200)
            plt.close(fig)
            print(f'Saved {filename}')
        '''

if __name__ == '__main__':
    main()
