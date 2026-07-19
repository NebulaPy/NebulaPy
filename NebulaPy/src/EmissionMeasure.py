import numpy as np
from astropy import units as u
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import track

logger = get_logger(__name__)

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")

class emissionMeasure():


    ######################################################################################
    # initializing
    ######################################################################################
    def __init__(self, Tmin, Tmax, Nbins, progress=False):

        self.progress = progress
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Nbins = Nbins

        logger.info("Generating DEM temperature bins")

        # Calculate the logarithmic width of each bin
        bin_width = (np.log10(self.Tmax) - np.log10(self.Tmin)) / self.Nbins
        # Half the width of a bin for adjusting bin edges
        self.half_bin_width = bin_width / 2

        # Generate the logarithmically spaced temperature bin edges.
        # This will create Nbins+1 edges to define the boundaries of Nbins.
        self.temperature_edges = np.linspace(np.log10(self.Tmin), np.log10(self.Tmax), self.Nbins + 1)

        # Create temperature bins by pairing adjacent edges.
        # Each bin is represented as [bin_min, bin_max].
        self.temperature_bins = [[self.temperature_edges[i], self.temperature_edges[i + 1]] for i in range(self.Nbins)]

        # Calculate the midpoints of each bin for potential further use.
        # These midpoints are the average of the logarithmic bin edges.
        self.Tb = np.array([(bin[0] + bin[1]) / 2 for bin in self.temperature_bins])

    ######################################################################################
    # generate differential emission measure indices todo: not verified
    ######################################################################################
    def generate_DEM_indices(self, temperature):

        logger.info("Mapping cells to DEM temperature bins")

        DEM_indices = np.full_like(temperature, fill_value=-1, dtype=np.int64)

        log_temperature = np.log10(temperature)

        for bin_idx in range(self.Nbins):

            bin_min = self.temperature_edges[bin_idx]
            bin_max = self.temperature_edges[bin_idx + 1]

            if bin_idx == self.Nbins - 1:
                mask = (log_temperature >= bin_min) & (log_temperature <= bin_max)
            else:
                mask = (log_temperature >= bin_min) & (log_temperature < bin_max)

            DEM_indices[mask] = bin_idx

        logger.info("DEM temperature-bin indexing completed")

        self.DEM_indices = DEM_indices

    ######################################################################################
    # differential emission measure for 2D grid
    ######################################################################################
    def DEM2D(self, temperature, ne, speciesDensities, volume, gridMask):

        # Generate temperature-bin index grid
        self.generate_DEM_indices(temperature)

        temperature = np.asarray(temperature, dtype=np.float64)
        ne = np.asarray(ne, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        gridMask = np.asarray(gridMask, dtype=np.float64)

        DEM = {}

        species_list = list(speciesDensities.items())

        for species, species_density in track(
                species_list,
                description="Computing species DEM",
                unit="species",
                enabled=self.progress,
        ):

            species_density = np.asarray(species_density, dtype=np.float64)

            # Safety check
            if temperature.shape != species_density.shape:
                raise NebulaError(
                    f" DEM-2D input arrays have inconsistent shapes for species {species}."
                )

            # Accumulate all temperature bins in one pass. The previous loop
            # constructed three full-grid masks for every bin and species.
            valid_cells = (
                (self.DEM_indices >= 0)
                & (self.DEM_indices < self.Nbins)
                & (species_density > 0.0)
            )

            weights = ne * gridMask
            weights *= species_density
            weights *= volume

            species_DEM = np.bincount(
                self.DEM_indices[valid_cells].ravel(),
                weights=weights[valid_cells].ravel(),
                minlength=self.Nbins,
            ).astype(np.float64, copy=False)

            DEM[species] = species_DEM

        self.DEM = DEM


    ###############################################################################
    def SAM_DEM(self, density, temperature, ne, mask, ngrid, volume, mesh_edges_min, mesh_edges_max, temp_bin, hw):
        # Function to calculate the differential emission measure of the nebula
        # See Green et al. (2019) - Bubble Nebula - paper for details.
        # temp_bin: array of temperature bins in logspace
        # hw:       half-width of bin

        # todo: this function need to be optimized and rewritten.
        # info: this fucntion is now used inside Xray spectrum.py
        density = density
        temp = temperature
        mask = mask
        ngrid = ngrid
        lim_max = (mesh_edges_max * u.cm).value
        lim_min = (mesh_edges_min * u.cm).value
        vol = volume

        dem_bin_all = np.zeros(len(temp_bin))


        for j in range(len(density)):
            denj = density[j]
            tempj = temp[j]
            maskj = mask[j]
            # info: replaced with my code to calculate the cell volume and ne
            #volj = self.SAMvolume2D(lim_max[j], lim_min[j], ngrid)
            volj = vol[j]
            #nei = denj * maskj
            #nei = nei * 1.2 * 0.715 / m_p  # assume 1.2 electrons per H ion
            nei = ne[j]
            # These are the two arrays I need:
            vol_den = volj * nei * nei
            log_masktemp = np.log10(tempj * maskj)
            # If masktemp[i,j,k] is in one bin range w, then add vol_dens[i,j,k] to dem[w]
            for w in range(len(temp_bin)):
              pick = np.zeros_like(log_masktemp)
              pick[(log_masktemp>=(temp_bin[w]-hw))&(log_masktemp<(temp_bin[w]+hw))] = 1
              dem_bin_all[w] += np.sum(vol_den * pick)

        return dem_bin_all
