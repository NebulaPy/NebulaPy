import os
import glob
import re
import math
import numpy as np
from astropy.io import fits
from scipy import integrate
import matplotlib.pyplot as plt
import NebulaPy.version as version
from NebulaPy.src import Constants as const
from NebulaPy.src.LoggingConfig import NebulaError, get_logger
from NebulaPy.src.Progress import update_progress

logger = get_logger(__name__)
# TODO: Include tolerance for mdot value in potsdam model
# TODO: Mdot value for SMC-OB-Vd3 and other SMC check
# TODO: Include these new model in wiki page

# TODO: few important info about the sed must be printed on to the
#  outfile like metalicity, massfractions, type, original mdot values,
#  original clumb size used etc

# TODO: In the output file, when writing pion format, name
#  has to be all with underscore rather than bar



class sed:

    def __init__(self, energy_bins, plot=None, pion=None, progress=False):

        # get database
        database = os.environ.get("NEBULAPYDB")
        # Check if the database exists, exit if missing
        if database is None:
            raise NebulaError("required database missing, install database to proceed")

        self.EnergyBins = energy_bins
        self.Plot = plot
        self.Pion = pion
        self.progress = progress
        self.container = {'energy_bins': self.EnergyBins,
                          'plot': self.Plot, 'pion': self.Pion}
        self.AtlasDatabase = os.path.join(database, "SED", "Atlas")
        self.PoWRDatabase = os.path.join(database, "SED", "PoWR")
        self.CMFGENDatabase = os.path.join(database, "SED", "CMFGEN")
        self.setup_lambda_bin()

    ##############################################################################
    def setup_lambda_bin(self):
        '''
        making wavelenght bins from the given energy bins
        :return:
        '''
        # making wave-lenght bins from the given energy bins
        given_lambdabins = []
        # Loop through the rows (outer loop)
        for Ebin in self.EnergyBins:
            # Loop through the columns (inner loop)
            lambdaBin = []
            for energy in Ebin:
                Lambda = const.EV_ANGSTROM / energy  # Waveleghts are given in Angstrom
                lambdaBin.append(Lambda)
            lambdaBin.reverse()
            given_lambdabins.append(lambdaBin)
        given_lambdabins.reverse()
        self.given_lambdabins = given_lambdabins
        self.container['lambda_bins'] = given_lambdabins
        self.container['wvl_unit'] = 'Angstrom'

    #########################################################################################
    # progress bar
    #########################################################################################
    def progress_bar(self, iteration, Nmodels, prefix='', fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total_files - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            fill       - Optional  : bar fill character (Str)
        """
        update_progress(
            key=prefix,
            description=prefix,
            completed=iteration + 1,
            total=Nmodels,
            unit="models",
            enabled=self.progress,
        )


    ######################################################################################
    # incomplete
    ##############################################################################
    def bundle_up_atlas_models(self, metallicity):
        '''
        TODO: no completed
        1.get the model and return it
        2.if the parameter are wrong return some advice

        :param metallicity:
        :param gravity:
        :return:
        '''

        sign = None
        if metallicity < 0:
            sign = 'm'
        if metallicity >= 0:
            sign = 'p'

        metallicity_str = str(metallicity).replace('.', '')

        # Construct model directory name
        model_dirname = 'ck' + sign + metallicity_str

        model_dir = os.path.join(self.AtlasDatabase, model_dirname)

        # Define the fits file pattern.
        model_file_pattern = 'ck' + sign + metallicity_str + '_*.fits'

        # Collecting all fits files with similar names.
        model_set = glob.glob(os.path.join(model_dir, model_file_pattern))

        self.model_name = 'ck' + sign + metallicity_str

        return model_set



    ######################################################################################
    # incomplete
    ######################################################################################
    def bundle_up_potsdam_models(self, metallicity, composition, mdot):
        '''

        :param metallicity:
        :param composition:
        :param mdot:
        :return:
        '''
        # generating model grid name
        grid_name = metallicity.lower().replace(".", "") + '-' + composition.lower()
        self.grid_name = grid_name
        # Construct path to the grid directory
        grid_dir = os.path.join(self.PoWRDatabase, grid_name + '-sed')
        # get model parameter file
        modelparameters_file = os.path.join(grid_dir, 'modelparameters.txt')

        # ******************************************************************
        # exact correct model from 'modelparameters.txt' file, remove model
        # which do not fall in the parameter space
        columns = ["MODEL", "T_EFF", "R_TRANS", "MASS", "LOG G", "LOG L",
                   "LOG MDOT", "V_INF"]
        modelparams = {column: [] for column in columns}

        with open(modelparameters_file, 'r') as file:
            lines = file.readlines()

            # Removing first few lines
            lines = lines[8:]

            for line in lines:
                parts = line.split()
                modelparams["MODEL"].append(parts[0])
                modelparams["T_EFF"].append(float(parts[1]))
                modelparams["R_TRANS"].append(float(parts[2]))
                modelparams["MASS"].append(float(parts[3]))
                modelparams["LOG G"].append(float(parts[4]))
                modelparams["LOG L"].append(float(parts[5]))
                modelparams["LOG MDOT"].append(float(parts[6]))
                modelparams["V_INF"].append(float(parts[7]))

        bundle_modelparams = {column: [] for column in columns}
        unique_temperatures = set(modelparams["T_EFF"])

        temp_to_closest_index = {}
        for temp in unique_temperatures:
            indices = [i for i, x in enumerate(modelparams["T_EFF"]) if x == temp]
            closest_index = min(indices, key=lambda i: abs(modelparams["LOG MDOT"][i] + abs(mdot)))
            temp_to_closest_index[temp] = closest_index

        # Sort temperatures
        sorted_temps = sorted(temp_to_closest_index.keys())

        # Populate filtered models according to sorted temperatures
        for temp in sorted_temps:
            closest_index = temp_to_closest_index[temp]
            for column in columns:
                bundle_modelparams[column].append(modelparams[column][closest_index])
        # ******************************************************************

        # two info to make plot title
        self.which_gird_model = bundle_modelparams['MODEL']
        self.Nmodels = len(bundle_modelparams['MODEL'])
        self.Teff = bundle_modelparams['T_EFF']

        # append bundled model to the container
        self.container.update(bundle_modelparams)

        # Add clumping factor to the container
        # Get clumping factor for the specific grid
        Clump = const.POWR_CLUMPING_FACTORS[grid_name]
        self.container['CLUMP'] = Clump

        # Calculating stellar radius for each models
        R_star = []
        for i in range(len(bundle_modelparams['MODEL'])):
            velocity = (
                float(bundle_modelparams['V_INF'][i])
                / const.POWR_REFERENCE_VELOCITY_KM_S
            )
            massloss = (
                10 ** float(bundle_modelparams['LOG MDOT'][i])
                * math.sqrt(Clump)
                * const.POWR_MASS_LOSS_SCALE
            )
            radius = float(bundle_modelparams['R_TRANS'][i]) / (velocity / massloss) ** (2 / 3)
            R_star.append(radius)
        # Adding calculated stellar radius to the bundle and container
        bundle_modelparams['R_STAR'] = R_star
        self.R_star = R_star
        self.container['R_STAR'] = R_star
        # Adding

        # ******************************************************************
        # Make file paths for all the models in the final bundle
        bundle_model_files = []
        for model in bundle_modelparams['MODEL']:
            model_filename = grid_name + '_' + model + '_sed.txt'
            model_file = os.path.join(grid_dir, model_filename)
            bundle_model_files.append(model_file)
        return bundle_model_files


    ######################################################################################
    # sort bundled model set by temperature
    ######################################################################################
    def sort_modelset_by_temperature(self, model_set):
        '''
        This will extract temperature values from the all the file name
         in the model set
        :param model_set:
        :return:
        '''
        Teff = []
        TeffFileSet = []
        for file in model_set:
            FileMatch = re.search(self.model_name + r'_(\d+)\.fits', file)
            temperature = FileMatch.group(1)
            if FileMatch:
                Teff.append(float(temperature))
                TeffFileSet.append((float(temperature), file))
        # Sorting.
        TeffFileSet.sort(key=lambda x: x[0])
        # Sorted finalized file set.
        model_set = [file for _, file in TeffFileSet]
        del TeffFileSet
        self.Nmodels = len(model_set)
        self.Teff = sorted(Teff)
        self.container['Teff'] = self.Teff
        return model_set

    ######################################################################################
    # Blackbody bin fraction
    ######################################################################################
    def blackbody_binfrac_func(self, T, E):
        factor = 2.0*const.PI / pow(const.PLANCK_CONSTANT, 3.0) / pow(const.SPEED_OF_LIGHT, 2.0) \
                 / const.STEFAN_BOLTZMANN_CONSTANT / pow(T, 4.0)
        return factor * pow(E, 3.0) / (math.exp(E / const.BOLTZMANN_CONSTANT / T) - 1.0)

    ######################################################################################
    # Blackbody flux lambda
    ######################################################################################
    def blackbody_flam(self, T, lam):
        factor = 2.0*const.PI*const.PLANCK_CONSTANT*pow(const.SPEED_OF_LIGHT, 2.0) / pow(lam*const.ANGSTROM_TO_CM, 4.0) / lam
        return factor / (math.exp(const.PLANCK_CONSTANT * const.SPEED_OF_LIGHT / (lam*const.ANGSTROM_TO_CM) / const.BOLTZMANN_CONSTANT / T) - 1.0)

    ######################################################################################
    # Binning Atlas Spectral Energy Distribution
    ######################################################################################
    def CastelliKuruczAtlas(self, metallicity, gravity):
        '''

        :param metallicity:
        :param gravity:
        :return:
        '''
        self.container['model'] = 'Atlas'
        self.Model = 'Atlas'
        self.container['metallicity'] = metallicity
        self.metallicity = metallicity
        self.container['gravity'] = gravity
        self.gravity = gravity

        # model set corresponding to a specific metallicity
        model_set = self.bundle_up_atlas_models(metallicity)
        # model set sorting according to accending T_eff of SED
        model_set = self.sort_modelset_by_temperature(model_set)


        # make column name from the input gravity parameter
        # TODO: do more, cross reference with existing info
        column_name = 'g' + str(gravity).replace('.', '')

        self.sed_set_name = 'atlas_' + self.model_name + column_name

        prefix_comment = self.sed_set_name + ' binning'
        model_flux_set = []
        model_lambda_set = []
        binned_flux_set = []
        total_flux_set = []


        for model_index, model in enumerate(model_set):

            self.progress_bar(model_index, self.Nmodels,
                              prefix=prefix_comment)

            # Fits File Reading #####################################################
            # Atlas model in PyMicroPION database are in fits file format.
            # The datas are obtained form: https://www.stsci.edu/hst/instrumentation/
            # reference-data-for-calibration-and-tools/astronomical-catalogs/castelli
            # -and-kurucz-atlas
            # Open the FITS file to read wavelength and flam
            with fits.open(model) as hdul:
                # The open function returns an object called an HDUList which is a
                # list-like collection of HDU objects. An HDU (Header Data Unit) is
                # the highest level component of the FITS file structure, consisting
                # of a header and (typically) a data array or table.

                # Files opened for reading can be verified and fixed with method
                # HDUList.verify. This method is typically invoked after opening the
                # file but before accessing any headers or data:
                hdul.verify('fix')

                # hdul[0] is the primary HDU, hdul[1] is the first extension HDU
                # The header in hdul[0] of the Atlas fits file contains comprehensive
                # data details, while the actual data is stored in hdul[1].

                # Retrieve the wavelength and FLAM from the designated FITS file.
                model_lambda = hdul[1].data['Wavelength']
                model_flux = hdul[1].data[column_name]

                # Binning the wavelength and flux of the original model.
                # Binning lambda values and flam values according to the Lambda
                # Bins (obtained from the input energy bin).
                binned_lambda = []
                binned_flux = []
                for bin in self.given_lambdabins:
                    sub_binned_lambda = []
                    sub_binned_flux = []
                    for lam, flux in zip(model_lambda, model_flux):
                        if bin[0] <= lam <= bin[1]:
                            sub_binned_lambda.append(lam)
                            sub_binned_flux.append(flux)
                    binned_lambda.append(sub_binned_lambda)
                    binned_flux.append(sub_binned_flux)
                    del sub_binned_lambda
                    del sub_binned_flux

                # calculating the normalization factor, perform integration across
                # the entire wavelength domain to obtain the total flux.
                total_flux = np.trapz(np.asarray(model_flux), np.asarray(model_lambda))
                # Append the total Flux into TotalFlux_BundledGrids
                total_flux_set.append(total_flux)

                model_lambda_set.append(model_lambda)
                model_flux_set.append(model_flux)

                # calculating flux in each binned flux by integrating within the bin
                # interval
                flux_bin = []
                for i in range(len(binned_lambda)):
                    flux_bin.append(np.trapz(np.asarray(binned_flux[i]),
                                             np.asarray(binned_lambda[i])))

                # reverse the order of the flux bins since we are interested in
                # obtaining flux in energy bins.
                flux_bin.reverse()
                # Normalizing flux within each bin.
                if total_flux == 0.0:
                    norm_flux_bin = np.zeros_like(flux_bin)
                else:
                    norm_flux_bin = flux_bin / total_flux
                # Removing binned flux array
                del flux_bin

                # Appending the result fractional flux for each sed effective temperature
                # to final binned flux set
                binned_flux_set.append(norm_flux_bin)
            # End of Fits File Reading ##################################################

        self.binned_flux_set = binned_flux_set
        self.container['binned_flux'] = binned_flux_set
        self.container['total_flux'] = total_flux_set
        self.model_lambda_set = model_lambda_set
        self.model_flux_set = model_flux_set
        # End of binning model set ######################################################

        if self.Plot is not None:
            self.plotter(self.Plot, binned_flux_set, 5.0, 80.0)

        # Gathering SED model info
        model_info = f"{self.sed_set_name + '.info'} = " \
                     f"\"#COMMENT: SED - {self.Model} Model Atmospheres\\n\""\
                     f"\"#COMMENT: SED parameters:\\n\""\
                     f"\"#COMMENT: LOG Z = {self.metallicity}\\n\""\
                     f"\"#COMMENT: LOG G = {self.gravity}\\n\";"
        if self.Pion is not None:
            self.pion_format(self.Pion, binned_flux_set, model_info)

    ######################################################################################
    # Binning Atlas Spectral Energy Distribution
    ######################################################################################
    def PotsdamWolfRayet(self, metallicity, composition, mdot):
        '''

        :param metallicity:
        :param composition:
        :param mdot:
        :return:
        '''

        self.container['model'] = 'PoWR'
        self.Model = 'PoWR'
        self.container['metallicity'] = metallicity
        self.metallicity = metallicity
        self.container['composition'] = composition
        self.composition = composition
        self.container['mdot'] = mdot
        self.mdot = mdot

        # model set corresponding to a specific metallicity, composition and mdot
        model_set = self.bundle_up_potsdam_models(metallicity, composition, mdot)
        # note: model set in already sort according to accending order of T_eff


        #**********************************************************
        # make sed set name
        self.sed_set_name = 'powr_' + self.grid_name \
                            + f'_mdot{str(self.mdot).replace(".", "").replace("-", "")}'

        # since the data in potsdam models are in log10, converting
        # lambda_bins to log10 basis.
        given_loglambda_bins = np.log10(self.given_lambdabins)

        prefix_comment = self.sed_set_name + ' binning'
        model_lambda_set = []
        model_flux_set = []
        binned_flux_set = []
        total_flux_set = []

        for model_index, model in enumerate(model_set):

            self.progress_bar(model_index, self.Nmodels,
                              prefix=prefix_comment)

            # Reading Model Txt File ###################################################
            with open(model, 'r') as file:
                model_loglambda = []
                model_logflux = []
                # Read the file and get log wavelength and log flux lambda (flam)
                for line in file:
                    # Strip leading and trailing whitespace from the line
                    line = line.strip()
                    # Check if the line is not empty
                    if line:
                        # Split the line into two parts using whitespace as the
                        # separator
                        columns = line.split()
                        # Convert the parts to float and append them to the
                        # respective arrays
                        model_loglambda.append(float(columns[0]))
                        model_logflux.append(float(columns[1]))

                # binning lambda values and flam values according for the given lambda bins
                binned_log_lambda = []
                binned_log_flux = []
                for bin in given_loglambda_bins:
                    sub_binned_loglambda = []
                    sub_binned_logflux = []
                    for loglam, logflam in zip(model_loglambda, model_logflux):
                        if bin[0] <= loglam <= bin[1]:
                            sub_binned_loglambda.append(loglam)
                            sub_binned_logflux.append(logflam)
                    binned_log_lambda.append(sub_binned_loglambda)
                    binned_log_flux.append(sub_binned_logflux)

                # Get the original form of binned wavelength and flux arrays from log scaled
                binned_lambda = [[10 ** lam for lam in row] for row in binned_log_lambda]
                binned_flux = [[10 ** flam for flam in row] for row in binned_log_flux]
                del binned_log_lambda
                del binned_log_flux

                # Get the original form of binned wavelength and flux from the
                # original model data
                model_lambda = [10 ** l for l in model_loglambda]
                model_flux = [10 ** flam for flam in model_logflux]
                del model_loglambda
                del model_logflux

                # Perform integration across the entire wavelength domain to obtain the
                # total flux.
                total_flux_10pc = np.trapz(np.asarray(model_flux), np.asarray(model_lambda))
                # However this is the total flux at 10 pc. The total flux at the stellar
                # surface is
                total_flux_Rstar = (
                    total_flux_10pc
                    * const.POWR_FLUX_DISTANCE_PC**2
                    / (
                        self.R_star[model_index]
                        * const.SOLAR_RADIUS
                        / const.PARSEC
                    ) ** 2.0
                )
                # Append the Total Flux into TotFluxSe
                total_flux_set.append(total_flux_Rstar)

                # remember, the flam in potsdam models are given at 10 pc
                model_lambda_set.append(model_lambda)
                model_flux_set.append(model_flux)


                # calculating flux in each binned flux by integrating within the bin
                # interval
                flux_bin = []
                for i in range(len(binned_lambda)):
                    flux_bin.append(np.trapz(np.asarray(binned_flux[i]),
                                             np.asarray(binned_lambda[i])))

                # reverse the order of the flux bins since we are interested in
                # obtaining flux in energy bins.
                flux_bin.reverse()

                # determining the normalized flux within each bin.
                norm_flux_bin = flux_bin / total_flux_10pc
                # Removing binned flux array
                del flux_bin

                # Appending the result fractional flux for each sed effective temperature
                # to final binned flux set
                binned_flux_set.append(norm_flux_bin)
            # End of Reading Model Txt File #############################################


        self.binned_flux_set = binned_flux_set
        self.container['binned_flux'] = binned_flux_set
        self.container['total_flux'] = total_flux_set
        self.model_lambda_set = model_lambda_set
        self.model_flux_set = model_flux_set
        # End of binning model set ######################################################

        if self.Plot is not None:
            self.plotter(self.Plot, binned_flux_set, 5.0, 80.0)


        # Gathering SED model info
        model_info = f"{self.sed_set_name + '.info'} = " \
                     f"\"#COMMENT: SED - {self.Model} Model Atmospheres\\n\"" \
                     f"\"#COMMENT: SED parameters:\\n\"" \
                     f"\"#COMMENT: Metallicity: {self.metallicity}\\n\"" \
                     f"\"#COMMENT: Composition: {self.composition}\\n\"" \
                     f"\"#COMMENT: Mdot: {self.mdot}\\n\";"
        if self.Pion is not None:
            self.pion_format(self.Pion, binned_flux_set, model_info)


    ######################################################################################
    # Binning Blackbody Spectral Energy Distribution
    ######################################################################################
    def Blackbody(self):
        '''

        :return:
        '''
        self.container['model'] = 'Blackbody'
        self.Model = 'Blackbody'
        prefix_comment = 'blackbody binning'
        self.Nmodels = len(const.BLACKBODY_TEMPERATURE_GRID)

        self.sed_set_name = 'blackbody'
        self.Teff = const.BLACKBODY_TEMPERATURE_GRID

        model_flux_set = []
        model_lambda_set = []
        binned_flux_set = []
        total_flux_set = []

        lambda_min = self.given_lambdabins[0][0]
        lambda_max = self.given_lambdabins[-1][1]
        model_lambda = np.linspace(lambda_min, lambda_max, 20)

        # **********************************************************
        # performing flux binning for black body spectrum for different temperatures
        for model_index, Teff in enumerate(const.BLACKBODY_TEMPERATURE_GRID):

            self.progress_bar(model_index, self.Nmodels,
                              prefix=prefix_comment)

            norm_flux_bin = []
            for Ebin in self.EnergyBins:
                norm_flux_bin.append(
                    integrate.quad(lambda E: self.blackbody_binfrac_func(Teff, E),
                                   Ebin[0]*const.EV_TO_ERG, Ebin[1]*const.EV_TO_ERG)[0])

            binned_flux_set.append(norm_flux_bin)
            total_flux = const.STEFAN_BOLTZMANN_CONSTANT * pow(Teff, 4.0)
            total_flux_set.append(total_flux)

            model_flux = []
            for lam in model_lambda:
                model_flux.append(self.blackbody_flam(Teff, lam))
            model_flux_set.append(model_flux)
            model_lambda_set.append(model_lambda)

        self.binned_flux_set = binned_flux_set
        self.container['binned_flux'] = binned_flux_set
        self.container['total_flux'] = total_flux_set
        self.model_lambda_set = model_lambda_set
        self.model_flux_set = model_flux_set
        # **********************************************************


        if self.Plot is not None:
            self.plotter(self.Plot, binned_flux_set, 5.0, 80.0)


        # Gathering SED model info
        model_info = f"{'blackbody.info'} = " \
                     f"\"#COMMENT: SED - {self.Model} Model Atmospheres\\n\";"
        if self.Pion is not None:
            self.pion_format(self.Pion, binned_flux_set, model_info)


    ######################################################################################
    # plotter
    ######################################################################################
    def plotter(self, plot_path, binned_flux_set, min_plot_energy, max_plot_energy):
        '''

        :param PlotDir:
        :return: None
        '''

        energy_bins = self.EnergyBins

        plot_dir = os.path.join(plot_path, self.sed_set_name)
        os.makedirs(plot_dir, exist_ok=True)
        if not plot_dir.endswith('/'):
            plot_dir += '/'

        # Calculate bin center value and bin width for each energy bin
        bin_widths = [bin_max - bin_min for bin_min, bin_max in energy_bins]
        bin_centers = [(bin_min + bin_max) / 2 for bin_min, bin_max in energy_bins]

        prefix_comment = self.sed_set_name + ' plotting'

        for model_index, Teff in enumerate(self.Teff, start=0):

            self.progress_bar(model_index, self.Nmodels, prefix=prefix_comment)

            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            # converting model lambda to electron volt unit
            model_energy = [const.EV_ANGSTROM / Lambda for Lambda in self.model_lambda_set[model_index]]

            # SubPlot 1: Plot the original model flux data
            axs[0].plot(model_energy, self.model_flux_set[model_index], label="Original Flam",
                        color='black', linestyle='-', linewidth=2)
            axs[0].set_xlabel("Energy, eV")
            if self.Model == 'Atlas':
                axs[0].set_ylabel(r'$\rm log \ F_{\lambda} \  (ergs \, cm^{-2} s^{-1} \AA^{-1})$')
                axs[0].set_title(f'Model: {self.Model} {self.model_name},  T_eff: {Teff} K,'
                                 f'  Log Z: {self.metallicity}, Log g: {self.gravity}', fontsize=12)
                axs[0].set_yscale('log')
            if self.Model == 'PoWR':
                axs[0].set_ylabel(r'$\rm \ F_{\lambda} \  (ergs \, cm^{-2} s^{-1} \AA^{-1})$ at 10 pc')
                axs[0].set_title(f'Grid: {self.grid_name.upper().replace("-", " ")},'
                                 f' Grid Model: {self.which_gird_model[model_index]},'
                                 f' T_eff: {Teff} K,'
                                 f' log Mdot: {self.mdot}', fontsize=12)
            if self.Model == 'CMFGEN':
                axs[0].set_ylabel(r'$\rm \ F_{\lambda} \  (ergs \, cm^{-2} s^{-1} \AA^{-1})$ at 1 kpc')
                axs[0].set_title(f'Grid: {self.grid_name.upper().replace("-", " ")},'
                                 f' Grid Model: {self.which_gird_model[model_index]},'
                                 f' T_eff: {Teff} K,'
                                 f' log Mdot: {self.mdot}', fontsize=12)

            if self.Model == 'Blackbody':
                axs[0].set_title(f'Model: {self.Model}, T_eff: {Teff} K', fontsize=12)

            axs[0].tick_params(axis="both", direction="inout", which="both",
                               bottom=True, top=True, left=True, right=True, length=3)
            axs[0].legend(loc='upper right')
            axs[0].set_xlim(min_plot_energy, max_plot_energy)
            axs[0].grid(True, linestyle='--', alpha=0.5)

            # SubPlot 2: Plot (bar plot) the binned data calculated by PyMicroPion
            axs[1].bar(bin_centers, binned_flux_set[model_index], width=bin_widths,
                       align='center', color='orange',
                       alpha=0.5, label=f"NebulaPy {version.__version__}")
            axs[1].set_xlabel("Energy, eV")
            axs[1].set_ylabel("log Fractional Binned Flux")
            axs[1].set_yscale('log')
            axs[1].tick_params(axis="both", direction="inout", which="both", bottom=True, top=True,
                               left=True, right=True, length=3)
            axs[1].legend(loc='upper right')
            axs[1].set_xlim(min_plot_energy, max_plot_energy)
            axs[1].grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            image_file = plot_dir + self.sed_set_name + '_' + str(int(Teff)) + ".png"
            plt.savefig(image_file, dpi=100)
            plt.close(fig)

    ######################################################################################
    # pion format
    ######################################################################################
    def pion_format(self, pion_format_path, binned_flux_set, model_info):

        logger.info("Saving binned %s SED in PION format", self.Model.lower())

        if not pion_format_path.endswith('/'):
            pion_format_path += '/'
        pion_format_filename = self.sed_set_name + '.txt'
        pion_format_file = os.path.join(pion_format_path, pion_format_filename)

        with open(pion_format_file, 'w') as outfile:
            # saving model info
            outfile.write(model_info)
            # saving energy bins
            outfile.write('\n')
            outfile.write("energy_bins = {\n")
            # saving energy bins
            for row in self.EnergyBins:
                outfile.write("    {")
                outfile.write(', '.join(map(lambda x: f"{x:.6e}", row)))
                outfile.write("},\n")
            outfile.write("};\n")
            # saving temperature and binned flux
            outfile.write(f"{self.sed_set_name}.data = {{\n")
            for index, row in enumerate(binned_flux_set):
                outfile.write("    {")
                # Include temp[index] as the first element of the row
                outfile.write(f"{self.Teff[index]:.6e}, ")
                outfile.write(', '.join(map(lambda x: f"{x:.6e}", row)))
                outfile.write("},\n")
            outfile.write("};\n")

    ######################################################################################
    # bundle up CMFGEN models
    ######################################################################################
    def bundle_up_cmfgen_models(self, metallicity, composition, mdot):

        # generating model grid name
        grid_name = metallicity.lower().replace(".", "") + '-' + composition.lower()
        self.grid_name = grid_name
        # Construct path to the grid directory
        grid_dir = self.CMFGENDatabase + grid_name + '-sed'
        # get model parameter file
        modelparameters_file = os.path.join(grid_dir, 'modelparameters.txt')

        # ******************************************************************
        # exact correct model from 'modelparameters.txt' file, remove model
        # which do not fall in the parameter space
        columns = ["MODEL", "STAR", "T_EFF", "R_TRANS", "LOG L", "LOG MDOT", "V_INF", "R_STAR"]
        modelparams = {column: [] for column in columns}

        with open(modelparameters_file, 'r') as file:
            lines = file.readlines()

            # Removing first few lines
            lines = lines[8:]

            for line in lines:
                parts = line.split()
                modelparams["MODEL"].append(parts[0])
                modelparams["STAR"].append(parts[1])
                modelparams["T_EFF"].append(float(parts[2]))
                modelparams["R_TRANS"].append(float(parts[3]))
                modelparams["LOG L"].append(float(parts[4]))
                modelparams["LOG MDOT"].append(float(parts[5]))
                modelparams["V_INF"].append(float(parts[6]))
                modelparams["R_STAR"].append(float(parts[7]))

        bundle_modelparams = {column: [] for column in columns}
        unique_temperatures = set(modelparams["T_EFF"])

        temp_to_closest_index = {}
        for temp in unique_temperatures:
            indices = [i for i, x in enumerate(modelparams["T_EFF"]) if x == temp]
            closest_index = min(indices, key=lambda i: abs(modelparams["LOG MDOT"][i] + abs(mdot)))
            temp_to_closest_index[temp] = closest_index

        # Sort temperatures
        sorted_temps = sorted(temp_to_closest_index.keys())

        # Populate filtered models according to sorted temperatures
        for temp in sorted_temps:
            closest_index = temp_to_closest_index[temp]
            for column in columns:
                bundle_modelparams[column].append(modelparams[column][closest_index])
        # ******************************************************************

        # two info to make plot title
        self.which_gird_model = bundle_modelparams['MODEL']
        self.Nmodels = len(bundle_modelparams['MODEL'])
        self.Teff = bundle_modelparams['T_EFF']
        self.R_star = bundle_modelparams['R_STAR']

        # append bundled model to the container
        self.container.update(bundle_modelparams)

        Clump = const.POWR_CLUMPING_FACTORS[grid_name]
        self.container['CLUMP'] = Clump

        # ******************************************************************
        # Make file paths for all the models in the final bundle
        bundle_model_files = []
        for model in bundle_modelparams['MODEL']:
            model_filename = grid_name + '_' + model + '_sed.txt'
            model_file = os.path.join(grid_dir, model_filename)
            bundle_model_files.append(model_file)
        return bundle_model_files

    ######################################################################################
    # Binning CMFGEN Spectral Energy Distribution
    ######################################################################################
    def CMFGEN(self, metallicity, composition, mdot):
        '''

        :param metallicity:
        :param composition:
        :param mdot:
        :return:
        '''
        self.container['model'] = 'CMFGEN'
        self.Model = 'CMFGEN'
        self.container['metallicity'] = metallicity
        self.metallicity = metallicity
        self.container['composition'] = composition
        self.composition = composition
        self.container['mdot'] = mdot
        self.mdot = mdot

        # model set corresponding to a specific metallicity, composition and mdot
        model_set = self.bundle_up_cmfgen_models(metallicity, composition, mdot)
        # note: model set in already sort according to accending order of T_eff

        # make sed set name
        self.sed_set_name = 'cmfgen_' + self.grid_name + f'-mdot{str(mdot).replace(".", "").replace("-", "")}'

        # **********************************************************

        # since the data in potsdam models are in log10, converting
        # lambda_bins to log10 basis.
        given_loglambda_bins = np.log10(self.given_lambdabins)

        prefix_comment = self.sed_set_name + ' binning'
        model_lambda_set = []
        model_flux_set = []
        binned_flux_set = []
        total_flux_set = []

        for model_index, model in enumerate(model_set):

            self.progress_bar(model_index, self.Nmodels, prefix=prefix_comment)

            # Reading Model Txt File ###################################################
            with open(model, 'r') as file:
                model_logflux = []
                model_frequency = []
                model_flux = []
                # Read the file and get log wavelength and log flux lambda (flam)
                for line in file:
                    # Strip leading and trailing whitespace from the line
                    line = line.strip()
                    # Check if the line is not empty
                    if line:
                        # Split the line into two parts using whitespace as the
                        # separator
                        columns = line.split()
                        # Convert the parts to float and append them to the
                        # respective arrays
                        model_frequency.append(float(columns[0])) # given in 10^{15} Herz
                        model_flux.append(float(columns[1])) # given in Janskys at 1kpc

                # Convert the frequencies (in 10^15 Hz) to angstroms
                model_lambda = [
                    const.SPEED_OF_LIGHT * const.CM_TO_ANGSTROM
                    / (freq * const.PETAHERTZ_TO_HERTZ)
                    for freq in model_frequency
                ]
                model_loglambda = np.log10(model_lambda)
                # Convert the fluxes (in jansky at 1 kpc) to erg s-1 cm-2 A-1, correct for the distance, and take log10
                model_flux = [
                    flux
                    * const.JANSKY_TO_CGS_FLUX_DENSITY
                    * const.SPEED_OF_LIGHT
                    * const.CM_TO_ANGSTROM
                    / lam**2
                    for flux, lam in zip(model_flux, model_lambda)
                ]
                model_logflux = np.log10(model_flux)
                # Note the model flux is calculated at 1 kpc

                # print to file the original data
                #with open('output.txt', 'w') as file:
                #    # Write x and y values separated by a tab
                #    for x, y in zip(model_lambda, model_flux):
                #        file.write(f"{x}\t{y}\n")

                # binning lambda values and flam values according for the given lambda bins
                binned_log_lambda = []
                binned_log_flux = []
                for bin in given_loglambda_bins:
                    sub_binned_loglambda = []
                    sub_binned_logflux = []
                    for loglam, logflam in zip(model_loglambda, model_logflux):
                        if bin[0] <= loglam <= bin[1]:
                            sub_binned_loglambda.append(loglam)
                            sub_binned_logflux.append(logflam)
                    binned_log_lambda.append(sub_binned_loglambda)
                    binned_log_flux.append(sub_binned_logflux)

                # Get the original form of binned wavelength and flux arrays from log scaled
                binned_lambda = [[10 ** lam for lam in row] for row in binned_log_lambda]
                binned_flux = [[10 ** flam for flam in row] for row in binned_log_flux]
                del binned_log_lambda
                del binned_log_flux

                # Perform integration across the entire wavelength domain to obtain the
                # total flux.
                total_flux_1kpc = np.trapz(np.asarray(model_flux), np.asarray(model_lambda))
                # However this is the total flux at 1 kpc. The total flux at the stellar
                # surface is
                total_flux_Rstar = total_flux_1kpc * const.CMFGEN_FLUX_DISTANCE_PC**2 / (
                        self.R_star[model_index] * const.SOLAR_RADIUS / const.PARSEC) ** 2.0
                # Append the Total Flux into TotFluxSet
                total_flux_set.append(total_flux_Rstar)

                # remember, the flam in potsdam models are given at 10 pc
                model_lambda_set.append(model_lambda)
                model_flux_set.append(model_flux)

                # calculating flux in each binned flux by integrating within the bin
                # interval
                flux_bin = []
                for i in range(len(binned_lambda)):
                    flux_bin.append(np.trapz(np.asarray(binned_flux[i]),
                                             np.asarray(binned_lambda[i])))

                # reverse the order of the flux bins since we are interested in
                # obtaining flux in energy bins.
                flux_bin.reverse()

                # determining the normalized flux within each bin.
                norm_flux_bin = flux_bin / total_flux_1kpc
                # Removing binned flux array
                del flux_bin

                # Appending the result fractional flux for each sed effective temperature
                # to final binned flux set
                binned_flux_set.append(norm_flux_bin)
            # End of Reading Model Txt File #############################################


        self.binned_flux_set = binned_flux_set
        self.container['binned_flux'] = binned_flux_set
        self.container['total_flux'] = total_flux_set
        self.model_lambda_set = model_lambda_set
        self.model_flux_set = model_flux_set
        # End of binning model set ######################################################


        if self.Plot is not None:
            self.plotter(self.Plot, binned_flux_set, 5.0, 100.0)

        # Gathering SED model info
        model_info = f"{self.sed_set_name + '.info'} = " \
                     f"\"#COMMENT: SED - {self.Model} Model Atmospheres\\n\"" \
                     f"\"#COMMENT: SED parameters:\\n\"" \
                     f"\"#COMMENT: Metallicity: {metallicity}\\n\"" \
                     f"\"#COMMENT: Composition: {composition}\\n\"" \
                     f"\"#COMMENT: Mdot: {mdot}\\n\";"

        if self.Pion is not None:
            self.pion_format(self.Pion, binned_flux_set, model_info)


    ################################################################################
