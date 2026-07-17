from urllib.parse import urljoin
import requests
import argparse
import os
import re
import tarfile
import time
import sys
from requests.exceptions import Timeout
import warnings
import importlib.resources as pkg_resources
import NebulaPy.data as compressed_data
import NebulaPy.version as version
from NebulaPy.src.LoggingConfig import configure_logging, get_logger
from NebulaPy.src.NebulaProgress import update_progress
warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class DownloadDatabase:

    def __init__(self, destination=None, verbose=False):

        if destination is None:
            destination = os.getcwd()  # set the current working directory as default location

        # Create a subdirectory inside the destination
        database_dir = os.path.join(destination, 'NebulaPy-DB')
        os.makedirs(database_dir, exist_ok=True)

        sed_database_dir = os.path.join(database_dir, 'SED')  # separate NebulaPy-DB and SED
        try:
            if not os.path.exists(sed_database_dir):
                os.makedirs(sed_database_dir)  # make both NebulaPy-DB and SED directories
                if verbose:
                    logger.info("Created database and SED directories")
            else:
                if verbose:
                    logger.info("Database directory already exists")
        except Exception as e:
            logger.exception("Unable to prepare the SED database directory")


        self.sed_database_dir = sed_database_dir
        self.verbose = verbose
        self.download_atlas_database()
        self.download_postdam_database()
        self.download_cmfgen_database()

        cool_database_dir = os.path.join(database_dir, 'Cooling')  # separate NebulaPy-DB and SED
        try:
            if not os.path.exists(cool_database_dir):
                os.makedirs(cool_database_dir)  # make both NebulaPy-DB and SED directories
                if verbose:
                    logger.info("Created cooling-table directory")
            else:
                if verbose:
                    logger.info("Cooling-table directory already exists")
        except Exception as e:
            logger.exception("Unable to prepare the cooling-table directory")

        self.cool_database_dir = cool_database_dir
        self.download_cooling_table()
        if verbose:
            logger.info("Database download completed")


    #########################################################################################
    # download progress bar
    #########################################################################################
    def download_progress(self, iteration, total_files, prefix='', fill='█', error=None):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total_files - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            fill       - Optional  : bar fill character (Str)
        """
        if error is None:
            update_progress(
                key=prefix,
                description=f"{prefix.strip()} download",
                completed=iteration,
                total=total_files,
                unit="files",
                enabled=self.verbose,
            )
        else:
            logger.error("%s", error)

    #########################################################################################
    # Verify atlas model in atlas database directory
    #########################################################################################
    def verify_atlas_model(self, model_dir, nfiles):
        """
        Verify whether the specified directory contains exactly `nfiles` and
        if all of those files are exactly 67 bytes in size.

        Args:
            model_dir (str): The path to the directory to be checked.
            nfiles (int): The expected number of files in the directory.

        Returns:
            bool: True if the directory contains exactly `nfiles` and all
             files are 69120 bytes in size, False otherwise.
        """
        try:
            files = os.listdir(model_dir)  # List all files in the specified directory
            if len(files) != nfiles:  # Check if the number of files matches the expected number
                return False  # Return False if the number of files does not match

            for file in files:  # Iterate over each file in the directory
                file_path = os.path.join(model_dir, file)  # Construct the full file path
                if os.path.isfile(file_path):  # Check if the current path is a file
                    if os.path.getsize(file_path) != 69120:  # Check if the file size is 69120 bytes
                        return False  # Return False if the file size is not 67 bytes
                else:
                    return False  # Return False if the current path is not a file

            return True  # Return True if all conditions are met
        except Exception as e:  # Catch any exceptions that occur
            logger.exception("Unable to verify Atlas model directory %s", model_dir)
            return False  # Return False if an exception occurs

    #########################################################################################
    # Verify atlas model file in atlas database
    #########################################################################################
    def verify_atlas_model_file_exist(self, model_dir, file, filesize):
        """
        Check if a file exists in the specified directory and if
        its size match

        Parameters:
        model_dir (str): The directory to check.
        file (str): The file to look for.

        Returns:
        bool: True if the file exists and its size is 67 kB, False otherwise.
        """
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            return file_size == filesize
        return False


    #########################################################################################
    # download atlas database
    #########################################################################################
    def download_atlas_database(self):
        atlas_base = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/'
        atlas_models = ['ckm05/', 'ckm10/', 'ckm15/', 'ckm20/', 'ckm25/', 'ckp00/',
                        'ckp02/', 'ckp05/']
        Nfiles = [76, 76, 76, 76, 76, 76, 76, 76]
        atlas_info = ['AA_README', 'catalog.fits']
        atlas_dir = os.path.join(self.sed_database_dir, 'Atlas/')
        os.makedirs(atlas_dir, exist_ok=True)

        # Downloading all atlas models ######################################################
        NModelExists = 0
        for model_index, model in enumerate(atlas_models):
            atlas_model_url = urljoin(atlas_base, model)
            model_name = model.replace('/', '')
            prefix_comment = f' Atlas model {model_name}'
            model_dir = os.path.join(atlas_dir, model)
            os.makedirs(model_dir, exist_ok=True)

            if self.verify_atlas_model(model_dir, Nfiles[model_index]):
                logger.info("Atlas model %s exists in the database", model_name)
                NModelExists += 1
                continue

            else:
                response = requests.get(atlas_model_url)
                href_links = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
                # filter out non-FITS files
                href_links = [link for link in href_links if link.endswith('.fits')]

                # Download files with .fits extensions
                for file_index, href in enumerate(href_links, start=1):
                    # check if the file already exist in database, then skip
                    if self.verify_atlas_model_file_exist(model_dir, href, 69120):
                        continue
                    else:
                        file_url = os.path.join(atlas_model_url, href)
                        err = self.download_file(file_url, model_dir)
                        self.download_progress(file_index, Nfiles[model_index],
                                               prefix=prefix_comment, error=err)
                        if err is not None:
                            logger.error(
                                "NebulaPy %s database download is incomplete; "
                                "restart with 'download-database'",
                                version.__version__,
                            )
                            sys.exit(1)


        if NModelExists == 8:
            logger.info("All Atlas models exist in the database")

        else:
            logger.info("Atlas model download completed successfully")
        # Downloading atlas info files ######################################################

    #########################################################################################
    # method to download single file
    #########################################################################################
    def download_file(self, url, dest_dir, timeout=10):
        try:
            # Create the destination directory if it does not exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Send a GET request to the URL with a specified timeout
            response = requests.get(url, stream=True, timeout=timeout)
            # Raise an error for bad HTTP status codes
            response.raise_for_status()

            # Extract the filename from the URL and create the full path
            filename = os.path.join(dest_dir, url.split('/')[-1])
            with open(filename, 'wb') as file:
                # Write the content to the file in chunks
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            return None

        except Timeout:
            # Handle the timeout exception and return an error message
            return " error: download timed out"
        except requests.RequestException as e:
            # Handle other HTTP exceptions and return an error message
            return f" error: {str(e)}"


    #########################################################################################
    # download postdam database
    #########################################################################################
    def download_postdam_database(self):

        # Get the 'PoWR.har.xz' path from NebulaPy package
        with pkg_resources.path(compressed_data, 'PoWR.tar.xz') as PoWR_xz_path:
            PoWR_tarfile = str(PoWR_xz_path)

        # Delay in seconds
        delay = 0.001  # Change this to the desired delay

        # Open the tar.xz file
        with tarfile.open(PoWR_tarfile, 'r:xz') as tar:
            # Iterate through the members of the tar archive
            for member in tar.getmembers():
                logger.debug("Extracting %s", member.name)
                tar.extract(member, path=self.sed_database_dir)
                time.sleep(delay)

        logger.info("PoWR files extracted to %s", self.sed_database_dir)

    #########################################################################################
    # download CMFGEN database
    #########################################################################################
    def download_cmfgen_database(self):

        # Get the 'PoWR.har.xz' path from NebulaPy package
        with pkg_resources.path(compressed_data, 'CMFGEN.tar.xz') as CMFGEN_xz_path:
            CMFGEN_tarfile = str(CMFGEN_xz_path)

        # Delay in seconds
        delay = 0.01  # Change this to the desired delay

        # Open the tar.xz file
        with tarfile.open(CMFGEN_tarfile, 'r:xz') as tar:
            # Iterate through the members of the tar archive
            for member in tar.getmembers():
                logger.debug("Extracting %s", member.name)
                tar.extract(member, path=self.sed_database_dir)
                time.sleep(delay)

        logger.info("CMFGEN files extracted to %s", self.sed_database_dir)

    #########################################################################################
    # download CMFGEN database
    #########################################################################################
    def download_cooling_table(self):

        # Get the 'PoWR.har.xz' path from NebulaPy package
        with pkg_resources.path(compressed_data, 'Chianti.tar.xz') as CoolTab_xz_path:
            CoolTab_tarfile = str(CoolTab_xz_path)

        # Delay in seconds
        delay = 0.01  # Change this to the desired delay

        # Open the tar.xz file
        with tarfile.open(CoolTab_tarfile, 'r:xz') as tar:
            # Iterate through the members of the tar archive
            for member in tar.getmembers():
                logger.debug("Extracting %s", member.name)
                tar.extract(member, path=self.cool_database_dir)
                time.sleep(delay)

        logger.info("Cooling-table files extracted to %s", self.cool_database_dir)


    @staticmethod
    def download_database(destination=None):
        """This method allows the download process to be called directly from the command line."""
        # Create the class instance to run the download process
        downloader = DownloadDatabase(destination, verbose=True)

    @staticmethod
    def run():
        """Main method to handle command line arguments and run the download process."""
        configure_logging()
        parser = argparse.ArgumentParser(description="Download NebulaPy database")
        parser.add_argument(
            'destination',
            nargs='?',
            default=None,
            help="The destination path where the database will be downloaded"
        )
        args = parser.parse_args()
        # Pass the destination and verbose flag to the download_database method
        DownloadDatabase.download_database(destination=args.destination)




