"""Official source metadata for the NebulaPy database installer."""

DATABASE_VERSION = "1.0.0"
DATABASE_NAME = f"nebulapy-db-{DATABASE_VERSION}"

ZENODO_RECORD_ID = "21677884"
ZENODO_DOI = "10.5281/zenodo.21677884"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
ZENODO_COMPONENTS = (
    {
        "name": "chianti_cooling_rates",
        "label": "Chianti Cooling Rates",
        "filename": "chianti_cooling_rates.zip",
        "data_files": 94,
        "sha256": "47c39d7377463ca59f7cc55117489dae6856b34e170f7b3cf39b79990c603fbd",
        "checksums": "SHA256SUMS-chianti_cooling_rates",
    },
    {
        "name": "cie_ion_fractions",
        "label": "CIE Ion Fraction",
        "filename": "cie_ion_fractions.zip",
        "data_files": 1,
        "sha256": "b2dabb69ebb16cdef323b1b010ae98d9d2d3b6ead79f9d89490e6587616dfc2b",
        "checksums": "SHA256SUMS-cie_ion_fractions",
    },
)

ATLAS_BASE_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/"
)
ATLAS_GRIDS = (
    "ckm05", "ckm10", "ckm15", "ckm20",
    "ckm25", "ckp00", "ckp02", "ckp05",
)

POWR_BASE_URL = "https://www.astro.physik.uni-potsdam.de/~wrh/PoWR"
POWR_GRID_PAGE = f"{POWR_BASE_URL}/powrgrid2.php"
POWR_DOWNLOAD_URL = f"{POWR_BASE_URL}/download.php"

# Official request ID, NebulaPy directory name, and website display label.
POWR_GRIDS = (
    ("OB-I", "mw-ob-i", "OB-I"),
    ("wc", "mw-wc", "MW WC"),
    ("wne", "mw-wne", "MW WNE"),
    ("wnl", "mw-wnl-h20", "MW WNL-H20"),
    ("wnl-h50", "mw-wnl-h50", "MW WNL-H50"),
    ("LMC-OB-I", "lmc-ob-i", "LMC-OB-I"),
    ("lmc-wc", "lmc-wc", "LMC WC"),
    ("lmc-wne", "lmc-wne", "LMC WNE"),
    ("lmc-wnl-h20", "lmc-wnl-h20", "LMC WNL-H20"),
    ("lmc-wnl-h40", "lmc-wnl-h40", "LMC WNL-H40"),
    ("SMC-OB-I", "smc-ob-i", "SMC-OB-I"),
    ("SMC-OB-II", "smc-ob-ii", "SMC-OB-II"),
    ("SMC-OB-III", "smc-ob-iii", "SMC-OB-III"),
    ("SMC-OB-Vd3", "smc-ob-vd3", "SMC-OB-Vd3"),
    ("smc-wc-2021", "smc-wc", "SMC WC"),
    ("smc-wne", "smc-wne", "SMC WNE"),
    ("smc-wnl-h20", "smc-wnl-h20", "SMC WNL-H20"),
    ("smc-wnl-h40", "smc-wnl-h40", "SMC WNL-H40"),
    ("smc-wnl-h60", "smc-wnl-h60", "SMC WNL-H60"),
    ("007-wc-2021", "z007-wc", "Z0.07 WC"),
    ("007-wne-2015", "z007-wne", "Z0.07 WNE"),
    ("007-wnl-h20-2015", "z007-wnl-h20", "Z0.07 WNL-H20"),
    ("007-wnl-h40-2015", "z007-wnl-h40", "Z0.07 WNL-H40"),
    ("007-wnl-h60-2015", "z007-wnl-h60", "Z0.07 WNL-H60"),
)

SOURCE_NOTICE = {
    "core": {
        "provider": "Zenodo",
        "collection": f"NebulaPy Database {DATABASE_VERSION}",
        "url": f"https://doi.org/{ZENODO_DOI}",
    },
    "atlas": {
        "provider": "Space Telescope Science Institute",
        "collection": "Castelli-Kurucz ATLAS9 ck04models",
        "url": ATLAS_BASE_URL,
    },
    "powr": {
        "provider": "University of Potsdam",
        "collection": "PoWR model grids",
        "url": f"{POWR_BASE_URL}/powrgrid1.php",
    },
}
