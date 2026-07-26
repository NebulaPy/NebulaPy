import NebulaPy.src as nebula

EnergyBins = [[7.902470e+00, 1.126030e+01],
              [1.126030e+01, 1.359840e+01],
              [1.359840e+01, 1.453410e+01],
              [1.453410e+01, 1.600000e+01]]

for i in range(60):
  EnergyBins.append([16.0+i,16.0+i+1.0])

plot_dir = '/mnt/local/jm/code/arun/NebulaPy/problems/atlas_n64'
pion_format = '/mnt/local/jm/code/arun/NebulaPy/problems/atlas_n64'

atlas_sed = nebula.sed(
    energy_bins=EnergyBins,
    verbose=True,
    plot=None,
    pion=pion_format
)

atlas_sed.CastelliKuruczAtlas(metallicity=0.0, gravity=4.5)
atlas_sed.CastelliKuruczAtlas(metallicity=0.0, gravity=4.0)
atlas_sed.CastelliKuruczAtlas(metallicity=0.0, gravity=3.5)

bb_sed = nebula.sed(
    energy_bins=EnergyBins,
    verbose=True,
    plot=None,
    pion=pion_format
)

bb_sed.Blackbody()


