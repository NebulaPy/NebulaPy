import NebulaPy.src as nebula

EnergyBins = [[7.902470e+00, 1.126030e+01],
              [1.126030e+01, 1.359840e+01],
              [1.359840e+01, 1.453410e+01],
              [1.453410e+01, 1.619920e+01],
              [1.619920e+01, 2.156450e+01],
              [2.156450e+01, 2.438310e+01],
              [2.438310e+01, 2.960130e+01],
              [2.960130e+01, 3.065100e+01],
              [3.065100e+01, 3.512110e+01],
              [3.512110e+01, 4.096300e+01],
              [4.096300e+01, 4.514180e+01],
              [4.514180e+01, 4.788780e+01],
              [4.788780e+01, 5.441780e+01],
              [5.441780e+01, 6.342330e+01],
              [6.342330e+01, 6.450000e+01],
              [6.450000e+01, 7.700000e+01]]

plot_dir = '/mnt/local/jm/code/arun/NebulaPy/problems/atlas_n16'
pion_format = '/mnt/local/jm/code/arun/NebulaPy/problems/atlas_n16'

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


