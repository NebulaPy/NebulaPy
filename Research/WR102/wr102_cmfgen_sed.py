import NebulaPy.src as nebula

EnergyBins = [
    [7.902470e+00, 1.126030e+01], [1.126030e+01, 1.359840e+01],
    [1.359840e+01, 1.453410e+01], [1.453410e+01, 1.619920e+01],
    [1.619920e+01, 2.156450e+01], [2.156450e+01, 2.438310e+01],
    [2.438310e+01, 2.960130e+01], [2.960130e+01, 3.065100e+01],
    [3.065100e+01, 3.512110e+01], [3.512110e+01, 4.096300e+01],
    [4.096300e+01, 4.514180e+01], [4.514180e+01, 4.788780e+01],
    [4.788780e+01, 5.441780e+01], [5.441780e+01, 6.342330e+01],
    [6.342330e+01, 7.854778e+01], [7.854778e+01, 1.000000e+02]
]

plot_dir = '/home/tony/Desktop/NebulaPy/problems'
pion_format = '/home/tony/Desktop/NebulaPy/problems'

cmfgen_sed = nebula.sed(
    database='/home/tony/Desktop/NebulaPy/NebulaPy-DB',
    energy_bins=EnergyBins,
    verbose=True,
    plot=plot_dir,
    pion=pion_format
)

cmfgen_sed.CMFGEN('Z0.86', 'WO', -5.0)
print(cmfgen_sed.container)
