
import numpy as np
import os
import matplotlib.pyplot as plt  # Plotting
from NebulaPy.src.CIE import cieMode
import ChiantiPy.core as ch

# Macbook
OutputDir = '/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/XraySpectrum'  # Output image directory
# Razer Blade
# OutputDir = '/home/tony/Desktop/CWBs-2026/Postprocessing/X-raySpectrum'


# CIE TEST ####################################################################
print(" Comparing NebulaPy and CHIANTI CIE ion fractions")
# Initialize nebulapy CIE module
CIE = cieMode(verbose=True)
CIE.loadCIEFile()



# Temperature grid
Temperature = np.logspace(np.log10(1e6), np.log10(1e8), 200)



# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

ax.set_title(f"{element} Ionisation Fractions (CIE)")

# Define colors for ions
colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    'tab:olive', 'tab:cyan',
    'royalblue', 'darkorange', 'forestgreen', 'crimson',
    'mediumorchid', 'saddlebrown', 'hotpink', 'dimgray',
    'yellowgreen', 'deepskyblue',
    'navy', 'gold', 'limegreen', 'firebrick',
    'indigo', 'teal'
]

# Loop over Fe ions
for idx, i in enumerate(range(24, 26)):
    chianti_name = f'{element}_{i}'

    # Select color
    color = colors[idx]

    # NebulaPy CIE fraction
    nebula_cie = CIE.getCIEFraction(
        chianti_name,
        Temperature
    )

    # Plot NebulaPy CIE
    ax.plot(
        np.log10(Temperature),
        np.log10(nebula_cie),
        linewidth=2,
        linestyle='-',
        color=color,
        label=f'{chianti_name} (NebulaPy)'
    )

    # -------------------------------------------------------------------------
    '''
    ax.plot(
        np.log10(Temperature),
        np.log10(chianti_ieq[i - 1]),
        linewidth=2,
        linestyle='--',
        color=color,
        label=f'{chianti_name} (CHIANTI)'
    )
    '''

# Axis formatting
# ax.set_ylim(-6, 0)
ax.set_xlabel(r'$\log_{10}(T\,[\mathrm{K}])$')
ax.set_ylabel(r'$\log_{10}(\mathrm{Ion\ Fraction})$')
# Grid and legend
ax.grid(alpha=0.3)

ax.legend(
    fontsize=8,
    ncol=5,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.05)
)

# Save figure
outfile = os.path.join(OutputDir, f"nebula_vs_chianti_cie_{element}.png")

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches='tight'
)

print(f" Saved figure: {outfile}")

plt.close(fig)