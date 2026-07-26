"""
Single Ion 2D Cooling Function Map
Description: Generate cooling function map from
2D PION simulation silo files
Author: Arun Mathew
Date: 17 Nov 2024
"""


import NebulaPy.src as nebula
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")



# Assume nebula and cooling class are already imported

ion_list = [
    'H', 'H1+',
    'He', 'He1+', 'He2+',
    'C', 'C1+', 'C2+', 'C3+', 'C4+', 'C5+', 'C6+',
    'N', 'N1+', 'N2+', 'N3+', 'N4+', 'N5+', 'N6+', 'N7+',
    'O', 'O1+', 'O2+', 'O3+', 'O4+', 'O5+', 'O6+', 'O7+', 'O8+',
    'Ne1+', 'Ne2+', 'Ne3+', 'Ne4+', 'Ne5+', 'Ne6+', 'Ne7+', 'Ne8+', 'Ne9+', 'Ne10+',
    'Si1+', 'Si2+', 'Si3+', 'Si4+', 'Si5+', 'Si6+', 'Si7+', 'Si8+', 'Si9+', 'Si10+',
    'Si11+', 'Si12+', 'Si13+', 'Si14+',
    'S', 'S1+', 'S2+', 'S3+', 'S4+', 'S5+', 'S6+', 'S7+', 'S8+', 'S9+', 'S10+', 'S11+',
    'S12+', 'S13+', 'S14+', 'S15+', 'S16+',
    'Fe4+', 'Fe5+', 'Fe6+', 'Fe7+', 'Fe8+', 'Fe9+', 'Fe10+', 'Fe11+', 'Fe12+', 'Fe13+',
    'Fe14+', 'Fe15+', 'Fe16+', 'Fe17+', 'Fe18+', 'Fe19+', 'Fe20+', 'Fe21+', 'Fe22+',
    'Fe23+', 'Fe24+', 'Fe25+', 'Fe26+'
]

temperature = [8000]  # K
ne = [100]            # cm^-3

cooling_rates = {}

# Loop through all ions and compute cooling rates
for ion in ion_list:
    cooling = nebula.cooling(pion_ion=ion, verbose=False)
    ion_cooling_rate = cooling.generate_cooling_rate_map(temperature=temperature, ne=ne)
    # store as scalar assuming single T, ne
    cooling_rates[ion] = ion_cooling_rate[0]

# Sort by cooling rate descending
sorted_coolants = sorted(cooling_rates.items(), key=lambda x: x[1], reverse=True)

# Top 10 major coolants
top_10_coolants = sorted_coolants[:10]

print("Top 10 major coolants at T={} K, ne={} cm^-3:".format(temperature[0], ne[0]))
for ion, rate in top_10_coolants:
    print(f"{ion}: {rate:.3e} erg cm^3 s^-1")