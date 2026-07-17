import NebulaPy.src as nebula
import pandas
import os

dummy_temperature = [3e+4]
dummy_ne = [1e+3]

output_path = '/Users/tony/Desktop/Multi-Ion-Bowshock/multi-ion-bowshock/NebulaPy/lines-transitions'

nebula_pyneb = nebula.pyneb(pion_ion='H', temperature=dummy_temperature, ne=dummy_ne, verbose=True)

spectroscopic_name = nebula_pyneb.Spectroscopic

lines = nebula_pyneb.get_allLines()

series = pandas.Series(lines)

# Convert each number to fixed-point string with 6 decimals
series_str = [f"{x:.6f}" for x in series]

filename = os.path.join(output_path, spectroscopic_name.replace(" ", "") + '_recomb_lines.txt')
print(f" saving {spectroscopic_name} recombination lines to file")

# Ensure output folder exists
os.makedirs(output_path, exist_ok=True)

# Write to file line by line
with open(filename, 'w') as f:
    f.write("\n".join(series_str))