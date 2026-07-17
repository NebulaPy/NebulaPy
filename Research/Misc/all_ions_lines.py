import NebulaPy.src as nebula
import pandas
import os

dummy_temperature = [2e+6]
dummy_ne = [1e+9]

output_path = '/home/tony/Desktop/multi-ion-bowshock/NebulaPy/lines-transitions'

nebula_chianti = nebula.chianti(pion_ion='H', temperature=dummy_temperature, ne=dummy_ne, verbose=True)

spectroscopic_name = nebula_chianti.chianti_ion.Spectroscopic

lines = nebula_chianti.get_allLines()
# Convert to a pandas Series for vertical display
series = pandas.Series(lines)
# Save the Series to a text file
filename = os.path.join(output_path, spectroscopic_name.replace(" ", "") + '_lines.txt')
print(f" saving {spectroscopic_name} lines to file ")
with open(filename, 'w') as f:
    f.write(series.to_string())
