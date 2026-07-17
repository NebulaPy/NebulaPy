import NebulaPy.src as nebula
import pandas
import os

dummy_temperature = [2e+6]
dummy_ne = [1e+9]

output_path = '/home/tony/Desktop/multi-ion-bowshock/NebulaPy/lines-transitions'

nebula_chianti = nebula.chianti(pion_ion='H', temperature=dummy_temperature, ne=dummy_ne, verbose=True)
spectroscopic_name = nebula_chianti.chianti_ion.Spectroscopic
allLineData = nebula_chianti.get_allLinesTransitions()

# Save the Series to a text file
filename = os.path.join(spectroscopic_name.replace(" ", "") + '_lines_transitions.txt')
outfile = os.path.join(output_path, filename)

print(f" Saving {spectroscopic_name} line transition info to file: {filename}")

Reference = "#" + " ".join(allLineData['Reference'])
# Write to file
with open(outfile, 'w') as f:
    # Write header
    f.write(Reference)
    f.write('\n\n')
    f.write(f"{'wvl':<{12}} {'Avalue':<{12}} {'From':<{25}} {'To':<{25}}\n")

    # Write each row
    for w, a, fr, to in zip(allLineData['wvl'], allLineData['Avalue'], allLineData['From'], allLineData['To']):
        f.write(f"{w:<{12}.4f} {a:<{12}.3e} {fr:<{25}} {to:<{25}}\n")
