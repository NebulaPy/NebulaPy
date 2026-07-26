import NebulaPy.src as nebula


'''
comparison of collisional and recombination line emissivities of Halpha ane Hbeta
'''

dummy_temperature = [1e+4, 3e+4]
dummy_ne = [1e+3, 1e+3]

output_path = '/home/tony/Desktop/multi-ion-bowshock/NebulaPy/recombination-line'

nebula_pyneb_ion = nebula.pyneb(pion_ion='H', temperature=dummy_temperature, ne=dummy_ne, verbose=True)
nebula_chianti_ion = nebula.chianti(pion_ion='H', temperature=dummy_temperature, ne=dummy_ne, verbose=True)

pyneb_lines = [6562.816, 4861.332]
print(f" recombination line: {nebula_pyneb_ion.get_recomb_line_emissivity_for_list(line_list=pyneb_lines)}")

chianti_lines = [6564.523, 4862.637]
print(f" collisional line: {nebula_chianti_ion.get_line_emissivity_for_list(line_list=chianti_lines)}")
