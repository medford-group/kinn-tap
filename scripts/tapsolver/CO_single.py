import json

from tapsolver import *

#Define reactor
new_reactor = reactor()

new_reactor.zone_lengths = {0: 2.80718, 1: 0.17364, 2: 2.80718}  # cm
new_reactor.zone_voids = {0: 0.4, 1: 0.4, 2: 0.4}  # -
new_reactor.reactor_radius = 1  # cm

new_reactor_species = reactor_species()
new_reactor_species.inert_diffusion = 16  # cm2/s
new_reactor_species.catalyst_diffusion = 16  # cm2/s
new_reactor_species.reference_temperature = 385.5  # K
new_reactor_species.reference_mass = 60  # 24.01 # amu
new_reactor_species.temperature = 385.65  # K
new_reactor_species.advection = 0

#Define gas phase species
CO = define_gas()
CO.mass = 28.01
CO.intensity = 1
CO.delay = 0
CO.noise = 0.0
new_reactor_species.add_gas('CO', CO)

O2 = define_gas()
O2.mass = 32
O2.intensity = 1
O2.delay = 0.0
O2.noise = 0.0
new_reactor_species.add_gas('O2', O2)

CO2 = define_gas()
CO2.mass = 44.01
CO2.intensity = 0
CO2.delay = 0.0
CO2.noise = 0.0
new_reactor_species.add_gas('CO2', CO2)

Ar = define_gas()
Ar.mass = 40
Ar.intensity = 1
Ar.delay = 0.0
Ar.noise = 0.0
new_reactor_species.add_inert_gas('Ar', Ar)

#Define adspecies
s = define_adspecies()
s.concentration = 0
new_reactor_species.add_adspecies('CO*', s)

s = define_adspecies()
s.concentration = 0
new_reactor_species.add_adspecies('O*', s)

s = define_adspecies()
s.concentration = 30
new_reactor_species.add_adspecies('*', s)

#Define kinetic mechanism
new_mechanism = mechanism()

new_mechanism.elementary_processes[0] = elementary_process('CO + * <-> CO*')
new_mechanism.elementary_processes[1] = elementary_process('O2 + 2* -> 2O*')
new_mechanism.elementary_processes[2] = elementary_process('CO* + O* <-> CO2 + 2*')
new_mechanism.elementary_processes[3] = elementary_process('CO + O* -> CO2 + *')

new_mechanism.elementary_processes[0].forward.k = 15
new_mechanism.elementary_processes[0].backward.k = 0.7
new_mechanism.elementary_processes[1].forward.k = 0.33
new_mechanism.elementary_processes[1].backward.k = 0
new_mechanism.elementary_processes[2].forward.k = 0.4
new_mechanism.elementary_processes[2].backward.k = 0.02
new_mechanism.elementary_processes[3].forward.k = 15.2
new_mechanism.elementary_processes[3].backward.k = 0


#Make a function
for j in new_mechanism.elementary_processes:
    new_mechanism.elementary_processes[j].forward.use = 'k'
    try:
        new_mechanism.elementary_processes[j].backward.use = 'k'
    except:
        pass

mechanism_constructor(new_mechanism)

CO_tapobject = TAPobject()
CO_tapobject.store_flux_data = True
CO_tapobject.store_thin_data = True
CO_tapobject.store_cat_flux = True
CO_tapobject.display_analytical = True
CO_tapobject.show_graph = True
CO_tapobject.mechanism = new_mechanism
CO_tapobject.reactor_species = new_reactor_species
CO_tapobject.reactor = new_reactor

# CO_tapobject.show_graph = False
CO_tapobject.output_name = 'CO_TAP_long_inert'

CO_tapobject.gasses_objective = ['CO', 'O2', 'CO2']  # ['Ar']
CO_tapobject.optimize = False

forward_problem(2.5, 1, CO_tapobject)
flux_graph(CO_tapobject)
