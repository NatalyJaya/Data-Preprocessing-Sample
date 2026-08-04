
"""
Colloid fixed in the centre of the simulation box with surface groups and respective counterions. 
The surface groups have charge regulation. Repulsive LJ with offset for the interactions involving the colloid. Includes visualization. 
Script with some defined parameters and a short run to test stuff
"""

import numpy as np
import csv

import espressomd
import sys
sys.path.insert(0,'/cluster/home/ritad/sugar_library/')
from sugar import sugar_library
import espressomd.interactions
import espressomd.electrostatics
import espressomd.reaction_methods

import time
from icosphere import icosphere

sg=sugar_library()
required_features = ["P3M", "LENNARD_JONES"]
espressomd.assert_features(required_features)

# Reduced units of energy and length
################################

units= sg.units

TEMPERATURE = 298 * units.K
WATER_PERMITTIVITY = 78.5 # at 298 K
KT = TEMPERATURE * units.k_B
PARTICLE_SIZE = 0.355 * units.nm
BJERRUM_LENGTH = (units.e**2 / (4 * units.pi * units.eps0 * WATER_PERMITTIVITY * KT)).to('nm')

units.define(f'reduced_energy = {TEMPERATURE} * boltzmann_constant')
units.define(f'reduced_length = {PARTICLE_SIZE}')
units.define(f'reduced_charge = 1*e')

#store the values of some parameters in dimensionless reduced units
KT_REDUCED = KT.to('reduced_energy').magnitude
BJERRUM_LENGTH_REDUCED = BJERRUM_LENGTH.to('reduced_length').magnitude
PARTICLE_SIZE_REDUCED = PARTICLE_SIZE.to('reduced_length').magnitude


# Systems parameters
#####################

N_ICOSAHEDRON = 4  #generates 162 vertices
N_ACID = 12+10*(N_ICOSAHEDRON**2-1)
C_ACID = 1e-3 * units.molar
SURFACE_CHARGE_DENS = 2.0 *units.nm**-2     # density of acidic groups
C_SALT = 1e-2 * units.molar # concentration of salt 
pKa = 7.6                   # acidic constant
pH = 7.5

# Simulation parameters
N_MD_STEPS = 1000
USE_ELECTROSTATICS = True
USE_P3M = True
VISUALIZE = False

#Number of samples to be collected
N_BLOCKS = 16 # number of blocks to be used in data analysis
DESIRED_BLOCK_SIZE = 150 # desired number of samples per block
NUM_SAMPLES = int(N_BLOCKS * DESIRED_BLOCK_SIZE) # number of reaction samples per pH value

#Calculate dependent parameters
R_COLL = np.sqrt(N_ACID / (4 * units.pi * SURFACE_CHARGE_DENS))  # Radius of colloid based on N_ACID and SURFACE_CHARGE_DENS
R_COLL_REDUCED = R_COLL.to('reduced_length').magnitude 
BOX_V = (N_ACID / (units.avogadro_constant * C_ACID)).to("nm**3") # Volume of the box based on N_ACID and C_ACID
BOX_L = np.cbrt(BOX_V)
BOX_L_REDUCED = BOX_L.to('reduced_length').magnitude
N_SALT = int((C_SALT * BOX_V * units.avogadro_constant))
print(f"Radius of the colloid (red units) = {R_COLL_REDUCED}")
print(f"BOX_L_REDUCED = {BOX_L_REDUCED}")
print(f"N_SALT = {N_SALT}")

# Outputs - file name including the most relevant parameters to be read by analysis script
obs_file=open(f'sgD-{SURFACE_CHARGE_DENS.magnitude}_Rnp-{round(R_COLL_REDUCED, 1)}_pH-{pH}_obs.csv', mode='w')
writer = csv.writer(obs_file)                                     

# Generating the configuration of small ions
def generate_trialvectors(mag):
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos(costheta)
    x = np.sin(theta) *np.cos(phi)
    y = np.sin(theta) *np.sin(phi)
    z = np.cos(theta)
    vec=np.array([x,y,z])
    vec=vec*mag
    return vec

# Initiallize ESPResSo
##################
system = espressomd.System(box_l=[BOX_L_REDUCED] * 3)

system.time_step = 1.0e-2
print("Tune skin")
system.cell_system.tune_skin(min_skin=0.1, max_skin =4.0, tol=0.1, int_steps=1000)
print(system.cell_system.get_state())
np.random.seed(seed= 42)            # initialize the random generator in numpy


species = ["Colloid", "Surf_A", "Surf_B", "Surf_HA", "Ction+"]
types = {"Colloid": 0, "Surf_A":1, "Surf_B": 2, "Surf_HA": 3, "Na": 4, "Cl": 5}
charges = {"Colloid": 0, "Surf_A": -1, "Surf_B": +1., "Surf_HA": 0, "Na": +1,  "Cl": -1}
radii = {"Colloid": R_COLL_REDUCED, "Surf_A": PARTICLE_SIZE_REDUCED/2, "Surf_B": PARTICLE_SIZE_REDUCED/2, "Surf_HA": PARTICLE_SIZE_REDUCED/2, 
        "Na": PARTICLE_SIZE_REDUCED/2,  "Cl": PARTICLE_SIZE_REDUCED/2}

# Lennard-Jones interactions
#########################
lj_eps = 5.0
lj_sig = PARTICLE_SIZE_REDUCED
for type_1 in types.values():
    if type_1 == types["Surf_A"] or type_1 == types["Surf_HA"]:       # ignores the surface groups (embedded in the colloid)
        continue
    for type_2 in types.values():
        if type_2 == types["Surf_A"] or type_2 == types["Surf_HA"]:   # ignores the surface groups (embedded in the colloid)
            continue 
        if type_1 >= type_2:
            if type_1==types["Colloid"] or type_2==types["Colloid"]:   # includes an offset in pairs involving the colloid
                system.non_bonded_inter[type_1, type_2].lennard_jones.set_params(epsilon=lj_eps, sigma=lj_sig, 
                    cutoff=2**(1./6.)*lj_sig, offset = R_COLL_REDUCED-PARTICLE_SIZE_REDUCED/2, shift="auto")
            else:
                system.non_bonded_inter[type_1, type_2].lennard_jones.set_params(epsilon = lj_eps, sigma=lj_sig, 
                    cutoff=2**(1./6.)*lj_sig, shift="auto")

#Generating the initial configuration 
############

#Generates a colloid in the center of the box 
system.part.add(pos=[BOX_L_REDUCED / 2.0] * 3, fix=[True, True, True],
                    q=charges["Colloid"], type=types["Colloid"])

#Adds surface groups using icosphere package, multiplying by colloid radius-0.5 (below the surface) and translating to the cell center (cubic box). 
vertices, faces = icosphere(nu=N_ICOSAHEDRON)
radius = [item*(R_COLL_REDUCED-0.5) for item in vertices]
coordinates = [item+(BOX_L_REDUCED/2) for item in radius]

for surfgroup in coordinates: 
    system.part.add(pos=surfgroup, fix=[True, True, True], 
            q=charges["Surf_A"], type=types["Surf_A"])

#Add the corresponding number of H ions
for i in range(N_ACID):
    system.part.add(pos=generate_trialvectors(mag=(R_COLL_REDUCED+0.5))+[BOX_L_REDUCED*0.5] * 3, 
            q=charges["Surf_B"], type=types["Surf_B"])

# Add salt ion pairs
for i in range(N_SALT):
    system.part.add(pos=generate_trialvectors(mag=(R_COLL_REDUCED+1.0))+[BOX_L_REDUCED*0.5] * 3, 
        type=types["Na"], q=charges["Na"])
for i in range(N_SALT):
    system.part.add(pos=generate_trialvectors(mag=(R_COLL_REDUCED+1.5))+[BOX_L_REDUCED*0.5] * 3, 
        type=types["Cl"], q=charges["Cl"])

print(f"The system contains {len(system.part)} particles")

#print(system.part.all())

if VISUALIZE:
    import espressomd.visualization
    visualizer = espressomd.visualization.openGLLive(system, bond_type_radius=[0.3], background_color=[1, 1, 1]) 

energy = system.analysis.energy()
print(f"Before Minimization: E_total = {energy['total']:.2e}")

# warm-up integration
###########################
system.integrator.set_steepest_descent(f_max=0.0, gamma=0.1, max_displacement=0.1)
system.integrator.run(500)
system.integrator.set_vv()  # to switch back to velocity Verlet

energy = system.analysis.energy()
print(f"After Minimization: E_total = {energy['total']:.2e}")

# activate the thermostat           
###########################         
system.thermostat.set_langevin(kT=KT_REDUCED, gamma=1.0, seed=24)
system.integrator.run(steps=1000)

# activate the electrostatics
###########################
if USE_ELECTROSTATICS:
    COULOMB_PREFACTOR=BJERRUM_LENGTH_REDUCED * KT_REDUCED
    if USE_P3M:
        coulomb = espressomd.electrostatics.P3M(prefactor=COULOMB_PREFACTOR, accuracy=1e-3)
    else:
        coloumb = espressomd.electrostatics.DH(prefactor=COULOMB_PREFACTOR, 
                kappa = KAPPA_REDUCED, r_cut = 1./KAPPA_REDUCED)
    system.actors.add(coulomb)
else:
    # this speeds up the simulation of dilute systems with small particle numbers
    system.cell_system.set_n_square()


# Initialize the reaction method
#########################
exclusion_range = PARTICLE_SIZE_REDUCED
RE = espressomd.reaction_methods.ConstantpHEnsemble(kT=KT_REDUCED, 
        exclusion_range=exclusion_range, seed = 77, constant_pH=pH, 
        exclusion_radius_per_type = {types["Colloid"]: radii["Colloid"], types["Surf_B"]: radii["Surf_B"], 
            types["Na"]: radii["Na"], types["Cl"]: radii["Cl"]}) 
RE.set_non_interacting_type(type=len(types))  # assumes that particle type starts with 0
RE.add_reaction(gamma=10**(-pKa),
                reactant_types=[types["Surf_HA"]],
                product_types=[types["Surf_A"], types["Surf_B"]],
                default_charges={types["Surf_HA"]: charges["Surf_HA"],
                     types["Surf_A"]: charges["Surf_A"],
                     types["Surf_B"]: charges["Surf_B"]}
                )

# Run
############
start_time = time.time()
N_step_time = 2

writer.writerow(['time', 'number_A', 'number_HA', 'alpha', 'surf_ch_dens'])  # prints the header

RE.reaction(reaction_steps=N_ACID) # Needed??

for sample in range(NUM_SAMPLES):
    
    system.integrator.run(N_MD_STEPS)
    
    RE.reaction(reaction_steps=N_ACID) # we should do at least one reaction attempt per reactive particle
    
    num_A = system.number_of_particles(type=types["Surf_A"])
    num_HA = system.number_of_particles(type=types["Surf_HA"])
    alpha = num_A/(num_A+num_HA)
    surf_charge_dens = num_A * charges["Surf_A"] *sg.e / (4 * np.pi * (R_COLL**2)) 

    obs = [system.time,num_A,num_HA,alpha,surf_charge_dens.to('mC / m**2').magnitude]
    writer.writerow(obs)

    if sample == int(N_step_time):
        N_step_time*=1.5
        sg.write_progress(step=sample, total_steps=NUM_SAMPLES)
obs_file.close()
print(f"Run completed")

if VISUALIZE:
    visualizer.run(1)
