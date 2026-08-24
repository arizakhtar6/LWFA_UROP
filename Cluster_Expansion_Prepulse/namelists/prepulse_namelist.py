import math
import os
import numpy as np
import scipy.constants 
from scipy.interpolate import RegularGridInterpolator

#### Physical constants
lambda0             = 0.8e-6
c                   = scipy.constants.c
omega0              = 2*math.pi*c/lambda0
eps0                = scipy.constants.epsilon_0
e                   = scipy.constants.e
me                  = scipy.constants.m_e
ncrit               = eps0*omega0**2*me/e**2
c_over_omega0       = lambda0/2./math.pi
reference_frequency = omega0
E0                  = me*omega0*c/e
electron_mass_MeV   = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]


#### Unit conversions
c_normalized        = 1.
um                  = 1.e-6/c_over_omega0
mm                  = 1.e-3/c_over_omega0
fs                  = 1.e-15*omega0
mm_mrad             = um
pC                  = 1.e-12/e
MeV                 = 1./electron_mass_MeV

# Resolution and iterations

cells_per_wavelength = 1024

dx = 2*math.pi/cells_per_wavelength
dy = dx

dt = 0.98/(np.sqrt(1/dx**2 + 1/dy**2))

Lx = 1*um
Ly = 1*um

nx = int(Lx/dx)
ny = int(Ly/dy)

# patches

n_patch_x = 128
n_patch_y = 16

#dt                  = 0.8*dx/c_normalized

Niterations         = 15000 # changed from 150k

EM_boundary_conditions = [["silver-muller","silver-muller"],["silver-muller","silver-muller"]]


gemini = np.loadtxt(
    "gemini_prepulse_clean.csv",
    delimiter=",",
    skiprows=1
)

I_norm = gemini[:,0]
t_ps = gemini[:,1]


mask = np.isfinite(I_norm) & np.isfinite(t_ps)

I_norm = I_norm[mask]
t_ps = t_ps[mask]


I_norm /= np.max(I_norm)


a0 = 1.2

# start simulation at -1 ps

t_start_ps = -1.0

t_sim_ps = t_ps - t_start_ps

laser_time = t_sim_ps*1e-12*omega0

simulation_time_ps = 3.0

def gemini_envelope(t):

    intensity = np.interp(
        t,
        laser_time,
        I_norm,
        left=0.0,
        right=0.0
    )

    return intensity # removed sqrt

cluster_density = 91.0 # in units of n_crit

cluster_radius = 0.01*um

cluster_spacing = 2*um

clusters = [[0.5*Lx, 0.5*Ly]]

def cluster_profile(x, y): # This gives a solid profile

    density = 0.0

    for xc, yc in clusters:

        r2 = (x-xc)**2 + (y-yc)**2

        if r2 <= cluster_radius**2:
            density += cluster_density

    return density

def carbon_profile(x,y):

    return cluster_profile(x,y)/10.0



def hydrogen_profile(x,y):

    return 4.0*cluster_profile(x,y)/10.0


#print("Simulation time: ", Niterations*dt/omega0, "s")
print("Simulation time: ", simulation_time_ps, "ps")
print("dx : ", dx)
print("Cluster profile: ", cluster_profile(0.5*Lx,0.5*Ly))



EM_boundary_conditions = [["silver-muller", "silver-muller"], ["silver-muller", "silver-muller"]]


Main(
    geometry = "2Dcartesian",
    timestep = dt,
    simulation_time = simulation_time_ps*1e-12*omega0,
    cell_length = [dx, dy],
    grid_length = [Lx, Ly], 
    number_of_patches = [n_patch_x, n_patch_y],
    EM_boundary_conditions = EM_boundary_conditions,
    print_every = 10000,
    reference_angular_frequency_SI = omega0,
)

Species(
    name="cluster_electrons",
    position_initialization="random",
    momentum_initialization="cold",
    particles_per_cell=0,
    mass=1,
    charge=-1.0,
    number_density=0.0,
    pusher="boris",
    boundary_conditions=[["remove","remove"], ["remove","remove"]]
)

# Neutral carbon

Species(
    name="carbon_clusters",
    position_initialization="random",
    momentum_initialization="cold",
    particles_per_cell=1,
    atomic_number=6,
    mass=12*1836,
    charge=0.0,
    number_density=carbon_profile,
    ionization_model="tunnel",
    ionization_electrons="cluster_electrons",
    pusher="boris",
    boundary_conditions=[["remove","remove"], ["remove","remove"]]
)

# Neutral hydrogen

Species(
    name="hydrogen_clusters",
    position_initialization="random",
    momentum_initialization="cold",
    particles_per_cell=1,
    atomic_number=1,
    mass=1836,
    charge=0.0,
    number_density=hydrogen_profile,
    ionization_model="tunnel",
    ionization_electrons="cluster_electrons",
    pusher="boris",
    boundary_conditions=[["remove","remove"], ["remove","remove"]]
)



# Collisional ionisation

Collisions(
    species1=["cluster_electrons"],
    species2=["carbon_clusters"],
    every=100,
    ionizing="cluster_electrons",
    debug_every=1000

)



Collisions(
    species1=["cluster_electrons"],
    species2=["hydrogen_clusters"],
    every=100,
    ionizing="cluster_electrons",
    debug_every=1000

)

laser_fwhm = 45.*fs
laser_waist = 17.*um
a0 = 1.2

LaserGaussian2D(
    box_side="xmin",
    a0=a0,
    omega=1.0,
    focus=[0.5*Lx, 0.5*Ly], # could change the x to 0.7
    waist=laser_waist,
    time_envelope=gemini_envelope
)


# Diagnostics

fields_probes = ['Ex', 'Ey', 'Rho', 'Rho_cluster_electrons', 'Bz']

DiagProbe(
    every   = 1000,
    origin  = [0., Ly/2],
    corners = [[Lx, Ly/2]],
    number  = [nx],
    fields  = fields_probes
)


# Plasma density movie

DiagFields(
    every=5000,
    fields=["Rho", "Ey", "Ex", "Rho_cluster_electrons"]
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 1000,
    flush_every = 10000,
    species = ["cluster_electrons"],
    axes = [
        ["ekin", 0, 1000, 300]
    ]
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 1000,
    flush_every = 10000,
    species = ["cluster_electrons"],
    axes = [
        ["x",0,Lx,64],
        ["y",0,Ly,64]
    ]
)

# Carbon ion density evolution

DiagParticleBinning(
    deposited_quantity="weight",
    every=1000,
    flush_every=10000,
    species=["carbon_clusters"],
    axes=[["x", 0, Lx, 64],
        ["y", 0, Ly, 64]]
)


# Hydrogen ion density evolution

DiagParticleBinning(
    deposited_quantity="weight",
    every=1000,
    flush_every=10000,
    species=["hydrogen_clusters"],
    axes=[["x", 0, Lx, 64],
        ["y", 0, Ly, 64]]
)

DiagTrackParticles(
    species="cluster_electrons",
    every=1000,
    flush_every=2000,
    attributes=["x", "y", "px", "py", "pz", "charge", "weight"]
)

DiagTrackParticles(
    species="carbon_clusters",
    every=1000,
    flush_every=2000,
    attributes=["x", "y", "px", "py", "pz", "charge", "weight"]
)

DiagTrackParticles(
    species="hydrogen_clusters",
    every=1000,
    flush_every=2000,
    attributes=["x", "y", "px", "py", "pz", "charge", "weight"]
)


DiagScalar(every=1000)

DiagPerformances(every=10000)