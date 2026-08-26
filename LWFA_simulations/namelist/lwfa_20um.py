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

cells_per_wavelength = 40

dx = 2*math.pi/cells_per_wavelength
dy = dx

dt = 0.95/(np.sqrt(1/dx**2 + 1/dy**2))

Lx = 64*um
Ly = 48*um

nx = int(Lx/dx)
ny = int(Ly/dy)


# patches

n_patch_x = 128
n_patch_y = 16

ne_background = 0.003     

cluster_electron_density = 0.85 - ne_background
cluster_radius = 0.032*um   
cluster_spacing = 5.0*um        

# Cluster region: 20 um radius around laser focus
cluster_region_radius = 20.*um

x_focus = 200.*um
y_focus = 0.5*Ly

clusters = []

x_positions = np.arange(
    x_focus - cluster_region_radius,
    x_focus + cluster_region_radius + cluster_spacing,
    cluster_spacing
)

y_positions = np.arange(
    y_focus - cluster_region_radius,
    y_focus + cluster_region_radius + cluster_spacing,
    cluster_spacing
)

for xc in x_positions:
    for yc in y_positions:

        # Keep only clusters inside 20 um radius of focus
        if (xc-x_focus)**2 + (yc-y_focus)**2 <= cluster_region_radius**2:
            clusters.append([xc, yc])

print("Number of clusters:", len(clusters))

# Localised cluster perturbations:

def cluster_perturbation(x, y):

    density = np.zeros_like(x) if hasattr(x, "__len__") else 0.0

    for xc, yc in clusters:

        r2 = (x-xc)**2 + (y-yc)**2
        density += np.where(r2 <= cluster_radius**2, cluster_electron_density, 0.0)

    return density


Niterations         = 50000 # Perhaps change to 30k?


EM_boundary_conditions = [["silver-muller","silver-muller"],["silver-muller","silver-muller"]]

print("Simulation time: ", Niterations*dt/omega0, "s")
print("dx : ", dx)

Main(
    geometry = "2Dcartesian",
    timestep = dt,
    simulation_time = Niterations*dt,
    cell_length = [dx, dy],
    grid_length = [Lx, Ly], 
    number_of_patches = [n_patch_x, n_patch_y],
    EM_boundary_conditions = EM_boundary_conditions,
    print_every = 100,
    reference_angular_frequency_SI = omega0,
)

# density_profile = trapezoidal(
#     max=ne_background,
#     xvacuum=100.*um,
#     xslope1=100.*um,
#     xslope2=0.*um
# )

# carbon_density_profile = trapezoidal(
#     max=carbon_density,
#     xvacuum=100.*um,
#     xslope1=100.*um,
#     xslope2=0.*um
# )

# hydrogen_density_profile = trapezoidal(
#     max=hydrogen_density,
#     xvacuum=100.*um,
#     xslope1=100.*um,
#     xslope2=0.*um
# )

ramp = trapezoidal(
    max=ne_background,
    xvacuum=100.*um,
    xslope1=100.*um,
    xslope2=0.*um
)


def total_density(x, y):

    return ramp(x, y) + cluster_perturbation(x, y)


def carbon_density(x, y):

    return total_density(x, y) / 10.0


def hydrogen_density(x, y):

    return 4.0 * total_density(x, y) / 10.0


Species(
    name = "ions",
    position_initialization = "random",
    momentum_initialization = "cold",
    particles_per_cell = 4,
    mass = 1836.0,
    charge = 1.0,
    number_density = ramp,
    pusher = "boris",
    boundary_conditions = [["remove", "remove"], ["remove", "remove"]],
)

Species(
    name = "electrons",
    position_initialization = "random", 
    momentum_initialization = "cold",
    particles_per_cell = 4,
    mass = 1.0,
    charge = -1.0,
    number_density = ramp,
    pusher = "boris",
    boundary_conditions = [["remove", "remove"], ["remove", "remove"]]
)

Species(
    name="carbon",
    position_initialization="random",
    momentum_initialization="cold",
    particles_per_cell=4,
    atomic_number=6,
    mass=12*1836,
    charge=0.0,
    number_density=carbon_density,
    ionization_model = "tunnel",
    ionization_electrons = "electrons",
    pusher="boris",
    boundary_conditions=[["remove","remove"], ["remove","remove"]]
)

Species(
    name="hydrogen",
    position_initialization="random",
    momentum_initialization="cold",
    particles_per_cell=4,
    atomic_number=1,
    mass=1836,
    charge=0.0,
    number_density=hydrogen_density,
    ionization_model = "tunnel",
    ionization_electrons = "electrons", 
    pusher = "boris",
    boundary_conditions=[["remove","remove"], ["remove","remove"]]
)

laser_fwhm = 45.*fs
laser_waist = 17.*um
a0 = 1.2*math.sqrt(2) # accounts for reduced geometry, otherwise pulse will not self focus

LaserGaussian2D(
    box_side="xmin",
    a0=a0,
    omega=1.0,
    focus=[200.*um, 0.5*Ly], # could change the x to 0.7
    waist=laser_waist,
    time_envelope=tgaussian(center=3*laser_fwhm, fwhm=laser_fwhm) # centre changed to 2 from 3*fwhm to better align with when pulse peak reaches focus
)

vg = math.sqrt(1 - ne_background)
time_start_moving_window = 3*laser_fwhm + 50.*um # should be correct

MovingWindow(
    time_start = time_start_moving_window,
    velocity_x = vg #- 1e-2
) 

print("Lx =",Lx)
# print("moving window start =",time_start_moving_window)
# print("Number density=", ne_background, carbon_density, hydrogen_density)

# Diagnostics

fields_probes = ['Ex', 'Ey', 'Rho', 'Rho_electrons']

DiagProbe(
    every   = 500,
    origin  = [0., Ly/2],
    corners = [[Lx, Ly/2]],
    number  = [nx],
    fields  = fields_probes
)

# Plasma density movie

DiagFields(
    every=500,
    flush_every =2000,
    fields=['Ey', 'Rho'],
)

DiagParticleBinning(
    deposited_quantity = "weight_charge",
    every = 100,
    species = ["electrons"],
    axes = [
        ["x", 0, Lx, 100], # check Lx is right here
        ["ekin", 0, 1000, 300]
    ]
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 100,
    species = ["electrons"],
    axes = [
        ["x", 0, Lx, 100], # check Lx is right here
        ["ekin", 0, 1000, 300]
    ]
)

# DiagTrackParticles(
#     species = "electrons",
#     every = 200,
#     flush_every = 2000,
#     attributes = [
#         "x", "y",
#         "px", "py",
#         "Ex", "Ey", "Bz",
#         "charge", "weight"
#     ]
# )

# These next two particle diagnostics are too expensive
# DiagTrackParticles(
#     species = "carbon",
#     every = 100,
#     attributes = [
#         "x", "y",
#         "px", "py",
#         "Ex", "Ey", "Bz",
#         "charge", "weight"
#     ]
# )

# DiagTrackParticles(
#     species = "hydrogen",
#     every = 100,
#     attributes = [
#         "x", "y",
#         "px", "py",
#         "Ex", "Ey", "Bz",
#         "charge", "weight"
#     ]
# )

DiagScalar(every=100)

DiagPerformances(every=1000)

