import math
import numpy as np

G  = 6.6730831e-8
c  = 2.99792458e10
MeV_fm3_to_pa = 1.6021766e35
c_km = 2.99792458e5 # km/s
mN = 1.67e-24 # g
mev_to_ergs = 1.602176565e-6
fm_to_cm = 1.0e-13
ergs_to_mev = 1.0/mev_to_ergs
cm_to_fm = 1.0/fm_to_cm
Msun = 1.988435e33
MeV_fm3_to_pa_cgs = 1.6021766e33
km_to_mSun = G/c**2

hbarc3 = 197.32700288295746**3

nucleon_mass = 938.04

pi = math.pi

# Fundamental conversions
MeV_fm = 197.326960
MeV_cm = MeV_fm * 1e-13
MeV_m = MeV_fm * 1e-15
MeV_km = MeV_cm * 1e-5
MeV_sec = 6.58212e-22

# Mass-energy conversion
MeV_per_gram = 5.60959e26
gram_per_MeV = 1 / MeV_per_gram
MeV_per_kg = 1000 * MeV_per_gram
kg_per_MeV = gram_per_MeV / 1000

# Dyne per MeV^2
dyne_per_MeV2 = gram_per_MeV * MeV_cm / MeV_sec**2
MeV2_per_dyne = 1 / dyne_per_MeV2

# Temperature-energy conversions
K_per_eV = 11605.
K_per_MeV = K_per_eV * 1e6
eV_per_K = 1 / K_per_eV
MeV_per_K = 1 / K_per_MeV

# Charge and energy
e_C = 1.60217646e-19  # Coulombs
J_per_eV = e_C
J_per_MeV = J_per_eV * 1e6
eV_per_J = 1 / J_per_eV
MeV_per_J = 1 / J_per_MeV

# Ergs
erg_per_MeV = 1e7 * J_per_MeV
MeV_per_erg = 1e-7 * MeV_per_J

# Speed of light and time
m_per_sec = 2.99792458e8
cm_per_sec = 100 * m_per_sec
sec_per_year = 31556925.9747

# Particle masses
Mneutron_MeV = 939.565378
Mneutron_gram = Mneutron_MeV * gram_per_MeV
Mproton_MeV = 938.272013
Mproton_gram = Mproton_MeV * gram_per_MeV
Melectron_MeV = 0.51100
Melectron_gram = Melectron_MeV * gram_per_MeV

# Weak interaction
G_Fermi = 1.16637e-11  # MeV^-2
Cabibbo_angle_rad = 13.02 * 2 * np.pi / 360

# Electromagnetism
epsilon_SI = 8.85418782e-12  # F/m
hbar_SI = 1.05457148e-34  # J·s
c_SI = 2.99792458e8  # m/s

# Magnetic conversions
MeV2_per_Tesla = np.sqrt(epsilon_SI * hbar_SI**3 * c_SI**5) / J_per_MeV**2
MeV2_per_Gauss = MeV2_per_Tesla / 1e4
Tesla_per_MeV2 = 1 / MeV2_per_Tesla
Gauss_per_MeV2 = 1 / MeV2_per_Gauss

# Nuclear densities
NuclearDensity_nucleons_per_fm3 = 0.16
NuclearDensity_nucleons_MeV3 = NuclearDensity_nucleons_per_fm3 * MeV_fm**3
NuclearDensity_quarks_MeV3 = 3 * NuclearDensity_nucleons_MeV3

# Energy density of nuclear matter
NuclearEnergyDensity_MeV4 = (Mneutron_MeV - 8) * NuclearDensity_nucleons_MeV3
NuclearEnergyDensity_g_per_cm3 = NuclearEnergyDensity_MeV4 / MeV_per_gram / MeV_cm**3

# Solar and astrophysical constants
MSolar_gram = 1.98847e33
MSolar_MeV = MSolar_gram * MeV_per_gram
BSolar = MSolar_MeV / Mneutron_MeV

# Newton's constant
NewtonG_SI = 6.67428e-11  # m^3 kg^-1 s^-2
NewtonG_MeV = NewtonG_SI / (hbar_SI * c_SI**5) * J_per_MeV**2  # MeV^-2