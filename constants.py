"""
Physical constants for hydrogen cloud simulation.
All values in SI units unless otherwise noted.
"""

import numpy as np

# Fundamental constants
h = 6.62607015e-34      # Planck constant [J·s]
hbar = h / (2 * np.pi)  # Reduced Planck constant [J·s]
c = 2.99792458e8        # Speed of light [m/s]
k_B = 1.380649e-23      # Boltzmann constant [J/K]
e = 1.602176634e-19     # Elementary charge [C]
m_e = 9.1093837015e-31  # Electron mass [kg]
m_p = 1.67262192369e-27 # Proton mass [kg]
m_H = 1.6735575e-27     # Hydrogen atom mass [kg]
epsilon_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
G = 6.67430e-11         # Gravitational constant [m³/(kg·s²)]
sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
a_rad = 4 * sigma_SB / c   # Radiation constant [J/(m³·K⁴)]

# Derived constants
R_inf = 1.0973731568160e7  # Rydberg constant [1/m]
a_0 = 5.29177210903e-11    # Bohr radius [m]

# Hydrogen specific
E_ion = 13.6 * e           # Ionization energy [J] (13.6 eV)
E_ion_eV = 13.6            # Ionization energy [eV]
nu_ion = E_ion / h         # Ionization threshold frequency [Hz]
lambda_ion = c / nu_ion    # Ionization threshold wavelength [m]

# Lyman-alpha transition (n=2 -> n=1)
E_Lya = 10.2 * e           # Lyman-alpha energy [J]
E_Lya_eV = 10.2            # Lyman-alpha energy [eV]
nu_Lya = E_Lya / h         # Lyman-alpha frequency [Hz]
lambda_Lya = c / nu_Lya    # Lyman-alpha wavelength [m]
A_Lya = 6.2649e8           # Einstein A coefficient for Ly-alpha [1/s]
f_Lya = 0.4162             # Oscillator strength for Ly-alpha

# 21 cm hyperfine transition
E_21cm = 5.87433e-6 * e    # 21 cm energy [J]
nu_21cm = 1.420405751768e9 # 21 cm frequency [Hz]
lambda_21cm = c / nu_21cm  # 21 cm wavelength [m]
A_21cm = 2.8843e-15        # Einstein A coefficient [1/s]
T_star_21cm = E_21cm / k_B # Characteristic temperature [K] (~0.068 K)

# Photoionization
sigma_ion_0 = 6.3e-22      # Photoionization cross-section at threshold [m²]

# Unit conversions
eV_to_J = e                # 1 eV in Joules
J_to_eV = 1 / e            # 1 Joule in eV
ly_to_m = 9.4607304725808e15  # 1 light year in meters
pc_to_m = 3.0856775814914e16  # 1 parsec in meters

# Cloud defaults
DEFAULT_RADIUS_LY = 1.0
DEFAULT_RADIUS_M = DEFAULT_RADIUS_LY * ly_to_m
DEFAULT_DENSITY = 1e6      # atoms per m³

# Temperature references
T_CMB = 2.725              # CMB temperature today [K]
T_room = 300               # Room temperature [K]
T_sun_surface = 5778       # Solar surface temperature [K]


def energy_level_H(n):
    """
    Energy of hydrogen level n relative to ionization (continuum = 0).
    Returns negative values for bound states.
    
    E_n = -13.6 eV / n²
    """
    return -E_ion / (n ** 2)


def energy_level_H_eV(n):
    """Energy of hydrogen level n in eV."""
    return -E_ion_eV / (n ** 2)


def transition_frequency(n_upper, n_lower):
    """
    Frequency of transition from n_upper to n_lower.
    """
    dE = energy_level_H(n_lower) - energy_level_H(n_upper)
    return dE / h


def transition_wavelength(n_upper, n_lower):
    """
    Wavelength of transition from n_upper to n_lower.
    """
    return c / transition_frequency(n_upper, n_lower)


def statistical_weight(n):
    """
    Statistical weight (degeneracy) of hydrogen level n.
    g_n = 2n² (factor of 2 for electron spin)
    """
    return 2 * n ** 2


# Print summary if run directly
if __name__ == "__main__":
    print("=== Physical Constants for H Cloud Simulation ===")
    print()
    print(f"Ionization energy: {E_ion_eV:.2f} eV")
    print(f"Ionization threshold: λ = {lambda_ion*1e9:.2f} nm, ν = {nu_ion:.3e} Hz")
    print(f"Lyman-α: λ = {lambda_Lya*1e9:.2f} nm, E = {E_Lya_eV:.2f} eV")
    print(f"21 cm: λ = {lambda_21cm*100:.2f} cm, ν = {nu_21cm/1e6:.2f} MHz")
    print(f"21 cm characteristic temp: T* = {T_star_21cm*1000:.2f} mK")
    print()
    print(f"Default cloud radius: {DEFAULT_RADIUS_LY} ly = {DEFAULT_RADIUS_M:.3e} m")
    print(f"Default density: {DEFAULT_DENSITY:.0e} atoms/m³")
