"""
Thermal equilibrium calculations.
Heating and cooling rates for the hydrogen cloud.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from constants import h, c, k_B, m_e, m_H, e, E_ion, nu_ion, sigma_ion_0
from blackbody import planck_spectral_radiance, mean_intensity
from atomic_physics import (photoionization_cross_section, 
                            einstein_A, transition_frequency,
                            line_cross_section_peak,
                            excitation_energy)
from equilibrium import (ionization_fraction, electron_density, 
                         neutral_density, ion_density,
                         level_population_fraction)


# ============================================
# Photoionization Heating
# ============================================

def photoionization_heating_rate(n_H, T_rad, T_gas):
    """
    Heating rate from photoionization.
    
    Each ionization deposits (hν - hν₀) as kinetic energy.
    
    H = n_H × ∫_{ν₀}^∞ (4πJ_ν/hν) × σ_bf(ν) × h(ν - ν₀) dν
    
    Parameters:
        n_H: neutral hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K] (for cross-section, usually minor effect)
    
    Returns:
        Heating rate [W/m³]
    """
    if n_H <= 0 or T_rad <= 0:
        return 0.0
    
    # Check if any ionizing photons exist
    x_threshold = h * nu_ion / (k_B * T_rad)
    if x_threshold > 50:
        return 0.0  # No ionizing photons
    
    # Numerical integration
    def integrand(nu):
        J_nu = mean_intensity(nu, T_rad)
        sigma = photoionization_cross_section(nu)
        excess_energy = h * (nu - nu_ion)
        return (4 * np.pi * J_nu / (h * nu)) * sigma * excess_energy
    
    # Integrate from threshold to reasonable upper limit
    nu_max = nu_ion * 10  # 10× threshold
    
    try:
        result, _ = quad(integrand, nu_ion, nu_max, limit=100)
        return n_H * max(result, 0.0)
    except:
        return 0.0


def photoionization_rate(T_rad):
    """
    Photoionization rate per neutral atom.
    
    Γ = ∫_{ν₀}^∞ (4πJ_ν/hν) × σ_bf(ν) dν
    
    Parameters:
        T_rad: radiation temperature [K]
    
    Returns:
        Rate [s⁻¹]
    """
    if T_rad <= 0:
        return 0.0
    
    x_threshold = h * nu_ion / (k_B * T_rad)
    if x_threshold > 50:
        return 0.0
    
    def integrand(nu):
        J_nu = mean_intensity(nu, T_rad)
        sigma = photoionization_cross_section(nu)
        return (4 * np.pi * J_nu / (h * nu)) * sigma
    
    nu_max = nu_ion * 10
    
    try:
        result, _ = quad(integrand, nu_ion, nu_max, limit=100)
        return max(result, 0.0)
    except:
        return 0.0


# ============================================
# Recombination Cooling
# ============================================

def recombination_coefficient_A(T):
    """
    Case A recombination coefficient (includes ground state).
    
    α_A(T) ≈ 4.18e-19 × (T/10⁴)^(-0.75) m³/s
    
    Parameters:
        T: electron temperature [K]
    
    Returns:
        Recombination coefficient [m³/s]
    """
    if T <= 0:
        return 0.0
    return 4.18e-19 * (T / 1e4)**(-0.75)


def recombination_coefficient_B(T):
    """
    Case B recombination coefficient (excludes ground state).
    
    Used when Lyman photons are trapped (optically thick to Lyman series).
    
    α_B(T) ≈ 2.56e-19 × (T/10⁴)^(-0.83) m³/s
    
    Parameters:
        T: electron temperature [K]
    
    Returns:
        Recombination coefficient [m³/s]
    """
    if T <= 0:
        return 0.0
    return 2.56e-19 * (T / 1e4)**(-0.83)


def recombination_cooling_rate(n_e, n_Hplus, T, case='B'):
    """
    Cooling rate from radiative recombination.
    
    L_rec = n_e × n_H+ × α(T) × k_B × T × β(T)
    
    where β(T) ≈ 0.75 accounts for the distribution of recombination energies.
    
    Parameters:
        n_e: electron density [m⁻³]
        n_Hplus: ion density [m⁻³]
        T: electron temperature [K]
        case: 'A' or 'B' for recombination case
    
    Returns:
        Cooling rate [W/m³]
    """
    if case == 'A':
        alpha = recombination_coefficient_A(T)
    else:
        alpha = recombination_coefficient_B(T)
    
    # Approximate cooling per recombination
    beta = 0.75  # Approximate
    
    return n_e * n_Hplus * alpha * k_B * T * beta


# ============================================
# Free-Free (Bremsstrahlung) Cooling
# ============================================

def bremsstrahlung_cooling_rate(n_e, n_Hplus, T):
    """
    Free-free (bremsstrahlung) cooling rate.
    
    L_ff = 1.42e-40 × g_ff × √T × n_e × n_H+ [W/m³]
    
    Parameters:
        n_e: electron density [m⁻³]
        n_Hplus: ion density [m⁻³]
        T: electron temperature [K]
    
    Returns:
        Cooling rate [W/m³]
    """
    if T <= 0:
        return 0.0
    
    # Gaunt factor (frequency-averaged)
    g_ff = 1.1 + 0.34 * np.exp(-(5.5 - np.log10(T))**2 / 3)
    g_ff = np.clip(g_ff, 1.0, 1.5)
    
    return 1.42e-40 * g_ff * np.sqrt(T) * n_e * n_Hplus


# ============================================
# Collisional Excitation Cooling
# ============================================

def collisional_excitation_cooling_rate(n_H, n_e, T):
    """
    Cooling from collisional excitation followed by radiative decay.
    
    Dominated by Lyman-α at moderate temperatures.
    
    L_exc = n_H × n_e × q_12(T) × E_12
    
    Parameters:
        n_H: neutral hydrogen density [m⁻³]
        n_e: electron density [m⁻³]
        T: temperature [K]
    
    Returns:
        Cooling rate [W/m³]
    """
    if T <= 0 or n_H <= 0 or n_e <= 0:
        return 0.0
    
    # Excitation energy for Ly-α (1→2)
    E_12 = excitation_energy(1, 2)  # ~10.2 eV
    
    # Collisional excitation rate coefficient
    x = E_12 / (k_B * T)
    if x > 50:
        return 0.0
    
    Omega_12 = 0.5  # Approximate collision strength
    g_1 = 2  # Statistical weight of ground state
    
    q_12 = 8.63e-12 * Omega_12 * np.exp(-x) / (g_1 * np.sqrt(T))
    
    return n_H * n_e * q_12 * E_12


# ============================================
# Compton Heating/Cooling
# ============================================

def compton_heating_rate(n_e, T_rad, T_gas):
    """
    Compton heating/cooling rate.
    
    Energy exchange between electrons and radiation field.
    Heats gas if T_rad > T_gas, cools if T_gas > T_rad.
    
    H_C = n_e × (4σ_T/m_e c) × a × T_rad⁴ × (T_rad - T_gas) × k_B
    
    Parameters:
        n_e: electron density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K]
    
    Returns:
        Heating rate [W/m³] (positive = heating, negative = cooling)
    """
    from constants import sigma_SB, a_rad
    
    # Thomson cross-section
    sigma_T = 6.652e-29  # m²
    
    # Compton y-parameter prefactor
    prefactor = (4 * sigma_T * k_B) / (m_e * c)
    
    # Radiation energy density
    u_rad = a_rad * T_rad**4
    
    # Net heating rate
    return n_e * prefactor * u_rad * (T_rad - T_gas)


# ============================================
# Total Heating and Cooling
# ============================================

def total_heating_rate(n_total, T_rad, T_gas):
    """
    Total heating rate.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K]
    
    Returns:
        Heating rate [W/m³]
    """
    # Get densities from ionization equilibrium at gas temperature
    n_H = neutral_density(n_total, T_gas)
    n_e = electron_density(n_total, T_gas)
    
    H = 0.0
    
    # Photoionization heating
    H += photoionization_heating_rate(n_H, T_rad, T_gas)
    
    # Compton heating (if T_rad > T_gas)
    H_compton = compton_heating_rate(n_e, T_rad, T_gas)
    if H_compton > 0:
        H += H_compton
    
    return H


def total_cooling_rate(n_total, T_rad, T_gas):
    """
    Total cooling rate.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K]
    
    Returns:
        Cooling rate [W/m³]
    """
    n_H = neutral_density(n_total, T_gas)
    n_e = electron_density(n_total, T_gas)
    n_Hplus = ion_density(n_total, T_gas)
    
    L = 0.0
    
    # Recombination cooling
    L += recombination_cooling_rate(n_e, n_Hplus, T_gas)
    
    # Bremsstrahlung cooling
    L += bremsstrahlung_cooling_rate(n_e, n_Hplus, T_gas)
    
    # Collisional excitation cooling (Ly-α)
    L += collisional_excitation_cooling_rate(n_H, n_e, T_gas)
    
    # Compton cooling (if T_gas > T_rad)
    H_compton = compton_heating_rate(n_e, T_rad, T_gas)
    if H_compton < 0:
        L += -H_compton
    
    return L


def net_heating_rate(n_total, T_rad, T_gas):
    """
    Net heating rate (heating - cooling).
    
    Positive = gas heats up
    Negative = gas cools down
    Zero = thermal equilibrium
    """
    H = total_heating_rate(n_total, T_rad, T_gas)
    L = total_cooling_rate(n_total, T_rad, T_gas)
    return H - L


# ============================================
# Equilibrium Temperature
# ============================================

def equilibrium_temperature(n_total, T_rad, T_min=1, T_max=1e6):
    """
    Find gas temperature where heating = cooling.
    
    Parameters:
        n_total: total density [m⁻³]
        T_rad: radiation temperature [K]
        T_min, T_max: search bounds [K]
    
    Returns:
        Equilibrium temperature [K], or T_rad if no solution
    """
    def f(T_gas):
        return net_heating_rate(n_total, T_rad, T_gas)
    
    # Check bounds
    f_min = f(T_min)
    f_max = f(T_max)
    
    # If same sign at both ends, return radiation temperature
    if f_min * f_max > 0:
        return T_rad
    
    try:
        T_eq = brentq(f, T_min, T_max, xtol=1.0)
        return T_eq
    except:
        return T_rad


def equilibrium_temperature_array(n_total, T_rad_array):
    """
    Equilibrium temperature for array of radiation temperatures.
    """
    return np.array([equilibrium_temperature(n_total, T) for T in T_rad_array])


# ============================================
# Cooling Time
# ============================================

def cooling_time(n_total, T_gas, T_rad):
    """
    Characteristic cooling time.
    
    t_cool = (3/2) × n × k_B × T / L
    
    Parameters:
        n_total: total density [m⁻³]
        T_gas: gas temperature [K]
        T_rad: radiation temperature [K]
    
    Returns:
        Cooling time [s]
    """
    L = total_cooling_rate(n_total, T_rad, T_gas)
    
    if L <= 0:
        return np.inf
    
    # Thermal energy density
    u_thermal = 1.5 * n_total * k_B * T_gas
    
    return u_thermal / L


def heating_time(n_total, T_gas, T_rad):
    """
    Characteristic heating time.
    
    t_heat = (3/2) × n × k_B × T / H
    """
    H = total_heating_rate(n_total, T_rad, T_gas)
    
    if H <= 0:
        return np.inf
    
    u_thermal = 1.5 * n_total * k_B * T_gas
    
    return u_thermal / H


# Test if run directly
if __name__ == "__main__":
    print("=== Thermal Equilibrium Test ===")
    print()
    
    n_total = 1e6  # m⁻³
    
    print(f"Cloud density: n = {n_total:.0e} m⁻³")
    print()
    
    print("Photoionization rate vs radiation temperature:")
    for T_rad in [3000, 10000, 30000, 50000]:
        gamma = photoionization_rate(T_rad)
        print(f"  T_rad = {T_rad:>6} K: Γ = {gamma:.2e} s⁻¹")
    
    print()
    print("Recombination coefficient vs gas temperature:")
    for T in [5000, 10000, 20000]:
        alpha = recombination_coefficient_B(T)
        print(f"  T = {T:>6} K: α_B = {alpha:.2e} m³/s")
    
    print()
    print("Equilibrium temperature for different radiation fields:")
    for T_rad in [100, 1000, 5000, 10000, 20000, 50000]:
        T_eq = equilibrium_temperature(n_total, T_rad)
        x_ion = ionization_fraction(n_total, T_eq)
        print(f"  T_rad = {T_rad:>6} K → T_eq = {T_eq:>8.1f} K, x_ion = {x_ion:.4f}")
