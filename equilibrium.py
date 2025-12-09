"""
Statistical equilibrium calculations.
Saha ionization equation, Boltzmann level populations.
"""

import numpy as np
from scipy.optimize import brentq
from constants import h, k_B, m_e, E_ion, E_ion_eV
from atomic_physics import (energy_level, statistical_weight, 
                            statistical_weight_ion, statistical_weight_electron)


# ============================================
# Partition Functions
# ============================================

def partition_function_H(T, n_max=10):
    """
    Partition function for neutral hydrogen.
    
    Z = Σ g_n × exp(-E_n/kT)
    
    At low T, Z ≈ g_1 = 2 (only ground state populated).
    At high T, sum diverges — need cutoff.
    
    Parameters:
        T: temperature [K]
        n_max: maximum level to include (cutoff)
    
    Returns:
        Partition function [dimensionless]
    """
    if T <= 0:
        return statistical_weight(1)  # Ground state only
    
    Z = 0.0
    E_1 = energy_level(1)  # Ground state energy (negative)
    
    for n in range(1, n_max + 1):
        g_n = statistical_weight(n)
        E_n = energy_level(n)
        dE = E_n - E_1  # Excitation energy (positive)
        
        x = dE / (k_B * T)
        if x < 100:  # Avoid underflow
            Z += g_n * np.exp(-x)
    
    return Z


def partition_function_Hplus():
    """Partition function for H⁺ (just a proton)."""
    return 1.0


def partition_function_electron(T):
    """
    Translational partition function for electrons.
    This is the thermal de Broglie factor.
    
    Z_e = (2πm_e kT / h²)^(3/2) × V
    
    For the Saha equation, we use the number density form.
    """
    return (2 * np.pi * m_e * k_B * T / h**2)**(3/2)


# ============================================
# Boltzmann Distribution
# ============================================

def boltzmann_ratio(n_upper, n_lower, T):
    """
    Ratio of populations n_upper/n_lower in thermal equilibrium.
    
    n_j/n_i = (g_j/g_i) × exp(-(E_j - E_i)/kT)
    
    For hydrogen, E_n = -13.6/n² eV, so E_upper > E_lower (less negative).
    The excitation energy ΔE = E_upper - E_lower > 0.
    
    Parameters:
        n_upper, n_lower: level numbers
        T: temperature [K]
    
    Returns:
        Population ratio [dimensionless]
    """
    if T <= 0:
        return 0.0 if n_upper > n_lower else np.inf
    
    g_upper = statistical_weight(n_upper)
    g_lower = statistical_weight(n_lower)
    
    # Excitation energy (positive for upper > lower)
    dE = energy_level(n_upper) - energy_level(n_lower)  # E_upper - E_lower > 0
    
    # Boltzmann factor: exp(-ΔE/kT)
    x = dE / (k_B * T)  # Positive
    
    if x > 100:
        return 0.0  # Exponentially suppressed
    
    return (g_upper / g_lower) * np.exp(-x)


def level_population_fraction(n, T, n_max=10):
    """
    Fraction of atoms in level n.
    
    f_n = g_n × exp(-E_n/kT) / Z
    
    Parameters:
        n: level number
        T: temperature [K]
        n_max: max level for partition function
    
    Returns:
        Population fraction [dimensionless]
    """
    if T <= 0:
        return 1.0 if n == 1 else 0.0
    
    Z = partition_function_H(T, n_max)
    g_n = statistical_weight(n)
    
    E_1 = energy_level(1)
    E_n = energy_level(n)
    dE = E_n - E_1
    
    x = dE / (k_B * T)
    if x > 100:
        return 0.0
    
    return g_n * np.exp(-x) / Z


def excited_fraction(T, n_max=10):
    """
    Total fraction of atoms in excited states (n > 1).
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Excited fraction [dimensionless]
    """
    f_ground = level_population_fraction(1, T, n_max)
    return 1.0 - f_ground


# ============================================
# Saha Equation
# ============================================

def saha_ratio(T):
    """
    Saha equation: ratio (n_H+ × n_e) / n_H at equilibrium.
    
    (n_H+ × n_e) / n_H = (g_H+ × g_e / g_H) × (2πm_e kT/h²)^(3/2) × exp(-χ/kT)
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Saha ratio [m⁻³]
    """
    if T <= 0:
        return 0.0
    
    # Statistical weights
    g_H = partition_function_H(T)
    g_Hplus = partition_function_Hplus()
    g_e = statistical_weight_electron()
    
    # Thermal de Broglie factor
    lambda_th = (2 * np.pi * m_e * k_B * T / h**2)**(3/2)
    
    # Ionization potential
    x = E_ion / (k_B * T)
    if x > 100:
        return 0.0
    
    return (g_Hplus * g_e / g_H) * lambda_th * np.exp(-x)


def ionization_fraction(n_total, T):
    """
    Solve for ionization fraction x = n_H+ / (n_H + n_H+).
    
    From Saha equation with charge neutrality (n_e = n_H+):
    
    x² / (1-x) = S(T) / n_total
    
    where S(T) is the Saha ratio.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T: temperature [K]
    
    Returns:
        Ionization fraction x [0, 1]
    """
    if T <= 0 or n_total <= 0:
        return 0.0
    
    S = saha_ratio(T)
    
    if S <= 0:
        return 0.0
    
    # Dimensionless Saha parameter
    y = S / n_total
    
    if y > 1e10:
        return 1.0  # Fully ionized
    if y < 1e-20:
        return 0.0  # Fully neutral
    
    # Solve quadratic: x² + y×x - y = 0
    # x = (-y + √(y² + 4y)) / 2
    discriminant = y**2 + 4*y
    x = (-y + np.sqrt(discriminant)) / 2
    
    return np.clip(x, 0.0, 1.0)


def ionization_fraction_array(n_total, T_array):
    """
    Ionization fraction for array of temperatures.
    
    Parameters:
        n_total: total density [m⁻³]
        T_array: array of temperatures [K]
    
    Returns:
        Array of ionization fractions
    """
    return np.array([ionization_fraction(n_total, T) for T in T_array])


def electron_density(n_total, T):
    """
    Electron density from ionization equilibrium.
    
    n_e = x × n_total
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T: temperature [K]
    
    Returns:
        Electron density [m⁻³]
    """
    x = ionization_fraction(n_total, T)
    return x * n_total


def neutral_density(n_total, T):
    """
    Neutral hydrogen density.
    
    n_H = (1-x) × n_total
    """
    x = ionization_fraction(n_total, T)
    return (1 - x) * n_total


def ion_density(n_total, T):
    """
    Ion (H⁺) density.
    
    n_H+ = x × n_total
    """
    return electron_density(n_total, T)  # Same by charge neutrality


# ============================================
# Characteristic Temperatures
# ============================================

def ionization_temperature(n_total, x_target=0.5):
    """
    Find temperature where ionization fraction equals target.
    
    Parameters:
        n_total: total density [m⁻³]
        x_target: target ionization fraction
    
    Returns:
        Temperature [K]
    """
    def f(T):
        return ionization_fraction(n_total, T) - x_target
    
    # Search between reasonable bounds
    T_low = 1000  # Below this, essentially no ionization
    T_high = 100000  # Above this, fully ionized
    
    try:
        T = brentq(f, T_low, T_high)
        return T
    except ValueError:
        return np.nan


def excitation_temperature_characteristic():
    """
    Characteristic temperature for first excited state.
    
    T_exc = (E_2 - E_1) / k_B = 10.2 eV / k_B ≈ 118,000 K
    """
    from atomic_physics import excitation_energy
    dE = excitation_energy(1, 2)
    return dE / k_B


def ionization_temperature_characteristic():
    """
    Characteristic temperature for ionization.
    
    T_ion = χ / k_B = 13.6 eV / k_B ≈ 158,000 K
    """
    return E_ion / k_B


# ============================================
# Summary Functions
# ============================================

def equilibrium_summary(n_total, T):
    """
    Compute full equilibrium state at given conditions.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T: temperature [K]
    
    Returns:
        Dictionary with equilibrium properties
    """
    x = ionization_fraction(n_total, T)
    n_e = x * n_total
    n_H = (1 - x) * n_total
    n_Hplus = x * n_total
    
    # Level populations in neutral hydrogen
    f_1 = level_population_fraction(1, T)
    f_2 = level_population_fraction(2, T)
    f_3 = level_population_fraction(3, T)
    
    return {
        'T': T,
        'n_total': n_total,
        'ionization_fraction': x,
        'n_e': n_e,
        'n_H': n_H,
        'n_Hplus': n_Hplus,
        'f_ground': f_1,
        'f_n2': f_2,
        'f_n3': f_3,
        'partition_function': partition_function_H(T),
        'saha_ratio': saha_ratio(T)
    }


# Test if run directly
if __name__ == "__main__":
    print("=== Statistical Equilibrium Test ===")
    print()
    
    # Our cloud parameters
    n_total = 1e6  # m⁻³
    
    print(f"Cloud density: n = {n_total:.0e} m⁻³")
    print()
    
    print("Characteristic temperatures:")
    print(f"  Excitation (n=1→2): T* = {excitation_temperature_characteristic():.0f} K")
    print(f"  Ionization: T* = {ionization_temperature_characteristic():.0f} K")
    
    T_half = ionization_temperature(n_total, 0.5)
    print(f"  50% ionization at n={n_total:.0e}: T = {T_half:.0f} K")
    print()
    
    print("Ionization fraction vs temperature:")
    temperatures = [100, 1000, 5000, 10000, 15000, 20000, 30000]
    for T in temperatures:
        x = ionization_fraction(n_total, T)
        f2 = level_population_fraction(2, T)
        print(f"  T = {T:>6} K: x_ion = {x:.6f}, f(n=2) = {f2:.2e}")
    
    print()
    print("Boltzmann ratio n₂/n₁:")
    for T in [1000, 10000, 50000, 100000]:
        ratio = boltzmann_ratio(2, 1, T)
        print(f"  T = {T:>6} K: n₂/n₁ = {ratio:.2e}")
