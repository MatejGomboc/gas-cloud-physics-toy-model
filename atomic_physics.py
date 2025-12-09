"""
Atomic physics of hydrogen.
Energy levels, transition rates, cross-sections, selection rules.
"""

import numpy as np
from constants import (h, c, k_B, m_e, m_H, e, epsilon_0, 
                       E_ion, E_ion_eV, nu_ion, 
                       A_Lya, f_Lya, A_21cm, T_star_21cm,
                       sigma_ion_0, R_inf, a_0)


# ============================================
# Energy Levels
# ============================================

def energy_level(n):
    """
    Energy of hydrogen level n (ground state = 1).
    Reference: continuum (ionized) = 0, bound states negative.
    
    E_n = -13.6 eV / n²
    
    Parameters:
        n: principal quantum number (integer ≥ 1)
    
    Returns:
        Energy [J]
    """
    return -E_ion / n**2


def energy_level_eV(n):
    """Energy of level n in eV."""
    return -E_ion_eV / n**2


def excitation_energy(n_lower, n_upper):
    """
    Energy required to excite from n_lower to n_upper.
    
    Parameters:
        n_lower: lower level
        n_upper: upper level
    
    Returns:
        Excitation energy [J]
    """
    return energy_level(n_upper) - energy_level(n_lower)


def excitation_energy_eV(n_lower, n_upper):
    """Excitation energy in eV."""
    return energy_level_eV(n_upper) - energy_level_eV(n_lower)


def transition_frequency(n_upper, n_lower):
    """
    Photon frequency for transition n_upper -> n_lower.
    
    ν = R_∞ × c × (1/n_lower² - 1/n_upper²)
    """
    return R_inf * c * (1/n_lower**2 - 1/n_upper**2)


def transition_wavelength(n_upper, n_lower):
    """Transition wavelength [m]."""
    return c / transition_frequency(n_upper, n_lower)


def transition_wavelength_nm(n_upper, n_lower):
    """Transition wavelength [nm]."""
    return transition_wavelength(n_upper, n_lower) * 1e9


# ============================================
# Statistical Weights
# ============================================

def statistical_weight(n):
    """
    Statistical weight (degeneracy) of level n.
    g_n = 2n² (includes spin)
    """
    return 2 * n**2


def statistical_weight_ion():
    """Statistical weight of H⁺ ion (just a proton)."""
    return 1


def statistical_weight_electron():
    """Statistical weight of free electron (spin degeneracy)."""
    return 2


# ============================================
# Einstein Coefficients
# ============================================

def einstein_A(n_upper, n_lower):
    """
    Einstein A coefficient (spontaneous emission rate).
    
    Uses semi-classical approximation for hydrogen.
    Accurate for low n, approximate for high n.
    
    Parameters:
        n_upper: upper level
        n_lower: lower level
    
    Returns:
        A coefficient [s⁻¹]
    """
    # Special case: Lyman-alpha (most accurate)
    if n_upper == 2 and n_lower == 1:
        return A_Lya
    
    # General formula (approximate)
    nu = transition_frequency(n_upper, n_lower)
    g_u = statistical_weight(n_upper)
    g_l = statistical_weight(n_lower)
    
    # Approximate oscillator strength (Kramers formula)
    f_lu = oscillator_strength(n_lower, n_upper)
    
    # Convert to A coefficient
    prefactor = 8 * np.pi**2 * e**2 / (m_e * c**3 * 4 * np.pi * epsilon_0)
    A = prefactor * nu**2 * (g_l / g_u) * f_lu
    
    return A


def oscillator_strength(n_lower, n_upper):
    """
    Absorption oscillator strength f_{lu}.
    
    Uses Kramers approximation for hydrogen.
    """
    if n_upper == 2 and n_lower == 1:
        return f_Lya
    
    # Kramers approximation
    nl, nu = n_lower, n_upper
    
    # Gaunt factor (order unity correction)
    g_bf = 1.0  # Simplified
    
    # Kramers formula
    f = (32 / (3 * np.sqrt(3) * np.pi)) * g_bf
    f *= 1 / (nl**3 * nu**3)
    f *= nl**2 * nu**2 / (nu**2 - nl**2)**3
    
    return max(f, 1e-10)  # Prevent negative/zero


def einstein_B_absorption(n_lower, n_upper):
    """
    Einstein B coefficient for absorption.
    B_lu = (c² / 2hν³) × (g_u / g_l) × A_ul
    """
    nu = transition_frequency(n_upper, n_lower)
    g_u = statistical_weight(n_upper)
    g_l = statistical_weight(n_lower)
    A = einstein_A(n_upper, n_lower)
    
    return (c**2 / (2 * h * nu**3)) * (g_u / g_l) * A


def einstein_B_emission(n_upper, n_lower):
    """
    Einstein B coefficient for stimulated emission.
    B_ul = (c² / 2hν³) × A_ul
    """
    nu = transition_frequency(n_upper, n_lower)
    A = einstein_A(n_upper, n_lower)
    
    return (c**2 / (2 * h * nu**3)) * A


# ============================================
# Absorption & Emission Cross-sections
# ============================================

def doppler_width(nu_0, T, mass=m_H):
    """
    Doppler width of a spectral line.
    
    Δν_D = (ν₀/c) × √(2kT/m)
    
    Parameters:
        nu_0: line center frequency [Hz]
        T: temperature [K]
        mass: particle mass [kg], default hydrogen
    
    Returns:
        Doppler width [Hz]
    """
    return (nu_0 / c) * np.sqrt(2 * k_B * T / mass)


def natural_width(A_ul):
    """
    Natural line width from spontaneous emission.
    
    Δν_nat = A_ul / (2π)
    
    Parameters:
        A_ul: Einstein A coefficient [s⁻¹]
    
    Returns:
        Natural width [Hz]
    """
    return A_ul / (2 * np.pi)


def voigt_parameter(nu_0, T, A_ul, mass=m_H):
    """
    Voigt parameter a = Δν_nat / Δν_D.
    
    a << 1: Doppler-dominated
    a >> 1: Lorentzian-dominated
    """
    return natural_width(A_ul) / doppler_width(nu_0, T, mass)


def line_cross_section_peak(n_lower, n_upper, T):
    """
    Peak absorption cross-section at line center.
    
    σ_0 = (πe²/m_e c) × f_lu × (1/Δν_D) × (1/√π)
    
    For Doppler-broadened line.
    
    Parameters:
        n_lower, n_upper: transition levels
        T: temperature [K]
    
    Returns:
        Peak cross-section [m²]
    """
    f = oscillator_strength(n_lower, n_upper)
    nu_0 = transition_frequency(n_upper, n_lower)
    delta_nu = doppler_width(nu_0, T)
    
    # Classical electron radius
    r_e = e**2 / (4 * np.pi * epsilon_0 * m_e * c**2)
    
    sigma_0 = np.sqrt(np.pi) * r_e * c * f / delta_nu
    
    return sigma_0


def line_profile_gaussian(nu, nu_0, T, mass=m_H):
    """
    Normalized Gaussian (Doppler) line profile φ(ν).
    
    ∫ φ(ν) dν = 1
    
    Parameters:
        nu: frequency [Hz]
        nu_0: line center [Hz]
        T: temperature [K]
        mass: particle mass [kg]
    
    Returns:
        Profile value [Hz⁻¹]
    """
    delta_nu = doppler_width(nu_0, T, mass)
    x = (nu - nu_0) / delta_nu
    
    return np.exp(-x**2) / (delta_nu * np.sqrt(np.pi))


def line_cross_section(nu, n_lower, n_upper, T):
    """
    Frequency-dependent absorption cross-section for a line.
    
    σ(ν) = σ_0 × φ(ν) × Δν_D × √π
    
    Parameters:
        nu: frequency [Hz]
        n_lower, n_upper: transition levels
        T: temperature [K]
    
    Returns:
        Cross-section [m²]
    """
    sigma_0 = line_cross_section_peak(n_lower, n_upper, T)
    nu_0 = transition_frequency(n_upper, n_lower)
    phi = line_profile_gaussian(nu, nu_0, T)
    delta_nu = doppler_width(nu_0, T)
    
    return sigma_0 * phi * delta_nu * np.sqrt(np.pi)


# ============================================
# Photoionization
# ============================================

def photoionization_cross_section(nu, n=1):
    """
    Photoionization cross-section from level n.
    
    σ_bf(ν) = σ₀ × (ν_n/ν)³ × g_bf
    
    where ν_n is the ionization threshold from level n.
    
    Parameters:
        nu: photon frequency [Hz]
        n: level being ionized (default: ground state)
    
    Returns:
        Cross-section [m²], 0 if below threshold
    """
    # Threshold frequency for level n
    nu_n = nu_ion / n**2
    
    if np.isscalar(nu):
        if nu < nu_n:
            return 0.0
    else:
        nu = np.asarray(nu)
        result = np.zeros_like(nu)
        mask = nu >= nu_n
        if not np.any(mask):
            return result
        nu = nu[mask] if len(nu[mask]) > 0 else nu
    
    # Kramers formula with Gaunt factor
    g_bf = gaunt_factor_bf(nu, n)
    
    sigma = sigma_ion_0 * n**5 * (nu_n / nu)**3 * g_bf
    
    if not np.isscalar(nu):
        result[mask] = sigma
        return result
    return sigma


def gaunt_factor_bf(nu, n):
    """
    Bound-free Gaunt factor (quantum correction to Kramers).
    
    Approximate formula, accurate to ~10%.
    """
    nu_n = nu_ion / n**2
    
    if np.isscalar(nu):
        if nu < nu_n:
            return 1.0
        x = nu / nu_n
    else:
        x = np.maximum(nu / nu_n, 1.0)
    
    # Approximate Gaunt factor
    g = 1.0 - 0.3456 * (x**(-1/3) - 1) / (x**(1/3) + 1)
    
    return np.clip(g, 0.5, 1.5)


# ============================================
# 21 cm Hyperfine Transition
# ============================================

def hyperfine_statistical_weights():
    """
    Statistical weights for 21 cm hyperfine states.
    
    Returns:
        (g_triplet, g_singlet) = (3, 1)
    """
    return 3, 1


def spin_temperature_ratio(T_s):
    """
    Population ratio n_1/n_0 (triplet/singlet) for spin temperature T_s.
    
    n_1/n_0 = 3 × exp(-T*/T_s) ≈ 3(1 - T*/T_s) for T_s >> T*
    
    T* = 0.068 K
    """
    if T_s <= 0:
        return 0.0
    
    return 3 * np.exp(-T_star_21cm / T_s)


def optical_depth_21cm(N_H, T_s, delta_v=1e3):
    """
    21 cm optical depth.
    
    τ_21 = (3c²A₁₀)/(32π ν²) × N_H/(T_s × Δv)
    
    Parameters:
        N_H: neutral hydrogen column density [m⁻²]
        T_s: spin temperature [K]
        delta_v: velocity dispersion [m/s]
    
    Returns:
        Optical depth [dimensionless]
    """
    from constants import nu_21cm
    
    tau = 3 * c**2 * A_21cm / (32 * np.pi * nu_21cm**2)
    tau *= N_H / (T_s * delta_v)
    
    return tau


# ============================================
# Collisional Rates
# ============================================

def collisional_excitation_rate_H(n_lower, n_upper, T, n_e):
    """
    Collisional excitation rate coefficient (electron-hydrogen).
    
    Approximate formula using van Regemorter (1962).
    
    Parameters:
        n_lower, n_upper: levels
        T: electron temperature [K]
        n_e: electron density [m⁻³]
    
    Returns:
        Rate [s⁻¹]
    """
    dE = excitation_energy(n_lower, n_upper)
    x = dE / (k_B * T)
    
    if x > 50:
        return 0.0  # Negligible
    
    f = oscillator_strength(n_lower, n_upper)
    g_l = statistical_weight(n_lower)
    
    # van Regemorter formula
    g_bar = 0.2  # Average Gaunt factor (approximate)
    
    q = 1.1e-8 * f * g_bar * np.exp(-x) / np.sqrt(T)  # cm³/s
    q *= 1e-6  # Convert to m³/s
    
    return q * n_e


def collisional_ionization_rate(T, n_e):
    """
    Collisional ionization rate from ground state.
    
    Parameters:
        T: electron temperature [K]
        n_e: electron density [m⁻³]
    
    Returns:
        Rate [s⁻¹]
    """
    x = E_ion / (k_B * T)
    
    if x > 50:
        return 0.0
    
    # Approximate (Lotz formula)
    S = 2.5e-12 * np.exp(-x) / np.sqrt(T)  # cm³/s
    S *= 1e-6  # m³/s
    
    return S * n_e


# Test if run directly
if __name__ == "__main__":
    print("=== Hydrogen Atomic Physics Test ===")
    print()
    
    # Energy levels
    print("Energy levels:")
    for n in [1, 2, 3, 4]:
        print(f"  n={n}: E = {energy_level_eV(n):.4f} eV, g = {statistical_weight(n)}")
    
    print()
    print("Key transitions:")
    transitions = [(2,1), (3,1), (3,2), (4,2)]
    names = ["Ly-α", "Ly-β", "H-α", "H-β"]
    for (nu, nl), name in zip(transitions, names):
        wl = transition_wavelength_nm(nu, nl)
        A = einstein_A(nu, nl)
        print(f"  {name} ({nu}->{nl}): λ = {wl:.2f} nm, A = {A:.2e} s⁻¹")
    
    print()
    print("Line cross-sections at T=100 K:")
    T = 100
    for (nu, nl), name in zip(transitions[:2], names[:2]):
        sigma = line_cross_section_peak(nl, nu, T)
        print(f"  {name}: σ_peak = {sigma:.2e} m²")
    
    print()
    print("Photoionization cross-section at threshold:")
    print(f"  Ground state: σ_bf = {photoionization_cross_section(nu_ion):.2e} m²")
