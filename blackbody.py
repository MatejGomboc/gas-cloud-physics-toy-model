"""
Blackbody radiation functions.
Planck distribution, photon statistics, and related quantities.
"""

import numpy as np
from constants import h, c, k_B, sigma_SB, a_rad, E_ion, nu_ion


def planck_spectral_radiance(nu, T):
    """
    Planck function B_ν(T) - spectral radiance.
    
    B_ν = (2hν³/c²) × 1/(exp(hν/kT) - 1)
    
    Parameters:
        nu: frequency [Hz] (scalar or array)
        T: temperature [K]
    
    Returns:
        Spectral radiance [W/(m²·sr·Hz)]
    """
    if T <= 0:
        return np.zeros_like(nu) if hasattr(nu, '__len__') else 0.0
    
    x = h * nu / (k_B * T)
    # Avoid overflow for large x
    x = np.clip(x, 0, 700)
    
    return (2 * h * nu**3 / c**2) / (np.exp(x) - 1)


def planck_wavelength(wavelength, T):
    """
    Planck function B_λ(T) - spectral radiance per wavelength.
    
    B_λ = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
    
    Parameters:
        wavelength: wavelength [m]
        T: temperature [K]
    
    Returns:
        Spectral radiance [W/(m²·sr·m)]
    """
    if T <= 0:
        return np.zeros_like(wavelength) if hasattr(wavelength, '__len__') else 0.0
    
    x = h * c / (wavelength * k_B * T)
    x = np.clip(x, 0, 700)
    
    return (2 * h * c**2 / wavelength**5) / (np.exp(x) - 1)


def spectral_energy_density(nu, T):
    """
    Spectral energy density u_ν(T).
    
    u_ν = (4π/c) × B_ν = (8πhν³/c³) × 1/(exp(hν/kT) - 1)
    
    Parameters:
        nu: frequency [Hz]
        T: temperature [K]
    
    Returns:
        Energy density per unit frequency [J/(m³·Hz)]
    """
    return (4 * np.pi / c) * planck_spectral_radiance(nu, T)


def photon_number_density(nu, T):
    """
    Spectral photon number density n_ph(ν, T).
    
    n_ph = u_ν / (hν) = (8πν²/c³) × 1/(exp(hν/kT) - 1)
    
    Parameters:
        nu: frequency [Hz]
        T: temperature [K]
    
    Returns:
        Photon number density per unit frequency [photons/(m³·Hz)]
    """
    if T <= 0:
        return np.zeros_like(nu) if hasattr(nu, '__len__') else 0.0
    
    x = h * nu / (k_B * T)
    x = np.clip(x, 0, 700)
    
    return (8 * np.pi * nu**2 / c**3) / (np.exp(x) - 1)


def photon_occupation_number(nu, T):
    """
    Bose-Einstein occupation number n̄(ν, T).
    
    n̄ = 1/(exp(hν/kT) - 1)
    
    This is the mean number of photons per mode.
    
    Parameters:
        nu: frequency [Hz]
        T: temperature [K]
    
    Returns:
        Mean occupation number [dimensionless]
    """
    if T <= 0:
        return np.zeros_like(nu) if hasattr(nu, '__len__') else 0.0
    
    x = h * nu / (k_B * T)
    x = np.clip(x, 0, 700)
    
    return 1.0 / (np.exp(x) - 1)


def mean_intensity(nu, T):
    """
    Mean intensity J_ν for isotropic radiation field.
    For a blackbody: J_ν = B_ν(T)
    
    Parameters:
        nu: frequency [Hz]
        T: temperature [K]
    
    Returns:
        Mean intensity [W/(m²·sr·Hz)]
    """
    return planck_spectral_radiance(nu, T)


def total_energy_density(T):
    """
    Total energy density of blackbody radiation.
    
    u = aT⁴ = (4σ/c)T⁴
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Energy density [J/m³]
    """
    return a_rad * T**4


def total_photon_density(T):
    """
    Total photon number density.
    
    n_γ = (2ζ(3)/π²) × (kT/ℏc)³ ≈ 20.28 × T³ [photons/m³ for T in K]
    
    where ζ(3) ≈ 1.202 is Riemann zeta function.
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Total photon density [photons/m³]
    """
    zeta_3 = 1.2020569031595942
    return (2 * zeta_3 / np.pi**2) * (k_B * T / (h / (2*np.pi) * c))**3


def wien_peak_frequency(T):
    """
    Peak frequency from Wien's displacement law.
    
    ν_peak = 2.821 × kT/h
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Peak frequency [Hz]
    """
    return 2.821439372 * k_B * T / h


def wien_peak_wavelength(T):
    """
    Peak wavelength from Wien's displacement law.
    
    λ_peak = 2.898 mm·K / T
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Peak wavelength [m]
    """
    if T <= 0:
        return np.inf
    return 2.897771955e-3 / T


def ionizing_photon_flux(T):
    """
    Flux of ionizing photons (ν > ν_ion = 13.6 eV/h).
    
    Φ_ion = ∫_{ν_ion}^∞ (c/4π) × n_ph(ν) dν
    
    Integrated numerically.
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Ionizing photon flux [photons/(m²·s·sr)]
    """
    if T <= 0:
        return 0.0
    
    # Check if ionizing photons are negligible
    x_threshold = h * nu_ion / (k_B * T)
    if x_threshold > 100:
        return 0.0  # Essentially zero
    
    # Integrate from threshold to ~20 kT above threshold
    nu_max = nu_ion + 20 * k_B * T / h
    nu_array = np.linspace(nu_ion, nu_max, 1000)
    
    integrand = (c / (4 * np.pi)) * photon_number_density(nu_array, T)
    
    # Trapezoidal integration
    flux = np.trapz(integrand, nu_array)
    return flux


def ionizing_photon_density(T):
    """
    Number density of ionizing photons (E > 13.6 eV).
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Ionizing photon density [photons/m³]
    """
    if T <= 0:
        return 0.0
    
    x_threshold = h * nu_ion / (k_B * T)
    if x_threshold > 100:
        return 0.0
    
    nu_max = nu_ion + 30 * k_B * T / h
    nu_array = np.linspace(nu_ion, nu_max, 1000)
    
    integrand = photon_number_density(nu_array, T)
    
    return np.trapz(integrand, nu_array)


def fraction_ionizing(T):
    """
    Fraction of photon energy above ionization threshold.
    
    Parameters:
        T: temperature [K]
    
    Returns:
        Fraction [dimensionless]
    """
    if T <= 0:
        return 0.0
    
    x_threshold = h * nu_ion / (k_B * T)
    if x_threshold > 50:
        return 0.0
    
    # Total energy density
    u_total = total_energy_density(T)
    if u_total == 0:
        return 0.0
    
    # Integrate energy density above threshold
    nu_max = nu_ion * 10  # Go to 10× threshold
    nu_array = np.linspace(nu_ion, nu_max, 1000)
    
    integrand = spectral_energy_density(nu_array, T)
    u_ionizing = np.trapz(integrand, nu_array)
    
    return u_ionizing / u_total


# Test if run directly
if __name__ == "__main__":
    print("=== Blackbody Radiation Test ===")
    print()
    
    temperatures = [2.725, 300, 3000, 10000, 30000]
    
    for T in temperatures:
        print(f"T = {T:,.0f} K:")
        print(f"  Peak wavelength: {wien_peak_wavelength(T)*1e9:.2f} nm")
        print(f"  Peak frequency: {wien_peak_frequency(T):.3e} Hz")
        print(f"  Total energy density: {total_energy_density(T):.3e} J/m³")
        print(f"  Total photon density: {total_photon_density(T):.3e} /m³")
        print(f"  Ionizing fraction: {fraction_ionizing(T):.3e}")
        print(f"  Ionizing photon density: {ionizing_photon_density(T):.3e} /m³")
        print()
