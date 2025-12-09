"""
Emission spectrum calculations for hydrogen cloud.

Calculates emissivity from three physical processes:
1. Line emission (bound-bound): Lyman, Balmer, Paschen series
2. Recombination continuum (free-bound): electron capture
3. Bremsstrahlung (free-free): electron-ion scattering

All emissivities are in units of W/(m³·Hz·sr) unless noted.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from constants import (h, c, k_B, m_e, m_H, e, epsilon_0,
                       E_ion, E_ion_eV, nu_ion, lambda_ion,
                       ly_to_m, DEFAULT_RADIUS_M, DEFAULT_DENSITY)
from atomic_physics import (einstein_A, transition_frequency, transition_wavelength,
                            statistical_weight, doppler_width, 
                            line_profile_gaussian, photoionization_cross_section,
                            gaunt_factor_bf)
from equilibrium import (ionization_fraction, level_population_fraction,
                         electron_density, neutral_density, ion_density,
                         partition_function_H)
from thermal import equilibrium_temperature


# ============================================
# Physical Constants for Emission
# ============================================

# Fine structure constant
alpha_fs = 1 / 137.036

# Classical electron radius
r_e = e**2 / (4 * np.pi * epsilon_0 * m_e * c**2)


# ============================================
# Line Emission (Bound-Bound)
# ============================================

def line_emissivity(nu: float, n_upper: int, n_lower: int, 
                    n_H: float, T: float) -> float:
    """
    Spectral emissivity for a single emission line.
    
    j_ν = (hν / 4π) × n_u × A_ul × φ(ν)
    
    where n_u is the density in the upper level.
    
    Parameters:
        nu: frequency [Hz]
        n_upper: upper level quantum number
        n_lower: lower level quantum number  
        n_H: neutral hydrogen density [m⁻³]
        T: gas temperature [K]
    
    Returns:
        Emissivity [W/(m³·Hz·sr)]
    """
    if n_H <= 0 or T <= 0:
        return 0.0
    
    # Line center frequency
    nu_0 = transition_frequency(n_upper, n_lower)
    
    # Population in upper level
    f_u = level_population_fraction(n_upper, T)
    n_u = n_H * f_u
    
    # Einstein A coefficient
    A_ul = einstein_A(n_upper, n_lower)
    
    # Line profile (normalized)
    phi = line_profile_gaussian(nu, nu_0, T, m_H)
    
    # Emissivity
    j_nu = (h * nu_0 / (4 * np.pi)) * n_u * A_ul * phi
    
    return j_nu


def line_emissivity_integrated(n_upper: int, n_lower: int,
                                n_H: float, T: float) -> float:
    """
    Frequency-integrated emissivity for a line.
    
    j = (hν / 4π) × n_u × A_ul  [W/(m³·sr)]
    
    Parameters:
        n_upper, n_lower: transition levels
        n_H: neutral hydrogen density [m⁻³]
        T: gas temperature [K]
    
    Returns:
        Integrated emissivity [W/(m³·sr)]
    """
    if n_H <= 0 or T <= 0:
        return 0.0
    
    nu_0 = transition_frequency(n_upper, n_lower)
    f_u = level_population_fraction(n_upper, T)
    n_u = n_H * f_u
    A_ul = einstein_A(n_upper, n_lower)
    
    return (h * nu_0 / (4 * np.pi)) * n_u * A_ul


def total_line_emissivity(nu: float, n_H: float, T: float,
                          n_max: int = 10) -> float:
    """
    Total line emissivity from all transitions.
    
    Includes Lyman (n→1), Balmer (n→2), Paschen (n→3), etc.
    
    Parameters:
        nu: frequency [Hz]
        n_H: neutral hydrogen density [m⁻³]
        T: gas temperature [K]
        n_max: maximum upper level to include
    
    Returns:
        Total line emissivity [W/(m³·Hz·sr)]
    """
    j_total = 0.0
    
    # Sum over all series
    for n_lower in range(1, n_max):
        for n_upper in range(n_lower + 1, n_max + 1):
            j_total += line_emissivity(nu, n_upper, n_lower, n_H, T)
    
    return j_total


# ============================================
# Recombination Continuum (Free-Bound)
# ============================================

def recombination_emissivity_to_level(nu: float, n: int,
                                       n_e: float, n_Hplus: float, 
                                       T: float) -> float:
    """
    Emissivity from recombination to level n.
    
    Uses Milne relation (detailed balance with photoionization):
    
    j_ν^fb = (2hν³/c²) × n_e × n_H+ × (h²/(2πm_e kT))^(3/2) 
             × (g_n/2) × σ_bf(ν) × exp(-h(ν - ν_n)/kT)
    
    where ν_n is the threshold frequency for level n.
    
    Parameters:
        nu: frequency [Hz]
        n: level being recombined to
        n_e: electron density [m⁻³]
        n_Hplus: ion density [m⁻³]
        T: electron temperature [K]
    
    Returns:
        Emissivity [W/(m³·Hz·sr)]
    """
    if n_e <= 0 or n_Hplus <= 0 or T <= 0:
        return 0.0
    
    # Threshold frequency for level n
    nu_n = nu_ion / n**2
    
    # Below threshold, no recombination to this level
    if nu < nu_n:
        return 0.0
    
    # Statistical weight
    g_n = statistical_weight(n)
    
    # Photoionization cross-section (from level n at this frequency)
    sigma_bf = photoionization_cross_section(nu, n)
    
    # Thermal de Broglie factor
    lambda_th_cubed = (h**2 / (2 * np.pi * m_e * k_B * T))**1.5
    
    # Excess energy factor
    x = h * (nu - nu_n) / (k_B * T)
    if x > 100:
        return 0.0
    
    # Emissivity (Milne relation)
    j_nu = (2 * h * nu**3 / c**2) * n_e * n_Hplus
    j_nu *= lambda_th_cubed * (g_n / 2) * sigma_bf * np.exp(-x)
    j_nu /= (4 * np.pi)  # Per steradian
    
    return j_nu


def total_recombination_emissivity(nu: float, n_e: float, n_Hplus: float,
                                    T: float, n_max: int = 20) -> float:
    """
    Total recombination continuum emissivity.
    
    Sums contributions from recombination to all levels.
    
    Parameters:
        nu: frequency [Hz]
        n_e: electron density [m⁻³]
        n_Hplus: ion density [m⁻³]
        T: electron temperature [K]
        n_max: maximum level to include
    
    Returns:
        Total recombination emissivity [W/(m³·Hz·sr)]
    """
    j_total = 0.0
    
    for n in range(1, n_max + 1):
        j_total += recombination_emissivity_to_level(nu, n, n_e, n_Hplus, T)
    
    return j_total


# ============================================
# Bremsstrahlung (Free-Free)
# ============================================

def gaunt_factor_ff(nu: float, T: float) -> float:
    """
    Free-free Gaunt factor (quantum correction).
    
    Approximate formula valid for most astrophysical conditions.
    
    Parameters:
        nu: frequency [Hz]
        T: temperature [K]
    
    Returns:
        Gaunt factor [dimensionless]
    """
    if T <= 0 or nu <= 0:
        return 1.0
    
    # Dimensionless parameters
    u = h * nu / (k_B * T)
    gamma_squared = 13.6 * e / (k_B * T)  # Z²Ry/kT for Z=1
    
    # Approximate Gaunt factor (Karzas & Latter approximation)
    if u < 1:
        g_ff = 1.0 + 0.1728 * (u**0.333) * (1 + 2/gamma_squared)
    else:
        g_ff = (np.sqrt(3) / np.pi) * np.log(4 / (1.781 * u))
        g_ff = max(g_ff, 1.0)
    
    return np.clip(g_ff, 0.8, 3.0)


def bremsstrahlung_emissivity(nu: float, n_e: float, n_Hplus: float, 
                               T: float) -> float:
    """
    Free-free (bremsstrahlung) emissivity.
    
    j_ν^ff = (8/3) × (2π/3)^(1/2) × (e²/4πε₀)³ × (1/m_e c²) 
             × (m_e/kT)^(1/2) × n_e × n_H+ × g_ff × exp(-hν/kT)
    
    Simplified form:
    j_ν^ff = 6.8e-51 × T^(-1/2) × n_e × n_H+ × g_ff × exp(-hν/kT) [W/(m³·Hz·sr)]
    
    Parameters:
        nu: frequency [Hz]
        n_e: electron density [m⁻³]
        n_Hplus: ion density [m⁻³]
        T: electron temperature [K]
    
    Returns:
        Emissivity [W/(m³·Hz·sr)]
    """
    if n_e <= 0 or n_Hplus <= 0 or T <= 0:
        return 0.0
    
    # Exponential cutoff
    x = h * nu / (k_B * T)
    if x > 100:
        return 0.0
    
    # Gaunt factor
    g_ff = gaunt_factor_ff(nu, T)
    
    # Coefficient (SI units)
    # Derived from full expression
    coeff = 6.8e-51  # W·m³·K^(1/2)·Hz⁻¹·sr⁻¹
    
    j_nu = coeff * T**(-0.5) * n_e * n_Hplus * g_ff * np.exp(-x)
    
    return j_nu


# ============================================
# Total Emission Spectrum
# ============================================

@dataclass
class EmissionComponents:
    """Container for emission spectrum components."""
    frequency: np.ndarray      # [Hz]
    wavelength: np.ndarray     # [m]
    line: np.ndarray           # Line emission [W/(m³·Hz·sr)]
    recombination: np.ndarray  # Recombination continuum
    bremsstrahlung: np.ndarray # Free-free emission
    total: np.ndarray          # Sum of all components


def compute_emission_spectrum(n_total: float, T_rad: float,
                               nu_min: float = 1e13, nu_max: float = 1e17,
                               n_points: int = 1000,
                               n_max_levels: int = 10,
                               T_gas: float = None) -> EmissionComponents:
    """
    Compute full emission spectrum for the cloud.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        nu_min, nu_max: frequency range [Hz]
        n_points: number of frequency points
        n_max_levels: maximum quantum number for lines
        T_gas: gas temperature [K], if None uses equilibrium or T_rad
    
    Returns:
        EmissionComponents with all spectral components
    """
    # Determine gas temperature
    if T_gas is None:
        T_eq = equilibrium_temperature(n_total, T_rad)
        # If equilibrium solver returns boundary value, use T_rad instead
        if T_eq <= 10:
            T_gas = T_rad
        else:
            T_gas = T_eq
    
    n_H = neutral_density(n_total, T_gas)
    n_e = electron_density(n_total, T_gas)
    n_Hplus = ion_density(n_total, T_gas)
    
    # Frequency grid (logarithmic)
    nu_array = np.logspace(np.log10(nu_min), np.log10(nu_max), n_points)
    lambda_array = c / nu_array
    
    # Initialize arrays
    j_line = np.zeros(n_points)
    j_recomb = np.zeros(n_points)
    j_brems = np.zeros(n_points)
    
    print(f"Computing emission spectrum...")
    print(f"  T_rad = {T_rad:.0f} K, T_gas = {T_gas:.0f} K")
    print(f"  x_ion = {n_e/n_total:.4f}")
    print(f"  n_H = {n_H:.2e} m⁻³, n_e = {n_e:.2e} m⁻³")
    
    for i, nu in enumerate(nu_array):
        # Line emission (only if neutral atoms exist)
        if n_H > 0:
            j_line[i] = total_line_emissivity(nu, n_H, T_gas, n_max_levels)
        
        # Recombination and bremsstrahlung (only if ionized)
        if n_e > 0 and n_Hplus > 0:
            j_recomb[i] = total_recombination_emissivity(nu, n_e, n_Hplus, 
                                                          T_gas, n_max_levels)
            j_brems[i] = bremsstrahlung_emissivity(nu, n_e, n_Hplus, T_gas)
    
    j_total = j_line + j_recomb + j_brems
    
    return EmissionComponents(
        frequency=nu_array,
        wavelength=lambda_array,
        line=j_line,
        recombination=j_recomb,
        bremsstrahlung=j_brems,
        total=j_total
    )


# ============================================
# Line Catalog
# ============================================

@dataclass
class SpectralLine:
    """Information about a spectral line."""
    name: str
    series: str
    n_upper: int
    n_lower: int
    wavelength_nm: float
    frequency_hz: float
    einstein_A: float


def get_line_catalog(n_max: int = 7) -> List[SpectralLine]:
    """
    Generate catalog of hydrogen emission lines.
    
    Parameters:
        n_max: maximum upper level
    
    Returns:
        List of SpectralLine objects
    """
    series_names = {1: 'Lyman', 2: 'Balmer', 3: 'Paschen', 
                    4: 'Brackett', 5: 'Pfund', 6: 'Humphreys'}
    greek = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ']
    
    lines = []
    
    for n_lower in range(1, min(n_max, 7)):
        series = series_names.get(n_lower, f'n={n_lower}')
        
        for i, n_upper in enumerate(range(n_lower + 1, n_max + 1)):
            letter = greek[i] if i < len(greek) else str(i + 1)
            name = f'{series}-{letter}'
            
            nu = transition_frequency(n_upper, n_lower)
            wl_nm = c / nu * 1e9
            A = einstein_A(n_upper, n_lower)
            
            lines.append(SpectralLine(
                name=name,
                series=series,
                n_upper=n_upper,
                n_lower=n_lower,
                wavelength_nm=wl_nm,
                frequency_hz=nu,
                einstein_A=A
            ))
    
    return lines


def print_line_catalog(n_max: int = 7):
    """Print formatted line catalog."""
    lines = get_line_catalog(n_max)
    
    print("\nHydrogen Emission Line Catalog")
    print("=" * 65)
    print(f"{'Line':<12} {'Transition':<12} {'λ (nm)':<12} {'A (s⁻¹)':<12}")
    print("-" * 65)
    
    current_series = None
    for line in lines:
        if line.series != current_series:
            if current_series is not None:
                print()
            current_series = line.series
        
        trans = f"{line.n_upper} → {line.n_lower}"
        print(f"{line.name:<12} {trans:<12} {line.wavelength_nm:<12.2f} {line.einstein_A:<12.2e}")


# ============================================
# Plotting Functions
# ============================================

def plot_emission_spectrum(spectrum: EmissionComponents, 
                           T_rad: float,
                           wavelength_units: str = 'nm',
                           save_path: str = None,
                           show_lines: bool = True) -> plt.Figure:
    """
    Create publication-quality emission spectrum plot.
    
    Parameters:
        spectrum: EmissionComponents from compute_emission_spectrum
        T_rad: radiation temperature (for title)
        wavelength_units: 'nm', 'um', or 'A' (angstrom)
        save_path: path to save figure
        show_lines: annotate major emission lines
    
    Returns:
        matplotlib Figure
    """
    # Unit conversion
    if wavelength_units == 'nm':
        wavelength = spectrum.wavelength * 1e9
        xlabel = 'Wavelength [nm]'
    elif wavelength_units == 'um':
        wavelength = spectrum.wavelength * 1e6
        xlabel = 'Wavelength [μm]'
    elif wavelength_units == 'A':
        wavelength = spectrum.wavelength * 1e10
        xlabel = 'Wavelength [Å]'
    else:
        wavelength = spectrum.wavelength * 1e9
        xlabel = 'Wavelength [nm]'
    
    # Convert to per-wavelength emissivity: j_λ = j_ν × (c/λ²)
    conversion = c / spectrum.wavelength**2
    j_line_lambda = spectrum.line * conversion
    j_recomb_lambda = spectrum.recombination * conversion
    j_brems_lambda = spectrum.bremsstrahlung * conversion
    j_total_lambda = spectrum.total * conversion
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Top panel: Full spectrum
    ax1 = axes[0]
    
    # Plot components
    ax1.loglog(wavelength, j_total_lambda, 'k-', linewidth=1.5, 
               label='Total', alpha=0.8)
    ax1.loglog(wavelength, j_line_lambda, 'b-', linewidth=1, 
               label='Lines (bound-bound)', alpha=0.7)
    ax1.loglog(wavelength, j_recomb_lambda, 'r-', linewidth=1,
               label='Recombination (free-bound)', alpha=0.7)
    ax1.loglog(wavelength, j_brems_lambda, 'g-', linewidth=1,
               label='Bremsstrahlung (free-free)', alpha=0.7)
    
    # Mark key wavelengths
    key_lines = [
        (121.6, 'Ly-α', 'blue'),
        (656.3, 'H-α', 'red'),
        (486.1, 'H-β', 'cyan'),
        (91.2, 'Lyman limit', 'purple'),
    ]
    
    if show_lines:
        ymin, ymax = ax1.get_ylim()
        for wl, name, color in key_lines:
            if wavelength.min() < wl < wavelength.max():
                ax1.axvline(wl, color=color, linestyle='--', alpha=0.5, linewidth=0.8)
                ax1.annotate(name, xy=(wl, ymax*0.5), fontsize=8, 
                            rotation=90, va='center', color=color)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Emissivity $j_λ$ [W/(m³·m·sr)]')
    ax1.set_title(f'Hydrogen Cloud Emission Spectrum (T_rad = {T_rad:.0f} K)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(wavelength.min(), wavelength.max())
    
    # Bottom panel: Relative contributions
    ax2 = axes[1]
    
    # Avoid division by zero
    j_total_safe = np.maximum(j_total_lambda, 1e-100)
    
    ax2.semilogx(wavelength, j_line_lambda / j_total_safe * 100, 'b-', 
                 label='Lines', linewidth=1.5)
    ax2.semilogx(wavelength, j_recomb_lambda / j_total_safe * 100, 'r-',
                 label='Recombination', linewidth=1.5)
    ax2.semilogx(wavelength, j_brems_lambda / j_total_safe * 100, 'g-',
                 label='Bremsstrahlung', linewidth=1.5)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Contribution [%]')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(wavelength.min(), wavelength.max())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_spectrum_vs_temperature(n_total: float,
                                  temperatures: List[float],
                                  nu_min: float = 1e14,
                                  nu_max: float = 5e15,
                                  n_points: int = 500,
                                  save_path: str = None) -> plt.Figure:
    """
    Plot emission spectra at multiple temperatures.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        temperatures: list of radiation temperatures [K]
        nu_min, nu_max: frequency range [Hz]
        n_points: number of points
        save_path: path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(temperatures)))
    
    for T_rad, color in zip(temperatures, colors):
        spectrum = compute_emission_spectrum(n_total, T_rad, 
                                              nu_min, nu_max, n_points)
        
        # Convert to wavelength
        wavelength_nm = spectrum.wavelength * 1e9
        j_lambda = spectrum.total * c / spectrum.wavelength**2
        
        x_ion = electron_density(n_total, equilibrium_temperature(n_total, T_rad)) / n_total
        label = f'T = {T_rad:.0f} K (x_ion = {x_ion:.2f})'
        
        ax.loglog(wavelength_nm, j_lambda, '-', color=color, 
                  linewidth=1.5, label=label)
    
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Emissivity $j_λ$ [W/(m³·m·sr)]')
    ax.set_title(f'Emission Spectrum vs Temperature (n = {n_total:.0e} m⁻³)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Mark Lyman and Balmer limits
    ax.axvline(91.2, color='purple', linestyle='--', alpha=0.5, label='Lyman limit')
    ax.axvline(364.6, color='red', linestyle='--', alpha=0.5, label='Balmer limit')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_line_strengths(n_total: float, T_rad: float,
                        n_max: int = 7,
                        save_path: str = None) -> plt.Figure:
    """
    Bar chart of integrated line strengths.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        n_max: maximum quantum number
        save_path: path to save figure
    
    Returns:
        matplotlib Figure
    """
    T_gas = equilibrium_temperature(n_total, T_rad)
    n_H = neutral_density(n_total, T_gas)
    
    lines = get_line_catalog(n_max)
    
    # Compute integrated emissivities
    strengths = []
    for line in lines:
        j_int = line_emissivity_integrated(line.n_upper, line.n_lower, n_H, T_gas)
        strengths.append(j_int)
    
    # Separate by series
    series_data = {}
    for line, strength in zip(lines, strengths):
        if line.series not in series_data:
            series_data[line.series] = {'names': [], 'strengths': [], 'wavelengths': []}
        series_data[line.series]['names'].append(line.name.split('-')[1])
        series_data[line.series]['strengths'].append(strength)
        series_data[line.series]['wavelengths'].append(line.wavelength_nm)
    
    fig, axes = plt.subplots(1, min(3, len(series_data)), figsize=(14, 5))
    if len(series_data) == 1:
        axes = [axes]
    
    series_colors = {'Lyman': 'blue', 'Balmer': 'red', 'Paschen': 'green',
                     'Brackett': 'orange', 'Pfund': 'purple'}
    
    for ax, (series_name, data) in zip(axes, list(series_data.items())[:3]):
        color = series_colors.get(series_name, 'gray')
        bars = ax.bar(data['names'], data['strengths'], color=color, alpha=0.7)
        
        ax.set_xlabel('Line')
        ax.set_ylabel('Integrated emissivity [W/(m³·sr)]')
        ax.set_title(f'{series_name} Series')
        ax.set_yscale('log')
        
        # Add wavelength labels
        for bar, wl in zip(bars, data['wavelengths']):
            ax.annotate(f'{wl:.0f} nm', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       fontsize=7, ha='center', va='bottom', rotation=45)
    
    plt.suptitle(f'Line Strengths at T_rad = {T_rad:.0f} K, T_gas = {T_gas:.0f} K')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


# ============================================
# Main Function
# ============================================

def main():
    """Demonstrate emission spectrum calculations."""
    
    print("=" * 70)
    print("HYDROGEN CLOUD EMISSION SPECTRUM")
    print("=" * 70)
    
    # Cloud parameters
    n_total = 1e6  # m⁻³
    
    # Print line catalog
    print_line_catalog(n_max=6)
    
    # Compute spectrum at a few temperatures
    print("\n" + "=" * 70)
    print("EMISSION SPECTRA AT DIFFERENT TEMPERATURES")
    print("=" * 70)
    
    temperatures = [2000, 5000, 10000, 20000]
    
    for T_rad in temperatures:
        print(f"\n--- T_rad = {T_rad} K ---")
        spectrum = compute_emission_spectrum(n_total, T_rad, 
                                              nu_min=1e14, nu_max=1e16,
                                              n_points=200)
        
        # Find peak
        idx_peak = np.argmax(spectrum.total)
        lambda_peak = spectrum.wavelength[idx_peak] * 1e9
        
        print(f"  Peak wavelength: {lambda_peak:.1f} nm")
        print(f"  Peak emissivity: {spectrum.total[idx_peak]:.2e} W/(m³·Hz·sr)")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Single temperature spectrum
    T_demo = 5000
    print(f"\n1. Detailed spectrum at T = {T_demo} K...")
    spectrum = compute_emission_spectrum(n_total, T_demo,
                                          nu_min=1e14, nu_max=3e16,
                                          n_points=1000)
    fig1 = plot_emission_spectrum(spectrum, T_demo, 
                                   wavelength_units='nm',
                                   save_path='emission_spectrum.png')
    
    # Temperature comparison
    print("\n2. Spectrum vs temperature...")
    fig2 = plot_spectrum_vs_temperature(n_total, 
                                         [2000, 4000, 6000, 10000, 20000],
                                         save_path='emission_vs_temperature.png')
    
    # Line strengths
    print("\n3. Line strength comparison...")
    fig3 = plot_line_strengths(n_total, 5000, n_max=7,
                                save_path='line_strengths.png')
    
    print("\n" + "=" * 70)
    print("DONE! Generated plots:")
    print("  - emission_spectrum.png")
    print("  - emission_vs_temperature.png")
    print("  - line_strengths.png")
    print("=" * 70)
    
    plt.show()
    
    return spectrum


if __name__ == "__main__":
    spectrum = main()
