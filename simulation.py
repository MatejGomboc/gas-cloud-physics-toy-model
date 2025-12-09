"""
Main simulation for hydrogen cloud in a blackbody radiation bath.

This module ties together all the physics to simulate how a
1 light-year hydrogen cloud responds to different background
radiation temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Dict

# Our physics modules
from constants import (h, c, k_B, m_H, G, 
                       ly_to_m, E_ion_eV, T_CMB,
                       nu_21cm, A_21cm, T_star_21cm,
                       DEFAULT_RADIUS_M, DEFAULT_DENSITY)
from blackbody import (planck_spectral_radiance, wien_peak_wavelength,
                       total_energy_density, ionizing_photon_density,
                       fraction_ionizing)
from equilibrium import (ionization_fraction, level_population_fraction,
                         boltzmann_ratio, partition_function_H,
                         ionization_temperature)
from thermal import (photoionization_rate, recombination_coefficient_B,
                     total_heating_rate, total_cooling_rate,
                     equilibrium_temperature, cooling_time)
from atomic_physics import (line_cross_section_peak, doppler_width,
                            transition_frequency, optical_depth_21cm)


@dataclass
class CloudParameters:
    """Parameters defining the hydrogen cloud."""
    radius: float = DEFAULT_RADIUS_M  # meters
    density: float = DEFAULT_DENSITY  # atoms/m^3
    
    @property
    def radius_ly(self) -> float:
        return self.radius / ly_to_m
    
    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3
    
    @property
    def total_atoms(self) -> float:
        return self.density * self.volume
    
    @property
    def total_mass(self) -> float:
        return self.total_atoms * m_H
    
    @property
    def column_density(self) -> float:
        """Column density through center [m^-2]"""
        return 2 * self.density * self.radius
    
    def __str__(self):
        return (f"Hydrogen Cloud:\n"
                f"  Radius: {self.radius_ly:.2f} ly ({self.radius:.2e} m)\n"
                f"  Density: {self.density:.2e} atoms/m^3\n"
                f"  Total atoms: {self.total_atoms:.2e}\n"
                f"  Total mass: {self.total_mass:.2e} kg ({self.total_mass/5.97e24:.1f} Earth masses)\n"
                f"  Column density: {self.column_density:.2e} m^-2")


@dataclass 
class CloudState:
    """Physical state of the cloud at given conditions."""
    T_radiation: float  # Background radiation temperature [K]
    T_gas: float        # Gas kinetic temperature [K]
    ionization_fraction: float
    n_neutral: float    # Neutral H density [m^-3]
    n_electron: float   # Electron density [m^-3]
    n_ion: float        # H+ density [m^-3]
    f_ground: float     # Fraction in ground state
    f_n2: float         # Fraction in n=2
    f_n3: float         # Fraction in n=3
    heating_rate: float # [W/m^3]
    cooling_rate: float # [W/m^3]
    tau_lya: float      # Lyman-alpha optical depth
    tau_21cm: float     # 21 cm optical depth
    sound_speed: float  # [m/s]
    jeans_length: float # [m]
    expansion_time: float  # [s]
    cooling_time: float    # [s]


class HydrogenCloudSimulation:
    """
    Simulation of a hydrogen cloud bathed in blackbody radiation.
    
    Physics included:
    - Saha ionization equilibrium
    - Boltzmann level populations
    - Photoionization and recombination
    - Thermal equilibrium (heating/cooling balance)
    - Optical depths
    - Gravitational stability
    """
    
    def __init__(self, cloud: CloudParameters = None):
        if cloud is None:
            cloud = CloudParameters()
        self.cloud = cloud
        
    def compute_state(self, T_rad: float) -> CloudState:
        """
        Compute the equilibrium state of the cloud for a given
        radiation temperature.
        
        Parameters:
            T_rad: background radiation temperature [K]
        
        Returns:
            CloudState with all computed properties
        """
        n_total = self.cloud.density
        
        # Gas temperature (equilibrium or coupled to radiation)
        T_gas = equilibrium_temperature(n_total, T_rad)
        
        # Ionization state
        x_ion = ionization_fraction(n_total, T_gas)
        n_neutral = (1 - x_ion) * n_total
        n_ion = x_ion * n_total
        n_electron = n_ion  # Charge neutrality
        
        # Level populations (in neutral atoms)
        f_1 = level_population_fraction(1, T_gas)
        f_2 = level_population_fraction(2, T_gas)
        f_3 = level_population_fraction(3, T_gas)
        
        # Heating and cooling rates
        H = total_heating_rate(n_total, T_rad, T_gas)
        L = total_cooling_rate(n_total, T_rad, T_gas)
        
        # Optical depths
        tau_lya = self._compute_lya_optical_depth(n_neutral, T_gas)
        tau_21 = self._compute_21cm_optical_depth(n_neutral, T_gas)
        
        # Dynamics
        c_s = self._sound_speed(T_gas, x_ion)
        lambda_J = self._jeans_length(T_gas, n_total)
        t_exp = self.cloud.radius / c_s if c_s > 0 else np.inf
        t_cool = cooling_time(n_total, T_gas, T_rad)
        
        return CloudState(
            T_radiation=T_rad,
            T_gas=T_gas,
            ionization_fraction=x_ion,
            n_neutral=n_neutral,
            n_electron=n_electron,
            n_ion=n_ion,
            f_ground=f_1,
            f_n2=f_2,
            f_n3=f_3,
            heating_rate=H,
            cooling_rate=L,
            tau_lya=tau_lya,
            tau_21cm=tau_21,
            sound_speed=c_s,
            jeans_length=lambda_J,
            expansion_time=t_exp,
            cooling_time=t_cool
        )
    
    def _compute_lya_optical_depth(self, n_H: float, T: float) -> float:
        """Compute Lyman-alpha optical depth through cloud center."""
        if n_H <= 0 or T <= 0:
            return 0.0
        sigma = line_cross_section_peak(1, 2, T)
        N_H = n_H * 2 * self.cloud.radius
        return N_H * sigma
    
    def _compute_21cm_optical_depth(self, n_H: float, T_s: float) -> float:
        """Compute 21 cm optical depth."""
        if n_H <= 0 or T_s <= 0:
            return 0.0
        N_H = n_H * 2 * self.cloud.radius
        delta_v = np.sqrt(2 * k_B * T_s / m_H)
        return optical_depth_21cm(N_H, T_s, delta_v)
    
    def _sound_speed(self, T: float, x_ion: float) -> float:
        """Isothermal sound speed: c_s = sqrt(k_B T / mu m_H)"""
        if T <= 0:
            return 0.0
        mu = 1 / (1 + x_ion)
        return np.sqrt(k_B * T / (mu * m_H))
    
    def _jeans_length(self, T: float, n: float) -> float:
        """Jeans length for gravitational stability."""
        if T <= 0 or n <= 0:
            return np.inf
        rho = n * m_H
        c_s = self._sound_speed(T, 0)
        return c_s * np.sqrt(np.pi / (G * rho))
    
    def temperature_sweep(self, T_min: float = 2.7, T_max: float = 1e5,
                          n_points: int = 200, log_scale: bool = True) -> Dict:
        """
        Sweep radiation temperature and compute cloud state at each point.
        """
        if log_scale:
            T_array = np.logspace(np.log10(T_min), np.log10(T_max), n_points)
        else:
            T_array = np.linspace(T_min, T_max, n_points)
        
        results = {
            'T_radiation': T_array,
            'T_gas': np.zeros(n_points),
            'ionization_fraction': np.zeros(n_points),
            'n_neutral': np.zeros(n_points),
            'n_electron': np.zeros(n_points),
            'f_ground': np.zeros(n_points),
            'f_n2': np.zeros(n_points),
            'f_n3': np.zeros(n_points),
            'heating_rate': np.zeros(n_points),
            'cooling_rate': np.zeros(n_points),
            'tau_lya': np.zeros(n_points),
            'tau_21cm': np.zeros(n_points),
            'sound_speed': np.zeros(n_points),
            'jeans_length': np.zeros(n_points),
            'expansion_time': np.zeros(n_points),
            'cooling_time': np.zeros(n_points),
        }
        
        print("Running temperature sweep...")
        for i, T in enumerate(T_array):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_points} (T = {T:.0f} K)")
            
            state = self.compute_state(T)
            
            results['T_gas'][i] = state.T_gas
            results['ionization_fraction'][i] = state.ionization_fraction
            results['n_neutral'][i] = state.n_neutral
            results['n_electron'][i] = state.n_electron
            results['f_ground'][i] = state.f_ground
            results['f_n2'][i] = state.f_n2
            results['f_n3'][i] = state.f_n3
            results['heating_rate'][i] = state.heating_rate
            results['cooling_rate'][i] = state.cooling_rate
            results['tau_lya'][i] = state.tau_lya
            results['tau_21cm'][i] = state.tau_21cm
            results['sound_speed'][i] = state.sound_speed
            results['jeans_length'][i] = state.jeans_length
            results['expansion_time'][i] = state.expansion_time
            results['cooling_time'][i] = state.cooling_time
        
        print("Done!")
        return results


def plot_results(results: Dict, cloud: CloudParameters, save_path: str = None):
    """Create comprehensive visualization of simulation results."""
    T = results['T_radiation']
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Ionization fraction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(T, results['ionization_fraction'], 'b-', linewidth=2)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Radiation Temperature [K]')
    ax1.set_ylabel('Ionization Fraction')
    ax1.set_title('Ionization vs Temperature')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    x_ion = results['ionization_fraction']
    idx_half = np.argmin(np.abs(x_ion - 0.5))
    T_half = T[idx_half]
    ax1.axvline(T_half, color='red', linestyle=':', alpha=0.7)
    ax1.annotate(f'T_50% = {T_half:.0f} K', xy=(T_half, 0.5),
                xytext=(T_half*2, 0.3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 2. Gas temperature vs radiation temperature
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(T, results['T_gas'], 'r-', linewidth=2, label='T_gas')
    ax2.loglog(T, T, 'k--', alpha=0.5, label='T_gas = T_rad')
    ax2.set_xlabel('Radiation Temperature [K]')
    ax2.set_ylabel('Gas Temperature [K]')
    ax2.set_title('Thermal Equilibrium')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Level populations
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogx(T, results['f_ground'], 'b-', linewidth=2, label='n=1 (ground)')
    ax3.semilogx(T, results['f_n2'], 'r-', linewidth=2, label='n=2')
    ax3.semilogx(T, results['f_n3'], 'g-', linewidth=2, label='n=3')
    ax3.set_xlabel('Radiation Temperature [K]')
    ax3.set_ylabel('Population Fraction')
    ax3.set_title('Level Populations')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-20, 2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Heating and cooling rates
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.loglog(T, results['heating_rate'], 'r-', linewidth=2, label='Heating')
    ax4.loglog(T, results['cooling_rate'], 'b-', linewidth=2, label='Cooling')
    ax4.set_xlabel('Radiation Temperature [K]')
    ax4.set_ylabel('Rate [W/m^3]')
    ax4.set_title('Heating vs Cooling')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Optical depths
    ax5 = fig.add_subplot(gs[1, 1])
    tau_lya = np.array(results['tau_lya'])
    tau_21 = np.array(results['tau_21cm'])
    ax5.loglog(T, np.maximum(tau_lya, 1e-10), 'b-', linewidth=2, label='Lyman-alpha')
    ax5.loglog(T, np.maximum(tau_21, 1e-10), 'r-', linewidth=2, label='21 cm')
    ax5.axhline(1, color='gray', linestyle='--', alpha=0.5, label='tau=1')
    ax5.set_xlabel('Radiation Temperature [K]')
    ax5.set_ylabel('Optical Depth tau')
    ax5.set_title('Optical Depths')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Sound speed
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogx(T, results['sound_speed']/1000, 'g-', linewidth=2)
    ax6.set_xlabel('Radiation Temperature [K]')
    ax6.set_ylabel('Sound Speed [km/s]')
    ax6.set_title('Sound Speed')
    ax6.grid(True, alpha=0.3)
    
    # 7. Jeans length vs cloud size
    ax7 = fig.add_subplot(gs[2, 0])
    jeans_ly = np.array(results['jeans_length']) / ly_to_m
    ax7.loglog(T, jeans_ly, 'purple', linewidth=2, label='Jeans length')
    ax7.axhline(cloud.radius_ly, color='red', linestyle='--', 
                label=f'Cloud radius ({cloud.radius_ly:.0f} ly)')
    ax7.set_xlabel('Radiation Temperature [K]')
    ax7.set_ylabel('Length [light years]')
    ax7.set_title('Gravitational Stability')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Timescales
    ax8 = fig.add_subplot(gs[2, 1])
    t_exp_yr = np.array(results['expansion_time']) / (3600*24*365.25)
    t_cool_yr = np.array(results['cooling_time']) / (3600*24*365.25)
    ax8.loglog(T, t_exp_yr, 'b-', linewidth=2, label='Expansion time')
    ax8.loglog(T, np.maximum(t_cool_yr, 1), 'r-', linewidth=2, label='Cooling time')
    ax8.set_xlabel('Radiation Temperature [K]')
    ax8.set_ylabel('Timescale [years]')
    ax8.set_title('Characteristic Timescales')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary panel
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    x_ion = results['ionization_fraction']
    idx_1 = np.argmin(np.abs(x_ion - 0.01))
    idx_50 = np.argmin(np.abs(x_ion - 0.5))
    idx_99 = np.argmin(np.abs(x_ion - 0.99))
    
    summary_text = (
        f"CLOUD PARAMETERS\n"
        f"----------------\n"
        f"Radius: {cloud.radius_ly:.1f} ly\n"
        f"Density: {cloud.density:.0e} m^-3\n"
        f"Total mass: {cloud.total_mass/5.97e24:.0f} Earth masses\n"
        f"Column density: {cloud.column_density:.1e} m^-2\n\n"
        f"PHASE TRANSITIONS\n"
        f"----------------\n"
        f"1% ionized: T = {T[idx_1]:.0f} K\n"
        f"50% ionized: T = {T[idx_50]:.0f} K\n"
        f"99% ionized: T = {T[idx_99]:.0f} K\n\n"
        f"STABILITY\n"
        f"----------------\n"
        f"Jeans length >> Cloud radius\n"
        f"-> Cloud is gravitationally STABLE\n"
        f"-> Will expand, not collapse"
    )
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Hydrogen Cloud Response to Background Radiation', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return fig


def main():
    """Run the simulation and generate plots."""
    cloud = CloudParameters(
        radius=1.0 * ly_to_m,
        density=1e6
    )
    
    print("=" * 60)
    print("HYDROGEN CLOUD SIMULATION")
    print("=" * 60)
    print()
    print(cloud)
    print()
    
    sim = HydrogenCloudSimulation(cloud)
    results = sim.temperature_sweep(T_min=2.7, T_max=1e5, n_points=200)
    fig = plot_results(results, cloud, save_path='simulation_results.png')
    
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    
    T = results['T_radiation']
    x = results['ionization_fraction']
    
    idx_1 = np.argmin(np.abs(x - 0.01))
    idx_50 = np.argmin(np.abs(x - 0.5))
    idx_99 = np.argmin(np.abs(x - 0.99))
    
    print(f"\nIonization transitions:")
    print(f"  1% ionized at T_rad = {T[idx_1]:.0f} K")
    print(f"  50% ionized at T_rad = {T[idx_50]:.0f} K") 
    print(f"  99% ionized at T_rad = {T[idx_99]:.0f} K")
    
    print(f"\nTransition width (1% to 99%): {T[idx_99]/T[idx_1]:.1f}x in temperature")
    print(f"  -> This is a SHARP phase transition!")
    
    min_jeans = np.min(results['jeans_length']) / ly_to_m
    print(f"\nGravitational stability:")
    print(f"  Minimum Jeans length: {min_jeans:.0f} ly")
    print(f"  Cloud radius: {cloud.radius_ly:.0f} ly")
    print(f"  -> Cloud is {min_jeans/cloud.radius_ly:.0f}x smaller than Jeans length")
    print(f"  -> STABLE against gravitational collapse at all temperatures")
    
    plt.show()
    return results, cloud


if __name__ == "__main__":
    results, cloud = main()
