"""
Time evolution of ionization fronts.

This module implements 1D spherical radiative transfer with time-dependent
ionization dynamics, allowing visualization of how ionization fronts
propagate through the hydrogen cloud.

Key physics:
- Radial discretization of the cloud into shells
- Photoionization rate attenuated by optical depth
- Recombination rate from Case B coefficient
- Time integration of ionization fraction per shell
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from constants import (h, c, k_B, nu_ion, sigma_ion_0, 
                       DEFAULT_RADIUS_M, DEFAULT_DENSITY, ly_to_m)
from thermal import (photoionization_rate, recombination_coefficient_B,
                     recombination_coefficient_A)
from blackbody import mean_intensity
from atomic_physics import photoionization_cross_section


# ============================================
# Spatial Grid
# ============================================

def create_radial_grid(R_cloud, N_shells, spacing='linear'):
    """
    Create radial grid for the cloud.
    
    Parameters:
        R_cloud: cloud radius [m]
        N_shells: number of radial shells
        spacing: 'linear' or 'log' (log puts more resolution at center)
    
    Returns:
        r_edges: shell boundary radii [m], shape (N_shells + 1,)
        r_centers: shell center radii [m], shape (N_shells,)
        dr: shell thicknesses [m], shape (N_shells,)
        volumes: shell volumes [m³], shape (N_shells,)
    """
    if spacing == 'log':
        # Logarithmic spacing (more resolution near center)
        r_min = R_cloud / 1000  # Avoid r=0
        r_edges = np.logspace(np.log10(r_min), np.log10(R_cloud), N_shells + 1)
    else:
        # Linear spacing
        r_edges = np.linspace(0, R_cloud, N_shells + 1)
    
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    
    # Shell volumes: V = (4π/3)(r_outer³ - r_inner³)
    volumes = (4 * np.pi / 3) * (r_edges[1:]**3 - r_edges[:-1]**3)
    
    return r_edges, r_centers, dr, volumes


# ============================================
# Optical Depth and Radiative Transfer
# ============================================

def compute_optical_depth_inward(n_H_shells, dr, nu=None):
    """
    Compute optical depth from outside inward (radiation entering cloud).
    
    τ(r) = ∫_r^R n_H(r') σ(ν) dr'
    
    We integrate from outside (shell N-1) inward (shell 0).
    
    Parameters:
        n_H_shells: neutral hydrogen density per shell [m⁻³], shape (N_shells,)
        dr: shell thicknesses [m], shape (N_shells,)
        nu: frequency [Hz], if None uses threshold frequency
    
    Returns:
        tau: optical depth at each shell center (from outside), shape (N_shells,)
    """
    if nu is None:
        sigma = sigma_ion_0
    else:
        sigma = photoionization_cross_section(nu)
    
    N = len(n_H_shells)
    tau = np.zeros(N)
    
    # Integrate from outside in
    # tau[N-1] = 0 (outermost shell, no attenuation yet)
    # tau[i] = sum of n_H * sigma * dr for shells i+1 to N-1
    cumulative = 0.0
    for i in range(N - 1, -1, -1):
        tau[i] = cumulative
        # Add this shell's contribution for the next (inner) shell
        cumulative += n_H_shells[i] * sigma * dr[i]
    
    return tau


def attenuated_photoionization_rate(T_rad, tau):
    """
    Photoionization rate attenuated by optical depth.
    
    Γ_local = Γ_∞ × exp(-τ)
    
    where Γ_∞ is the unattenuated rate for blackbody at T_rad.
    
    Parameters:
        T_rad: radiation temperature [K]
        tau: optical depth (from outside), shape (N_shells,)
    
    Returns:
        Gamma: local photoionization rate per shell [s⁻¹], shape (N_shells,)
    """
    Gamma_inf = photoionization_rate(T_rad)
    return Gamma_inf * np.exp(-tau)


# ============================================
# Rate Equations
# ============================================

def ionization_rate_equation(x_ion, n_total, T_rad, T_gas, tau):
    """
    Time derivative of ionization fraction.
    
    dx/dt = Γ(τ) × (1 - x) - α(T) × n_total × x²
    
    First term: photoionization of neutrals
    Second term: recombination of ions+electrons
    
    Parameters:
        x_ion: ionization fraction per shell, shape (N_shells,)
        n_total: total density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K] (for recombination coefficient)
        tau: optical depth per shell, shape (N_shells,)
    
    Returns:
        dx_dt: time derivative of ionization fraction, shape (N_shells,)
    """
    # Photoionization rate (attenuated)
    Gamma = attenuated_photoionization_rate(T_rad, tau)
    
    # Recombination coefficient (Case B - Lyman photons trapped)
    alpha = recombination_coefficient_B(T_gas)
    
    # Rate equation
    # dx/dt = Γ × (1 - x) - α × n_e × x
    # With n_e = n_total × x:
    # dx/dt = Γ × (1 - x) - α × n_total × x²
    
    ionization_term = Gamma * (1 - x_ion)
    recombination_term = alpha * n_total * x_ion**2
    
    return ionization_term - recombination_term


def compute_neutral_density(x_ion, n_total):
    """Neutral density from ionization fraction."""
    return n_total * (1 - x_ion)


# ============================================
# Time Evolution
# ============================================

def evolve_ionization(n_total, T_rad, T_gas, R_cloud, N_shells=100,
                      t_max=None, x_init=None, rtol=1e-6, atol=1e-10,
                      max_step=None):
    """
    Evolve ionization state of the cloud over time.
    
    Parameters:
        n_total: total hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas kinetic temperature [K]
        R_cloud: cloud radius [m]
        N_shells: number of radial shells
        t_max: maximum evolution time [s], if None estimated automatically
        x_init: initial ionization fraction (scalar or array), default 1e-10
        rtol, atol: tolerances for ODE solver
        max_step: maximum time step [s]
    
    Returns:
        result: dictionary with evolution data
    """
    # Create grid
    r_edges, r_centers, dr, volumes = create_radial_grid(R_cloud, N_shells)
    
    # Initial conditions
    if x_init is None:
        x0 = np.full(N_shells, 1e-10)  # Nearly neutral
    elif np.isscalar(x_init):
        x0 = np.full(N_shells, x_init)
    else:
        x0 = np.array(x_init)
    
    # Estimate characteristic timescales
    Gamma_0 = photoionization_rate(T_rad)
    alpha = recombination_coefficient_B(T_gas)
    
    if Gamma_0 > 0:
        t_ion = 1.0 / Gamma_0  # Ionization timescale
    else:
        t_ion = np.inf
    
    t_rec = 1.0 / (alpha * n_total)  # Recombination timescale
    
    t_char = min(t_ion, t_rec)
    
    if t_max is None:
        # Evolve for ~10 characteristic times or until equilibrium
        t_max = 20 * t_char
    
    if max_step is None:
        max_step = t_char / 10
    
    # ODE system
    def rhs(t, x):
        # Compute current optical depth
        n_H = compute_neutral_density(x, n_total)
        tau = compute_optical_depth_inward(n_H, dr)
        
        # Rate equation
        dxdt = ionization_rate_equation(x, n_total, T_rad, T_gas, tau)
        return dxdt
    
    # Solve
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 200)
    
    solution = solve_ivp(
        rhs, t_span, x0,
        method='LSODA',  # Good for stiff problems
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )
    
    # Compute derived quantities
    n_H_final = compute_neutral_density(solution.y[:, -1], n_total)
    tau_final = compute_optical_depth_inward(n_H_final, dr)
    
    return {
        'r_centers': r_centers,
        'r_edges': r_edges,
        'dr': dr,
        'volumes': volumes,
        't': solution.t,
        'x_ion': solution.y,  # Shape: (N_shells, N_times)
        'n_total': n_total,
        'T_rad': T_rad,
        'T_gas': T_gas,
        'R_cloud': R_cloud,
        't_ion': t_ion,
        't_rec': t_rec,
        'tau_final': tau_final,
        'success': solution.success,
        'message': solution.message
    }


def find_ionization_front(x_ion, r_centers, x_threshold=0.5):
    """
    Find position of ionization front (where x = threshold).
    
    Parameters:
        x_ion: ionization fraction profile, shape (N_shells,)
        r_centers: radial positions [m], shape (N_shells,)
        x_threshold: ionization fraction defining the front
    
    Returns:
        r_front: front position [m], or NaN if not found
    """
    # Find where ionization crosses threshold (from outside in)
    # Front is between last shell above threshold and first below
    
    above = x_ion >= x_threshold
    
    if np.all(above):
        return 0.0  # Fully ionized
    if np.all(~above):
        return r_centers[-1]  # Not yet ionized
    
    # Find transition
    idx = np.where(above[:-1] & ~above[1:])[0]
    if len(idx) == 0:
        idx = np.where(~above[:-1] & above[1:])[0]
    
    if len(idx) > 0:
        i = idx[0]
        # Linear interpolation
        x1, x2 = x_ion[i], x_ion[i+1]
        r1, r2 = r_centers[i], r_centers[i+1]
        if x2 != x1:
            r_front = r1 + (x_threshold - x1) * (r2 - r1) / (x2 - x1)
        else:
            r_front = 0.5 * (r1 + r2)
        return r_front
    
    return np.nan


def track_front_position(result, x_threshold=0.5):
    """
    Track ionization front position over time.
    
    Parameters:
        result: output from evolve_ionization
        x_threshold: ionization fraction defining the front
    
    Returns:
        t: time array [s]
        r_front: front position array [m]
    """
    r_centers = result['r_centers']
    x_ion = result['x_ion']
    t = result['t']
    
    r_front = np.array([
        find_ionization_front(x_ion[:, i], r_centers, x_threshold)
        for i in range(len(t))
    ])
    
    return t, r_front


# ============================================
# Visualization
# ============================================

def plot_ionization_evolution(result, save_path=None):
    """
    Plot ionization front evolution as a space-time diagram.
    
    Parameters:
        result: output from evolve_ionization
        save_path: if provided, save figure to this path
    """
    r = result['r_centers'] / ly_to_m  # Convert to light-years
    t = result['t'] / (3600 * 24 * 365.25)  # Convert to years
    x_ion = result['x_ion']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Space-time diagram
    ax1 = axes[0, 0]
    T, R = np.meshgrid(t, r)
    pcm = ax1.pcolormesh(T, R, x_ion, shading='auto', cmap='plasma',
                         vmin=0, vmax=1)
    plt.colorbar(pcm, ax=ax1, label='Ionization fraction')
    ax1.set_xlabel('Time [years]')
    ax1.set_ylabel('Radius [light-years]')
    ax1.set_title('Ionization Front Propagation')
    
    # Add front tracking
    t_front, r_front = track_front_position(result)
    ax1.plot(t_front / (3600 * 24 * 365.25), r_front / ly_to_m, 
             'w--', linewidth=2, label='I-front (x=0.5)')
    ax1.legend(loc='upper right')
    
    # 2. Radial profiles at different times
    ax2 = axes[0, 1]
    n_times = len(t)
    time_indices = [0, n_times//4, n_times//2, 3*n_times//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    for i, idx in enumerate(time_indices):
        label = f't = {t[idx]:.1e} yr'
        ax2.plot(r, x_ion[:, idx], color=colors[i], label=label, linewidth=2)
    
    ax2.set_xlabel('Radius [light-years]')
    ax2.set_ylabel('Ionization fraction')
    ax2.set_title('Radial Ionization Profiles')
    ax2.legend(loc='best')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # 3. Front position vs time
    ax3 = axes[1, 0]
    ax3.plot(t_front / (3600 * 24 * 365.25), r_front / ly_to_m, 
             'b-', linewidth=2)
    ax3.set_xlabel('Time [years]')
    ax3.set_ylabel('Front position [light-years]')
    ax3.set_title('Ionization Front Position')
    ax3.grid(True, alpha=0.3)
    
    # 4. Front velocity
    ax4 = axes[1, 1]
    dt = np.diff(t_front)
    dr_front = -np.diff(r_front)  # Negative because front moves inward
    v_front = np.zeros_like(r_front)
    v_front[1:] = dr_front / dt
    v_front[0] = v_front[1]
    
    # Convert to km/s
    v_front_kms = v_front / 1000
    
    ax4.plot(t_front / (3600 * 24 * 365.25), v_front_kms, 'r-', linewidth=2)
    ax4.set_xlabel('Time [years]')
    ax4.set_ylabel('Front velocity [km/s]')
    ax4.set_title('Ionization Front Velocity')
    ax4.grid(True, alpha=0.3)
    
    # Add info text
    info_text = (
        f"Cloud: R = {result['R_cloud']/ly_to_m:.2f} ly, "
        f"n = {result['n_total']:.0e} m⁻³\n"
        f"T_rad = {result['T_rad']:.0f} K, T_gas = {result['T_gas']:.0f} K\n"
        f"t_ion = {result['t_ion']:.2e} s, t_rec = {result['t_rec']:.2e} s"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_multi_temperature(temperatures, n_total=DEFAULT_DENSITY, 
                           R_cloud=DEFAULT_RADIUS_M, save_path=None):
    """
    Compare ionization front evolution at different radiation temperatures.
    
    Parameters:
        temperatures: list of radiation temperatures [K]
        n_total: total density [m⁻³]
        R_cloud: cloud radius [m]
        save_path: if provided, save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(temperatures)))
    
    for i, T_rad in enumerate(temperatures):
        print(f"Evolving T_rad = {T_rad} K...")
        
        # Assume T_gas ~ T_rad for simplicity
        T_gas = min(T_rad, 20000)  # Cap gas temperature
        
        result = evolve_ionization(
            n_total=n_total,
            T_rad=T_rad,
            T_gas=T_gas,
            R_cloud=R_cloud,
            N_shells=100
        )
        
        t, r_front = track_front_position(result)
        
        # Convert units
        t_years = t / (3600 * 24 * 365.25)
        r_ly = r_front / ly_to_m
        
        # Front position
        axes[0].plot(t_years, r_ly, color=colors[i], linewidth=2,
                     label=f'T = {T_rad} K')
        
        # Final profile
        r = result['r_centers'] / ly_to_m
        axes[1].plot(r, result['x_ion'][:, -1], color=colors[i], 
                     linewidth=2, label=f'T = {T_rad} K')
    
    axes[0].set_xlabel('Time [years]')
    axes[0].set_ylabel('Front position [light-years]')
    axes[0].set_title('Ionization Front Propagation')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Radius [light-years]')
    axes[1].set_ylabel('Ionization fraction')
    axes[1].set_title('Final Ionization Profiles')
    axes[1].legend(loc='best')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Ionization Fronts: n = {n_total:.0e} m⁻³, R = {R_cloud/ly_to_m:.1f} ly',
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================
# Equilibrium Timescale Analysis
# ============================================

def strömgren_radius(n_total, T_rad, Q=None):
    """
    Estimate Strömgren radius for reference.
    
    R_S = (3 Q / (4π α_B n²))^(1/3)
    
    For a blackbody source, Q is the ionizing photon rate.
    This is a rough estimate for comparison.
    
    Parameters:
        n_total: hydrogen density [m⁻³]
        T_rad: radiation temperature [K]
        Q: ionizing photon rate [s⁻¹], if None estimated from T_rad
    
    Returns:
        R_S: Strömgren radius [m]
    """
    alpha_B = recombination_coefficient_B(T_rad)
    
    if Q is None:
        # Estimate Q from photoionization rate
        Gamma = photoionization_rate(T_rad)
        # For a uniform cloud, Q ~ Gamma * V * n (rough)
        # This is just for order-of-magnitude
        Q = Gamma * 1e50  # Arbitrary normalization
    
    R_S = (3 * Q / (4 * np.pi * alpha_B * n_total**2))**(1/3)
    
    return R_S


def ionization_equilibrium_time(n_total, T_rad, T_gas):
    """
    Estimate time to reach ionization equilibrium.
    
    Parameters:
        n_total: density [m⁻³]
        T_rad: radiation temperature [K]
        T_gas: gas temperature [K]
    
    Returns:
        t_eq: equilibrium timescale [s]
    """
    Gamma = photoionization_rate(T_rad)
    alpha = recombination_coefficient_B(T_gas)
    
    if Gamma <= 0:
        return np.inf
    
    # Equilibrium reached when dx/dt → 0
    # Timescale is roughly 1/Gamma for ionization-dominated
    # or 1/(alpha * n) for recombination-dominated
    
    t_ion = 1.0 / Gamma if Gamma > 0 else np.inf
    t_rec = 1.0 / (alpha * n_total)
    
    return min(t_ion, t_rec)


# ============================================
# Main - Demo and Testing
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Time Evolution of Ionization Fronts")
    print("=" * 60)
    print()
    
    # Cloud parameters
    n_total = 1e6  # m⁻³ (same as default in other modules)
    R_cloud = 1.0 * ly_to_m  # 1 light year
    
    print(f"Cloud parameters:")
    print(f"  Radius: {R_cloud/ly_to_m:.1f} light-years")
    print(f"  Density: {n_total:.0e} m⁻³")
    print()
    
    # Test with moderately hot radiation field
    T_rad = 30000  # K - hot enough to ionize
    T_gas = 10000  # K - typical H II region temperature
    
    print(f"Radiation field: T_rad = {T_rad} K")
    print(f"Gas temperature: T_gas = {T_gas} K")
    print()
    
    # Characteristic timescales
    t_eq = ionization_equilibrium_time(n_total, T_rad, T_gas)
    print(f"Equilibrium timescale: {t_eq:.2e} s = {t_eq/(3600*24*365.25):.2e} years")
    print()
    
    print("Running time evolution...")
    result = evolve_ionization(
        n_total=n_total,
        T_rad=T_rad,
        T_gas=T_gas,
        R_cloud=R_cloud,
        N_shells=100
    )
    
    print(f"Integration: {result['message']}")
    print(f"Time span: 0 to {result['t'][-1]/(3600*24*365.25):.2e} years")
    print()
    
    # Track front
    t, r_front = track_front_position(result)
    print(f"Initial front position: {r_front[0]/ly_to_m:.3f} ly")
    print(f"Final front position: {r_front[-1]/ly_to_m:.3f} ly")
    print()
    
    # Plot
    print("Generating plots...")
    plot_ionization_evolution(result, save_path='ionization_front_evolution.png')
    
    # Multi-temperature comparison
    print("\nComparing different radiation temperatures...")
    plot_multi_temperature(
        temperatures=[10000, 20000, 30000, 50000],
        n_total=n_total,
        R_cloud=R_cloud,
        save_path='ionization_front_comparison.png'
    )
    
    print("\nDone!")
