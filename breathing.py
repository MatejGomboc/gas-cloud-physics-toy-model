"""
Time-varying radiation field simulations.

What happens when we wobble the background radiation temperature?
Watch ionization "breathe" in and out as the cloud responds!

Key insight: To see actual breathing, you need:
1. High density (n ~ 10^10 m^-3) for fast recombination (t_rec ~ 10 years)
2. Truly cold "off" temperatures (< 1000 K) where ionization is negligible
3. Period comparable to or longer than t_rec

At low densities (n ~ 10^6 m^-3), t_rec ~ 100,000 years and the cloud
stays ionized through any realistic temperature oscillation.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from constants import k_B, ly_to_m, T_CMB
from thermal import photoionization_rate, recombination_coefficient_B

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a1a'
plt.rcParams['axes.facecolor'] = '#0a0a1a'
plt.rcParams['savefig.facecolor'] = '#0a0a1a'


def evolve_0d(n_total, T_rad_func, T_gas, t_array):
    """
    Simple 0D ionization evolution (no spatial structure).
    
    Solves: dx/dt = Gamma(T) * (1-x) - alpha * n * x^2
    
    Parameters:
        n_total: hydrogen density [m^-3]
        T_rad_func: function T_rad(t) returning temperature at time t
        T_gas: gas kinetic temperature [K] (for recombination rate)
        t_array: time points to evaluate [s]
    
    Returns:
        x: ionization fraction at each time point
    """
    alpha = recombination_coefficient_B(T_gas)
    
    def rhs(x, t):
        x = np.clip(x, 1e-15, 1-1e-15)
        T_rad = T_rad_func(t)
        Gamma = photoionization_rate(T_rad)
        return Gamma * (1 - x) - alpha * n_total * x**2
    
    x0 = 1e-10
    solution = odeint(rhs, x0, t_array)
    
    return solution.flatten()


def plot_breathing(t, x, T_rad, title, save_path=None, t_rec=None):
    """
    Plot breathing simulation results.
    
    Parameters:
        t: time array [s]
        x: ionization fraction array
        T_rad: radiation temperature array [K]
        title: plot title
        save_path: if provided, save figure
        t_rec: recombination timescale [s] for annotation
    """
    t_yr = t / (3600 * 24 * 365.25)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                              gridspec_kw={'height_ratios': [1, 2]})
    
    # Temperature panel
    axes[0].fill_between(t_yr, 0, T_rad/1000, color='#ff6b6b', alpha=0.4)
    axes[0].plot(t_yr, T_rad/1000, color='#ff6b6b', linewidth=2)
    axes[0].set_ylabel('T_rad [1000 K]', fontsize=12, color='white')
    axes[0].set_title(title, fontsize=14, color='white')
    axes[0].set_xlim(t_yr.min(), t_yr.max())
    axes[0].axhline(T_CMB/1000, color='cyan', linestyle=':', alpha=0.5, 
                    label=f'CMB ({T_CMB:.1f} K)')
    axes[0].legend(loc='upper right')
    
    # Ionization panel
    axes[1].fill_between(t_yr, 0, x, color='#00ff88', alpha=0.3)
    axes[1].plot(t_yr, x, color='#00ff88', linewidth=2.5)
    axes[1].set_xlabel('Time [years]', fontsize=12, color='white')
    axes[1].set_ylabel('Ionization Fraction', fontsize=12, color='white')
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].axhline(0.5, color='white', linestyle='--', alpha=0.3)
    
    if t_rec is not None:
        t_rec_yr = t_rec / (3600 * 24 * 365.25)
        axes[1].annotate(f't_rec = {t_rec_yr:.1f} yr', xy=(0.02, 0.95), 
                         xycoords='axes fraction', fontsize=11, color='yellow')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def run_breathing_demo(save_prefix='breathing'):
    """
    Run demonstration of ionization breathing with various temperature patterns.
    
    Returns dictionary of results.
    """
    print("=" * 60)
    print("Ionization Breathing Demonstration")
    print("=" * 60)
    
    # Use high density for fast recombination
    n_total = 1e10  # m^-3
    T_gas = 8000  # K
    
    alpha = recombination_coefficient_B(T_gas)
    t_rec = 1.0 / (alpha * n_total)
    t_rec_yr = t_rec / (3600 * 24 * 365.25)
    print(f"\nDensity: n = {n_total:.0e} m^-3")
    print(f"Recombination time: t_rec = {t_rec_yr:.2f} years")
    print()
    
    results = {}
    
    # =====================================================
    # 1. Slow sinusoidal breathing
    # =====================================================
    print("1. Slow sinusoidal (period = 10 * t_rec)")
    
    T_base, T_amp = 15000, 14000  # 1k to 29k K
    period = 10 * t_rec
    t_max = 5 * period
    t = np.linspace(0, t_max, 2000)
    
    def T_sine(t):
        return T_base + T_amp * np.sin(2 * np.pi * t / period)
    
    x1 = evolve_0d(n_total, T_sine, T_gas, t)
    T1 = np.array([T_sine(ti) for ti in t])
    
    results['slow_sine'] = {'t': t, 'x': x1, 'T': T1}
    plot_breathing(t, x1, T1, 
                   'Slow Sinusoidal Breathing (Period = 10 * t_rec)',
                   f'{save_prefix}_sine_slow.png', t_rec)
    
    # =====================================================
    # 2. Fast sinusoidal - cloud can't follow
    # =====================================================
    print("\n2. Fast sinusoidal (period = 0.5 * t_rec)")
    
    period_fast = 0.5 * t_rec
    t_max_fast = 20 * period_fast
    t2 = np.linspace(0, t_max_fast, 2000)
    
    def T_sine_fast(t):
        return T_base + T_amp * np.sin(2 * np.pi * t / period_fast)
    
    x2 = evolve_0d(n_total, T_sine_fast, T_gas, t2)
    T2 = np.array([T_sine_fast(ti) for ti in t2])
    
    results['fast_sine'] = {'t': t2, 'x': x2, 'T': T2}
    plot_breathing(t2, x2, T2,
                   'Fast Oscillation - Cloud Cannot Follow! (Period = 0.5 * t_rec)',
                   f'{save_prefix}_sine_fast.png', t_rec)
    
    # =====================================================
    # 3. Square wave pulsing
    # =====================================================
    print("\n3. Square wave pulses (50% duty cycle)")
    
    T_hot, T_cold = 30000, 100
    period_sq = 5 * t_rec
    t3 = np.linspace(0, 6 * period_sq, 2000)
    
    def T_square(t):
        phase = (t % period_sq) / period_sq
        return T_hot if phase < 0.5 else T_cold
    
    x3 = evolve_0d(n_total, T_square, T_gas, t3)
    T3 = np.array([T_square(ti) for ti in t3])
    
    results['square'] = {'t': t3, 'x': x3, 'T': T3}
    plot_breathing(t3, x3, T3,
                   'Square Wave Pulsing (Period = 5 * t_rec)',
                   f'{save_prefix}_square.png', t_rec)
    
    # =====================================================
    # 4. Short hot pulses
    # =====================================================
    print("\n4. Short hot pulses (10% duty cycle)")
    
    period_p = 3 * t_rec
    t4 = np.linspace(0, 8 * period_p, 2000)
    
    def T_pulses(t):
        phase = (t % period_p) / period_p
        return 50000 if phase < 0.1 else 100
    
    x4 = evolve_0d(n_total, T_pulses, T_gas, t4)
    T4 = np.array([T_pulses(ti) for ti in t4])
    
    results['pulses'] = {'t': t4, 'x': x4, 'T': T4}
    plot_breathing(t4, x4, T4,
                   'Short Hot Pulses (10% duty cycle, 50k K peaks)',
                   f'{save_prefix}_pulses.png', t_rec)
    
    # =====================================================
    # 5. Summary figure
    # =====================================================
    print("\n5. Creating summary figure...")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    
    data = [
        (t, x1, T1, 'Slow Sine'),
        (t2, x2, T2, 'Fast Sine'),
        (t3, x3, T3, 'Square Wave'),
        (t4, x4, T4, 'Short Pulses')
    ]
    
    for i, (ti, xi, Ti, label) in enumerate(data):
        t_yr = ti / (3600 * 24 * 365.25)
        
        # Temperature
        axes[i, 0].fill_between(t_yr, 0, Ti/1000, color='#ff6b6b', alpha=0.4)
        axes[i, 0].plot(t_yr, Ti/1000, color='#ff6b6b', linewidth=1.5)
        axes[i, 0].set_ylabel('T [kK]', fontsize=10)
        axes[i, 0].set_title(f'{label}: Temperature', fontsize=11, color='white')
        
        # Ionization
        axes[i, 1].fill_between(t_yr, 0, xi, color='#00ff88', alpha=0.3)
        axes[i, 1].plot(t_yr, xi, color='#00ff88', linewidth=2)
        axes[i, 1].set_ylabel('x_ion', fontsize=10)
        axes[i, 1].set_ylim(-0.02, 1.05)
        axes[i, 1].set_title(f'{label}: Ionization Response', fontsize=11, color='white')
    
    axes[3, 0].set_xlabel('Time [years]', fontsize=11)
    axes[3, 1].set_xlabel('Time [years]', fontsize=11)
    
    plt.suptitle(f'Ionization Breathing at n = 10^10 m^-3 (t_rec = {t_rec_yr:.1f} yr)', 
                 fontsize=14, color='white', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_summary.png")
    plt.show()
    
    results['params'] = {
        'n_total': n_total,
        'T_gas': T_gas,
        't_rec': t_rec
    }
    
    print("\nDone!")
    return results


# Custom temperature functions for user experiments
def make_sine_temperature(T_base, T_amplitude, period):
    """Create a sinusoidal temperature function."""
    def T(t):
        return T_base + T_amplitude * np.sin(2 * np.pi * t / period)
    return T


def make_square_temperature(T_hot, T_cold, period, duty_cycle=0.5):
    """Create a square wave temperature function."""
    def T(t):
        phase = (t % period) / period
        return T_hot if phase < duty_cycle else T_cold
    return T


def make_pulse_temperature(T_peak, T_base, period, pulse_width):
    """Create a pulsed temperature function."""
    def T(t):
        phase = (t % period) / period
        return T_peak if phase < pulse_width else T_base
    return T


if __name__ == "__main__":
    run_breathing_demo()
