"""
Focused analysis of the ionization phase transition.
Demonstrates the sharpness of the transition and compares to true phase transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from constants import k_B, E_ion, E_ion_eV, ly_to_m
from equilibrium import ionization_fraction, saha_ratio, boltzmann_ratio


def analyze_ionization_transition():
    """
    Detailed analysis of the ionization transition.
    """
    n_total = 1e6  # Our cloud density
    
    # Fine temperature grid around the transition
    T_coarse = np.logspace(2, 5, 500)
    T_fine = np.linspace(2000, 6000, 1000)
    
    # Compute ionization
    x_coarse = np.array([ionization_fraction(n_total, T) for T in T_coarse])
    x_fine = np.array([ionization_fraction(n_total, T) for T in T_fine])
    
    # Find transition temperatures
    def find_transition_T(x_array, T_array, target):
        idx = np.argmin(np.abs(x_array - target))
        return T_array[idx]
    
    T_01 = find_transition_T(x_fine, T_fine, 0.01)
    T_10 = find_transition_T(x_fine, T_fine, 0.10)
    T_50 = find_transition_T(x_fine, T_fine, 0.50)
    T_90 = find_transition_T(x_fine, T_fine, 0.90)
    T_99 = find_transition_T(x_fine, T_fine, 0.99)
    
    print("=" * 60)
    print("IONIZATION TRANSITION ANALYSIS")
    print("=" * 60)
    print(f"\nCloud density: n = {n_total:.0e} m^-3")
    print(f"\nCharacteristic ionization temperature: E_ion/k_B = {E_ion/k_B:.0f} K")
    print(f"\nTransition temperatures:")
    print(f"  1% ionized:  T = {T_01:.0f} K")
    print(f"  10% ionized: T = {T_10:.0f} K")
    print(f"  50% ionized: T = {T_50:.0f} K")
    print(f"  90% ionized: T = {T_90:.0f} K")
    print(f"  99% ionized: T = {T_99:.0f} K")
    
    print(f"\nTransition sharpness:")
    print(f"  10% -> 90%: dT = {T_90 - T_10:.0f} K (ratio: {T_90/T_10:.2f})")
    print(f"  1% -> 99%:  dT = {T_99 - T_01:.0f} K (ratio: {T_99/T_01:.2f})")
    
    # Compare to a "soft" transition (linear) and Fermi function
    T_c = T_50  # Critical temperature
    width = (T_90 - T_10) / 2
    
    x_fermi = 1 / (1 + np.exp(-(T_fine - T_c) / (width/4)))
    x_linear = np.clip((T_fine - T_01) / (T_99 - T_01), 0, 1)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full temperature range (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(T_coarse, x_coarse, 'b-', linewidth=2.5)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(T_50, color='red', linestyle=':', alpha=0.7)
    ax1.fill_between([T_01, T_99], 0, 1, alpha=0.2, color='orange', 
                     label='Transition zone')
    ax1.set_xlabel('Radiation Temperature [K]', fontsize=12)
    ax1.set_ylabel('Ionization Fraction', fontsize=12)
    ax1.set_title('Ionization vs Temperature (Log Scale)', fontsize=14)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom on transition (linear scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(T_fine, x_fine, 'b-', linewidth=2.5, label='Saha equilibrium')
    ax2.plot(T_fine, x_fermi, 'r--', linewidth=2, alpha=0.7, label='Fermi function')
    ax2.plot(T_fine, x_linear, 'g:', linewidth=2, alpha=0.7, label='Linear')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(T_50, color='gray', linestyle=':', alpha=0.3)
    
    for T_mark, x_mark, label in [(T_10, 0.1, '10%'), (T_50, 0.5, '50%'), (T_90, 0.9, '90%')]:
        ax2.plot(T_mark, x_mark, 'ko', markersize=8)
        ax2.annotate(label, xy=(T_mark, x_mark), xytext=(T_mark+100, x_mark), fontsize=10)
    
    ax2.set_xlabel('Temperature [K]', fontsize=12)
    ax2.set_ylabel('Ionization Fraction', fontsize=12)
    ax2.set_title('Zoom on Transition Zone', fontsize=14)
    ax2.set_xlim(2000, 6000)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Derivative (shows sharpness)
    dx_dT = np.gradient(x_fine, T_fine)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(T_fine, dx_dT * 1000, 'purple', linewidth=2)
    ax3.axvline(T_50, color='red', linestyle=':', alpha=0.7, label=f'T_50% = {T_50:.0f} K')
    ax3.set_xlabel('Temperature [K]', fontsize=12)
    ax3.set_ylabel('dx/dT x 1000 [K^-1]', fontsize=12)
    ax3.set_title('Transition Sharpness (Derivative)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    idx_peak = np.argmax(dx_dT)
    T_peak = T_fine[idx_peak]
    dx_peak = dx_dT[idx_peak]
    ax3.annotate(f'Peak: dx/dT = {dx_peak*1000:.2f}/1000 K\nat T = {T_peak:.0f} K',
                xy=(T_peak, dx_peak*1000), xytext=(T_peak+500, dx_peak*1000*0.8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
    
    # Plot 4: Comparison to phase transitions
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    magnetization_text = (
        "ANALOGY TO PHASE TRANSITIONS\n"
        "============================\n\n"
        "The ionization transition is analogous to:\n\n"
        "1. FERROMAGNETIC TRANSITION\n"
        "   Order parameter: Magnetization M -> Ionization x\n"
        "   Control parameter: Temperature T\n"
        "   Critical point: Curie temp -> ~3200 K\n\n"
        "2. LIQUID-GAS TRANSITION\n"
        "   Saha equation has same form as\n"
        "   law of mass action for:\n"
        "   H <-> H+ + e-\n\n"
        "KEY DIFFERENCE:\n"
        "============================\n"
        "* True phase transitions have SINGULARITIES\n"
        "  in thermodynamic quantities\n"
        "* Our transition is a CROSSOVER:\n"
        "  - Smooth but exponentially sharp\n"
        "  - No latent heat\n"
        "  - No symmetry breaking\n"
        "* Still: transition width << T_c\n"
        f"  Width/T_c = {(T_90-T_10)/T_50:.2f} = {(T_90-T_10)/T_50*100:.0f}%\n\n"
        "This is why we see 'step-like' behavior!"
    )
    
    ax4.text(0.05, 0.95, magnetization_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'The Ionization "Phase Transition" at n = {n_total:.0e} m^-3', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('ionization_transition.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to ionization_transition.png")
    
    plt.tight_layout()
    return fig


def density_dependence():
    """
    Show how the transition temperature depends on density.
    This is a key prediction of the Saha equation!
    """
    densities = np.logspace(0, 12, 13)  # 1 to 10^12 m^-3
    T_transitions = []
    
    for n in densities:
        T_array = np.linspace(1000, 50000, 1000)
        x_array = np.array([ionization_fraction(n, T) for T in T_array])
        idx = np.argmin(np.abs(x_array - 0.5))
        T_transitions.append(T_array[idx])
    
    T_transitions = np.array(T_transitions)
    
    print("\n" + "=" * 60)
    print("DENSITY DEPENDENCE OF TRANSITION")
    print("=" * 60)
    print("\nT_50% vs density:")
    for n, T in zip(densities, T_transitions):
        print(f"  n = {n:>10.0e} m^-3 -> T_50% = {T:>6.0f} K")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(densities, T_transitions, 'bo-', linewidth=2, markersize=10)
    
    ax.axvline(1e6, color='red', linestyle='--', alpha=0.7)
    ax.axhline(T_transitions[6], color='red', linestyle='--', alpha=0.7)
    ax.plot(1e6, T_transitions[6], 'r*', markersize=20, label='Our cloud')
    
    ax.set_xlabel('Density [m^-3]', fontsize=12)
    ax.set_ylabel('50% Ionization Temperature [K]', fontsize=12)
    ax.set_title('Transition Temperature vs Density\n(Saha Equation Prediction)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    ax.annotate('Higher density\n-> Higher T needed\nto ionize',
               xy=(1e10, T_transitions[10]), xytext=(1e8, T_transitions[10]+3000),
               fontsize=11, arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.savefig('density_dependence.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to density_dependence.png")
    
    return fig


def electronic_excitation_transition():
    """
    Analyze the electronic excitation transition (n=1 -> n=2).
    """
    T_array = np.logspace(3, 5.5, 500)
    
    n2_n1 = np.array([boltzmann_ratio(2, 1, T) for T in T_array])
    n3_n1 = np.array([boltzmann_ratio(3, 1, T) for T in T_array])
    
    T_exc = 10.2 * 1.6e-19 / k_B  # 10.2 eV in Kelvin
    
    print("\n" + "=" * 60)
    print("ELECTRONIC EXCITATION TRANSITION")
    print("=" * 60)
    print(f"\nExcitation energy (1->2): 10.2 eV = {T_exc:.0f} K")
    print(f"\nPopulation ratios n2/n1:")
    
    for T in [5000, 10000, 20000, 50000, 100000]:
        ratio = boltzmann_ratio(2, 1, T)
        print(f"  T = {T:>6} K: n2/n1 = {ratio:.2e}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(T_array, n2_n1, 'b-', linewidth=2, label='n2/n1')
    ax.loglog(T_array, n3_n1, 'r-', linewidth=2, label='n3/n1')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Equal population')
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5, label='1% of ground')
    ax.axvline(T_exc, color='green', linestyle='--', alpha=0.7, label=f'T* = {T_exc:.0f} K')
    
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('Population Ratio', fontsize=12)
    ax.set_title('Electronic Excitation: Another Sharp Transition', fontsize=14)
    ax.set_ylim(1e-20, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('excitation_transition.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to excitation_transition.png")
    
    return fig


if __name__ == "__main__":
    fig1 = analyze_ionization_transition()
    fig2 = density_dependence()
    fig3 = electronic_excitation_transition()
    plt.show()
