"""
Generate visualization plots for ionization front propagation.
Creates publication-quality figures with dark theme styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from constants import ly_to_m, DEFAULT_DENSITY, DEFAULT_RADIUS_M
from time_evolution import evolve_ionization, track_front_position, find_ionization_front

# Set nice style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0a0a1a'
plt.rcParams['axes.facecolor'] = '#0a0a1a'
plt.rcParams['savefig.facecolor'] = '#0a0a1a'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.color'] = '#ffffff'


def create_custom_cmap():
    """Create a custom colormap for ionization (neutral=blue, ionized=orange/yellow)."""
    colors = ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd', 
              '#ffd166', '#ef8354', '#d90429']
    return LinearSegmentedColormap.from_list('ionization', colors, N=256)


def plot_spectacular_evolution(result, save_path='ionization_spectacular.png'):
    """Create a spectacular visualization of ionization front evolution."""
    
    r = result['r_centers'] / ly_to_m
    t = result['t'] / (3600 * 24 * 365.25)
    x_ion = result['x_ion']
    
    fig = plt.figure(figsize=(16, 14))
    
    # Main space-time diagram
    ax_main = fig.add_axes([0.08, 0.35, 0.55, 0.55])
    
    cmap = create_custom_cmap()
    T, R = np.meshgrid(t, r)
    pcm = ax_main.pcolormesh(T, R, x_ion, shading='gouraud', cmap=cmap, vmin=0, vmax=1)
    
    # Track ionization front
    t_front, r_front = track_front_position(result)
    ax_main.plot(t_front / (3600 * 24 * 365.25), r_front / ly_to_m, 
                 color='#00ff88', linewidth=3, linestyle='--', 
                 label='Ionization Front (50%)', alpha=0.9)
    
    # Add glow effect to front line
    for lw, alpha in [(8, 0.1), (5, 0.2), (3, 0.4)]:
        ax_main.plot(t_front / (3600 * 24 * 365.25), r_front / ly_to_m, 
                     color='#00ff88', linewidth=lw, alpha=alpha)
    
    ax_main.set_xlabel('Time [years]', fontsize=14, color='white')
    ax_main.set_ylabel('Radius [light-years]', fontsize=14, color='white')
    ax_main.set_title('Ionization Front Propagation Through Hydrogen Cloud', 
                      fontsize=16, color='white', pad=15)
    ax_main.legend(loc='upper right', fontsize=11, framealpha=0.7)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.65, 0.35, 0.02, 0.55])
    cbar = plt.colorbar(pcm, cax=cbar_ax)
    cbar.set_label('Ionization Fraction', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Radial profile snapshots
    ax_profiles = fig.add_axes([0.73, 0.35, 0.24, 0.55])
    
    n_times = len(t)
    time_indices = [0, n_times//6, n_times//3, n_times//2, 2*n_times//3, -1]
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(time_indices)))
    
    for i, idx in enumerate(time_indices):
        t_val = t[idx]
        if t_val < 1:
            label = f't = {t_val*365.25:.1f} days'
        elif t_val < 100:
            label = f't = {t_val:.1f} yr'
        else:
            label = f't = {t_val:.0f} yr'
        ax_profiles.fill_between(r, 0, x_ion[:, idx], color=colors[i], alpha=0.3)
        ax_profiles.plot(r, x_ion[:, idx], color=colors[i], label=label, linewidth=2)
    
    ax_profiles.set_xlabel('Radius [ly]', fontsize=12, color='white')
    ax_profiles.set_ylabel('Ionization Fraction', fontsize=12, color='white')
    ax_profiles.set_title('Radial Profiles', fontsize=13, color='white')
    ax_profiles.legend(loc='best', fontsize=9, framealpha=0.7)
    ax_profiles.set_ylim(-0.02, 1.02)
    ax_profiles.set_xlim(0, r.max())
    
    # Front position plot
    ax_front = fig.add_axes([0.08, 0.08, 0.38, 0.20])
    t_yrs = t_front / (3600 * 24 * 365.25)
    r_ly = r_front / ly_to_m
    
    ax_front.fill_between(t_yrs, 0, r_ly, color='#00ff88', alpha=0.2)
    ax_front.plot(t_yrs, r_ly, color='#00ff88', linewidth=2.5)
    ax_front.set_xlabel('Time [years]', fontsize=11, color='white')
    ax_front.set_ylabel('Front Position [ly]', fontsize=11, color='white')
    ax_front.set_title('I-Front Penetration Depth', fontsize=12, color='white')
    
    # Front velocity plot
    ax_vel = fig.add_axes([0.55, 0.08, 0.42, 0.20])
    dt = np.diff(t_front)
    dr_front = -np.diff(r_front)
    v_front = np.zeros_like(r_front)
    v_front[1:] = dr_front / dt
    v_front[0] = v_front[1]
    v_front_kms = v_front / 1000
    
    ax_vel.fill_between(t_yrs, 0, v_front_kms, color='#ff6b6b', alpha=0.2)
    ax_vel.plot(t_yrs, v_front_kms, color='#ff6b6b', linewidth=2.5)
    ax_vel.set_xlabel('Time [years]', fontsize=11, color='white')
    ax_vel.set_ylabel('Front Velocity [km/s]', fontsize=11, color='white')
    ax_vel.set_title('I-Front Velocity (inward)', fontsize=12, color='white')
    
    # Add peak velocity annotation
    v_max = np.max(v_front_kms)
    t_max = t_yrs[np.argmax(v_front_kms)]
    ax_vel.annotate(f'Peak: {v_max:.0f} km/s', xy=(t_max, v_max), 
                    xytext=(t_max + t_yrs.max()*0.1, v_max*0.8),
                    color='#ff6b6b', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b', alpha=0.7))
    
    # Info box
    info_text = (
        f"Cloud Parameters:\n"
        f"  Radius: {result['R_cloud']/ly_to_m:.1f} light-year\n"
        f"  Density: n = {result['n_total']:.0e} m^-3\n"
        f"  T_radiation: {result['T_rad']:,} K\n"
        f"  T_gas: {result['T_gas']:,} K\n"
        f"  t_ionization: {result['t_ion']:.2e} s\n"
        f"  t_recombination: {result['t_rec']:.2e} s"
    )
    fig.text(0.02, 0.02, info_text, fontsize=9, color='#aaaaaa',
             family='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8, edgecolor='#333355'))
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_temperature_comparison(temperatures=None, n_total=DEFAULT_DENSITY, 
                                R_cloud=DEFAULT_RADIUS_M, save_path='temperature_comparison.png'):
    """Compare ionization fronts at different radiation temperatures."""
    
    if temperatures is None:
        temperatures = [15000, 25000, 40000, 60000, 100000]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main comparison plot
    ax1 = fig.add_axes([0.08, 0.45, 0.58, 0.48])
    ax2 = fig.add_axes([0.72, 0.45, 0.25, 0.48])
    ax3 = fig.add_axes([0.08, 0.08, 0.40, 0.30])
    ax4 = fig.add_axes([0.55, 0.08, 0.42, 0.30])
    
    # Custom colormap for temperatures
    temp_colors = plt.cm.hot(np.linspace(0.3, 0.9, len(temperatures)))
    
    results = []
    for i, T_rad in enumerate(temperatures):
        print(f"Computing T_rad = {T_rad:,} K...")
        T_gas = min(T_rad, 20000)
        result = evolve_ionization(
            n_total=n_total, T_rad=T_rad, T_gas=T_gas,
            R_cloud=R_cloud, N_shells=150
        )
        results.append(result)
        
        t, r_front = track_front_position(result)
        t_yrs = t / (3600 * 24 * 365.25)
        r_ly = r_front / ly_to_m
        
        # Space-time curves
        ax1.plot(t_yrs, r_ly, color=temp_colors[i], linewidth=2.5, 
                 label=f'{T_rad/1000:.0f}k K')
        
        # Final profiles
        r = result['r_centers'] / ly_to_m
        ax2.plot(r, result['x_ion'][:, -1], color=temp_colors[i], linewidth=2)
    
    ax1.set_xlabel('Time [years]', fontsize=12, color='white')
    ax1.set_ylabel('I-Front Position [light-years]', fontsize=12, color='white')
    ax1.set_title('Ionization Front vs Radiation Temperature', 
                  fontsize=14, color='white')
    ax1.legend(loc='upper right', fontsize=10, title='T_rad', title_fontsize=10)
    
    ax2.set_xlabel('Radius [ly]', fontsize=11, color='white')
    ax2.set_ylabel('x_ion', fontsize=11, color='white')
    ax2.set_title('Final State', fontsize=12, color='white')
    ax2.set_ylim(-0.02, 1.02)
    
    # Timescales vs temperature
    T_range = np.array(temperatures)
    t_ions = []
    t_recs = []
    final_depths = []
    
    for result in results:
        t_ions.append(result['t_ion'])
        t_recs.append(result['t_rec'])
        t, r_front = track_front_position(result)
        final_depths.append((result['R_cloud'] - r_front[-1]) / ly_to_m)
    
    ax3.semilogy(T_range/1000, t_ions, 'o-', color='#00ff88', linewidth=2, 
                 markersize=8, label='t_ionization')
    ax3.semilogy(T_range/1000, t_recs, 's-', color='#ff6b6b', linewidth=2, 
                 markersize=8, label='t_recombination')
    ax3.set_xlabel('Radiation Temperature [1000 K]', fontsize=11, color='white')
    ax3.set_ylabel('Timescale [s]', fontsize=11, color='white')
    ax3.set_title('Characteristic Timescales', fontsize=12, color='white')
    ax3.legend(fontsize=10)
    
    # Final penetration depth
    ax4.bar(np.arange(len(temperatures)), final_depths, color=temp_colors, 
            edgecolor='white', linewidth=0.5)
    ax4.set_xticks(np.arange(len(temperatures)))
    ax4.set_xticklabels([f'{T/1000:.0f}k' for T in temperatures])
    ax4.set_xlabel('T_rad [K]', fontsize=11, color='white')
    ax4.set_ylabel('Penetration Depth [ly]', fontsize=11, color='white')
    ax4.set_title('Final I-Front Penetration', fontsize=12, color='white')
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_3d_evolution(result, save_path='evolution_3d.png'):
    """Create a 3D visualization of the ionization evolution."""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    r = result['r_centers'] / ly_to_m
    t = result['t'] / (3600 * 24 * 365.25)
    x_ion = result['x_ion']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for cleaner visualization
    r_idx = np.linspace(0, len(r)-1, 50, dtype=int)
    t_idx = np.linspace(0, len(t)-1, 40, dtype=int)
    
    R, T_mesh = np.meshgrid(r[r_idx], t[t_idx])
    X = x_ion[r_idx, :][:, t_idx].T
    
    cmap = create_custom_cmap()
    surf = ax.plot_surface(T_mesh, R, X, cmap=cmap, alpha=0.85,
                           linewidth=0, antialiased=True)
    
    # Add ionization front line on surface
    t_front, r_front = track_front_position(result)
    t_front_yrs = t_front / (3600 * 24 * 365.25)
    r_front_ly = r_front / ly_to_m
    ax.plot(t_front_yrs, r_front_ly, 0.5 * np.ones_like(r_front_ly), 
            color='#00ff88', linewidth=3, label='I-Front')
    
    ax.set_xlabel('Time [years]', fontsize=11, color='white', labelpad=10)
    ax.set_ylabel('Radius [ly]', fontsize=11, color='white', labelpad=10)
    ax.set_zlabel('Ionization Fraction', fontsize=11, color='white', labelpad=10)
    ax.set_title('3D View: Ionization Front Evolution', fontsize=14, color='white', pad=20)
    
    # Set view angle
    ax.view_init(elev=25, azim=45)
    
    # Style the 3D axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='x_ion', pad=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def plot_schematic_diagram(save_path='cloud_schematic.png'):
    """Create a schematic diagram showing the ionization process."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw the cloud as nested circles
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Background cloud (neutral)
    r_cloud = 1.0
    ax.fill(r_cloud * np.cos(theta), r_cloud * np.sin(theta), 
            color='#1b4965', alpha=0.6, label='Neutral H')
    
    # Ionization front
    r_front = 0.5
    ax.fill(r_cloud * np.cos(theta), r_cloud * np.sin(theta), 
            color='#1b4965', alpha=0.3)
    ax.fill(r_front * np.cos(theta), r_front * np.sin(theta), 
            color='#ff6b35', alpha=0.0)
    
    # Outer ionized region
    ax.fill_between(np.cos(theta) * r_cloud, np.sin(theta) * r_cloud, 
                    np.cos(theta) * r_front, np.sin(theta) * r_front,
                    color='#ff9500', alpha=0.5, label='Ionized H+')
    
    # Ionization front ring
    ax.plot(r_front * np.cos(theta), r_front * np.sin(theta), 
            color='#00ff88', linewidth=3, linestyle='--', label='I-Front')
    
    # Incoming radiation arrows
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        r_start = 1.4
        r_end = 1.05
        ax.annotate('', xy=(r_end * np.cos(angle), r_end * np.sin(angle)),
                    xytext=(r_start * np.cos(angle), r_start * np.sin(angle)),
                    arrowprops=dict(arrowstyle='->', color='#ffd700', lw=2))
    
    # Labels
    ax.text(0, 0, 'Neutral\nCore', ha='center', va='center', 
            fontsize=14, color='white', fontweight='bold')
    ax.text(0.75, 0.75, 'H+ + e-', ha='center', va='center', 
            fontsize=12, color='white')
    ax.text(1.5, 0, 'UV Radiation\n(hv > 13.6 eV)', ha='center', va='center', 
            fontsize=11, color='#ffd700')
    
    # Title and legend
    ax.set_title('Ionization Front in a Hydrogen Cloud', fontsize=16, color='white', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add physics explanation
    physics_text = (
        "Physics:\n"
        "- UV photons (E > 13.6 eV) ionize neutral hydrogen\n"
        "- Front propagates inward as outer layers become transparent\n"
        "- Equilibrium: ionization rate = recombination rate\n"
        "- Front velocity ~ 10-1000 km/s depending on flux"
    )
    ax.text(-1.7, -1.1, physics_text, fontsize=10, color='#aaaaaa',
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9, edgecolor='#333355'))
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Ionization Front Visualizations")
    print("=" * 60)
    print()
    
    # Run main simulation
    print("Running main simulation (T_rad = 30,000 K)...")
    result = evolve_ionization(
        n_total=1e6,
        T_rad=30000,
        T_gas=10000,
        R_cloud=ly_to_m,
        N_shells=150
    )
    print(f"  Status: {result['message']}")
    print()
    
    # Generate plots
    print("Generating spectacular evolution plot...")
    plot_spectacular_evolution(result, 'ionization_spectacular.png')
    
    print("\nGenerating temperature comparison...")
    plot_temperature_comparison(save_path='temperature_comparison.png')
    
    print("\nGenerating 3D evolution plot...")
    plot_3d_evolution(result, 'evolution_3d.png')
    
    print("\nGenerating schematic diagram...")
    plot_schematic_diagram('cloud_schematic.png')
    
    print("\nAll plots generated successfully!")
