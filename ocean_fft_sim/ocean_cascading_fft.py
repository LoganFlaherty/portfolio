import ocean_fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import time

# Constants
N = ocean_fft.N
L = 256.0
L2 = 1024.0
PHILLIPS_A = 18.0 # Amps

# Vars
wind_speed = 60.0 # Controls wave length
wind_angle = np.radians(45.0) # Direction of waves
is_running = True

def plot_ocean():
    # Init plot
    plt.ion()
    fig = plt.figure(figsize=(8, 8), layout='tight')
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.mpl_connect('close_event', on_close)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0)) # Remove the gray background panes
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    base_x, base_z = np.meshgrid(np.linspace(0, L2, N), np.linspace(0, L2, N))

    # Init ocean
    k_x, k_z, k_mag, h0_k, _ = ocean_fft.init_ocean(L, wind_speed, wind_angle, PHILLIPS_A)
    k_x2, k_z2, k_mag2, h0_k2, _ = ocean_fft.init_ocean(L2, wind_speed, wind_angle, PHILLIPS_A)

    # FFT loop
    start_time = time.time()
    while is_running:
        t = (time.time() - start_time) * ocean_fft.SPEED
        height_sum, disp_x_sum, disp_z_sum = sum_fft_maps(h0_k, k_x, k_z, k_mag, 
                                                          h0_k2, k_x2, k_z2, k_mag2, t)

        # Construct plot grid
        final_x = base_x + disp_x_sum
        final_z = base_z + disp_z_sum
        final_y = height_sum

        # Plotting
        ax.clear() # Clear previous frame
        ax.plot_wireframe(final_x, final_z, final_y, rstride=2, cstride=2, color='#1da2d8', linewidth=0.5)
        ax.set_xlim(0, L2)
        ax.set_ylim(0, L2)
        ax.set_zlim(-20, 20)
        ax.set_title(f"Ocean Simulation     Time: {t:.2f}s")
        plt.draw()
        plt.pause(0.01)

def sum_fft_maps(h0_k, k_x, k_z, k_mag, h0_k2, k_x2, k_z2, k_mag2, t):
    # Calculate the three maps for the two layers for this time step
    height, disp_x, disp_z = ocean_fft.get_wave_maps(h0_k, k_x, k_z, k_mag, t, ocean_fft.CHOPPINESS)
    height2, disp_x2, disp_z2 = ocean_fft.get_wave_maps(h0_k2, k_x2, k_z2, k_mag2, t, ocean_fft.CHOPPINESS)

    # Tile smaller map
    height_tiled = tile_layer(height, L, L2, N)
    dx_tiled = tile_layer(disp_x, L, L2, N)
    dz_tiled = tile_layer(disp_z, L, L2, N)
    
    # Combine layers
    height_sum = height_tiled + height2
    disp_x_sum = dx_tiled + disp_x2
    disp_z_sum = dz_tiled + disp_z2
    return height_sum, disp_x_sum, disp_z_sum

def tile_layer(small_map, small_L, large_L, N):
    # Create the grid for the large view
    x = np.linspace(0, large_L, N)
    z = np.linspace(0, large_L, N)
    grid_x, grid_z = np.meshgrid(x, z)
    
    # Convert large coordinates to 'indices' in the small map
    scale_ratio = small_L / N
    indices_x = (grid_x % small_L) / scale_ratio
    indices_z = (grid_z % small_L) / scale_ratio
    
    # Sample the small map at these repeating indices
    coords = np.array([indices_z.ravel(), indices_x.ravel()])
    tiled = map_coordinates(small_map, coords, order=1, mode='wrap')
    return tiled.reshape((N, N))

def on_close(event):
    global is_running
    is_running = False

if __name__ == '__main__':
    plot_ocean()