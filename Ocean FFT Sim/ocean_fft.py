import numpy as np
from scipy.fft import ifft2

# Constants
G = 9.81 # Gravity
N = 256 # Grid resolution
CHOPPINESS = 1.0 # Sharpness of wave peak, which fluxes x
SPEED = 1.5 # Speed of the sim

def init_ocean(L, wind_speed, wind_angle, phillips_a):
    # Create 1D frequency arrays
    k_values = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    k_x, k_z = np.meshgrid(k_values, k_values)
    k_mag = np.sqrt(k_x ** 2 + k_z ** 2)

    ph_k = phillips_spectrum(k_x, k_z, wind_speed, wind_angle, phillips_a)
    noise = gen_noise()
    h0_k = np.sqrt(2) * noise * np.sqrt(ph_k)

    return k_x, k_z, k_mag, h0_k, noise

def phillips_spectrum(k_x, k_z, wind_speed, wind_dir_angle, a_const):
    # Define the characteristic length L using wind speed (U)
    L_cutoff = wind_speed**2 / G
    
    k_mag_sq = k_x**2 + k_z**2
    k_mag = np.sqrt(k_mag_sq)

    wind_x = np.cos(wind_dir_angle)
    wind_z = np.sin(wind_dir_angle)

    # Normalize k for dot product
    k_dot_W = (k_x * wind_x + k_z * wind_z) / np.where(k_mag == 0, 1.0, k_mag) 
    
    ph = np.zeros_like(k_mag_sq)
    mask = k_mag > 1e-6 # Avoid division by zero at k=0

    # The practical Phillips spectrum for graphics incorporating L
    term1 = np.exp(-1.0 / (k_mag_sq[mask] * L_cutoff**2))
    term2 = 1.0 / k_mag_sq[mask]**2
    term3 = k_dot_W[mask]**2 # Waves moving against wind are damped
    
    ph[mask] = a_const * term1 * term2 * term3
    ph *= np.exp(-k_mag_sq * 0.1) # Damping
    return ph

def gen_noise():
    # Generate two independent complex Gaussian fields
    gaussian_rand_real = np.random.normal(0.0, 1.0, (N, N))
    gaussian_rand_imag = np.random.normal(0.0, 1.0, (N, N))
    xi = gaussian_rand_real + 1j * gaussian_rand_imag
    return xi

def get_wave_maps(h0_k, k_x, k_z, k_mag, t, choppiness):
    omega_k = np.sqrt(G * k_mag)
    h_k_t = h0_k * np.exp(1j * omega_k * t)
    
    # Height Map - IFFT on h_k_t
    height_map = np.real(ifft2(h_k_t))
    
    # Displacement Maps (Choppiness)
    k_mag_masked = np.where(k_mag == 0, 1.0, k_mag)
    
    # Note: 1j * k / |k| corresponds to the gradient in Fourier space
    Dx_k_t = h_k_t * (1j * k_x / k_mag_masked) * choppiness
    Dz_k_t = h_k_t * (1j * k_z / k_mag_masked) * choppiness

    disp_x_map = np.real(ifft2(Dx_k_t))
    disp_z_map = np.real(ifft2(Dz_k_t))
    
    return height_map, disp_x_map, disp_z_map