import ocean_fft
import ocean_cascading_fft
import numpy as np
import os

# Constants
OUTPUT_DIR = "ocean_data_shards"
SHARD_SIZE = 500  # Save to disk every 500 samples
WIND_SPEEDS = np.linspace(10.0, 70.0, 10)
WIND_ANGLES = np.linspace(0, 2*np.pi, 8)
TIME_STEPS  = np.linspace(0, 60.0, 20)
PHILLIPS_A = ocean_cascading_fft.PHILLIPS_A
L = ocean_cascading_fft.L
L2 = ocean_cascading_fft.L2

def gen_cfft_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    k_x, k_z, k_mag, h0_k, noise = ocean_fft.init_ocean(L, 10.0, np.radians(0.0), PHILLIPS_A)
    k_x2, k_z2, k_mag2, h0_k2, noise2 = ocean_fft.init_ocean(L2, 10.0, np.radians(0.0), PHILLIPS_A)

    x = []
    y = []
    shard_cnt = 0
    sample_cnt = 0
    for wind_speed in WIND_SPEEDS:
        for wind_angle in WIND_ANGLES:

            ph = ocean_fft.phillips_spectrum(k_x, k_z, wind_speed, wind_angle, PHILLIPS_A)
            ph2 = ocean_fft.phillips_spectrum(k_x, k_z, wind_speed, wind_angle, PHILLIPS_A)
            h0_k = np.sqrt(2) * noise * np.sqrt(ph)
            h0_k2 = np.sqrt(2) * noise2 * np.sqrt(ph2)

            for t in TIME_STEPS:
                height_sum, _, _ = ocean_cascading_fft.sum_fft_maps(h0_k, k_x, k_z, k_mag,
                                                                    h0_k2, k_x2, k_z2, k_mag2, t)
            
                # Pack data
                # Normalize inputs to be in 0-1 range for better ML convergence
                x.append([wind_speed/70.0, wind_angle/(2*np.pi), t/60.0])
                # Normalize height roughly to -1 to 1 (assuming z limit of 20m)
                y.append(height_sum.astype(np.float32)/20.0)

                sample_cnt += 1
                if sample_cnt == SHARD_SIZE:
                    save_shard(shard_cnt, x, y)
                    shard_cnt += 1
                    sample_cnt = 0
                    x = []
                    y = []
    
    # Save any remaining data
    if sample_cnt > 0:
        save_shard(shard_cnt, x, y)

def save_shard(index, x, y):
    filename = os.path.join(OUTPUT_DIR, f"shard_{index:04d}.npz")
    
    # Convert list to numpy arrays
    X = np.array(x, dtype=np.float32)
    Y = np.array(y, dtype=np.float32)
    Y = Y[:, np.newaxis, :, :] # Add the channel dimension (1)
    
    np.savez_compressed(filename, x=X, y=Y)
    print(f"Saved {filename} | Shard Shape: {Y.shape}")

if __name__ == '__main__':
    gen_cfft_data()