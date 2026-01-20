# Ocean FFT Sim
This project focuses on the dynamic simulation of an ocean using the Fast Fourier Transform (FFT) technique, 
which models the oceans surface as a sum of sin waves across various frequencies and directions. 
Furthermore, the algorithm takes into account wind speed, wing angle, and time dialation (the time relative to the simulation denoted as "speed" in the source). 
It was built using Python 3.11 with standard math libraries and matplot for visualization.

To run the simulation for yourself, you'll be using the file "ocean_cascasing_fft.py". 
This script creates a dynamic "real" ocean simulation by summing mutiple simulation layers of different resolutions created by the "ocean_fft.py" script. 
The movement of the ocean can be adjusted with the vars "wind_speed" and "wind_angle" found in "ocean_cascasing_fft.py", while "speed" is found in "ocean_fft.py".

The strucutre of the simulation could be improved greatly, and will be, but it is still in a work in progress evident by the "fft_data.py" script.  
The goal of this project is to ultimately use this simulation to generate a dataset to train an ML model on to hopefully achieve a model capable of completely simulating the ocean itself given the config variables. 
This is in hopes to educate myself, but to also measure the performance differences between running the algorithm itself versus the model. 
If the model does show improved performance, I could see it being useful in game development applications after reaching a production quality.
