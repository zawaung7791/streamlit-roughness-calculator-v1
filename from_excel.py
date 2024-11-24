import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew
from scipy.stats import kurtosis

# Function to perform Gaussian smoothing on a 2D array
def gaussian_smoothing(data, sigma):
    return gaussian_filter(data, sigma=sigma)

# Example 2D array
data = xl("input!B3:QI451")


# Perform Gaussian smoothing with sigma = 1
smoothed_data = gaussian_smoothing(data, sigma=xl("D1"))

# Function to calculate the average deviation of a 2D array
def average_deviation_2d(array):
    mean = np.mean(array)
    deviations = np.abs(array - mean)
    avg_deviation = np.mean(deviations)
    return avg_deviation

# Example 2D array
data = xl("B1")

# Calculate the average deviation
avg_dev = average_deviation_2d(data)

# Function to calculate the standard deviation of a 2D array
def standard_deviation_2d(array):
    return np.std(array)

# Example 2D array
data = xl("B1")

# Calculate the standard deviation
std_dev = standard_deviation_2d(data)



# Function to calculate the skewness of a 2D array
def skewness_2d(array):
    # Flatten the 2D array to 1D
    flattened_array = array.flatten()
    return skew(flattened_array)

# Example 2D array
data = xl("B1")

# Calculate the skewness
skewness_value = skewness_2d(data)



# Function to calculate the kurtosis of a 2D array
def kurtosis_2d(array):
    # Flatten the 2D array to 1D
    flattened_array = array.flatten()
    return kurtosis(flattened_array)

# Example 2D array
data = xl("B1")

# Calculate the kurtosis
kurtosis_value = kurtosis_2d(data)

# Function to calculate the average of the average absolute slope for the rows in a 2D array
def average_of_average_absolute_slope(array):
    avg_absolute_slopes = []
    for row in array:
        # Calculate the absolute slope for each row
        absolute_slope = np.abs(np.diff(row))
        # Calculate the average absolute slope for the row
        avg_absolute_slope = np.mean(absolute_slope)
        avg_absolute_slopes.append(avg_absolute_slope)
    # Calculate the average of the average absolute slopes
    avg_of_avg_absolute_slope = np.mean(avg_absolute_slopes)
    return avg_of_avg_absolute_slope

# Example 2D array
data = xl("B1")
# Calculate the average of the average absolute slope for the rows
avg_of_avg_abs_slope = average_of_average_absolute_slope(data)

# Function to calculate the average of the average absolute slope for the columns in a 2D array
def average_of_average_absolute_slope_columns(array):
    avg_absolute_slopes = []
    for col in array.T:  # Transpose the array to iterate over columns
        # Calculate the absolute slope for each column
        absolute_slope = np.abs(np.diff(col))
        # Calculate the average absolute slope for the column
        avg_absolute_slope = np.mean(absolute_slope)
        avg_absolute_slopes.append(avg_absolute_slope)
    # Calculate the average of the average absolute slopes
    avg_of_avg_absolute_slope = np.mean(avg_absolute_slopes)
    return avg_of_avg_absolute_slope

# Example 2D array
data = xl("B1")
# Calculate the average of the average absolute slope for the columns
avg_of_avg_abs_slope_columns = average_of_average_absolute_slope_columns(data)

=2*PI()*B2*F1/B6 # wavelength x
# B2 = AveDev Ra, F1 = spacing, B6 = slope x

=2*PI()*B2*F1/B7
# B2 = AveDev Ra, F1 = spacing, B7 = slope y

# Function to calculate the standard deviation for each row of a 2D array and find the average
def average_std_deviation_rows(array):
    std_devs = []
    for row in array:
        # Calculate the standard deviation for each row
        std_dev = np.std(row)
        std_devs.append(std_dev)
    # Calculate the average of the standard deviations
    avg_std_dev = np.mean(std_devs)
    return avg_std_dev

# Example 2D array
data = xl("B1")

# Calculate the average of the standard deviations for the rows
avg_std_dev_rows = average_std_deviation_rows(data)

# Function to calculate the standard deviation for each column of a 2D array and find the average
def average_std_deviation_columns(array):
    std_devs = []
    for col in array.T:  # Transpose the array to iterate over columns
        # Calculate the standard deviation for each column
        std_dev = np.std(col)
        std_devs.append(std_dev)
    # Calculate the average of the standard deviations
    avg_std_dev = np.mean(std_devs)
    return avg_std_dev

# Example 2D array
data = xl("B1")
# Calculate the average of the standard deviations for the columns
avg_std_dev_columns = average_std_deviation_columns(data)



# Function to calculate the skew for each row of a 2D array
def skew_rows(array):
    skews = []
    for row in array:
        # Calculate the skew for each row
        row_skew = skew(row)
        skews.append(row_skew)
    return skews

# Example 2D array
data = xl("B1")
# Calculate the skew for each row
skew_values = np.mean(skew_rows(data))

# Function to calculate the skew for each column of a 2D array
def skew_columns(array):
    skews = []
    for col in array.T:  # Transpose the array to iterate over columns
        # Calculate the skew for each column
        col_skew = skew(col)
        skews.append(col_skew)
    return skews

# Example 2D array
data = xl("B1")

# Calculate the skew for each column
skew_values = np.mean(skew_columns(data))

import numpy as np
array = xl("B1")
gy, gx = np.gradient(array)
gnorm = np.sqrt(gx**2 + gy**2)
sharpness = np.average(gnorm)

# 2d image gradient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create a 2D array
data = xl("B1"), E15 min, E16 max
plt.imshow(data, vmin = xl("E15"), vmax = xl("G15"), extent = [1, 14, 14, 1], cmap='viridis')
plt.colorbar()
plt.title(xl("C15"))
plt.show()

# 3d image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample 2D array
data = xl("B1")

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate X and Y data based on the shape of the array
x = np.arange(data.shape[0])
y = np.arange(data.shape[1])
x, y = np.meshgrid(x, y)

# Plot the surface
ax.plot_surface(x, y, data.T, cmap='viridis')

# Hide the X and Y axis values
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()

# Sample 2D array
data = xl("B1")

# Find the minimum value in the 2D array
min_value = np.min(data)

# Sample 2D array
data = xl("B1")

# Find the maximum value in the 2D array
max_value = np.max(data)