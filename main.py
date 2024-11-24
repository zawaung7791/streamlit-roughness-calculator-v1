import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew
from scipy.stats import kurtosis

#############################################################################################
# streamlit setup code
#############################################################################################

# change streamlit layout
st.set_page_config(layout="centered")

st.title("Streamlit Roughness Calculator")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Profile", "2D Graph", "3D Graph"])

# create sidebar
st.sidebar.title("File Upload")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt'])
st.sidebar.text("By Stephen Downing, converted by Zaw Aung 2024")

# textboxes for parameters
st.sidebar.header("Parameters")
sigma = st.sidebar.number_input("Sigma", value=1.0)
spacing = st.sidebar.number_input("Spacing", value=1.0)
image_min = st.sidebar.number_input("Image Min", value=0.0)
image_max = st.sidebar.number_input("Image Max", value=1.0)
graph_min = st.sidebar.number_input("Graph Min", value=0.0)
graph_max = st.sidebar.number_input("Graph Max", value=1.0)
twod_title =  st.sidebar.text_input("2D Title", value="2D Image")

#############################################################################################
# functions for calculating roughness
#############################################################################################

if uploaded_file is not None:
    try: 
        # Check the type and content of uploaded_file
        #print(f"Uploaded file type: {type(uploaded_file)}")
        file_size = uploaded_file.getbuffer().nbytes
        #print(f"Uploaded file size: {file_size}")
        file_content = uploaded_file.getvalue().decode('utf-8')
        #print("File content:")
        #print(file_content)

        # Check if the file is empty
        if file_size == 0:
            raise ValueError("Uploaded file is empty")

        # Check if the file content is not just whitespace
        if not file_content.strip():
            raise ValueError("Uploaded file contains only whitespace")

        # Clean up the file content
        cleaned_content = "\n".join(
            ["\t".join(line.split()) for line in file_content.splitlines() if line.strip()]
        )

        # Write the cleaned content to a temporary file for debugging
        with open("temp_uploaded_file.txt", "w") as temp_file:
            temp_file.write(cleaned_content)


        # Convert uploaded file to DataFrame without headers
        data = pd.read_csv("temp_uploaded_file.txt", header=None, delimiter="\t")
       # print(type(data))
       # print(data.head())  # Debugging line to check the content of the DataFrame


        # Function to perform Gaussian smoothing on a 2D array
        def gaussian_smoothing(data, sigma):
            smoothed = gaussian_filter(data.to_numpy(), sigma=sigma)
            print(type(smoothed))  # Debugging line to check the type
            return smoothed

        # Perform Gaussian smoothing with sigma = 1
        smoothed_data = gaussian_smoothing(data, sigma=sigma)
        print(type(smoothed_data))
        # Function to calculate the average deviation of a 2D array
        def average_deviation_2d(array):
            mean = np.mean(array)
            deviations = np.abs(array - mean)
            avg_deviation = np.mean(deviations)
            return avg_deviation
        
        # Calculate the average deviation
        avg_dev = average_deviation_2d(smoothed_data)

        # Function to calculate the standard deviation of a 2D array
        def standard_deviation_2d(array):
            return np.std(array)
        
        # Calculate the standard deviation
        std_dev = standard_deviation_2d(smoothed_data)

        # Function to calculate the skewness of a 2D array
        def skewness_2d(array):
            # Flatten the 2D array to 1D
            flattened_array = array.flatten()
            return skew(flattened_array)

        # Calculate the skewness
        skewness_value = skewness_2d(smoothed_data)

        # Function to calculate the kurtosis of a 2D array
        def kurtosis_2d(array):
            # Flatten the 2D array to 1D
            flattened_array = array.flatten()
            return kurtosis(flattened_array)

        # Calculate the kurtosis
        kurtosis_value = kurtosis_2d(smoothed_data)

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

        # Calculate the average of the average absolute slope for the rows
        avg_of_avg_abs_slope = average_of_average_absolute_slope(smoothed_data)

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

        # Calculate the average of the average absolute slope for the columns
        avg_of_avg_abs_slope_columns = average_of_average_absolute_slope_columns(smoothed_data)

        # wavelength x
        wavelength_x = 2 * np.pi * avg_dev * spacing / avg_of_avg_abs_slope

        # wavelength y
        wavelength_y = 2 * np.pi * avg_dev * spacing / avg_of_avg_abs_slope_columns

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

        # Calculate the average of the standard deviations for the rows
        avg_std_dev_rows = average_std_deviation_rows(smoothed_data)

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

        # Calculate the average of the standard deviations for the columns
        avg_std_dev_columns = average_std_deviation_columns(smoothed_data)

        # Function to calculate the skew for each row of a 2D array
        def skew_rows(array):
            skews = []
            for row in array:
                # Calculate the skew for each row
                row_skew = skew(row)
                skews.append(row_skew)
            return skews

        # Calculate the skew for each row
        skew_values_x = np.mean(skew_rows(smoothed_data))

        # Function to calculate the skew for each column of a 2D array
        def skew_columns(array):
            skews = []
            for col in array.T:  # Transpose the array to iterate over columns
                # Calculate the skew for each column
                col_skew = skew(col)
                skews.append(col_skew)
            return skews

        # Calculate the skew for each column
        skew_values_y = np.mean(skew_columns(smoothed_data))

        # Calculate the sharpness of the image
        gy, gx = np.gradient(smoothed_data)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)

        # Find the minimum value in the 2D array
        min_value = np.min(smoothed_data) 

        # Find the maximum value in the 2D array
        max_value = np.max(smoothed_data)   

    except pd.errors.EmptyDataError:
        print("Error loading file: No columns to parse from file")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
#############################################################################################
# streamlit code
#############################################################################################

with tab1:
    st.header("Data Profile")

    # If the file is uploaded, show the file name
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            numeric_df = df.select_dtypes(include=['number'])
            category_df = df.select_dtypes(include=['object', 'datetime'])

            st.text("Parameters")

            # Display the values in a table
            values_df = pd.DataFrame({
                "Metric": ["Minimum Value", "Maximum Value", "Average Deviation", "Standard Deviation", "Skewness", "Kurtosis", "Wavelength X", "Wavelength Y", "Average Standard Deviation Rows", "Average Standard Deviation Columns", "Skew Values X", "Skew Values Y", "Sharpness"],
                "Value": [round(min_value, 2), round(max_value, 2), round(avg_dev, 2), round(std_dev, 2), round(skewness_value, 2), round(kurtosis_value, 2), round(wavelength_x, 2), round(wavelength_y, 2), round(avg_std_dev_rows, 2), round(avg_std_dev_columns, 2), round(skew_values_x, 2), round(skew_values_y, 2), round(sharpness, 2)]
            })
            st.write(values_df)

                        # Display dataframe info
            st.text("Dataframe Info")
            info_df = pd.DataFrame({
                "Datatype": df.dtypes,
                "Count": df.count(),
                "Distinct Count": df.nunique(),
                "Null Values": df.isnull().sum(),
                "Blanks": (df == '').sum()
            })
            st.write(info_df)

            # Display the dataframe
            st.text("Data")
            st.write(df)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.write("No file uploaded yet.")

with tab2:
    if uploaded_file is not None:
        st.header("2D")
        st.text(f"{twod_title}")
        # 2d image generation
        fig = px.imshow(data, zmin=graph_min, zmax=graph_max, color_continuous_scale='Viridis', title=twod_title,
                         width=800, height=800)
        st.plotly_chart(fig)
    else:
        st.write("No data loaded for 2D image")

with tab3:
    if uploaded_file is not None:
        st.header("3D")
        # generate Topographical 3D Surface Plot using plotly
        fig = go.Figure(data=[go.Surface(z=smoothed_data, colorscale='Viridis')])   
        fig.update_layout(title='3D Topographical Surface Plot', autosize=False, width=800, height=800)
        st.plotly_chart(fig)


    else:
        st.write("No data loaded for 3D Image.")
