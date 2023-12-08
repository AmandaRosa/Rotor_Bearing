import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import statistics
from itertools import combinations
import re
from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os

def get_skewness(signal):
    return skew(signal)

def get_kurtosis(signal):
    return kurtosis(signal)

def get_shape_factor(signal):
    N = len(signal)
    return np.sqrt(((signal**2).sum()/N) / ((abs(signal)).sum()/N))

def get_variance(signal):
    return statistics.variance(signal)

def get_std(signal):
    return statistics.stdev(signal)

def get_rms_acceleration(signal):
    N = len(signal)
    return np.sqrt(1/N * (signal**2).sum())

def get_peak_acceleration(signal):
    return max(abs(signal))

def get_crest_factor(signal):
    return get_peak_acceleration(signal)/get_rms_acceleration(signal)

def get_frequency_centre(signal):
    return ((np.diff(signal)*signal).sum()) / (2 * np.pi * np.sum(signal**2))

def get_mean_square_frequency(signal):
    return  np.sum(np.diff(signal)**2) / (4 * np.pi**2 * np.sum(signal**2))

def get_root_mean_square_frequency(signal):
    return  np.sqrt(get_mean_square_frequency(signal))

def get_root_variance_frequency(signal):
    return  np.sqrt(get_mean_square_frequency(signal) - get_frequency_centre(signal)**2)




if __name__ == '__main__':

    # Specify the directory path
    new_directory = 'Results_Dataset3/SOM'
    parent_dir = os.path.abspath('.')
    path = os.path.join(parent_dir, new_directory)

    # Create the directory if it doesn't exist
    try:
        if not os.path.exists(new_directory):
            os.makedirs(path)
            print(f"Directory '{new_directory}' created successfully.")
        else:
            print(f"Directory '{new_directory}' already exists.")
    except: 
        pass

    path_signals = '../../Dados/Rotor_Bearing/Dataset3/bearing_signals/bearing_signals.csv'
    df_signals =pd.read_csv(path_signals)

    path_classes = '../../Dados/Rotor_Bearing/Dataset3/bearing_signals/bearing_classes.csv'
    df_classes =pd.read_csv(path_classes)

    print('LOADED DATASET!')
    info = ''
    np.savetxt('./Results_Dataset3/SOM/results.txt',[info], fmt='%s', header='     Methods               True Positive          Accuracy(%)            Trial')

    ids = ['experiment_id', 'bearing_1_id', 'bearing_2_id']
    np_ids = df_signals[ids].values

    experiments_ids = df_signals['experiment_id'].unique()

    np_classes = df_classes.values

    list_classes = list()

    timestamp = df_signals['timestamp'].values

    results = []

    # Extracting columns for the second NumPy array
    parameters = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'rpm']
    np_parameters = df_signals[parameters].values

    for i in np_classes:
        list_classes.append(i[0].split(";"))

    ## BEARING 1
    # Convert list_classes to a dictionary for efficient lookups
    class_dict = {item[0]: item[1] for item in list_classes}

    # Compare the third column of np_ids with the first index of list_classes
    mask = np.isin(np_ids[:, 1].astype(str), list(class_dict.keys()))

    # Update the third column in np_ids where there is a match
    np_ids[mask, 1] = [class_dict[str(item)] for item in np_ids[mask, 1]]

    ## ---------------------------------------------------------------------##

    ## BEARING 2
    # Convert list_classes to a dictionary for efficient lookups
    class_dict = {item[0]: item[1] for item in list_classes}

    # Compare the third column of np_ids with the first index of list_classes
    mask = np.isin(np_ids[:, 2].astype(str), list(class_dict.keys()))

    # Update the third column in np_ids where there is a match
    np_ids[mask, 2] = [class_dict[str(item)] for item in np_ids[mask, 2]]

    # Create a set of unique rows based on the first column
    unique_rows, unique_indices = np.unique(np_ids[:, 0], return_index=True)

    # Extract the unique rows
    output_array = np_ids[unique_indices][:][:,-1]

    features_function = [get_skewness, get_kurtosis, get_shape_factor, get_variance, get_std, get_rms_acceleration,
                     get_peak_acceleration, get_crest_factor, get_mean_square_frequency,
                     get_root_mean_square_frequency]
    
    # features_function = [get_skewness, get_kurtosis]

    list_features_function = []

    # Iterate over different sizes of combinations
    for r in range(1, len(features_function) + 1):
        # Generate all combinations of size r
        combinations_r = combinations(features_function, r)
        
        # Extend the list with each combination
        list_features_function.extend(combinations_r)

    # Convert combinations to lists for printing
    list_features_function = [list(combination) for combination in list_features_function]

    # Print all combinations
    # Print all combinations with indices

    for index, combination in enumerate(list_features_function):
        names_methods = [re.search(r'function (.*?) at', str(item)).group(1) for item in combination]
        functions = [method.split('_')[1::] for method in names_methods if method.startswith('get_')]
        new_directory = '_'.join('_'.join(inner_list) for inner_list in functions)
        parent_dir = os.path.abspath('./Results_Dataset3/SOM/')
        path = os.path.join(parent_dir, new_directory)
        if not os.path.exists(path):
            os.makedirs(path)

        data_features = []
        i = 0
        for exp in experiments_ids:
            print(exp)
            experiment = df_signals[(df_signals['experiment_id']==exp)]
            feature_a1_x = []
            feature_a1_y = []
            feature_a1_z = []
            feature_a2_x = []
            feature_a2_y = []
            feature_a2_z = []
            feature_rpm = []

            for func in combination:
                a1_x = func(experiment['a1_x'])
                a1_y = func(experiment['a1_y'])
                a1_z = func(experiment['a1_z'])
                a2_x = func(experiment['a2_x'])
                a2_y = func(experiment['a2_y'])
                a2_z = func(experiment['a2_z'])
                rpm = func(experiment['rpm'])

                if type(a1_x) == list:
                    feature_a1_x+=a1_x
                    feature_a1_y+=a1_y
                    feature_a1_z+=a1_z
                    feature_a2_x+=a2_x
                    feature_a2_y+=a2_y
                    feature_a2_z+=a2_z
                    feature_rpm +=rpm
                    
                else:
                    feature_a1_x.append(a1_x)
                    feature_a1_y.append(a1_y)
                    feature_a1_z.append(a1_z)
                    feature_a2_x.append(a2_x)
                    feature_a2_y.append(a2_y)
                    feature_a2_z.append(a2_z)
                    feature_rpm.append(rpm)

            
            data_features.append([feature_a1_x, feature_a1_y, feature_a1_z, feature_a2_x, feature_a2_y, feature_a2_z, feature_rpm, [output_array[exp-1]]])

        features_list = list()

        for i in data_features:
            y = np.concatenate([np.array(x) for x in i])
            features_list.append(y)

        features_list = np.array(features_list)
        dim_ = len(list_features_function[index])*7+1

        som = SOM(n=1,m=2,dim=dim_, max_iter=100000) 

        index_methods = list(range(0,int((len(features_list[0]))/7)))

        for i in range(0,3*len(index_methods)):   # X, Y e Z

            for trial in range(1,4):

                som.fit(features_list)
                predictions = som.predict(features_list)

                # Create a 1x2 subplot grid
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

                index1 = i
                index2 = index1+3*(len(index_methods))

                # Extract features
                x = features_list[:, index1]
                y = features_list[:, index2]
                label = features_list[:, -1]

                # Plot the first subplot (Actual Classes)
                scatter1 = axs[0].scatter(x, y, c=label, cmap=ListedColormap(['red', 'green']))
                axs[0].set_title('Actual Classes')
                axs[0].set_xlabel(f'{parameters[index1%6]} - {functions[int(i/3)]}')  # Add X axis label
                axs[0].set_ylabel(f'{parameters[index2%6]} - {functions[int(i/3)]}')  # Add Y axis label
                classes = ['Unhealthy', 'Healthy']

                # Create a custom legend
                legend_elements1 = [Line2D([0], [0], marker='o', color='w', label=f'Class {classes[i]}',
                                            markerfacecolor=['red', 'green'][i], markersize=10) for i in range(2)]

                # Add legend to the first subplot
                axs[0].legend(handles=legend_elements1, loc='upper right')

                # Plot the second subplot (SOM Predictions)
                scatter2 = axs[1].scatter(x, y, c=predictions, cmap=ListedColormap(['#ff7f0e', '#1f77b4']))
                axs[1].set_title('SOM Predictions')
                axs[1].set_xlabel(f'{parameters[index1%6]} - {functions[int(i/3)]}')  # Add X axis label
                axs[1].set_ylabel(f'{parameters[index2%6]} - {functions[int(i/3)]}')  # Add Y axis label
                classes = ['1', '2']

                # Create a custom legend
                legend_elements2 = [Line2D([0], [0], marker='o', color='w', label=f'Class {classes[i]}',
                                            markerfacecolor=['#ff7f0e', '#1f77b4'][i], markersize=10) for i in range(2)]

                # Add legend to the second subplot
                axs[1].legend(handles=legend_elements2, loc='upper right')

                # Set a global title for the entire figure
                fig.suptitle(f'Comparison of Actual Classes and SOM Predictions', fontsize=16)

                # Add a subtitle below the subplots
                # fig.text(0.5, 0.04, f'Feature: {functions}', ha='center', fontsize=12
                true_positive = np.where((predictions == 1) & (output_array == 1))[0]

                acc = predictions == output_array
                accuracy = np.mean(np.array(acc))*100
                vector_info = f'{new_directory} | {len(true_positive)} | {accuracy:.2f} | {trial}\n'

                ## LEGENDA
                fig.text(0.5, 0.03, f'Acc: {accuracy:.2f} %', ha='center', fontsize=8)

                with open('./Results_Dataset3/SOM/results.txt', 'a') as f:
                    f.write(vector_info)

                # Adjust layout for better spacing
                plt.tight_layout()

                # Save the figure
                plt.savefig(f'./Results_Dataset3/SOM/{new_directory}/image_{new_directory}_fig{i}_trial_{trial}_plot_{functions[int(i/3)]}.png')
