import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from scipy.stats import skew
from scipy.stats import kurtosis
import statistics
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import re

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
    return ((np.diff(signal)*signal[:-1]) / (2 * np.pi * np.sum(signal**2)))

def get_mean_square_frequency(signal):
    return  np.sum(np.diff(signal)**2) / (4 * np.pi**2 * np.sum(signal**2))

def get_root_mean_square_frequency(signal):
    return  np.sqrt(get_mean_square_frequency(signal))

def get_root_variance_frequency(signal):
    return  np.sqrt(get_mean_square_frequency(signal) - get_frequency_centre(signal)**2)



if __name__ == '__main__':
    
    # Specify the directory path
    new_directory = 'Results_Dataset11/SOM'
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
    info = ''
    np.savetxt('./Results_Dataset11/SOM/results.txt',[info], fmt='%s', header=' Methods            Accuracy(%)            Trial ')  
    
    letters = ['H', 'I', 'O'] # 0-healthy, I-inner fault, O-outer fault
    speeds = ['A', 'B', 'C', 'D']
    trials = ['1','2', '3']
    state = 0
    dataset = []
    for letter in letters:
        for speed in speeds:
            for trial in trials:
                path_signal = f'../../Dados/Rotor_Bearing/Dataset11/Data/{letter}-{speed}-{trial}.mat'
                df_signal = scipy.io.loadmat(path_signal)
                channel_1 = df_signal['Channel_1'].flatten()
                signal = []
                signal.append([channel_1, state])
                dataset.append(signal)
        state +=1
    print('LOADED DATASET!')
    data = []
    X=[]
    y=[]
    for i in dataset:
        data.append(i[0])

    for i in data:
        X.append(i[0])
        y.append(i[1])

    features_function = [get_skewness, get_kurtosis, get_shape_factor, get_variance, get_std, get_rms_acceleration,
                    get_peak_acceleration, get_crest_factor, get_mean_square_frequency,
                    get_root_mean_square_frequency]
    
    # features_function = [get_frequency_centre]

    list_features_function = []

    # Iterate over different sizes of combinations
    for r in range(1, len(features_function) + 1):
        # Generate all combinations of size r
        combinations_r = combinations(features_function, r)
        
        # Extend the list with each combination
        list_features_function.extend(combinations_r)

    # Convert combinations to lists for printing
    list_features_function = [list(combination) for combination in list_features_function]

    experiments_ids = list(range(0,len(dataset)))

    for index, combination in enumerate(list_features_function):
        names_methods = [re.search(r'function (.*?) at', str(item)).group(1) for item in combination]
        functions = [method.split('_')[1::] for method in names_methods if method.startswith('get_')]
        new_directory = '_'.join('_'.join(inner_list) for inner_list in functions)
        parent_dir = os.path.abspath('./Results_Dataset11/SOM/')
        path = os.path.join(parent_dir, new_directory)
        if not os.path.exists(path):
            os.makedirs(path)
        data_features = []
        i = 0

        for exp in experiments_ids:
            print(exp)
            experiment = dataset[exp][0][0]
            feature_accelerometer = []

            for func in combination:
                
                accelerometer = func(dataset[exp][0][0])

                if type(accelerometer) == list:
                    feature_accelerometer+=accelerometer
                    
                else:
                    feature_accelerometer.append(accelerometer)

            
            data_features.append([feature_accelerometer, [dataset[exp][0][1]]])

        features_list = list()

        for i in data_features:
            y = np.concatenate([np.array(x) for x in i])
            features_list.append(y)

        features_list = np.array(features_list)

        dim_ = len(list_features_function[index])+1

        som = SOM(n=1,m=3,dim=dim_, max_iter=100000) 

        index_methods = list(range(0,int(len(features_list[0]))-1))

        X = features_list[:,0:len(index_methods)]
        y = features_list[:,-1]

        for i in range(0,len(index_methods)):   # X, Y e Z

            for trial in range(1,6):

                som.fit(features_list)
                predictions = som.predict(features_list)

                # Create a 1x2 subplot grid
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

                index1 = i

                # Extract features
                x = list(range(0,36))
                y = features_list[:, index1]
                label = features_list[:, -1]

                matches = [label == pred for label, pred in zip(label, predictions)]
                accuracy = np.mean(np.array(matches))*100
                vector_info = f'{new_directory} | {accuracy:.2f} | {trial}\n'

                # Plot the first subplot (Actual Classes)
                scatter1 = axs[0].scatter(x, y, c=label, cmap=ListedColormap(['green', 'brown', 'black']))
                axs[0].set_title('Actual Classes')
                axs[0].set_xlabel(f'experiment')  # Add X axis label
                axs[0].set_ylabel(f'accelerometer - {functions[i]}')  # Add Y axis label
                classes = ['Healthy', 'Unhealthy 1', 'Unhealthy 2']

                # Create a custom legend
                legend_elements1 = [Line2D([0], [0], marker='o', color='w', label=f'Class {classes[i]}',
                                            markerfacecolor=['green', 'brown', 'black'][i], markersize=10) for i in range(3)]

                # Add legend to the first subplot
                axs[0].legend(handles=legend_elements1, loc='best')

                # Plot the second subplot (SOM Predictions)
                scatter2 = axs[1].scatter(x, y, c=predictions, cmap=ListedColormap(['red', 'brown', 'black']))
                axs[1].set_title('SOM Predictions')
                axs[1].set_xlabel(f'experiment')  # Add X axis label
                axs[1].set_ylabel(f'accelerometer - {functions[i]}')  # Add Y axis label
                classes = ['1', '2', '3']

                # Create a custom legend
                legend_elements2 = [Line2D([0], [0], marker='o', color='w', label=f'Class {classes[i]}',
                                            markerfacecolor=['red', 'brown', 'black'][i], markersize=10) for i in range(3)]

                # Add legend to the second subplot
                axs[1].legend(handles=legend_elements2, loc='best')

                # Set a global title for the entire figure
                fig.suptitle(f'Comparison of Actual Classes and SOM Predictions', fontsize=16)

                # Add a subtitle below the subplots
                fig.text(0.5, 0.03, f'Acc: {accuracy:.2f} %', ha='center', fontsize=8)

                # Adjust layout for better spacing
                plt.tight_layout()

                with open('./Results_Dataset11/SOM/results.txt', 'a') as f:
                    f.write(vector_info)

                # Save the figure
                plt.savefig(f'./Results_Dataset11/SOM/{new_directory}/image_{new_directory}_fig{i}_trial_{trial}_plot_{functions[i]}.png')


