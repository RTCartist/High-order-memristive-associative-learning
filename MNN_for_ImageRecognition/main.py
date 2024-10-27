import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import os
import matplotlib as mpl

# Simulated SDC memristor parameters
params = {
    'alphaoff': 1,
    'alphaon': 1,
    'voff': -0.16,
    'von': 0.14,
    'koff': -18.33,
    'kon': 2.82,
    'Ron': 20000,
    'Roff': 190000,
    'won': 0,
    'woff': 1,
    'wini': 1
}

# Memristor array simulation function
# weight = params['wini'] * np.ones((height, width)) 
# resistances_over_time = []
def simulate_memristor_array(voltage, weights, params, dt):
    height, width = weights.shape
    for y in range(height):
        for x in range(width):
            modulation_voltage = voltage[y, x]
            weights[y, x] = update_state(weights[y, x], modulation_voltage, dt, params)
    resistances = calculate_resistance(weights, params)
    return weights, resistances

def calculate_resistance(w, params):
    return params['Ron'] * (params['Roff'] / params['Ron']) ** w

def update_state(w, V, dt, params):
    if V < params['voff']:
        dwdt = params['koff'] * ((V / params['voff']) - 1) ** params['alphaoff']
    elif V > params['von']:
        dwdt = params['kon'] * ((V / params['von']) - 1) ** params['alphaon']
    else:
        dwdt = 0

    new_w = w - dwdt * dt
    return np.clip(new_w, 0, 1)

# Associative neural network parameters
learning_rate = 0.415

row_size = 20
col_size = 20
n_neurons = row_size * col_size  # Number of neurons representing pixels (20x20 image)
# Weight matrix (initialize with all values as 1, shape (20, 20))
weights = params['wini'] * np.ones((row_size, col_size))

# Set paths for input and teacher image folders
input_folder = 'training/input/'
teacher_folder = 'training/teacher/'
test_folder = 'test/'

# Get sorted list of image file names from both folders
input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))
teacher_files = sorted([f for f in os.listdir(teacher_folder) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))

# Check that both folders contain the same number of images
if len(input_files) != len(teacher_files):
    raise ValueError("The number of input images and teacher images must be the same.")

# Track weight changes
weight_history = []
resistances_over_time = []

for input_file, teacher_file in zip(input_files, teacher_files):
    # Load teacher and input images
    teacher_img_full = cv2.imread(os.path.join(teacher_folder, teacher_file), cv2.IMREAD_GRAYSCALE)
    if teacher_img_full is None:
        raise FileNotFoundError(f"Teacher image file '{teacher_file}' not found. Please check the file path.")
    # teacher_img_full = teacher_img_full / 255.0
    teacher_img_full = (teacher_img_full - np.min(teacher_img_full)) / (np.max(teacher_img_full) - np.min(teacher_img_full))
    height, width = teacher_img_full.shape
    # Aggregate the teacher image to match the size of the weight matrix
    teacher_image_region = cv2.resize(teacher_img_full, (col_size, row_size), interpolation=cv2.INTER_AREA)
    teacher_image = teacher_image_region.flatten()

    input_img_full = cv2.imread(os.path.join(input_folder, input_file), cv2.IMREAD_GRAYSCALE)
    if input_img_full is None:
        raise FileNotFoundError(f"Input image file '{input_file}' not found. Please check the file path.")
    input_img_full = (input_img_full - np.min(input_img_full)) / (np.max(input_img_full) - np.min(input_img_full))
    height, width = input_img_full.shape
    # Aggregate the input image to match the size of the weight matrix
    input_image_region = cv2.resize(input_img_full, (col_size, row_size), interpolation=cv2.INTER_AREA)
    input_image = input_image_region.flatten()

    # Step 1: Compare each value in the teacher image with all values in the input image
    delta_weights = np.zeros_like(weights)
    for i in range(len(teacher_image)):
        count = 0
        for j in range(len(input_image)):
            if abs(teacher_image[i] - input_image[j]) < 0.006:
                count += 1
        # Update corresponding weight based on the count
        if count > 4:
            delta_weights[i // col_size, i % col_size] = count * learning_rate

    # Update memristor array weights and resistances
    weights, resistances = simulate_memristor_array(delta_weights, weights, params, 0.001)
    
    # Store weights for visualization
    weight_history.append(weights.copy())
    resistances_over_time.append(resistances.copy())

weight_history = np.array(weight_history)
resistances_over_time = np.array(resistances_over_time)

print(weight_history[:, 2, 2])

# Plot the change in weights for a representative connection
plt.plot(weight_history[:, 2, 2], label='Weight from pixel 2,2')
plt.xlabel('Trial')
plt.ylabel('Weight Value')
plt.title('Pavlov Associative Learning with Image Comparison')
plt.legend()
plt.show()

# Visualize the final weight matrix
def plot_weight_matrix(weights, title, vmin, vmax):
    plt.figure(figsize=(12, 9))
    # sns.heatmap(weights, annot=True, cmap='viridis', cbar=True)
    sns.heatmap(weights, annot=True, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

# Plot the initial and final weight matrix
plot_weight_matrix(weights, 'Final Weight Matrix After Training', 0, 0.7)

# Calculate and plot modulation value (difference from initial weight)
modulation_values = weights - np.ones((row_size, col_size))
plot_weight_matrix(modulation_values, 'Modulation Values of the Weights After Training', -1 ,1)

# Test validation part
# Get sorted list of image file names from the test folder
test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))

# Track errors for each test image
test_errors = []
test_ssim_value = []

for test_file in test_files:
    # Load test image
    test_img_full = cv2.imread(os.path.join(test_folder, test_file), cv2.IMREAD_GRAYSCALE)
    if test_img_full is None:
        raise FileNotFoundError(f"Test image file '{test_file}' not found. Please check the file path.")
    # test_img_full = test_img_full / 255.0
    test_img_full = (test_img_full - np.min(test_img_full)) / (np.max(test_img_full) - np.min(test_img_full))
    height, width = test_img_full.shape
    # Aggregate the test image to match the size of the weight matrix
    test_image_region = cv2.resize(test_img_full, (col_size, row_size), interpolation=cv2.INTER_AREA)

    # Calculate error as sum of squared differences
    error = np.sum(np.minimum(np.abs(np.tanh(test_image_region.flatten() - weights.flatten())), np.abs(np.tanh(0.5 - test_image_region.flatten() - weights.flatten()))))
    test_errors.append(error)

# Plot the calculated errors
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_errors) + 1), test_errors, marker='o', linestyle='-', color='b')
plt.xlabel('Test Image Index')
plt.ylabel('Sum of Squared Errors')
plt.title('Validation Errors for Test Images')
plt.show()

# Configure plots to use Arial font and a style suitable for scientific journals
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 26
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.titlesize'] = 30

# Plot the resistance matrix
# Plotting the final weight matrix
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=False, cmap='viridis', cbar=True, vmin=0, vmax=0.7)
plt.title('Final Weight Matrix After Training')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.tight_layout()
plt.savefig('final_weight_matrix.png', dpi=600, bbox_inches='tight')  
plt.show()

# Initialize second_order_voltage array with zeros
second_order_voltage = np.zeros((1, len(test_errors)))
recognition_weights = params['wini'] * np.ones((1, len(test_errors)))

for i in range(len(test_files)):
    second_order_voltage[0, i] = (75 - test_errors[i]) * 1

second_weights, second_resistances = simulate_memristor_array(second_order_voltage, recognition_weights, params, 0.01)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(second_weights.flatten()) + 1), second_weights.flatten(), marker='o', linestyle='-', color='b')
plt.xlabel('Test Image Index')
plt.ylabel('Sum of Squared Errors')
plt.title('Validation Errors for Test Images')
plt.show()
