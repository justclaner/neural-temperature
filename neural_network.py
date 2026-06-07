import generate_training
import math
import random
import json
import time

from pathlib import Path
script_dir = Path(__file__).resolve().parent
model_file_name = "model.json"
model_file_path = script_dir / model_file_name

# excludes 0 as a possible return value
def random_uniform_start(a, b):
    if a == b:
        return a
    elif a > b:
        a ^= b
        b ^= a
        a ^= b
    diff = b - a
    return diff * (1.0 - random.random()) + a

hidden_layer_size = 4
weights = [[random_uniform_start(-1000, 1000) for _ in range(hidden_layer_size)] for _ in range(2)]
biases = [[random_uniform_start(-1000, 1000) for _ in range(hidden_layer_size)] for _ in range(2)]

def save_model(filename = model_file_path):
    with open(filename, "w") as f:
        json.dump({"weights": weights, "bias": biases}, f)
    
def load_model(filename = model_file_path):
    global weights, biases
    with open(filename, "r") as f:
        data = json.load(f)
    weights = data["weights"]
    biases = data["biases"]

def predict(num):
    a_nums = [math.tanh(w * num + b) for w, b in zip(weights[0], biases[0])]
    return sum(w * a + b for a, w, b in zip(a_nums, weights[1], biases[1]))

def train(trials, auto_save = True):
    global weights, biases
    if trials < 1:
        return
    training_data = []
    with open(generate_training.file_path, "r") as f:
        for line in f:
            fahrenheit, celsius = line.strip().split("\t")
            training_data.append((fahrenheit, celsius))
    
    # back propagation
    w_nudge = [[0 for _ in range(hidden_layer_size)] for _ in range(2)]
    b_nudge = [[0 for _ in range(hidden_layer_size)] for _ in range(2)]
    for _ in range(trials):
        for fahrenheit, y in training_data:
            # prediction = a^2_0
            prediction = predict(fahrenheit)
            # cost = (prediction - y) ** 2

            # nudge second layer weights and biases
            for k in range(hidden_layer_size):
                a = math.tanh(fahrenheit * weights[0][k] + biases[0])
                dC_dw = a * 2 * (prediction - y)
                w_nudge[1][k] -= dC_dw / trials
                b_nudge[1][k] -= 2 * (prediction - y) / trials
            
            # nudge first layer weights and biases
            for j in range(hidden_layer_size):
                z = fahrenheit * weights[0][j] + biases[0]
                w = weights[1][j]
                sech = (1 / math.cosh(z)) ** 2
                w_nudge[0][j] -= fahrenheit * sech * w * 2 * (prediction - y)
                b_nudge[0][j] -= sech * w * 2 * (prediction - y)
    
    for i in range(len(w_nudge)):
        for j in range(len(w_nudge[i])):
            weights[i][j] += w_nudge[i][j]
            biases[i][j] += b_nudge[i][j]
    
    if auto_save:
        save_model()

try:
    load_model()
    print("Loaded existing model.")
except FileNotFoundError:
    print("Creating new model.")
    
if __name__ == "__main__":

    start_time = time.perf_counter()
    print("Starting training...")
    train(100)
    print(f"Training took {time.perf_counter() - start_time}s!")