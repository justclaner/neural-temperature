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
weights = [[random_uniform_start(-1, 1) for _ in range(hidden_layer_size)] for _ in range(2)]
biases = [[random_uniform_start(-1, 1) for _ in range(hidden_layer_size)] for _ in range(2)]

def save_model(filename = model_file_path):
    with open(filename, "w") as f:
        json.dump({"weights": weights, "biases": biases}, f)
    
def load_model(filename = model_file_path):
    global weights, biases
    with open(filename, "r") as f:
        data = json.load(f)
    weights = data["weights"]
    biases = data["biases"]

INPUT_MIN = generate_training.min_num
INPUT_MAX = generate_training.max_num
def normalize(f):
    return (f - INPUT_MIN) / (INPUT_MAX - INPUT_MIN) * 2 - 1

OUTPUT_MIN = generate_training.fahrenheitToCelsius(INPUT_MIN)  # -184.4°C
OUTPUT_MAX = generate_training.fahrenheitToCelsius(INPUT_MAX)  #  148.9°C

def normalize_output(c):
    return (c - OUTPUT_MIN) / (OUTPUT_MAX - OUTPUT_MIN) * 2 - 1  # to [-1, 1]

def denormalize_output(n):
    return (n + 1) / 2 * (OUTPUT_MAX - OUTPUT_MIN) + OUTPUT_MIN

def predict(num):
    num = normalize(num)
    a_nums = [math.tanh(w * num + b) for w, b in zip(weights[0], biases[0])]
    raw = sum(w * a + b for a, w, b in zip(a_nums, weights[1], biases[1]))
    return denormalize_output(raw)

LEARNING_RATE = 0.01
def train(trials, auto_save = True):
    global weights, biases
    if trials < 1:
        return
    training_data = []
    with open(generate_training.file_path, "r") as f:
        for line in f:
            fahrenheit, celsius = line.strip().split("\t")
            training_data.append((float(fahrenheit), float(celsius)))
    
    # back propagation
    for _ in range(trials):
        w_nudge = [[0 for _ in range(hidden_layer_size)] for _ in range(2)]
        b_nudge = [[0 for _ in range(hidden_layer_size)] for _ in range(2)]
        for fahrenheit, y in training_data:
            norm_f = normalize(fahrenheit)
            norm_y = normalize_output(y)          # normalize the target too
            prediction_raw = sum(                 # raw network output, before denormalize
                w * math.tanh(norm_f * w0 + b0) + b
                for w0, b0, w, b in zip(weights[0], biases[0], weights[1], biases[1])
            )
            error = prediction_raw - norm_y       # error in normalized space

            for k in range(hidden_layer_size):
                a = math.tanh(norm_f * weights[0][k] + biases[0][k])
                w_nudge[1][k] += a * 2 * error / len(training_data)
                b_nudge[1][k] += 2 * error / len(training_data)

            for j in range(hidden_layer_size):
                z = norm_f * weights[0][j] + biases[0][j]
                sech2 = (1 / math.cosh(z)) ** 2
                w_nudge[0][j] += norm_f * sech2 * weights[1][j] * 2 * error / len(training_data)
                b_nudge[0][j] += sech2 * weights[1][j] * 2 * error / len(training_data)
    
        for i in range(len(w_nudge)):
            for j in range(len(w_nudge[i])):
                weights[i][j] -= w_nudge[i][j] * LEARNING_RATE
                biases[i][j] -= b_nudge[i][j] * LEARNING_RATE
    
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
    train(10000)
    print(f"Training took {time.perf_counter() - start_time}s!")