import math
import random
import time
from pathlib import Path

min_num = -300
max_num = 300
tests = 1000

script_dir = Path(__file__).resolve().parent
file_name = "training1.txt"
file_path = script_dir / file_name

def fahrenheitToCelsius(num):
    return (num - 32) * 5 / 9

if __name__ == "__main__":
    start_time = time.perf_counter()
    print("Started creating training data...")

    output = ""
    for _ in range(tests):
        fahrenheit = random.uniform(min_num, max_num)
        celsius = fahrenheitToCelsius(fahrenheit)
        output += f"{fahrenheit}\t{celsius}\n"
    print(f"Finished creating training data after {time.perf_counter() - start_time}s!")

    start_time = time.perf_counter()
    print("Started writing training data to file...")
    with open(file_path, "w") as f:
        f.write(output)
    print(f"Finished writing training data to file after {time.perf_counter() - start_time}s!")
