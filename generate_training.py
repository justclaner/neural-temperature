import random
import time
from pathlib import Path

MIN_NUM = -300
MAX_NUM = 300
NUM_TESTS = 1000

script_dir = Path(__file__).resolve().parent
file_name = "training1.txt"
OUTPUT_FILE_PATH = script_dir / file_name

def fahrenheitToCelsius(num):
    return (num - 32) * 5 / 9

if __name__ == "__main__":
    start_time = time.perf_counter()
    print("Started creating training data...")

    output = ""
    for _ in range(NUM_TESTS):
        fahrenheit = random.uniform(MIN_NUM, MAX_NUM)
        celsius = fahrenheitToCelsius(fahrenheit)
        output += f"{fahrenheit}\t{celsius}\n"
    print(f"Finished creating training data after {time.perf_counter() - start_time}s!")

    start_time = time.perf_counter()
    print("Started writing training data to file...")
    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(output)
    print(f"Finished writing training data to file after {time.perf_counter() - start_time}s!")
