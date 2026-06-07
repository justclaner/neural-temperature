import neural_network
import generate_training
import random

MIN_TEST_NUM = -300
MAX_TEST_NUM = 300

CORRECT = 0
INCORRECT = 0

ACCEPTED_PERCENT_ERROR = 10
MAX_INCORRECT_DISPLAYED = 5

NUM_TRIALS = 1000
for _ in range(NUM_TRIALS):
    f = random.uniform(MIN_TEST_NUM, MAX_TEST_NUM)
    c = neural_network.predict(f)
    expected = generate_training.fahrenheitToCelsius(f)
    percent_error = 100 * abs(expected - c) / expected
    prediction_correct = percent_error <= ACCEPTED_PERCENT_ERROR
    CORRECT += prediction_correct
    if not prediction_correct and INCORRECT < MAX_INCORRECT_DISPLAYED:
        print(f"{f:.2f}F was predicted incorrectly to be {c:.2f}C! ({percent_error:.1f}% off from {expected:.2f}C)")
        INCORRECT += 1

print(f"{100 * CORRECT / NUM_TRIALS}% accuracy!")