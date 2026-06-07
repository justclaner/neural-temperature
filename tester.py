import neural_network
import generate_training
import random
min_test_num = -300
max_test_num = 300
correct = 0

accepted_percent_error = 10
max_incorrect_displayed = 5
incorrect = 0
trials = 1000
for _ in range(trials):
    f = random.uniform(min_test_num, max_test_num)
    c = neural_network.predict(f)
    expected = generate_training.fahrenheitToCelsius(f)
    percent_error = 100 * abs(expected - c) / expected
    prediction_correct = percent_error <= accepted_percent_error
    correct += prediction_correct
    if not prediction_correct and incorrect < max_incorrect_displayed:
        print(f"{f:.2f}F was predicted incorrectly to be {c:.2f}C! ({percent_error:.1f}% off from {expected:.2f}C)")
        incorrect += 1

print(f"{100 * correct / trials}% accuracy!")