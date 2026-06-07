import neural_network
import generate_training
import random
min_test_num = -300
max_test_num = 300
correct = 0

error_delta = 10
max_incorrect_displayed = 5
incorrect = 0
trials = 1000
for _ in range(trials):
    f = random.uniform(min_test_num, max_test_num)
    c = neural_network.predict(f)
    expected = generate_training.fahrenheitToCelsius(f)
    prediction_correct = abs(expected - c) <= error_delta
    correct += prediction_correct
    if not prediction_correct and incorrect < max_incorrect_displayed:
        print(f"{f} degrees fahrenheit was predicted incorrectly to be {c} degrees celsius!")
        incorrect += 1

print(f"{100 * correct / trials}% accuracy!")