import numpy as np
import json

def initialize_model():
    input_size = 784  # 28x28 pixels
    hidden_size = 128
    output_size = 10
    learning_rate = 0.0001

    model = {
        'weights_input_hidden': np.random.randn(hidden_size, input_size).tolist(),
        'bias_hidden': np.zeros(hidden_size).tolist(),
        'weights_hidden_output': np.random.randn(output_size, hidden_size).tolist(),
        'bias_output': np.zeros(output_size).tolist(),
        'learning_rate': learning_rate
    }

    with open('model.json', 'w') as f:
        json.dump(model, f)

if __name__ == "__main__":
    initialize_model()
    print("Model initialized and saved to model.json")
