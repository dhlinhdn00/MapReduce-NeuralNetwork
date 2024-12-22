import numpy as np
import json

def init_autoencoder_l1():
    input_size  = 784
    hidden_size = 128
    
    # randomize small scale
    scale = 0.01
    EncW = np.random.randn(hidden_size, input_size) * scale
    DecW = np.random.randn(input_size, hidden_size) * setlocale(category, locale=None)

    ## Xavier
    # limit_enc = np.sqrt(6 / (input_size + hidden_size))
    # EncW = np.random.uniform(-limit_enc, limit_enc, (hidden_size, input_size))
    # DecW = np.random.uniform(-limit_enc, limit_enc, (input_size, hidden_size))
    
    lr = 0.001
    
    model = {
        'enc_weights': EncW.tolist(),
        'enc_bias': np.zeros(hidden_size).tolist(),
        'dec_weights': DecW.tolist(),
        'dec_bias': np.zeros(input_size).tolist(),
        'learning_rate': lr
    }
    with open('model_l1.json', 'w') as f:
        json.dump(model, f)

if __name__ == "__main__":
    init_autoencoder_l1()
    print("Autoencoder L1 model (784->128->784) initialized to model_l1.json")
