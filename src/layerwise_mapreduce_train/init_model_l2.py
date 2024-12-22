import json
import numpy as np

def init_model_l2():
    # Load pretrained from model_l1.json
    with open('model_l1.json','r') as f:
        m_l1 = json.load(f)

    W1 = np.array(m_l1['enc_weights'])  # shape(128,784)
    b1 = np.array(m_l1['enc_bias'])     # shape(128,)

    # Thêm W2,b2: (10,128)
    W2 = np.random.randn(10,128)*0.01
    b2 = np.zeros(10)
    lr = 0.0001

    # Lưu model
    model_l2 = {
      "enc_weights": W1.tolist(),
      "enc_bias": b1.tolist(),
      "cls_weights": W2.tolist(),
      "cls_bias": b2.tolist(),
      "learning_rate": lr
    }
    with open("model_l2.json",'w') as f2:
        json.dump(model_l2,f2)

if __name__=="__main__":
    init_model_l2()
    print("Init model_l2.json (W1,b1 from L1; W2,b2 random).")
