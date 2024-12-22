import json

def prepare_finetune():
    with open('model_l2.json','r') as f:
        m = json.load(f)
    m_final = {
      "weights_input_hidden": m['enc_weights'],
      "bias_hidden": m['enc_bias'],
      "weights_hidden_output": m['cls_weights'],
      "bias_output": m['cls_bias'],
      "learning_rate": m['learning_rate']
    }
    with open('model_finetune.json','w') as f2:
        json.dump(m_final,f2)
    print("model_finetune.json ready")

if __name__=="__main__":
    prepare_finetune()
