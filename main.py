import numpy as np
import random
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt = True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')

sentence = '근육이 커지기 위해서는'

# 문장 생성 기본

input_ids = tokenizer.encode(sentence)
input_ids = tf.convert_to_tensor([input_ids])

output = model.generate(input_ids,
                        max_length=128,
                        repetition_penalty=2.0,
                        use_cache=True)
output_ids = output.numpy().tolist()[0]

output = tokenizer.decode(output_ids)


# Top5 무작위 선별

input_ids = tokenizer.encode(sentence)

while len(input_ids) < 50:
    output = model(np.array([input_ids]))
    top5 = tf.math.top_k(output.logits[0, -1], k = 5)
    token_id = random.choice(top5.indices.numpy())
    input_ids.append(token_id)
    print(f':::    {tokenizer.decode(token_id)}\n\n')


print(f'###\n\n{tokenizer.decode(input_ids)}')