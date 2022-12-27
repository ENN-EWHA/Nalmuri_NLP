# import numpy as np
# from flask import Flask, request, jsonify
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('7emotions_model.pt','rb'))

# @app.route('/api',methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     prediction = model.predict([[np.array(data['sentence'])]])
#     output = prediction[0]
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(port=5000, debug=True) v


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from flask import Flask, jsonify,request
from urllib import parse

from flask_restful import reqparse
from flask import Flask, jsonify

import numpy as np
import pickle as p
import json

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


device = torch.device('cpu')
bertmodel, vocab = get_pytorch_kobert_model()

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]


# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# loss_fn = nn.CrossEntropyLoss()

# t_total = len(train_dataloader) * num_epochs
# warmup_step = int(t_total * warmup_ratio)

# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# #정확도 측정을 위한 함수 정의
# def calc_accuracy(X,Y):
#     max_vals, max_indices = torch.max(X, 1)
#     train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
#     return train_acc
    




class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

model = torch.load('model/7emotions_model.pt',map_location=torch.device('cpu')) #전체 모델을 통째로 불러온다. 클래스 선언 필수
model.load_state_dict(torch.load('model/7emotions_model_state_dict.pt',map_location=torch.device('cpu'))) #state_dict를 불러 온 후, 모델에 저장

# checkpoint = torch.load('model/7emotions_all.tar',map_location=torch.device('cpu')) #dict 불러오기
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])

app = Flask(__name__)


@app.route('/api',methods=['POST'])
def predict():
    args = request.json
    sentence = args['sentence']
    sentence = parse.unquote(sentence, 'utf8')

    

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    
    model.eval()


    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람")
            elif np.argmax(logits) == 2:
                test_eval.append("분노")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔")
            elif np.argmax(logits) == 4:
                test_eval.append("중립")
            elif np.argmax(logits) == 5:
                test_eval.append("행복")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오")

        output = ">> 감정 : " + test_eval[0]
        return jsonify(output)

if __name__ == '__main__':
    # modelfile = 'model/7emotions_model.pt'
    # model = torch.load(modelfile, 'rb',map_location=torch.device('cpu'))
    app.run(port=5000, debug=True)    
