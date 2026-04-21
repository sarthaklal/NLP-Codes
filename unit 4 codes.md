import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out

# Example
model = RNNClassifier(vocab_size=5000, embed_dim=128, hidden_size=64, output_size=2)
x = torch.randint(0, 5000, (32, 20))  # batch=32, seq=20
output = model(x)
print(output.shape)



import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Example
model = LSTMClassifier(5000, 128, 64, 2)
x = torch.randint(0, 5000, (32, 20))
output = model(x)
print(output.shape)




from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
texts = ["I love this!", "This is bad"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)

print(predictions)






from transformers import RobertaTokenizer, RobertaForSequenceClassificationimport torchtokenizer = RobertaTokenizer.from_pretrained('roberta-base')model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)texts = ["Amazing experience", "Worst ever"]inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")outputs = model(**inputs)logits = outputs.logitspredictions = torch.argmax(logits, dim=1)print(predictions)
