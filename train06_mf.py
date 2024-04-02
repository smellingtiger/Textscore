import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from model import SimpleNN


class TextScoreDataset(Dataset):
    def __init__(self, train_file, val_file, transformer_model_name='uer/sbert-base-chinese-nli'):
        self.model = SentenceTransformer(transformer_model_name)
        self.model = self.model.to("cuda")

        # 加载训练集
        self.train_data = []
        with open(train_file, 'r', encoding='utf-8') as file:
            raw_train = json.loads(file.read())
        tot_train = len(raw_train)
        for index, item in enumerate(raw_train):
            print(f"encoding training data...{index}/{tot_train}", end="\r")
            title = item["title"]
            title_embedding = torch.tensor(self.model.encode(title))
            content = item["cleaned_text"]
            content_embedding = torch.tensor(self.model.encode(content))
            id_embedding = torch.tensor(self.model.encode(str(item["public_id"])))
            authorid_embedding = torch.tensor(self.model.encode(str(item["author_id"])))
            combined_embedding = torch.cat((title_embedding, content_embedding, id_embedding, authorid_embedding))
            self.train_data.append((combined_embedding, item["label"]))
        print()

        # 加载验证集
        self.val_data = []
        with open(val_file, 'r', encoding='utf-8') as file:
            raw_val = json.loads(file.read())
        tot_val = len(raw_val)
        for index, item in enumerate(raw_val):
            print(f"encoding validation data...{index}/{tot_val}", end="\r")
            title = item["title"]
            title_embedding = torch.tensor(self.model.encode(title))
            content = item["cleaned_text"]
            content_embedding = torch.tensor(self.model.encode(content))
            id_embedding = torch.tensor(self.model.encode(str(item["public_id"])))
            authorid_embedding = torch.tensor(self.model.encode(str(item["author_id"])))
            combined_embedding = torch.cat((title_embedding, content_embedding, id_embedding, authorid_embedding))
            self.val_data.append((combined_embedding, item["label"]))
        print()

    def __len__(self):
        return len(self.train_data) + len(self.val_data)

    def __getitem__(self, index):
        if index < len(self.train_data):
            return self.train_data[index]
        else:
            return self.val_data[index - len(self.train_data)]


# 参数设置
input_size = 768 + 768 + 768 + 768  # 假设原有嵌入特征的大小为768，id嵌入的大小也为768
hidden_size = 1024
dropout_rate = 0.1
batch_size = 32
learning_rate = 0.0005
num_epochs = 100

# 实例化模型、损失函数和优化器
model = SimpleNN(input_size, hidden_size, dropout_rate=dropout_rate)
model = model.to("cuda")
criterion = nn.BCELoss().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 数据集文件路径
train_file = 'train06_mf.json'
val_file = 'val06_mf.json'

# 实例化数据集对象
dataset = TextScoreDataset(train_file=train_file, val_file=val_file)

# 计算训练集和验证集的划分索引
train_size = len(dataset.train_data)
val_size = len(dataset.val_data)

# 创建训练集和验证集的数据加载器
train_loader = DataLoader(Subset(dataset, range(train_size)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, range(train_size, train_size + val_size)), batch_size=batch_size, shuffle=False)

# 早停法设置
best_val_acc = 0.0
patience = 15
wait = 0

for epoch in range(num_epochs):
    model.train()
    loss_epoch = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to("cuda")
        labels = torch.tensor([float(label) for label in labels]).to("cuda")
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        predicted = outputs >= 0.5
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    loss_epoch /= len(train_loader)
    acc_epoch = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.5f}, Acc: {acc_epoch:.5f}')

    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:  # 使用验证集的数据加载器
            inputs = inputs.to("cuda")
            labels = torch.tensor([float(label) for label in labels]).to("cuda")
            outputs = model(inputs).squeeze()
            predicted = outputs >= 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc_val = correct / total
    f1_val = f1_score(true_labels, predicted_labels)
    print(f'Validation Accuracy: {acc_val:.5f}')
    print(f'Validation F1 Score: {f1_val:.5f}')

    # 早停检查
    if acc_val >= best_val_acc:
        best_val_acc = acc_val
        wait = 0
        torch.save(model.state_dict(), 'best_model05_4f.pth')
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step()
