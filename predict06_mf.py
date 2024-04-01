import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, confusion_matrix


class TextScoreDataset(Dataset):
    def __init__(self, test_file, transformer_model_name='sbert-base-chinese-nli'):
        self.model = SentenceTransformer(transformer_model_name)
        self.model = self.model.to("cuda")

        # 加载测试集
        self.test_data = []
        with open(test_file, 'r', encoding='utf-8') as file:
            raw_test = json.loads(file.read())
        tot_test = len(raw_test)
        for index, item in enumerate(raw_test):
            print(f"encoding validation data...{index}/{tot_test}", end="\r")
            title = item["title"]
            title_embedding = torch.tensor(self.model.encode(title))
            content = item["cleaned_text"]
            content_embedding = torch.tensor(self.model.encode(content))
            id_embedding = torch.tensor(self.model.encode(str(item["public_id"])))
            authorid_embedding = torch.tensor(self.model.encode(str(item["author_id"])))
            combined_embedding = torch.cat((title_embedding, content_embedding, id_embedding, authorid_embedding))
            self.test_data.append((combined_embedding, item["label"]))
        print()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index]


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def main():
    model_path = 'best_model06_mf.pth'
    inference_file = 'test06_mf.json'

    model = SimpleNN(input_size=3072, hidden_size=1024, dropout_rate=0.1)
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")
    model.eval()

    dataset = TextScoreDataset(inference_file, 'sbert-base-chinese-nli')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to("cuda")
            labels = torch.tensor([float(label) for label in labels]).to("cuda")
            outputs = model(inputs).squeeze()
            predicted = outputs >= 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy().flatten())  # 将 predicted 转换为 NumPy 数组，展平并添加到列表中
            true_labels.extend(labels.cpu().numpy().flatten())

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    tn, fp, fn, tp = cm.ravel()
    a1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # 正样本中预测正确的概率（即召回率）
    a2 = tp / (tp + fp) if (tp + fp) > 0 else 0  # 预测出的所有正例中实际为正例的概率（即精确度）
    a3 = fn / (tp + fn) if (tp + fn) > 0 else 0  # 正样本中预测错误的概率
    a4 = fn / (fn + tn) if (fn + tn) > 0 else 0  # 预测出的所有负例中实际为正例的概率
    a5 = fp / (fp + tn) if (fp + tn) > 0 else 0  # 负样本中预测错误的概率
    a6 = fp / (tp + fp) if (tp + fp) > 0 else 0  # 预测出的所有正例中实际为负例的概率
    a7 = tn / (fp + tn) if (fp + tn) > 0 else 0  # 负样本中预测正确的概率
    a8 = tn / (fn + tn) if (fn + tn) > 0 else 0  # 预测出的所有负例中实际为负例的概率
    f1 = f1_score(true_labels, predicted_labels)

    print(f"tn(真负例):{tn}, fp(假正例):{fp}, fn(假负例):{fn}, tp(真正例):{tp}")
    print("混淆矩阵:")
    print(cm)
    print(f"正样本中预测正确的概率（即召回率）: {a1:.4f}")
    print(f"预测出的所有正例中实际为正例的概率（即精确度）: {a2:.4f}")
    print(f"正样本中预测错误的概率: {a3:.4f}")
    print(f"预测出的所有负例中实际为正例的概率: {a4:.4f}")
    print(f"负样本中预测错误的概率: {a5:.4f}")
    print(f"预测出的所有正例中实际为负例的概率: {a6:.4f}")
    print(f"负样本中预测正确的概率: {a7:.4f}")
    print(f"预测出的所有负例中实际为负例的概率: {a8:.4f}")
    print(f"F1分数: {f1:.4f}")


if __name__ == "__main__":
    main()
