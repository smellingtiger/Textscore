import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.1):
        super(SimpleNN, self).__init__()
        # 第一层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 第三层
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 输出层
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        x = self.gelu1(self.fc1(x))
        x = self.dropout1(x)
        identity = x
        
        x = self.gelu2(self.fc2(x))
        x = self.dropout2(x)
        x = x + identity
        identity = x
        
        x = self.gelu3(self.fc3(x))
        x = self.dropout3(x)
        x = x + identity
        identity = x
        
        x = self.fc4(x)
        return torch.sigmoid(x)