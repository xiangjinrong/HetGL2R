import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 设置随机种子
seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# 数据加载函数
def load_data(embed_path, score_path):
    embeddings = torch.load(embed_path)
    scores = {}
    with open(score_path, 'r') as f:
        for line in f:
            node_id, score = line.strip().split()
            scores[node_id] = float(score)
    common_nodes = [n for n in embeddings.keys() if n in scores]

    embed_list = [torch.tensor(embeddings[n]) for n in common_nodes]
    score_list = [scores[n] for n in common_nodes]

    return torch.stack(embed_list), torch.tensor(score_list), common_nodes


# 数据集定义
class TrainingDataset(Dataset):
    def __init__(self, embeddings, scores, num_lists=50, list_length=10):
        super().__init__()
        self.embeddings = embeddings
        self.scores = scores
        self.num_nodes = len(embeddings)
        self.num_lists = num_lists
        self.list_length = list_length
        self.indices = [torch.randperm(self.num_nodes)[:list_length] for _ in range(num_lists)]

    def __len__(self):
        return self.num_lists

    def __getitem__(self, idx):
        indices = self.indices[idx]
        return self.embeddings[indices], self.scores[indices]


# 排序模型
class RankingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size, list_len, _ = x.shape
        x = x.view(-1, x.size(-1))
        scores = self.mlp(x).view(batch_size, list_len)
        return scores


# 修改后的KL散度损失函数
def kl_divergence_loss(pred, true):
    pred_prob = torch.softmax(pred, dim=1)
    true_prob = torch.softmax(true, dim=1)

    # 计算KL散度
    kl_div = true_prob * (torch.log(true_prob + 1e-10) - torch.log(pred_prob + 1e-10))
    loss = torch.sum(kl_div, dim=1)
    return loss.mean()


# 训练配置参数
embed_dim = 256
m = 50
K =9
batch_size = 32
hidden_dim = 128
epochs = 400
patience = 20

# 加载数据
train_embeds, train_scores, train_nodes = load_data(r'D:\Project\PythonProject\HetGL2R\data\rn_node_embeddings.pt',
                                                    r'D:\Project\PythonProject\HetGL2R\data\real_rn.txt')
test_embeds, test_scores, test_nodes = load_data(r'D:\Project\PythonProject\HetGL2R\data\sj_node_embeddings.pt',
                                                 r'D:\Project\PythonProject\HetGL2R\data\real_sj.txt')

# 划分训练集和验证集
train_embeds, valid_embeds, train_scores, valid_scores = train_test_split(
    train_embeds, train_scores, test_size=0.2, random_state=seed
)

# 创建数据加载器
train_dataset = TrainingDataset(train_embeds, train_scores, m, K)
valid_dataset = TrainingDataset(valid_embeds, valid_scores, num_lists=20, list_length=K)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RankingModel(train_embeds.size(1), hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

best_loss = float('inf')
patience_counter = 0

# 训练循环
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    # 训练阶段
    for embeds, scores in train_loader:
        embeds = embeds.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()
        pred = model(embeds)
        loss = kl_divergence_loss(pred, scores)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * embeds.size(0)

    # 验证阶段
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for embeds, scores in valid_loader:
            embeds = embeds.to(device)
            scores = scores.to(device)
            pred = model(embeds)
            loss = kl_divergence_loss(pred, scores)
            total_valid_loss += loss.item() * embeds.size(0)

    # 计算平均损失
    avg_train_loss = total_train_loss / len(train_dataset)
    avg_valid_loss = total_valid_loss / len(valid_dataset)

    print(f'Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_valid_loss:.4f}')

    # 早停机制
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_kl.pth')
        print(f'New best model saved with Loss: {best_loss:.4f}')
    else:
        patience_counter += 1
        print(f'Patience counter: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break


# 测试评估
@torch.no_grad()
def save_predictions(model, embeds, nodes):
    model.eval()
    pred_scores = model(embeds.unsqueeze(0)).squeeze(0)

    # 归一化到 [0, 1] 区间
    pred_scores = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min() + 1e-8)

    with open('pre_scores_sj.txt', 'w') as f:
        for node_id, score in zip(nodes, pred_scores.cpu().numpy()):
            f.write(f"{node_id} {score:.6f}\n")


# 加载最佳模型并保存预测结果
model.load_state_dict(torch.load('best_model_kl.pth'))
test_embeds = test_embeds.to(device)
save_predictions(model, test_embeds, test_nodes)
print("Predictions saved to pre_scores_sj.txt")