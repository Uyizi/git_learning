import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from model import HAGNN
from utils import load_data, HierarchyProcessor

DATASET = 'pubmed'
ROOT_PATH = "./data" 

HIDDEN = 4      
HEADS = 8         
DROPOUT = 0.5
LR = 0.005         
EPOCHS = 1000      
THRESHOLD = 0.001  
WEIGHT_DECAY = 5e-4 

torch.set_float32_matmul_precision('high')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_once(attempt_id, device, seed):
    set_seed(seed)
    print(f"\n[Attempt {attempt_id}] Seed: {seed}")
    
    # 加载数据
    adj_raw, features, labels = load_data(DATASET, ROOT_PATH)
    
    processor = HierarchyProcessor(adj_raw, num_nodes=features.shape[0])
    ranks = processor.calculate_spring_rank()
    source_hubs = processor.identify_source_hubs(ranks)
    
    adj_aug, hub_mask = processor.adaptive_context_extension(source_hubs)
    
    adj_aug = adj_aug.to(device)
    hub_mask = hub_mask.to(device)
    features = features.to(device)
    labels = labels.to(device)
    nclass = labels.max().item() + 1 
    
    # 数据集划分 
    num_nodes = features.shape[0]
    
    if DATASET == 'cora':
        # Cora 140 训练 / 500 验证 / 1000 测试
        idx_train = torch.arange(140).to(device)
        idx_val = torch.arange(200, 500).to(device)
        idx_test = torch.arange(500, 1500).to(device)
        
    elif DATASET == 'citeseer':
        # Citeseer  120 训练/ 500 验证 / 1000 测试
        idx_train = torch.arange(120).to(device)
        idx_val = torch.arange(200, 500).to(device)
        idx_test = torch.arange(500, 1500).to(device)
        
    elif DATASET == 'wiki-cs':
        # Wiki-CS 比例划分 (例如 60% / 20% / 20%)
        indices = torch.randperm(num_nodes).to(device)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        idx_train = indices[:train_size]
        idx_val = indices[train_size : train_size + val_size]
        idx_test = indices[train_size + val_size :]
    
    elif DATASET == 'pubmed':
        idx_train = torch.arange(60).to(device)
        idx_val = torch.arange(60, 560).to(device)
        idx_test = torch.arange(num_nodes - 1000, num_nodes).to(device)

    model = HAGNN(nfeat=features.shape[1], 
                  nhid=HIDDEN, 
                  nclass=labels.max().item()+1, 
                  dropout=DROPOUT, 
                  alpha=0.2, 
                  nheads=HEADS, 
                  threshold=THRESHOLD).to(device)
                  
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    final_test_acc = 0.0
    bad_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        output = model(features, adj_aug, hub_mask)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            output = model(features, adj_aug, hub_mask)
            val_acc = (output[idx_val].max(1)[1] == labels[idx_val]).float().mean()
            test_acc = (output[idx_test].max(1)[1] == labels[idx_test]).float().mean()
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                bad_counter = 0
        
                
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}: Val Acc {val_acc:.4f} | Best Test {final_test_acc:.4f}")
                
    print(f"Final Result for Seed {seed}: {final_test_acc:.4f}")
    return final_test_acc.item()

if __name__ == "__main__":
    device = torch.device('cpu')
    
    seeds = [42, 1, 777, 2024, 888, 114514, 1919810]
    results = []
    
    for i, s in enumerate(seeds):
        res = train_once(i+1, device, s)
        results.append(res)
    
    print("\n" + "="*30)
    print(f"({DATASET}):")
    print(f"    最高准确率 (Max): {np.max(results):.4f}")
    print(f"    平均准确率 (Mean): {np.mean(results):.4f}")
    print("="*30)
