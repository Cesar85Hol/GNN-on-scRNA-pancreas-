import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate
    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Second GCN layer (outputs logits for classes)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.5):
        super(GATNet, self).__init__()
        # First layer: GATConv with multiple heads, outputs hidden_channels features *per head*
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout_rate)
        # Second layer: GATConv with one head that outputs out_channels (no concat in final layer)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1, concat=True, dropout=dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat2(x, edge_index)
        return x

def evaluate_model_on_graph(model_class, edge_index, X, y, num_features, hidden_dim, num_classes,n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        # Split indices for this fold
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        # Initialize model fresh for each fold
        model = model_class(num_features, hidden_dim, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        # Training loop
        for epoch in range(1, 101):  # 100 epochs
            optimizer.zero_grad()
            out = model(X_tensor, torch.tensor(edge_index, dtype=torch.long))
            # Compute loss only on training nodes
            loss = F.cross_entropy(out[train_idx], y_tensor[train_idx])
            loss.backward()
            optimizer.step()
            # (Optional) you could add early stopping or validation here
            
        # Evaluation on test nodes
        model.eval()
        out = model(X_tensor, torch.tensor(edge_index, dtype=torch.long))
        preds = out.argmax(dim=1).detach().cpu().numpy()
        y_true = y_tensor.detach().cpu().numpy()
        test_preds = preds[test_idx]  # predictions for test nodes
        test_true = y_true[test_idx]
        # Compute metrics for this fold
        acc = accuracy_score(test_true, test_preds)
        prec = precision_score(test_true, test_preds)
        rec = recall_score(test_true, test_preds)
        f1 = f1_score(test_true, test_preds)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        print(f"Fold {fold}: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, F1={f1:.4f}")
        fold += 1
    # Average metrics over folds
    for m in metrics:
        metrics[m] = np.mean(metrics[m])
    print(f"Average over {n_folds} folds: " +
          f"Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, " +
          f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    return metrics

