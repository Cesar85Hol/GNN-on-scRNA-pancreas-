import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE

# Definiamo l'encoder GCN come modulo nn.Module
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GCNEncoder, self).__init__()
        # Primo strato GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Secondo strato GCN - produce 2*latent dim (per mu e logvar)
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_var = GCNConv(hidden_channels, latent_channels)
        
    def forward(self, x, edge_index):
        # Primo passaggio GCN + ReLU
        h = F.relu(self.conv1(x, edge_index))
        # Calcolo di mu e log_sigma
        mu = self.conv_mu(h, edge_index)
        log_var = self.conv_var(h, edge_index)
        return mu, log_var

def train(model, G_data, optimizer,epochs=100):
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        # encode: ottiene mu, log_var internamente e campiona z
        z = model.encode(G_data.x, G_data.edge_index)
        # calcola la loss di ricostruzione (edge) - PyG VGAE assume grafo non pesato
        recon_loss = model.recon_loss(z, G_data.edge_index)
        # calcola la perdita KL (regolarizzazione)
        kl_loss = model.kl_loss()
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")
    return loss.item()
