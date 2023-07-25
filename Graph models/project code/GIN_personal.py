#Imports 
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU


class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)
        
        return X
        


class GIN(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        #extraction of parameters from configuration file 
        #following the code format of 
        #https://github.com/diningphil/gnn-comparison
        self.config = config
        self.hidden_dim = [config['hidden_units'][0]] + config['hidden_units']
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.n_layers = len(self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(GINConv(hidden_dim))
        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        self.out_proj = torch.nn.Linear(hidden_dim*(1+n_layers), output_dim)

    def forward(self, A, X):
        X = self.in_proj(X)

        hidden_states = [X]
        
        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=2).sum(dim=1)

        X = self.out_proj(X)

        return X