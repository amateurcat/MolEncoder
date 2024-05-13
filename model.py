import torch
from torch.nn import Linear, BatchNorm1d, Sequential, CELU, ReLU
import torch.nn.functional as F 
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


MODEL_PARAMS = {
    "transformed_size": 128,
    "n_attention_heads": 4,
    "n_tranformer_layers": 4,
    "dropout_rate": 0.1,
    "BatchNorm": False, 
    "NN_hidden_size": 512,
    "output_dim": 1
}

class MolEncoder(torch.nn.Module):
    def __init__(self, atom_embedding_size, bond_embedding_size, model_params=MODEL_PARAMS):
        super(MolEncoder, self).__init__()
        self.atom_embedding_size = atom_embedding_size
        self.bond_embedding_size = bond_embedding_size
        self.transformed_size = model_params["transformed_size"]
        self.n_attention_heads = model_params["n_attention_heads"]
        self.n_tranformer_layers = model_params["n_tranformer_layers"]
        self.dropout_rate = model_params["dropout_rate"]
        self.BatchNorm = model_params["BatchNorm"]
        self.NN_hidden_size = model_params["NN_hidden_size"]
        self.output_dim = model_params["output_dim"]

        self.transformer = []
        # first transformer layer, from atom_embedding_size to transformed_size
        # do NOT try to combine these four instances into a Sequential instance
        # you will have trouble with the forward method
        self.transformer.append(TransformerConv(self.atom_embedding_size, 
                                              self.transformed_size, 
                                              heads=self.n_attention_heads, 
                                              dropout=self.dropout_rate,
                                              edge_dim=self.bond_embedding_size,
                                              beta=False))
        
        self.transformer.append(Linear(self.transformed_size*self.n_attention_heads, self.transformed_size))
        self.transformer.append(CELU())
        if self.BatchNorm:
            self.transformer.append(BatchNorm1d(self.transformed_size))

        for i in range(self.n_tranformer_layers-1):
            self.transformer.append(TransformerConv(self.transformed_size, 
                                              self.transformed_size, 
                                              heads=self.n_attention_heads, 
                                              dropout=self.dropout_rate,
                                              edge_dim=self.bond_embedding_size,
                                              beta=False))
            self.transformer.append(Linear(self.transformed_size*self.n_attention_heads, self.transformed_size))
            self.transformer.append(CELU())
            if self.BatchNorm:
                self.transformer.append(BatchNorm1d(self.transformed_size))

        self.transformer = Sequential(*self.transformer)

        # for convenience, here is a 3-layer dense NN to do regression
        self.encoded_size = self.transformed_size * 2 * self.n_tranformer_layers
        self.NN = Sequential(
            Linear(self.encoded_size, self.NN_hidden_size),
            ReLU(),
            Linear(self.NN_hidden_size, self.NN_hidden_size),
            ReLU(),
            Linear(self.NN_hidden_size, self.output_dim)
        )

    def encode(self, batch, batch_index):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        mol_embedding = []

        for layer in self.transformer:
            if isinstance(layer, TransformerConv):
                x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, CELU):
                x = layer(x)
                # take the node embedding after CELU to make the global embedding
                mol_embedding.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
            else:
                x = layer(x)

        return torch.cat(mol_embedding, dim=1)
    
    def forward(self, batch, batch_index):
        x = self.encode(batch, batch_index)
        x = self.NN(x)
        return x
    

