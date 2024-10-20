import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        #raise NotImplementedError
    
    def fit(self, X, epochs=100, batch_size=32):
        #TODO: 5%
        
        #load data
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float))
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        #error: average squared error
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        loss_arr = []
        for epoch in (range(epochs)):
            for batch in dataLoader:
                #clear the gradients
                optimizer.zero_grad()
                #will find the forward function, do the forward propagation
                output = self(batch[0])
                #loss
                loss = loss_fn(output, batch[0])
                loss_arr.append(loss.item())
                #backpropagation
                loss.backward()
                #update the weights
                optimizer.step()
        return loss_arr
        #raise NotImplementedError
    
    def transform(self, X):
        #TODO: 2%
        #not training, so no need to calculate gradients
        with torch.no_grad():
            encoded = self.encoder(torch.tensor(X, dtype=torch.float))
        encoded = encoded.numpy()
        return encoded
        #raise NotImplementedError
    
    def reconstruct(self, X):
        #TODO: 2%
        with torch.no_grad():
            decoded = self.decoder(self.encoder(X.clone().detach()))
        decoded = decoded.numpy()
        return decoded
        #raise NotImplementedError


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        #noise = torch.randn(x.size()) * self.noise_factor
        #x should be a tensor
        x = x.clone().detach()
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
        raise NotImplementedError
    
    def fit(self, X, epochs=100, batch_size=32):
        #TODO: 4%
        
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float))
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        loss_arr = []
        
        for epoch in range(epochs):
            for batch in dataLoader:
                noisy_batch = self.add_noise(batch[0])
                #clear the gradients
                optimizer.zero_grad()
                #will find the forward function, do the forward propagation
                output = self(noisy_batch)
                #loss
                loss = loss_fn(output, batch[0])
                loss_arr.append(loss.item())
                #backpropagation
                loss.backward()
                #update the weights
                optimizer.step()
        return loss_arr
                
        #raise NotImplementedError