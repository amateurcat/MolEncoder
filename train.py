# Example to train on y[4] in QM9 dataset
# which are HOMO LUMO gaps

import torch, wandb
from torch_geometric.data import Dataset, DataLoader
from torch.nn import MSELoss
import numpy as np
from tqdm import tqdm
from torch_optimizer import Lamb

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)  
        optimizer.zero_grad() 
        pred = model(batch, batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y[:,4].float())
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()
        step += 1
    return running_loss/step

def evaluate_one_epoch(model, val_loader, loss_fn, device):
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(val_loader)):
        batch.to(device)  
        pred = model(batch, batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y[:,4].float())
        running_loss += loss.item()
        step += 1
    return running_loss/step

def train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, max_epochs=10, early_stop_lr=1e-7, save_to='best_model.pt', run_name='run', device='cuda:0'):
    wandb.init(project="testfield", name=run_name)
    wandb.run.log_code(__file__)   
    best_loss = 1e10
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1} of {max_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_one_epoch(model, val_loader, loss_fn, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_to)
            wandb.save(save_to)

        scheduler.step(val_loss)

        print(f"Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, current_lr: {optimizer.param_groups[0]['lr']}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "best_val_loss" : best_loss, "lr": optimizer.param_groups[0]['lr']})

        if optimizer.param_groups[0]['lr'] < early_stop_lr:
            print("Stopping early, learning rate below threshold.")
            break


if __name__ == "__main__":
    from model import MolEncoder
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='qm9_AIM.pt')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_lr", type=float, default=1e-6)
    parser.add_argument("--save_to", type=str, default="test.pt")
    parser.add_argument("--run_name", type=str, default="test")
    args = parser.parse_args()

    dataset = torch.load(args.data_file)   #this is a list of torch_geometric.data.Data objects
    np.random.seed(42)
    np.random.shuffle(dataset)

    #separate 5% for validation
    N = len(dataset)
    train_loader = DataLoader(dataset[:int(0.95*N)], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[int(0.95*N):], batch_size=args.batch_size, shuffle=True)
    
    model = MolEncoder(256, 4)
    optimizer = Lamb(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = MSELoss()
    model.to(args.device)
    train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, args.max_epochs, args.early_stop_lr, args.save_to, args.run_name, args.device)







