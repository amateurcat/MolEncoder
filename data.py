# simple script to encode the QM9 dataset with AIMNet2
import torch

qm9 = torch.load('./qm9_dataset.pt')
model_file = './b973c_gas_0.jpt'
device = 'cuda:0'
save_to = 'qm9_AIM_fortest.pt'
model = torch.jit.load(model_file, map_location=device)


save_new = []
for data in qm9[:100]:
    coord = data.pos.unsqueeze(0).to(device)
    numbers = data.z.unsqueeze(0).to(device)
    charge = torch.as_tensor([0]).to(device)
    _in = dict(
            coord=coord,
            numbers=numbers,
            charge=charge
        )
    _out = model(_in)
    data.x = _out['aim'].squeeze(0).cpu()
    save_new.append(data)

torch.save(save_new, save_to)