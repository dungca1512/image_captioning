import torch
model = torch.load("van_large_839.pth.tar")
model.keys()
torch.save(model['state_dict'], 'van_b3.pth.tar')
