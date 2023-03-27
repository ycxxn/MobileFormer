import torch

from mobileformer.mobile_former import mobile_former_26m

model = mobile_former_26m()
model = model.eval()

ckpt_path = "ckpt/mobile-former-26m.pth"
ckpt = torch.load(ckpt_path)["state_dict"]
model.load_state_dict(ckpt)

x = torch.randn(1,3,224,224)

out = model(x)

for o in out:
    print(o.shape)
