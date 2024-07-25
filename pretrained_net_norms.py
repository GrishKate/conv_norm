from torchvision import models
import torch
import torch.nn as nn
import pandas as pd
from bounds import compute_ryu_2019, compute_singla_2021, compute_tensor_norm_einsum

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_resnet_18 = models.resnet18(pretrained=True).to(device).eval()
df = pd.DataFrame(columns=['cout', 'cin', 'h', 'w', 's', 'H', 'W',
                           'toeplitz', 'fantastic4', 'tensor', 'f4/t', 'tensor/t'])
img_sizes = [224, 112, 56, 56, 56, 56, 28, 56, 28, 28, 28, 14, 28, 14, 14, 14, 7, 14, 7, 7]
cnt = 0
num_iters = 100
with torch.no_grad():
    for name, module in model_resnet_18.named_modules():
        if isinstance(module, nn.Conv2d):
            kernel = module.weight.detach()
            cout, cin, h, w = kernel.shape
            stride = module.stride
            img_size = img_sizes[cnt]
            exact_toeplitz = compute_ryu_2019(kernel, pad_to=[img_size, img_size], num_iters=num_iters, s=stride[0],
                                              return_time=False).item()
            tensor_norm = compute_tensor_norm_einsum(kernel, pad_to=None, num_iters=num_iters, s=stride[0],
                                                     return_time=False,
                                                     times=5).item()
            fantastic4_sigma = compute_singla_2021(kernel, pad_to=None, num_iters=num_iters, s=1,
                                                   return_time=False).item()
            df.loc[len(df.index)] = [cout, cin, h, w, stride[0], img_size, img_size,
                                     exact_toeplitz, fantastic4_sigma, tensor_norm,
                                     fantastic4_sigma / exact_toeplitz, tensor_norm / exact_toeplitz]
            cnt += 1
            df.to_csv('resnet18.csv', index=False)
print(df.to_markdown())

model = models.vgg19(pretrained=True).to(device).eval()
df = pd.DataFrame(columns=['cout', 'cin', 'h', 'w', 's', 'H', 'W',
                           'toeplitz', 'fantastic4', 'tensor', 'f4/t', 'tensor/t'])
img_sizes = [224, 224, 112, 112] + [56] * 4 + [28] * 4 + [14] * 4
cnt = 0
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            kernel = module.weight.detach()
            cout, cin, h, w = kernel.shape
            stride = module.stride
            img_size = img_sizes[cnt]
            exact_toeplitz = compute_ryu_2019(kernel, pad_to=[img_size, img_size], num_iters=num_iters, s=stride[0],
                                              return_time=False).item()
            tensor_norm = compute_tensor_norm_einsum(kernel, pad_to=None, num_iters=num_iters, s=stride[0],
                                                     return_time=False).item()
            fantastic4_sigma = compute_singla_2021(kernel, pad_to=None, num_iters=num_iters, s=stride[0],
                                                   return_time=False).item()
            df.loc[len(df.index)] = [cout, cin, h, w, stride[0], img_size, img_size,
                                     exact_toeplitz, fantastic4_sigma, tensor_norm,
                                     fantastic4_sigma / exact_toeplitz, tensor_norm / exact_toeplitz]
            cnt += 1
            df.to_csv('vgg19.csv', index=False)
print(df.to_markdown())
