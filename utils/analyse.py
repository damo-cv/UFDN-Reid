import torch
from torchvision import transforms

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    conv = 0
    fc = 0
    others = 0
    for name, m in model.named_modules():
        # skip non-leaf modules
        if len(list(m.children())) > 0:
            continue
        num = sum(p.numel() for p in m.parameters())
        if isinstance(m, torch.nn.Conv2d):
            conv += num
        elif isinstance(m, torch.nn.Linear):
            fc += num
        else:
            others += num
    M = 1e6
    print('total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'.format(total/M, conv/M, fc/M, others/M))

def count_flops(model, input_shape):
    flops_dict = {}
    def make_conv2d_hook(name):
        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[1] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)
        return conv2d_hook

    def make_fc_hook(name):
        def fc_hook(m, input):
            n, _ = input[0].size(0), input[0].size(
                1)
            flops = n * m.in_features * m.out_features
            flops_dict[name] = int(flops)
        return fc_hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.Linear):
            h = m.register_forward_pre_hook(make_fc_hook(name))
            hooks.append(h)
    input = torch.zeros(*input_shape).cuda()

    model.eval()
    with torch.no_grad():
        output = model(input, None)
    model.train()
    total_flops = 0
    for k, v in flops_dict.items():
        total_flops += v
    print(('total FLOPs: {:.2f}M'.format(total_flops/1e6)))
    for h in hooks:
        h.remove()