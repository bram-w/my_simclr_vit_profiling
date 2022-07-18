from torch import nn
import torch
import time

def calc_time(model, feed):
    start_time = time.time()
    output = model(feed)
    return time.time() - start_time

def param_count(m):
    return sum([p.numel() for p in m.parameters()])


def conv_block(in_channels, out_channels, groups=1):
    assert not out_channels % 8
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups),
        # nn.BatchNorm2d(out_channels),
        # if have 6400 channels with 100 groups, want 800 groups here
        # if have 3200 channels with 100 groups, want 400 groups here (8/)
        nn.GroupNorm(out_channels // 8, out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_block_alt(in_channels, out_channels, groups=1):
    assert not out_channels % 8
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, groups=groups),
        # nn.BatchNorm2d(out_channels),
        # if have 6400 channels with 100 groups, want 800 groups here
        # if have 3200 channels with 100 groups, want 400 groups here (8/)
        nn.GroupNorm(out_channels // 8, out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )


def conv_block_resnet(in_channels, out_channels, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, groups=groups),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.LeakyReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.GroupNorm(out_channels // 8, out_channels),
            # nn.ReLU()
            )

def make_protonet_v2(num_groups):

    # start by replicating resnet
    conv1 = nn.Conv2d(3, 64*num_groups, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    norm1 = nn.GroupNorm(64 *num_groups // 8, 64 * num_groups)
    relu1 = nn.LeakyReLU()
    maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    encoder = nn.Sequential(
            conv1, norm1, relu1, maxpool1,
            conv_block_resnet(64*num_groups, 128*num_groups, groups=num_groups),
            conv_block_resnet(128*num_groups, 256*num_groups, groups=num_groups),
            conv_block_resnet(256*num_groups, 256*num_groups, groups=num_groups),
            nn.AdaptiveAvgPool2d((1, 1)), # shape is now bs x hidden_channels x 1 x 1
            nn.Conv2d(256*num_groups, num_groups, 1, groups=num_groups),
            nn.Flatten()
            )
    return encoder


def make_simple_protonet(in_channels, hidden_channels, 
                         groups=1, use_alt=False):
    # TODO: Standardize with transformer approach of defining width separate
    block_call = conv_block_alt if use_alt else conv_block
    encoder = nn.Sequential(
        block_call(in_channels, hidden_channels),
        block_call(hidden_channels, hidden_channels, groups=groups),
        block_call(hidden_channels, hidden_channels, groups=groups),
        block_call(hidden_channels, hidden_channels, groups=groups),
        block_call(hidden_channels, hidden_channels, groups=groups),
        nn.AdaptiveAvgPool2d((1, 1)), # shape is now bs x hidden_channels x 1 x 1
        nn.Conv2d(hidden_channels, groups, 1, groups=groups),
        nn.Flatten()
    )
    return encoder
    

def test():
    num_experts = 100
    p = make_simple_protonet(3, 64, 1).cuda()
    px = make_simple_protonet(3, 6400, 100, groups=num_experts).cuda()
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(num_classes=1, pretrained=False).cuda()
    feed_data = torch.randn(1, 3, 224, 224).cuda()

    total_serial_time = 0
    total_parallel_time = 0
    total_mobile_time = 0
    for test_i in range(10):
        serial_time = sum([calc_time(p, feed_data) for _ in range(num_experts)])
        mobile_time = sum([calc_time(m, feed_data) for _ in range(num_experts)])
        parallel_time = calc_time(px, feed_data)
        if not test_i: continue
        total_serial_time += serial_time
        total_parallel_time += parallel_time
        total_mobile_time += mobile_time

    print(total_serial_time / total_parallel_time)
    print(total_mobile_time / total_parallel_time)
    print(total_mobile_time / total_serial_time)

if __name__ == '__main__':
    test()


############
def orig_load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )
    # Get a linear layer here
    return Protonet(encoder)
