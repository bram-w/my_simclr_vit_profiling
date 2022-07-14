from torch import nn
import torch
import time

def calc_time(model, feed):
    start_time = time.time()
    output = model(feed)
    return time.time() - start_time

def param_count(m):
    return sum([p.numel() for p in m.parameters()])


# need to consider batchnorm 2d here
def conv_block(in_channels, out_channels, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def make_simple_protonet(in_channels, hidden_channels, output_dim,
                         groups=1):
    # TODO: Standardize with transformer approach of defining width separate
    encoder = nn.Sequential(
        conv_block(in_channels, hidden_channels),
        conv_block(hidden_channels, hidden_channels, groups=groups),
        conv_block(hidden_channels, hidden_channels, groups=groups),
        conv_block(hidden_channels, hidden_channels, groups=groups),
        conv_block(hidden_channels, hidden_channels, groups=groups),
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
