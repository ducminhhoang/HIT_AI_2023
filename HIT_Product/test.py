from images_unit import imsave, imload, get_transformer
from network import TransformNetwork
import torch
from upscale import up_scale_x3_normal_fast
def load_transform_network(args):
    transform_network = TransformNetwork()
    transform_network.load_state_dict(torch.load(args.model_load_path, map_location=torch.device('cpu')))
    return transform_network

def network_test(args):
    device = ('cuda' if args.cuda_device_no >= 0 else 'cpu')
    transform_network = load_transform_network(args).to('cpu')

    input_image = imload(args.test_content, args.imsize)
    with torch.no_grad():
        output_image = transform_network(input_image)
    imsave(output_image, args.output)
    up_scale_x3_normal_fast(args.output, args.output)
    return None
