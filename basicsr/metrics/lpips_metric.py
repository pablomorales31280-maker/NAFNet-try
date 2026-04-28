import numpy as np
import torch
import lpips

from basicsr.metrics.metric_util import reorder_image, to_y_channel


_lpips_models = {}


def _get_lpips_model(net='alex', device='cuda'):
    key = (net, device)
    if key not in _lpips_models:
        model = lpips.LPIPS(net=net).to(device)
        model.eval()
        _lpips_models[key] = model
    return _lpips_models[key]


def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, net='alex', **kwargs):
    """
    Calculate LPIPS between two images.

    Args:
        img (ndarray): Image with range [0, 255].
        img2 (ndarray): Image with range [0, 255].
        crop_border (int): Cropped pixels in each edge.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): Whether to use Y channel only.
        net (str): LPIPS backbone, e.g. 'alex' or 'vgg'.

    Returns:
        float: LPIPS value. Lower is better.
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}'
    assert input_order in ['HWC', 'CHW'], f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW".'

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    if img.ndim == 2:
        img = img[..., None]
        img2 = img2[..., None]

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
        img2 = np.repeat(img2, 3, axis=2)

    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    img = img * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = img.to(device)
    img2 = img2.to(device)

    model = _get_lpips_model(net=net, device=device)

    with torch.no_grad():
        value = model(img, img2).item()

    return float(value)