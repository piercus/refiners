import torch

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, ColorsTokenizer
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1UNet
from refiners.fluxion.adapters.histogram import HistogramExtractor, HistogramDistance, HistogramEncoder


def test_histogram_extractor() -> None:
    color_bits = 3
    color_size = 2 ** color_bits
    img = torch.randint(0, color_size, (1, 3, 224, 224), dtype=torch.uint8).float()

    extractor = HistogramExtractor(color_bits=color_bits)
    
    histogram = extractor(img)
    assert histogram.shape == (1, color_size, color_size, color_size)
    assert abs(histogram.sum().item() - 1.0) < 1e-4, "histogram sum should equal 1.0"
    
    img_black = torch.zeros((1, 3, 224, 224), dtype=torch.uint8).float()
    histogram_black = extractor(img_black)
    assert abs(histogram_black[0,0,0,0]  - 1.0) < 1e-4, "histogram_zero should be 1.0 at 0,0,0,0"
    assert abs(histogram_black.sum()  - 1.0) < 1e-4, "histogram sum should equal 1.0"
    
    img_white = torch.ones((1, 3, 224, 224)) * (color_size - 1)
    histogram_white = extractor(img_white)
    assert abs(histogram_white[0,-1,-1,-1]  - 1.0) < 1e-4, "histogram_white should be 1.0 at -1,-1,-1,-1"
    assert abs(histogram_white.sum()  - 1.0) < 1e-4, "histogram sum should equal 1.0"
    
def test_histogram_distance() -> None:
    distance = HistogramDistance()
    color_bits = 2
    color_size = 2 ** color_bits
    batch_size = 2

    histo1 = torch.rand((batch_size, color_size, color_size, color_size))
    sum1 = histo1.sum()
    histo1 = histo1 / sum1
    
    histo2 = torch.rand((batch_size, color_size, color_size, color_size))
    sum2 = histo2.sum()
    histo2 = histo2 / sum2
    
    dist_same = distance(histo1, histo1)
    assert dist_same == 0.0, "distance between himself should be 0.0"


def test_histogram_encoder() -> None:
    
    batch_size = 2
    patch_size = 8
    color_bits = 6
    cube_size = 2 ** color_bits
    histo1 = torch.rand((batch_size, 1, cube_size, cube_size, cube_size))
    sum1 = histo1.sum()
    histo1 = histo1 / sum1
    
    output_dim = 768
    
    encoder = HistogramEncoder(color_bits=color_bits, output_dim=output_dim, patch_size=patch_size)
    embedding = encoder(histo1)
    assert embedding.shape == (batch_size, output_dim), "embedding shape should be (batch_size, ouput_dim)"