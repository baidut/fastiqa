import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def blend_output(input_image, output, alpha=0.8, vmin=None, vmax=None, resample=Image.BILINEAR):
    """qmap.blend().save('qmap.jpg')"""
  # input image: pil image
  # output: local scores matrix
    def stretch(image, minimum, maximum):
        if maximum is None:
            maximum = image.max()
        if minimum is None:
            minimum = image.min()
        image = (image - minimum) / (maximum - minimum)
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    cm = plt.get_cmap('magma')
    # min-max normalize the image, you can skip this step
    qmap_matrix = output
    qmap_matrix = 100*stretch(np.array(qmap_matrix), vmin, vmax)
    qmap_matrix = (np.array(qmap_matrix) * 255 / 100).astype(np.uint8)
    colored_map = cm(qmap_matrix)
    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    heatmap = Image.fromarray((colored_map[:, :, :3] * 255).astype(np.uint8))
    sz = input_image.size
    heatmap = heatmap.resize(sz, resample=resample)

    return Image.blend(input_image, heatmap, alpha=alpha)
