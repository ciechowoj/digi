import subprocess
import os.path
import imageio
import numpy
import matplotlib
import matplotlib.pyplot as pyplot

def dcraw(file):
    # w - white balance
    # 16-bit linear
    # -t 0 do not flip an image
    # -T tiff
    output = "{}.tiff".format(os.path.splitext(file)[0])
    if not os.path.isfile(output):
        subprocess.check_call(["dcraw", "-w", "-4", "-t", "0", "-T", file])

def dcraw_meta(path):
    lines = subprocess.check_output(["dcraw", "-i", "-v", path]).decode("utf-8").splitlines()
    lines = [[x.strip() for x in line.split(":", 1)] for line in lines if ":" in line]
    lines = { x[0] : x[1] for x in lines }

    if "Aperture" in lines:
        lines["Aperture"] = 1 / float(lines["Aperture"].split("/")[1])

    if "ISO speed" in lines:
        lines["ISO speed"] = float(lines["ISO speed"])

    if "Shutter" in lines:
        shutter = lines["Shutter"].split()[0]
        shutter = shutter.split("/")
        lines["Shutter"] = float(shutter[0]) / float(shutter[1])

    return lines

def Ce(path):
    meta = dcraw_meta(path)
    result = meta["Aperture"] ** 2 / (meta["Shutter"] * meta["ISO speed"])
    print(result)
    return result

def gray(image):
    return numpy.dot(image, [0.2126, 0.7152, 0.0722])

def depth_model(Is):
    shape = Is.shape
    Is = Is.copy().ravel()
    L = Is <= 0.5
    G = Is > 0.5

    Is[L] = numpy.sqrt(1 / Is[L] - 1)
    Is[G] = 2 * (1 - Is[G])

    Is.reshape(shape)
    return Is


# computeDepth( shading_image, scale_factor, level)
#   if (level > 1)
#       shading_image2 = scaleDown(shading_image, GAUSSIAN, LAYER_RATIO=3)
#       nextDepth = computeDepth(shading_image2, scale_factor*LAYER_RATIO, level-1)
#       depth = scaleUP(nextDepth, BILINEAR/BICUBIC)
#       shading_image3 = scaleUP(shading_image2)
#       shading_image4 = shading_image/shading_image3
#       shading_image4 = shading_image4 / 2.
#   return (addDepth(depth, shading_image4, scale_factor)
# 
# addDepth(depth, shading_image, scale_factor)
#   return (depth + scale_factor*depthModel(shading_image) -1. //depth model is equation 13


dcraw("data/ex0/am.CR2")
dcraw("data/ex0/f1.CR2")
dcraw("data/calib/calib.CR2")

Nf = 1 / 2 ** 16
Id = imageio.imread("data/ex0/am.tiff").astype(numpy.float32) * Ce("data/ex0/am.CR2") * Nf
If = imageio.imread("data/ex0/f1.tiff").astype(numpy.float32) * Ce("data/ex0/f1.CR2") * Nf
Ic = imageio.imread("data/calib/calib.tiff").astype(numpy.float32) * Ce("data/calib/calib.CR2") * Nf

Ia = gray((If - Id) / Ic)
Is = gray(Id) / Ia
Is = Is / Is.mean((0, 1)) * 0.5

depth_model(Is)

pyplot.imshow(numpy.clip(Is * 255, 0, 255).astype(numpy.uint8), cmap = "gray")
pyplot.show()

