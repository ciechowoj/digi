import subprocess
import os.path
import io
import imageio
import numpy
import math
import matplotlib
import matplotlib.pyplot as pyplot
import skimage.transform
import scipy.ndimage.filters

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
        lines["Aperture"] = float(lines["Aperture"].split("/")[1])

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

    return Is.reshape(shape)

def add_depth(depth, Is, scale_factor):
    return depth + scale_factor * (depth_model(Is) - 1)

def compute_depth(Is, scale_factor, level = None):
    if level == None:
        pad = 0 # max(Is.shape) // 2
        depth = compute_depth(numpy.pad(Is, pad, mode='edge'), scale_factor, math.log(min(Is.shape), 3) - 1)
        return depth[pad:1-pad, pad:1-pad]

    if level > 4:
        Is2 = skimage.transform.pyramid_reduce(Is, downscale = 3)
        next_depth = compute_depth(Is2, scale_factor * 3, level - 1)
        depth = skimage.transform.resize(next_depth, Is.shape, order = 1)
        Is3 = skimage.transform.resize(Is2, Is.shape, order = 1)
        Is4 = Is / Is3
        Is4 = Is4 * 0.5
        return add_depth(depth, Is4, scale_factor)
    else:
        return numpy.zeros_like(Is)

def write_obj(path, depth):
    mat = os.path.splitext(path)[0]
    mtl = "{}.mtl".format(mat)

    scale = 1.0 / (max(depth.shape) - 1)
    tx = 1.0 / (depth.shape[1] - 1)
    ty = 1.0 / (depth.shape[0] - 1)

    w = depth.shape[1]
    h = depth.shape[0]
    w1 = w - 1
    h1 = h - 1

    result = io.StringIO()
    result.write("mtllib {}\n".format(mtl))
    result.write("usemtl {}\n".format(mat))

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            result.write("v {:.3} {:.3} {:.3}\n".format(x * scale, -y * scale, -depth[y, x]))
    
    result.write("\n")

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            a = (0, -scale, -depth[max(0, y - 1), x] + depth[y, x])
            b = (-scale, 0, -depth[y, max(0, x - 1)] + depth[y, x])
            c = (0, scale, -depth[min(h1, y + 1), x] + depth[y, x])
            d = (scale, 0, -depth[y, min(w1, x + 1)] + depth[y, x])
            
            a = numpy.cross(a, b)
            b = numpy.cross(c, d)

            a = a / numpy.linalg.norm(a)
            b = b / numpy.linalg.norm(b)

            n = (a + b) * 0.5
            n = n / numpy.linalg.norm(n)

            result.write("vn {:.3} {:.3} {:.3}\n".format(n[0], n[1], n[2]))

    result.write("\n")

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            result.write("vt {:.3} {:.3}\n".format(x * tx, 1 - y * ty))

    result.write("\n")

    for y in range(h1):
        for x in range(w1):
            a = y * w + x + 1
            b = y * w + x + 2
            c = (y + 1) * w + x + 1
            d = (y + 1) * w + x + 2
            result.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(a, a, a, d, d, d, b, b, b))
            result.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(a, a, a, c, c, c, d, d, d))

    open(path, "w+").write(result.getvalue())

    mtl = open(mtl, "w+")
    mtl.write("newmtl {}\n".format(mat))
    mtl.write("Ka 0.000 0.000 0.000\n")
    mtl.write("Kd 1.000 1.000 1.000\n")
    mtl.write("Ks 0.000 0.000 0.000\n")
    mtl.write("map_Kd {}.jpg\n".format(mat))

dcraw("data/ex0/am.CR2")
dcraw("data/ex0/f1.CR2")
dcraw("data/calib/calib.CR2")

Nf = 1 / 2 ** 16
Id = imageio.imread("data/ex0/am.tiff").astype(numpy.float32) * Ce("data/ex0/am.CR2") * Nf
If = imageio.imread("data/ex0/f1.tiff").astype(numpy.float32) * Ce("data/ex0/f1.CR2") * Nf
Ic = imageio.imread("data/calib/calib.tiff").astype(numpy.float32) * Ce("data/calib/calib.CR2") * Nf

Ia = (If - Id) / Ic * 0.1
Is = gray(Id) / gray(Ia)
Is = numpy.clip(Is / Is.mean((0, 1)) * 0.5, 0, 1)

imageio.imwrite("albedo.jpg", Ia)
imageio.imwrite("shading.jpg", Is)

depth_small = compute_depth(skimage.transform.pyramid_reduce(Is, downscale = 8), 0.005)
albedo_small = skimage.transform.pyramid_reduce(Ia, downscale = 8)

write_obj("depth.obj", depth_small)
imageio.imwrite("depth.jpg", albedo_small * 4)
