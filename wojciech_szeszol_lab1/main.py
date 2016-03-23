import random
import math
import imageio
import numpy
import subprocess
import matplotlib.pyplot as pyplot

def estimate_response(images, num_samples, lambda_):
    def generate_samples(num_samples, num_images, width, height):
        sq_num_samples = int(math.ceil(math.sqrt(num_samples)))
        num_samples = sq_num_samples * sq_num_samples
        samples = numpy.empty([3, num_samples, num_images])

        cell_width = width / sq_num_samples
        cell_height = height / sq_num_samples

        for y in range(sq_num_samples):
            for x in range(sq_num_samples):
                i = min((x + random.random()) * cell_width, width - 1)
                j = min((y + random.random()) * cell_height, height - 1)
                
                # print(i, j)

                for k in range(len(images)):
                    for c in range(3):
                        samples[c, y * sq_num_samples + x, k] = images[k][j, i, c]

        return samples

    def all_equal(x):
        return x == [] or x.count(x[0]) == len(x)

    assert images != []
    assert all_equal([x.shape for x in images])

    height, width, _ = images[0].shape

    samples = generate_samples(num_samples, len(images), width, height)

    times = [image.meta['EXIF_EXIF']['ExposureTime'] for image in images]
    times = [x[0] / x[1] for x in times]

    def estimate_response(samples, times, l):
        samples_min = numpy.min(samples)
        samples_max = numpy.max(samples)
        samples_avg = 0.5 * (samples_min + samples_max)

        def w(x):
            return x - samples_min if x <= samples_avg else samples_max - x

        def numpy_w(x):
            y = numpy.copy(x).ravel()
            y[y <= samples_avg] = x - samples_min
            y[y > samples_avg] = samples_max - x
            return y.reshape(x.shape)

        num_images = samples.shape[1]
        num_samples = samples.shape[0]

        n = 256
        A = numpy.zeros(((num_images * num_samples) + n + 1, n + num_samples))
        b = numpy.zeros((A.shape[0],))

        k = 0
        for i in range(num_samples):
            for j in range(num_images):
                w_ij = w(samples[i, j])
                A[k, samples[i, j]] = w_ij
                A[k, n + i] = -w_ij
                b[k] = w_ij * math.log(times[j])
                k = k + 1

        A[k, 129] = 1;
        k = k + 1

        for i in range(n - 1):
            A[k, i] = l * w(i + 1)
            A[k, i + 1] = -2 * l * w(i + 1)
            A[k, i + 2] = l * w(i + 1)
            k = k + 1

        x = numpy.linalg.lstsq(A, b)

        return x[0][:n], numpy_w

    return numpy.array([estimate_response(x, times, lambda_) for x in samples])
    
def write_response(response, path):
    pyplot.figure()
    pyplot.plot(response[0][0], color = "red")
    pyplot.plot(response[1][0], color = "green")
    pyplot.plot(response[2][0], color = "blue")
    pyplot.ylabel('response')
    pyplot.savefig(path)

def make_hdr(images, response):
    g = response
    times = [image.meta['EXIF_EXIF']['ExposureTime'] for image in images]
    times = numpy.array([math.log(x[0] / x[1]) for x in times])

    images = numpy.array(images)

    images = numpy.transpose(images, [3, 0, 1, 2])
    
    def make_hdr(channel, g, w):
        W = w(channel)
        return numpy.sum(W * (g[channel] - times.reshape((3, 1, 1))), 0) / numpy.sum(W, 0)

    images = numpy.array([make_hdr(c, f[0], f[1]) for c, f in zip(images, g)])

    return numpy.exp(numpy.transpose(images, [1, 2, 0]).astype(numpy.float32))

def save_exr(image, path):
    floats = []
    for pixel in image.ravel():
        floats.append("{}".format(pixel))
    process = subprocess.Popen(["./makexr.bin"], stdin = subprocess.PIPE)
    process.communicate("{} {} {} {}".format(path, image.shape[1], image.shape[0], " ".join(floats)).encode('utf-8'))

probes = ["probe0.JPG", "probe1.JPG", "probe2.JPG"]
probe1_paths = ["data/probe1/{}".format(x) for x in probes]
probe2_paths = ["data/probe2/{}".format(x) for x in probes]

probe1_images = [imageio.imread(x) for x in probe1_paths]
probe2_images = [imageio.imread(x) for x in probe2_paths]

def crop_probe1(probe):
    for i in range(len(probe)):
        probe[i] = probe[i][520:,1139:-1185,:]
        probe[i] = probe[i][0:probe[i].shape[1] - 16,:,:]

def crop_probe2(probe):
    for i in range(len(probe)):
        probe[i] = probe[i][460:,1055:-1159,:]
        probe[i] = probe[i][0:probe[i].shape[1] - 16,:,:]

response1 = estimate_response(probe1_images, 250, 4)
response2 = estimate_response(probe1_images, 250, 4)

write_response(response1, "response1.png")
write_response(response2, "response2.png")

crop_probe1(probe1_images)
crop_probe2(probe2_images)

probe1_hdr = make_hdr(probe1_images, response1)
probe2_hdr = make_hdr(probe2_images, response1)

def long_to_mball_coords(x, y):
    theta = (1 - y) * math.pi
    phi = x * 2 * math.pi

    x = math.sin(theta) * math.cos(phi)
    y = math.cos(theta)
    z = -math.sin(theta) * math.sin(phi)

    u, v = 0, 0
    try:
        u = x / math.sqrt(2 * (1 + z))
        v = y / math.sqrt(2 * (1 + z))
    except:
        pass

    return u, v

def mball_to_long_probe(mball, width, height, xshift = 0, yshift = 0):
    result = numpy.empty((height, width, 3))

    inv_width = 1 / width
    inv_height = 1 / height

    mball_height, mball_width, _ = mball.shape

    for y in range(height):
        for x in range(width):
            src_x, src_y = long_to_mball_coords((x + xshift) * inv_width, (y + yshift) * inv_height)
            src_x = min(max(0, int((src_x + 1) * 0.5 * mball_width)), mball_width - 1)
            src_y = min(max(0, int((src_y + 1) * 0.5 * mball_height)), mball_height - 1)
            result[y, x] = mball[src_y, src_x]

    return result

probe1_hdr = mball_to_long_probe(probe1_hdr, 1024, 512)
probe2_hdr = mball_to_long_probe(probe2_hdr, 1024, 512, 660, -10)

mask = imageio.imread("mask.jpg").astype(numpy.float32) / 255.0

probe = probe1_hdr * mask + probe2_hdr * (1 - mask)
save_exr(probe, "probe.exr")

imageio.imwrite("probe.png", (numpy.clip((probe) * 20, 0, 255).astype(numpy.uint8)))
