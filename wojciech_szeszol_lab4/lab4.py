import numpy
import numpy.linalg
import scipy.linalg
import scipy.misc
import skimage.draw

def projection(x2, x3):
    n = len(x2)
    m = 12

    A = numpy.zeros((n * 2, m))

    for i in range(n):
        A[i * 2 + 0][0] = x3[i][0]
        A[i * 2 + 0][1] = x3[i][1]
        A[i * 2 + 0][2] = x3[i][2]
        A[i * 2 + 0][3] = 1
        A[i * 2 + 0][4] = 0
        A[i * 2 + 0][5] = 0
        A[i * 2 + 0][6] = 0
        A[i * 2 + 0][7] = 0
        A[i * 2 + 0][8] = -x2[i][0] * x3[i][0]
        A[i * 2 + 0][9] = -x2[i][0] * x3[i][1]
        A[i * 2 + 0][10] = -x2[i][0] * x3[i][2]
        A[i * 2 + 0][11] = -x2[i][0]

        A[i * 2 + 1][0] = 0
        A[i * 2 + 1][1] = 0
        A[i * 2 + 1][2] = 0
        A[i * 2 + 1][3] = 0
        A[i * 2 + 1][4] = x3[i][0]
        A[i * 2 + 1][5] = x3[i][1]
        A[i * 2 + 1][6] = x3[i][2]
        A[i * 2 + 1][7] = 1
        A[i * 2 + 1][8] = -x2[i][1] * x3[i][0]
        A[i * 2 + 1][9] = -x2[i][1] * x3[i][1]
        A[i * 2 + 1][10] = -x2[i][1] * x3[i][2]
        A[i * 2 + 1][11] = -x2[i][1]

    U, S, V = numpy.linalg.svd(A)

    M = V.T[:, -1]

    return M.reshape((3, 4))

def load_points(path):
    return numpy.array([[float(x) for x in line.split()] for line in open(path, "r")])

def decompose(P):
    M = P[:, :3]
    T = P[:, 3]

    K, R = scipy.linalg.rq(M)

    return K, R, numpy.linalg.inv(K).dot(T.reshape((3, 1)))

def fundamental(p, q):
    n = len(p)
    m = 9

    A = numpy.empty((n, m))

    for i in range(n):
        A[i][0] = p[i][0] * q[i][0]
        A[i][1] = p[i][0] * q[i][1]
        A[i][2] = p[i][0]
        A[i][3] = p[i][1] * q[i][0]
        A[i][4] = p[i][1] * q[i][1]
        A[i][5] = p[i][1]
        A[i][6] = q[i][0]
        A[i][7] = q[i][1]
        A[i][8] = 1

    U, S, V = numpy.linalg.svd(A)

    M = V.T[:, -1]
    U, S, V = numpy.linalg.svd(M.reshape((3, 3)))

    S[2] = 0

    return numpy.dot(numpy.dot(U, numpy.diag(S)), V)

def task1():
    M = projection(
        load_points("data/pts2d-norm-pic_a.txt"),
        load_points("data/pts3d-norm.txt"))

    print("task1:")
    print("M = ", M)
    print()


def task2():
    P = projection(
        load_points("data/pts2d-norm-pic_a.txt"),
        load_points("data/pts3d-norm.txt"))

    print("task2:")
    for x, n in zip(decompose(P), ["K =", "R =", "T ="]):
        print(n, x)
    print()


def task3():
    F = fundamental(
        load_points("data/pts2d-pic_a.txt"),
        load_points("data/pts2d-pic_b.txt"))

    print("task4:")
    print("F =", F)
    print()


def draw_line(image, a, b):
    rr, cc, val = skimage.draw.line_aa(a[0], a[1], b[0], b[1])

    for r, c, v in zip(rr, cc, val):
        if 0 <= r and r < image.shape[0] and 0 <= c and c < image.shape[1]:
            image[r, c, :] = v

def task4():
    a = load_points("data/pts2d-pic_a.txt")
    b = load_points("data/pts2d-pic_b.txt")

    F = fundamental(a, b)

    aImage = scipy.misc.imread("data/pic_a.jpg")
    bImage = scipy.misc.imread("data/pic_b.jpg")

    for point in b:
        line = numpy.dot(F, [point[0], point[1], 1])

        p = (0, int((-line[2]) / line[0]))
        x = bImage.shape[1] - 1
        q = (int(x), int((-line[1] * x - line[2]) / line[0]))

        draw_line(aImage, p, q)

    for point in a:
        line = numpy.dot([point[0], point[1], 1], F)

        p = (0, int((-line[2]) / line[0]))
        x = bImage.shape[1] - 1
        q = (int(x), int((-line[1] * x - line[2]) / line[0]))

        draw_line(bImage, p, q)


    scipy.misc.imsave("pic_a.result.jpg", aImage)
    scipy.misc.imsave("pic_b.result.jpg", bImage)


task1()
task2()
task3()
task4()
