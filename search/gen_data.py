import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == "__main__":
    x_list = []
    y_list = []
    z_list = []
    r_list = []

    N = 1000000

    f = open("../shared/dataset.txt", "w")
    rr = np.random.uniform(0, 30000, N)
    fai = np.random.uniform(0, np.pi, N)
    theta = np.random.uniform(0, np.pi * 2, N)
    cnt = 0

    xx = rr * np.sin(theta) * np.cos(fai)
    yy = rr * np.sin(theta) * np.cos(fai)
    zz = rr * np.cos(theta)

    for i in range(N):
        x = xx[i]
        y = yy[i]
        z = zz[i]
        r = np.sqrt(x * x + y * y + z * z)

        if z > 0:
            continue
        if r > 30000:
            continue
        cnt += 1

        p1 = np.array([[400], [0], [0]])
        p2 = np.array([[-400], [0], [0]])
        p3 = np.array([[0], [400 * np.sqrt(3)], [0]])

        T = np.array([[x], [y], [z]])

        op1 = p1 + T
        op2 = p2 + T
        op3 = p3 + T

        e1 = op1 / np.linalg.norm(op1)
        e2 = op2 / np.linalg.norm(op2)
        e3 = op3 / np.linalg.norm(op3)

        print("[{}] [INPUT]e1=({:.6f},{:.6f},{:.6f}) e2=({:.6f},{:.6f},{:.6f}) e3=({:.6f},{:.6f},{:.6f}) "
              "[OUTPUT](tx,ty,tz)=({:.6f},{:.6f},{:.6f})".format(i, e1[0][0], e1[1][0], e1[2][0], e2[0][0], e2[1][0], e2[2][0], e3[0][0], e3[1][0], e3[2][0], x, y, z))

        f.write("{}\n".format(cnt))
        f.write("{:.6f} {:.6f} {:.6f}\n".format(e1[0][0], e1[1][0], e1[2][0]))
        f.write("{:.6f} {:.6f} {:.6f}\n".format(e2[0][0], e2[1][0], e2[2][0]))
        f.write("{:.6f} {:.6f} {:.6f}\n".format(e3[0][0], e3[1][0], e3[2][0]))
        f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        r_list.append(r)

    f.write("-1\n")
    f.close()

    plt.suptitle("Distribution")
    plt.subplot(2, 2, 1)
    plt.title("x")
    plt.hist(np.array(x_list), 1000)
    plt.subplot(2, 2, 2)
    plt.title("y")
    plt.hist(np.array(y_list), 1000)
    plt.subplot(2, 2, 3)
    plt.title("z")
    plt.hist(np.array(z_list), 1000)
    plt.subplot(2, 2, 4)
    plt.title("r")
    plt.hist(np.array(r_list), 1000)
    plt.show()
