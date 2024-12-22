#! /bin/python3

import os
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from intvalpy import Interval, Tol, precision
from intvalpy_fix import IntLinIncR2


precision.extendedPrecisionQ = True


# using twin arithmetics
def regression_type_2(x_new, y_ex_up, y_ex_down, y_in_up, y_in_down):
    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        # y_ex_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_ex_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        # y_ex_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        # y_in_up >= X_mat * b >= y_in_down
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])

    # now we have matrix X * b = Y, but with some "additional" rows
    # we can walk over all rows and if some of them is less than 0, we can just remove it at all
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    to_remove = []
    if tol_val < 0:
        # if Tol value is less than 0, we must iterate over all rows and add some changes to Y_vec, so Tol became 0
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]])
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del X_mat[i]
            del Y_vec[i]

    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)

    vertices1 = IntLinIncR2(X_mat_interval, Y_vec_interval)
    vertices2 = IntLinIncR2(X_mat_interval, Y_vec_interval, consistency='tol')

    plt.xlabel("b0")
    plt.ylabel("b1")
    b_uni_vertices = []
    for v in vertices1:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_uni_vertices += [(x[i], y[i]) for i in range(len(x))]
            plt.fill(x, y, linestyle='-', linewidth=1, color='gray', alpha=0.5, label="Uni")
            plt.scatter(x, y, s=0, color='black', alpha=1)

    b_tol_vertices = []
    for v in vertices2:
        # если пересечение с ортантом не пусто
        if len(v) > 0:
            x, y = v[:, 0], v[:, 1]
            b_tol_vertices += [(x[i], y[i]) for i in range(len(x))]
            plt.fill(x, y, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
            plt.scatter(x, y, s=10, color='black', alpha=1)

    plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
    plt.legend()
    return b_vec, (y_in_down, y_in_up), (y_ex_down, y_ex_up), to_remove, b_uni_vertices, b_tol_vertices


def build_plots(name, x_new, y_ex_up, y_ex_down, y_in_up, y_in_down):
    try:
        os.mkdir((f'_pics/{name}/'))
    except:
        pass

    # method 2
    plt.figure()
    plt.title(f"Uni and Tol method 2 for {name}")
    b_vec2, y_in, y_ex, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(x_new, y_ex_up, y_ex_down, y_in_up, y_in_down)
    print(name, 2, b_vec2[0], b_vec2[1], len(to_remove))
    x2 = [0.0, 1.0, 2.0]
    plt.savefig(f'_pics/{name}/uni_tol.png')

    plt.figure()
    plt.title(f"Y(x) method 2 for {name}")
    plt.grid()
    for i in range(len(x2)):
        plt.plot([x2[i], x2[i]], [y_ex[0][i], y_ex[1][i]], color="gray", zorder=1)
        plt.plot([x2[i], x2[i]], [y_in[0][i], y_in[1][i]], color="blue", zorder=2)

    plt.plot([-1, 3], [b_vec2[1] + b_vec2[0] * -1, b_vec2[1] + b_vec2[0] * 3], label="Argmax Tol", color="red",
             zorder=1000)

    x2 = [-1] + x2 + [3]

    for i in range(len(x2) - 1):
        x0 = x2[i]
        x1 = x2[i + 1]
        max_idx = 0
        min_idx = 0
        max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        min_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_uni_vertices)):
            val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
        y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
        y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
        y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightgray", linewidth=0)

        max_idx = 0
        min_idx = 0
        max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        min_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
        for j in range(len(b_tol_vertices)):
            val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
            if max_val < val:
                max_idx = j
                max_val = val
            if min_val > val:
                min_idx = j
                min_val = val

        y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
        y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
        y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
        y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
        plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi], facecolor="lightblue", linewidth=0)

    # plt.xlim((-0.6, 0.6))
    # plt.ylim((-0.6, 0.6))
    plt.savefig(f'_pics/{name}/method2.png')


if __name__ == "__main__":
    build_plots("ideal", [0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0])
    build_plots("up1", [0.0, 1.0, 2.0], [4.0, 6.0, 6.0], [0.0, 2.0, 2.0], [3.0, 5.0, 5.0], [1.0, 3.0, 3.0])
    build_plots("up2", [0.0, 1.0, 2.0], [4.0, 7.0, 6.0], [0.0, 3.0, 2.0], [3.0, 6.0, 5.0], [1.0, 4.0, 3.0])
    build_plots("up3", [0.0, 1.0, 2.0], [4.0, 8.0, 6.0], [0.0, 4.0, 2.0], [3.0, 7.0, 5.0], [1.0, 5.0, 3.0])
