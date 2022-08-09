# -*- coding: utf-8 -*-


if __name__ == '__main__':
    # 1
    with open("./9.1/result4_1.txt", "w", encoding="utf-8") as f:
        for i in range(1, 101):
            f.write(str(i))
            if i == 100:
                break
            f.write("$")
    # 2
    with open("./9.1/result4_2.txt", "w", encoding="utf-8") as f:
        for i in range(0, 101):
            f.write("%d,%d,%d" % (i, i + 100, i * (i + 100)))
            if i == 100:
                break
            f.write("\n")
