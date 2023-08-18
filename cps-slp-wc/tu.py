from matplotlib import pyplot as plt

w_2_values = []
set1 = [969, 1488, 50, 618, 677, 178, 1174, 35, 713, 111, 671, 99, 2745, 1642, 178, 2755, 2929, 949, 1990, 3489, 2937,
        299, 1786, 955, 1266, 114, 464, 639, 86, 86, 1057, 443, 1556, 162, 1445, 1488, 1290, 202, 1677, 198, 1406,
        557, 71, 1165, 1421, 1542, 1872, 86, 2098, 224]
# set1 =
# set1 =
# set1 =
# set1 =
# set1 =
# set1 =
# set1 =
# set1 =
# set1 =
mmm = [0.01, 0.02, 0.03, 0.04]
for i in range(50):
    w_2_values.append(i)
for j in range(4):
    w = mmm[j]

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.title('w_1:%.2f Variation of safe with w_2' % w)
    plt.plot(w_2_values, set1, 'g--o', label='w_1=0.01')
    plt.xlabel('w_2')
    plt.ylabel('Values')
    plt.legend()
    # plt.show()
    plt.title('w_1:0.01 Variation of safe with w_2')  # ............................
    plt.subplot(222)
    plt.plot(w_2_values, set1, 'b-', label='hunenergy_plt')
    plt.xlabel('w_2')
    plt.ylabel('Values')
    plt.title('w_1:0.01 Variation of energy with w_2')  # ............................
    plt.legend()
    # plt.show()
    plt.subplot(223)
    plt.plot(w_2_values, set1, 'r:.', label='hundelay_plt')
    plt.xlabel('w_2')
    plt.ylabel('Values')
    plt.title('w_1:0.01 Variation of delay with w_2')  # ............................
    plt.legend()
    # plt.show()
    plt.subplot(224)
    plt.plot(w_2_values, set1, 'r:.', label='hundelay_plt')
    plt.xlabel('w_2')
    plt.ylabel('Values')
    plt.title('w_1:0.01 Variation of delay with w_2')  # ............................
    plt.legend()
    plt.suptitle("w_2:%.2f, w_1:%.2f" % (w, w))
    # plt.draw()
    i+=1
    plt.savefig(r'D:\project\cps-slp-wc\graph\three\w_1\8.6\w_1-{}.png' .format(i))
    # cv2.imwrite("",)
    plt.clf()
    # plt.show()

   #i1 =
   # cv2.imshow("",i1)
   # cv2.imwrite("",i1)

