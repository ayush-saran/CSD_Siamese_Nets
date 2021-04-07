def reference(dict, length):
    ref = [None]*len(dict[1])

    for i in range(0, len(dict[1])):
        ref[i] = 0
    print(ref)
    for i in range(len(dict[1])):
        for j in range(1, length):
            ref[i] = ref[i] + dict[j][i]
        ref[i] = ref[i]/length

    return ref

def append(ref_dict, arr):
    ref_dict.update({(len(ref_dict) + 1): arr})
    return ref_dict

dict = {1: [1, 2, 3, 4, 5],
        2: [1.5, 2.5, 3.5, 4.5, 5.5],
        3: [1.7, 0.8, 1.3, 4.3, 5.0],
        4: [1.9, 2.3, 3.4, 4.8, 5.1],
        5: [1.4, 2.1, 3.2, 4.9, 5.9],
        6: [1.1, 2.8, 3.6, 4.5, 5.2],
        7: [1.14, 2.14, 3.14, 4.34, 5.65],
        8: [0.99, 1.98, 3.21, 4.21, 5.13],
        9: [1.01, 2.1, 3.54, 4.65, 5.76],
        10: [1.9, 2.1, 3.9, 4.1, 5.5]
        }


a = []
a = reference(dict, len(dict))
print(a)

ref_dict = {}
ref_dict = append(ref_dict, a)
print(ref_dict)
