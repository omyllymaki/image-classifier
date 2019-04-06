from random import shuffle


def divide_data_to_sets(data, *proportions):
    n_images = len(data)
    indices = list(range(n_images))
    shuffle(indices)

    output = []
    for proportion in proportions:
        n = int(proportion * n_images)
        set_indices = indices[:n]
        data_set = [data[i] for i in set_indices]
        output.append(data_set)

    return output