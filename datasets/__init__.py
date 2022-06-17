import datasets.food

def select(dataset, opt, data_path):

    if 'food' in dataset:
        return food.Give(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : food101, food172 & food500!'.format(dataset))
