# encoding: utf-8
"""
@author:  qianwen
@contact: qianwen2018@ia.ac.cn
"""

from .VeRi import VeRi
from .VehicleID import VehicleID
from .VeRi_wild import VeRi_WILD
from .dataset_loader import ImageDataset

__factory = {
   	'VeRi': VeRi,
    'VehicleID': VehicleID,
    'VeRi_WILD': VeRi_WILD,
}

#image_id, vid, cid, tid
def concat_dataset(dataset_list):
    vid_container, image_ids, cid_container, tid_container = list(), list(), list(), list()
    vid_bias, cid_bias, tid_bias = 0, 0, 0
    for i, dataset in enumerate(dataset_list):
        if i > 0: 
            vid_bias += max(vid_container) + 1
            cid_bias += max(cid_container) + 1
            tid_bias += max(tid_container) + 1
        vid_container += [vid_bias + data[1] for data in dataset]
        image_ids += [data[0] for data in dataset]
        cid_container += [cid_bias + data[2] for data in dataset]
        tid_container += [tid_bias + data[3] for data in dataset]
    new_dataset = [(image_ids[i], vid_container[i], cid_container[i], tid_container[i]) for i in range(len(vid_container))]
    return new_dataset, len(set(vid_container))


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    # if name == 'AIC+VeRi':
    #     dataset_AIC = AIC(*args, **kwargs)
    #     dataset_VeRi = VeRi(*args, **kwargs)
    #     dataset = BaseImageDataset()
    #     dataset.train, num_train_pids = concat_dataset([dataset_AIC.train, dataset_VeRi.train])
    #     dataset.num_train_pids = num_train_pids
    #     dataset.query = dataset_AIC.query
    #     dataset.gallery, _ = concat_dataset([dataset_AIC.gallery, dataset_VeRi.gallery])
    #     print("=> merge dataset loaded")
    #     dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)
    #     return dataset
    return __factory[name](*args, **kwargs)