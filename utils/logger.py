# import logging
# import os
# import sys


# def setup_logger(name, save_dir, distributed_rank):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     # don't log results for the non-master process
#     if distributed_rank > 0:
#         return logger

#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
    
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     if save_dir:
#         fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='a')
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     return logger

import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='a')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger