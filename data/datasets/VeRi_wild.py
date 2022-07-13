# encoding: utf-8
"""
@author:  qianwen
@contact: qianwen2018@ia.ac.cn
"""


import glob
import re

import os.path as osp
# from multiprocessing import Pool
import json
import numpy as np

from .bases import BaseImageDataset

def res_(line):
    return json.loads(line.strip())

class VeRi_WILD(BaseImageDataset):

    def __init__(self, root='.', verbose=True,offline_path="", **kwargs):
        super(VeRi_WILD, self).__init__()

        self.list_file = root

        train_path='/home/qianwen.qian/data/data_list/VeRi-wild/train_viewpoint2.list'
        # gallery_path='/home/qianwen.qian/data/data_list/VeRi-wild/test_3000_viewpoint2.list'
        # query_path='/home/qianwen.qian/data/data_list/VeRi-wild/query_3000_viewpoint2.list'
        # gallery_path='/home/qianwen.qian/data/data_list/VeRi-wild/test_5000_viewpoint2.list'
        # query_path='/home/qianwen.qian/data/data_list/VeRi-wild/query_5000_viewpoint2.list'
        gallery_path='/home/qianwen.qian/data/data_list/VeRi-wild/test_10000_viewpoint2.list'
        query_path='/home/qianwen.qian/data/data_list/VeRi-wild/query_10000_viewpoint2.list'

        if not offline_path=="":
            train_path=osp.join(offline_path,train_path.split("/")[-1])
            test_path=osp.join(offline_path,test_path.split("/")[-1])
            query_path=osp.join(offline_path,query_path.split("/")[-1])

        train, num_train_pids, num_train_imgs, num_train_cams, num_train_tids  = self._process_dir(train_path, process_type='train')

        self._process_dir(gallery_path, process_type='get')

        gallery, num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_tids  = self._process_dir(gallery_path, process_type='test')

        query, num_query_pids, num_query_imgs, num_query_cams, num_query_tids  = self._process_dir(query_path, process_type='query')


        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_tids = num_train_pids, num_train_imgs, num_train_cams, num_train_tids
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_tids = num_query_pids, num_query_imgs, num_query_cams, num_query_tids
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_tids = num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_tids

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_class=self.num_train_pids+self.num_query_pids

        self.test_tid_map = dict()
        self.test_vid_map = dict()

        if verbose:
            print("=> VehicleID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _process_dir(self, list_file, process_type='train'):
        # multiprocessing.set_start_method(‘spawn’, force=True)


        # pool = Pool(processes=96)
        results = list()
        with open(list_file, 'rb') as f:
            lines = f.readlines()
        for line in lines:
            results.append(res_(line))


        # with open(list_file, 'rb') as f:
        #     lines = f.readlines()
        # results = pool.map(res_ , lines)
        cid_list = list()
        tid_list = list()
        vid_list = list()
        image_ids = list()
        dates = list()
        dsets = list()
        colors = list()
        types = list()
        score_map_paths = list()
        for res in results:
            # image_id = res['image_id']
            image_id = "/home/qianwen.qian/data/VeRI-Wild/images/"+res['portrait_image_url_bak'].split("images/")[-1]
            cid = int(res['cameraID'][1:])
            tid = int(res['trackID'])
            if 'viewpoint' in res:
                tid = np.argmax(res['viewpoint'][1:]) + 1
                if tid >=3: tid = 3
            vid = int(res['vehicleID'])
            date = res['date']
            dset = res['dataset']
            color = int(res['color']) if 'color' in res else -1
            type_ = int(res['type']) if 'type' in res else -1
            score_map_path = res['score_map_path'] if 'score_map_path' in res else ''
            if color == -1: color = 10 
            if type_ == -1: type_ = 9

            if dset == 'VeRi': tid = 0 
            elif dset == 'VehicleX': tid = 1

            image_ids.append(image_id)
            cid_list.append(cid)
            tid_list.append(tid)
            vid_list.append(vid)
            dates.append(date)
            dsets.append(dset)
            colors.append(color)
            types.append(type_)
            score_map_paths.append(score_map_path)

        cid_container = sorted(set(cid_list))
        tid_container = sorted(set(tid_list))
        vid_container = sorted(set(vid_list))

        if process_type == 'get':
            vid2label = {vid: label for label, vid in enumerate(vid_container)}
            self.test_vid_map = vid2label
            return 
        elif process_type == 'test':
            tid2label = {tid: label for label, tid in enumerate(tid_container)}
            self.test_tid_map = tid2label
            vid2label = self.test_vid_map
            # vid2label = {vid: label for label, vid in enumerate(vid_container)}
            # self.test_vid_map = vid2label
        elif process_type == 'query':
            vid2label = self.test_vid_map
            tid2label = {tid: label+len(self.test_tid_map.keys()) for label, tid in enumerate(tid_container)}
        elif process_type == 'train':
            tid2label = {tid: label for label, tid in enumerate(tid_container)}
            vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = list()
        for i, image_id in enumerate(image_ids):
            cid = cid_list[i]
            tid = tid2label[tid_list[i]]
            vid = vid2label[vid_list[i]]
            date = dates[i]
            dset = dsets[i]
            color = colors[i]
            type_ = types[i]
            score_map_path = score_map_paths[i]
            dataset.append((image_id, vid, cid, tid, date, dset, score_map_path, None))
        return dataset, len(vid_container), len(image_ids), len(cid_container), len(tid_container)

