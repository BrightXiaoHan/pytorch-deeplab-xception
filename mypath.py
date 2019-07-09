class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'links/PascalVoc'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'links/benchmark_RELEASE'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return 'links/cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return 'links/coco'
        elif dataset == 'agriculture':
            return 'links/Agriculture'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
