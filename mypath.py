class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'agriculture':
            return 'links/Agriculture'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
