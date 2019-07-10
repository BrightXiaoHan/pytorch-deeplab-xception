from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, agriculture
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == "agriculture":
        ds = agriculture.AgricutureSegmentation(args, args.agriculture_cropsize, args.agriculture_cropstride, split='train')
        train_set, val_set = ds.split()
        test_set = agriculture.AgricutureSegmentation(args, args.agriculture_cropsize, args.agriculture_cropsize, split='test')

        num_class = ds.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

