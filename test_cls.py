import argparse
import importlib
import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

import provider
from data_utils.ModelNetDataLoader import ModelNeth5DataLoader

# import seaborn as sns

NUM_REPEATS = 1


def dump_per_class_accuracy_csv(fname, array):
    np.savetxt('{}.csv'.format(fname), array, fmt='%.6f', delimiter=',')


class DataLoaderX(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='ssg_normal', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=True,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=10,
                        help='Aggregate classification scores with voting [default: 3]')
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = torch.from_numpy(xyz).float().cuda()
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)
        return pc


def test(model, loader, num_class=40, vote_num=1):
    # confusion_matrix = torch.zeros(num_class, num_class)
    # pointscale = PointcloudScale(scale_low=0.8, scale_high=1.18)
    best_instance_acc = 0.
    best_class_acc = 0.
    for i in range(NUM_REPEATS):
        mean_correct = []
        class_acc = np.zeros((num_class, 3))
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            in_points, in_target = data
            in_target = in_target[:, 0]
            # in_points, target = points.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).float()
            classifier = model.eval()
            vote_pool = torch.zeros(in_target.size()[0], num_class).cuda()
            for _ in range(vote_num):
                # points.data = pointscale(points.data)
                points = in_points.data.numpy()
                # points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points, target = points.cuda(non_blocking=True).float(), in_target.cuda(non_blocking=True).float()
                points = points.transpose(2, 1)
                pred, _ = classifier(points)
                vote_pool += pred
            pred = vote_pool / vote_num
            pred_choice = pred.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[int(cat), 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[int(cat), 1] += 1
            # for t, p in zip(target.view(-1), pred_choice.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
        # break
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc_ = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)
        print('Voting %d, test acc: %.6f, class acc: %.6f' % (i, instance_acc * 100, class_acc_ * 100))
        if instance_acc > best_instance_acc:
            best_instance_acc = instance_acc
        if class_acc_ > best_class_acc:
            best_class_acc = class_acc_
    print('Final Voting test acc: %.6f, class acc: %.6f' % (best_instance_acc * 100, best_class_acc * 100))
    # print(confusion_matrix)
    # names = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
    # plt.figure(figsize=(10,10))
    # plot_confusion_matrix(confusion_matrix, names)
    # np.save("./confusion_matrix_vote_1", confusion_matrix)


#     ax = plt.subplot()
#     sns.heatmap(confusion_matrix, annot=True, ax=ax, fmt='g', cmap='Greens');  # annot=True to annotate cells
#
#     # labels, title and ticks
#     ax.set_xlabel('Predicted labels')
#     ax.set_ylabel('True labels')
#     # cax = ax.matshow(confusion_matrix)
#     name = """airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox
# """
#     name = name.split(' ')
#     ax.set_xticks(list(range(40)))
#     ax.set_yticks(list(range(40)))
#     ax.xaxis.set_ticklabels(name)
#     ax.yaxis.set_ticklabels(name)
#     plt.xticks(rotation=70)
#     plt.yticks(rotation=0)
#     plt.show()

# return instance_acc, class_acc_, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/'
    TEST_DATASET = ModelNeth5DataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = DataLoaderX(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)

    with torch.no_grad():
        test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
        # dump_per_class_accuracy_csv(str(experiment_dir) + '/class_acc_' + str(class_acc_), class_acc)
        # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc_))


if __name__ == '__main__':
    args = parse_args()
    main(args)
