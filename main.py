# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset_2gpu import TSNDataSet
from ops.models_2gpu import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance

from tensorboardX import SummaryWriter

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)


    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    selection_set = None
    dataset_TSN = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                             new_length=data_length,
                             modality=args.modality,
                             image_tmpl=prefix,
                             transform=torchvision.transforms.Compose([
                                 train_augmentation,
                                 Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                 ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                 normalize,
                   ]), dense_sample=args.dense_sample, selection_set=selection_set)

    train_loader = torch.utils.data.DataLoader(
        dataset_TSN,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_loader_select = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, all_in=True),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    train_loader_1 = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, dense_segments=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    best_model = model
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        if epoch > 5:
            cluster_set = None
            print('selection of training (cluster at 3rd Res block, 32 to 16 slope clustering)') # train_loader_1
            train(train_loader_1, model, criterion, optimizer, epoch, log_training, tf_writer, cluster_set=cluster_set)
        else:
            train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            if is_best:
                best_model = model

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, cluster_set=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        # compute output
        if cluster_set is not None:
            cluster_set = torch.tensor(cluster_set, dtype=torch.float).cuda()
            output = model(input_var, cluster_set=cluster_set[i*args.batch_size:i*args.batch_size+args.batch_size])
        else:
            output = model(input_var, merge=(epoch > 5))  # merge=(epoch > 3)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec, pred = accuracy(output.data, target, topk=(1, 2))
        prec1, prec5 = prec
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # activations = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activations[name] = output.detach()
    #     return hook
    #
    # model.module.new_fc.register_forward_hook(get_activation('new_fc'))
    # model.module.base_model.conv1.register_forward_hook(get_activation('conv1'))
    # model.module.base_model.layer1[1].conv2.register_forward_hook(get_activation('layer1.1.conv2'))
    # model.module.base_model.layer2[1].conv2.register_forward_hook(get_activation('layer2.1.conv2'))
    # model.module.base_model.layer3[1].conv2.register_forward_hook(get_activation('layer3.1.conv2'))
    # model.module.base_model.layer4[1].conv2.register_forward_hook(get_activation('layer4.1.conv2'))


    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec, pred = accuracy(output.data, target, topk=(1, 2))
            prec1, prec5 = prec

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

        # tf_writer.add_histogram('conv1', activations['conv1'].cpu().numpy(), epoch)
        # tf_writer.add_histogram('layer1.1.conv2', activations['layer1.1.conv2'].cpu().numpy(), epoch)
        # tf_writer.add_histogram('layer2.1.conv2', activations['layer2.1.conv2'].cpu().numpy(), epoch)
        # tf_writer.add_histogram('layer3.1.conv2', activations['layer3.1.conv2'].cpu().numpy(), epoch)
        # tf_writer.add_histogram('layer4.1.conv2', activations['layer4.1.conv2'].cpu().numpy(), epoch)
        # tf_writer.add_histogram('new_fc', activations['new_fc'].cpu().numpy(), epoch)

    return top1.avg


def data_validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    tmp = [x.strip().split(' ') for x in open(args.train_list)]

    activations = dict()
    activations['layer4'] = dict()

    def get_activation(name):
        def hook(model, input, output):
            activations[name][output.get_device()] = output.detach().cpu()

        return hook

    model.module.base_model.layer4.register_forward_hook(get_activation('layer4'))


    # switch to evaluate mode
    model.eval()

    selection_set = {}
    cluster_set = []
    print('hamming dis, KModes')
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input = input.view(-1, 3, input.size(-2), input.size(-1))
            # print(input.size())

            # compute output
            output = model(input)

            batch_size = activations['layer4'][0].size(0)
            activations_set = activations['layer4'][0].view(batch_size, -1).numpy()
            activ = activations['layer4'][1].view(batch_size, -1).numpy()
            activations_set = np.concatenate((activations_set, activ), axis=0)
            activations_sign_set = activations_set
            activations_sign_set[activations_sign_set == np.float(0)] = np.float(-1)

            # PCA
            # pca = PCA(n_components=len(activations_set))
            # pca.fit(activations_set)
            # activations_set = pca.transform(activations_set)
            #
            # Sklearn Kmeans clustering
            kmeans = KModes(n_clusters=args.num_segments, init='Huang',
                            n_init=5, max_iter=5).fit_predict(np.sign(activations_sign_set))
            kmeans_labels = kmeans.labels_
            selection_dict = {}
            for j in range(len(kmeans_labels)):
                if kmeans_labels[j] not in selection_dict:
                    selection_dict[kmeans_labels[j]] = [j]
                else:
                    selection_dict[kmeans_labels[j]] += [j]

            final_selection = []
            cluster = []
            for k, v in selection_dict.items():
                numbers = len(v)
                final_selection.append(v[np.random.randint(numbers)])
                cluster.append(numbers)
            final_selection.sort()

            # accumulated distance selection
            # euclidean = distance.euclidean(activations_set[0], activations_set[1])
            # accum_dis = [euclidean] * 2
            # for j in range(1, activations_set.shape[0] - 1):
            #     euclidean = distance.euclidean(activations_set[j], activations_set[j + 1])
            #     accum_dis.append(euclidean + accum_dis[-1])

            # hamming = distance.hamming(np.sign(activations_sign_set[0]), np.sign(activations_sign_set[1]))
            # accum_dis = [hamming] * 2
            # for j in range(1, activations_sign_set.shape[0] - 1):
            #     hamming = distance.hamming(np.sign(activations_sign_set[j]), np.sign(activations_sign_set[j + 1]))
            #     accum_dis.append(hamming + accum_dis[-1])

            # dis_index = accum_dis[-1] / args.num_segments
            #
            # cnt = 1
            # clusters = []
            # clus = []
            # for k in range(activations_set.shape[0]):
            #
            #     if accum_dis[k] <= dis_index * cnt:
            #         clus.append(k)
            #     else:
            #         clusters.append(clus)
            #         clus = []
            #         cnt += 1
            #         clus.append(k)
            # clusters.append(clus)
            #
            # final_selection = []
            # cluster = []
            # for c in clusters:
            #     cluster.append(len(c))
            #     final_selection.append(random.choice(c))

            # print('final selection:', final_selection)
            # print('cluster:', cluster)
            selection_set[tmp[i][0]] = final_selection
            cluster_set.append(cluster)
    return selection_set, cluster_set


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
