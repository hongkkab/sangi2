# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import sys
#sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import AverageMeter as AverageMeter
from utils.utils import save_checkpoint as save_checkpoint
from utils.utils import Config as Config
import cpm_model_noncenter
import sangi_data
#import Mytransforms
import torchvision.transforms as transforms

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='../ckpt/cpm_latest.pth.tar',type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--train_dir', default='../dataset/train/sangi/', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--model_name', default='../ckpt/cpm', type=str,
                        help='model name to save parameters')

    return parser.parse_args()


def construct_model(args):

    model = cpm_model_noncenter.CPM(k=14)
    # load pretrained model
    # state_dict = torch.load(args.pretrained)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def get_parameters(model, config, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
            {'params': lr_2, 'lr': config.base_lr * 2.},
            {'params': lr_4, 'lr': config.base_lr * 4.},
            {'params': lr_8, 'lr': config.base_lr * 8.}]

    return params, [1., 2., 4., 8.]



def train_val(model, args):

    train_dir = args.train_dir
    val_dir = args.val_dir

    config = Config(args.config)
    cudnn.benchmark = True

    # train
    train_loader = torch.utils.data.DataLoader(
        sangi_data.Sangi_Data(train_dir, 8),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)
    # val
    if args.val_dir is not None and config.test_interval != 0:
        # val
        val_loader = torch.utils.data.DataLoader(
            sangi_data.Sangi_Data(val_dir, 8),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    params, multiple = get_parameters(model, config, False)

    optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(6)]
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model

    heat_weight = 46 * 46 * 15 / 1.0

    while iters < config.max_iter:

        for i, (input, heatmap) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy,
                                                 policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            heatmap = heatmap.cuda(async=True)
            #centermap = centermap.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            #centermap_var = torch.autograd.Variable(centermap)


            #heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var)
            loss1 = criterion(heat1, heatmap_var) * heat_weight
            loss2 = criterion(heat2, heatmap_var) * heat_weight
            loss3 = criterion(heat3, heatmap_var) * heat_weight
            loss4 = criterion(heat4, heatmap_var) * heat_weight
            loss5 = criterion(heat5, heatmap_var) * heat_weight
            loss6 = criterion(heat6, heatmap_var) * heat_weight


            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            losses.update(loss.data[0], input.size(0))
            for cnt, l in enumerate(
                    [loss1, loss2, loss3, loss4, loss5, loss6]):
                losses_list[cnt].update(l.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                      'Learning rate = {2}\n'
                      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0, 6):
                    print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                          .format(cnt + 1, loss1=losses_list[cnt]))

                print (time.strftime(
                '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(6):
                    losses_list[cnt].reset()

            save_checkpoint({
                'iter': iters,
                'state_dict': model.state_dict(),
            }, 0, args.model_name)
            torch.save(model,'model')

            # val
            if args.val_dir is not None and config.test_interval != 0 and iters % config.test_interval == 0:

                model.eval()
                for j, (input, heatmap) in enumerate(val_loader):
                    heatmap = heatmap.cuda(async=True)
                    #centermap = centermap.cuda(async=True)

                    input_var = torch.autograd.Variable(input)
                    heatmap_var = torch.autograd.Variable(heatmap)
                    #centermap_var = torch.autograd.Variable(centermap)

                    heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var)

                    loss1 = criterion(heat1, heatmap_var) * heat_weight
                    loss2 = criterion(heat2, heatmap_var) * heat_weight
                    loss3 = criterion(heat3, heatmap_var) * heat_weight
                    loss4 = criterion(heat4, heatmap_var) * heat_weight
                    loss5 = criterion(heat5, heatmap_var) * heat_weight
                    loss6 = criterion(heat6, heatmap_var) * heat_weight

                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                    losses.update(loss.data[0], input.size(0))
                    for cnt, l in enumerate(
                            [loss1, loss2, loss3, loss4, loss5, loss6]):
                        losses_list[cnt].update(l.data[0], input.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()
                    is_best = losses.avg < best_model
                    best_model = min(best_model, losses.avg)
                    save_checkpoint({
                        'iter': iters,
                        'state_dict': model.state_dict(),
                    }, is_best, args.model_name)
                    torch.save(model,'model')
                    if j % config.display == 0:
                        print('Test Iteration: {0}\t'
                              'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                              'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                              'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                            j, config.display, batch_time=batch_time,
                            data_time=data_time, loss=losses))
                        for cnt in range(0, 6):
                            print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                                  .format(cnt + 1, loss1=losses_list[cnt]))

                        print (time.strftime(
                            '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                            time.localtime()))
                        batch_time.reset()
                        losses.reset()
                        for cnt in range(6):
                            losses_list[cnt].reset()

                model.train()

def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    #map_6 = maps[0]

    kpts = []
    #for m in map_6[1:]:
    for m in maps[1:] :
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
		
    return kpts

def draw_paint(img_path, kpts, iters, index):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255]]
    limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10, 11], [7, 6], [12, 3], [12, 2], [2, 1], [1, 0], [3, 4],
               [4, 5]]

    im = cv2.imread(img_path)
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    #cv2.imshow('test_example', im)
    #cv2.waitKey(0)
    write_name='output/heatmap_%d_%d.jpg' % (iters+1, index+1)

    cv2.imwrite(write_name, im)

if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model = construct_model(args)
    train_val(model, args)
