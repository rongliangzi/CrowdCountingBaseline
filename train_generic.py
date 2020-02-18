import torch
import torchvision
import torch.nn as nn
import os
import glob
from modeling import *
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from dataset import *
import torch.nn.functional as F
from utils.functions import *
from utils import pytorch_ssim
import argparse
from tqdm import tqdm
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_loader(train_path, test_path, downsample_ratio, args):
    train_img_paths = []
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        train_img_paths.append(img_path)
    bg_img_paths = []
    for bg_img_path in glob.glob(os.path.join('/home/datamining/Datasets/CrowdCounting/bg/', '*.jpg')):
        bg_img_paths.append(bg_img_path)
    if args.use_bg:
        train_img_paths += bg_img_paths
    test_img_paths = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_img_paths.append(img_path)
    
    if args.loss=='bayes':
        bayes_dataset = Crowd(train_path, args.crop_size, downsample_ratio, False, 'train')
        train_loader = torch.utils.data.DataLoader(bayes_dataset, collate_fn=bayes_collate, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(Crowd(test_path, args.crop_size, downsample_ratio, False, 'val'),batch_size=1, num_workers=8, pin_memory=True)
    elif args.bn>0:
        bn_dataset=PatchSet(train_img_paths, transform, c_size=(args.crop_size,args.crop_size), crop_n=args.random_crop_n)
        train_loader = torch.utils.data.DataLoader(bn_dataset, collate_fn=my_collate_fn, shuffle=True, batch_size=args.bs, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(RawDataset(test_img_paths, transform, mode='one', downsample_ratio=downsample_ratio, test=True), shuffle=False, batch_size=1, pin_memory=True)
    else:
        single_dataset=RawDataset(train_img_paths, transform, args.crop_mode, downsample_ratio, args.crop_scale)
        train_loader = torch.utils.data.DataLoader(single_dataset, shuffle=True, batch_size=1, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(RawDataset(test_img_paths, transform, mode='one', downsample_ratio=downsample_ratio, test=True), shuffle=False, batch_size=1, pin_memory=True)
    
    return train_loader, test_loader, train_img_paths, test_img_paths


def main(args):
    # use gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cur_device=torch.device('cuda:{}'.format(args.gpu))
    if args.loss=='bayes':
        root = '/home/datamining/Datasets/CrowdCounting/sha_bayes_512/'
        train_path = root+'train/'
        test_path = root+'test/'
    elif args.bn:
        root = '/home/datamining/Datasets/CrowdCounting/sha_512_a/'
        train_path = root+'train/'
        test_path = root+'test/'
    else:
        if args.dataset=='sha':
            root = '/home/datamining/Datasets/CrowdCounting/shanghaitech/part_A_final/'
            train_path = root+'train_data/images'
            test_path = root+'test_data/images/'
        elif args.dataset=='shb':
            root = '/home/datamining/Datasets/CrowdCounting/shb_1024_f15/'
            train_path = root+'train/'
            test_path = root+'test/'
        elif args.dataset =='qnrf':
            root = '/home/datamining/Datasets/CrowdCounting/qnrf_1024_a/'
            train_path = root+'train/'
            test_path = root+'test/'
        
    downsample_ratio = args.downsample
    train_loader, test_loader, train_img_paths, test_img_paths = get_loader(train_path, test_path, downsample_ratio, args)
    
    model_dict = {'VGG16_13': M_CSRNet, 'DefCcNet':DefCcNet, 'Res50_back3':Res50, 'InceptionV3':Inception3CC, 'CAN':CANNet}
    model_name = args.model
    dataset_name = args.dataset
    net = model_dict[model_name](downsample=args.downsample, bn=args.bn>0, objective=args.objective, sp=(args.sp>0),se=(args.se>0),NL=args.nl)
    net.cuda()
    if args.bn>0:
        save_name = '{}_{}_{}_bn{}_ps{}_{}'.format(model_name, dataset_name, str(int(args.bn)), str(args.crop_size),args.loss)
    else:
        save_name = '{}_d{}{}{}{}{}_{}_{}_cr{}_{}{}{}{}{}{}'.format(model_name, str(args.downsample), '_sp' if args.sp else '', '_se' if args.se else '', '_'+args.nl if args.nl!='relu' else '', '_vp' if args.val_patch else '', dataset_name, args.crop_mode, str(args.crop_scale), args.loss, '_wu' if args.warm_up else '', '_cl' if args.curriculum=='W' else '', '_v'+str(int(args.value_factor)) if args.value_factor!=1 else '', '_amp'+str(args.amp_k) if args.objective=='dmp+amp' else '', '_bg' if args.use_bg else '')
    save_path = "/home/datamining/Models/CrowdCounting/"+save_name+".pth"
    logger = get_logger('logs/'+save_name+'.txt')
    for k, v in args.__dict__.items():  # save args
        logger.info("{}: {}".format(k, v))
    if os.path.exists(save_path) and args.resume:
        net.load_state_dict(torch.load(save_path))
        print('{} loaded!'.format(save_path))
    
    value_factor=args.value_factor
    freq = 100
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == 'SGD':
        # not converage
        optimizer=torch.optim.SGD(net.parameters(),lr=args.lr, momentum=0.95, weight_decay=args.decay)
    
    if args.loss=='bayes':
        bayes_criterion=Bay_Loss(True, cur_device)
        post_prob=Post_Prob(sigma=8.0,c_size=args.crop_size,stride=1,background_ratio=0.15,use_background=True,device=cur_device)
    else:
        mse_criterion = nn.MSELoss().cuda()
    
    if args.scheduler == 'plt':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=10, verbose=True)
    elif args.scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=0)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.8)
    elif args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.scheduler == 'cyclic' and args.optimizer == 'SGD':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr*0.01, max_lr=args.lr, step_size_up=25,)
    elif args.scheduler == 'None':
        scheduler = None
    else:
        print('scheduler name error!')
    
    if args.val_patch:
        best_mae, best_rmse = val_patch(net, test_loader, value_factor)
    elif args.loss=='bayes':
        best_mae, best_rmse = val_bayes(net, test_loader, value_factor)
    else:
        best_mae, best_rmse = val(net, test_loader, value_factor)
    if args.scheduler=='plt':
        scheduler.step(best_mae)
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    for epoch in range(args.epochs):
        if args.crop_mode == 'curriculum':
            # every 20%, change the dataset
            if (epoch+1) % (args.epochs//5) == 0:
                print('change dataset')
                single_dataset = RawDataset(train_img_paths, transform, args.crop_mode, downsample_ratio, args.crop_scale, (epoch+1.0+args.epochs//5)/args.epochs)
                train_loader = torch.utils.data.DataLoader(single_dataset, shuffle=True, batch_size=1, num_workers=8)
        
        train_loss = 0.0
        if args.loss=='bayes':
            epoch_mae = AverageMeter()
            epoch_mse = AverageMeter()
        net.train()
        if args.warm_up and epoch < args.warm_up_steps:
            linear_warm_up_lr(optimizer, epoch, args.warm_up_steps,args.lr)
        for it, data in enumerate(train_loader):
            if args.loss == 'bayes':
                inputs, points, targets, st_sizes=data
                img = inputs.to(cur_device)
                st_sizes = st_sizes.to(cur_device)
                gd_count = np.array([len(p) for p in points], dtype=np.float32)
                points = [p.to(cur_device) for p in points]
                targets = [t.to(cur_device) for t in targets]
            else:
                img, target, _, amp_gt = data
                img = img.cuda()
                target = value_factor*target.float().unsqueeze(1).cuda()
                amp_gt = amp_gt.cuda()
            #print(img.shape)
            optimizer.zero_grad()
            
            #print(target.shape)
            if args.objective == 'dmp+amp':
                output, amp = net(img)
                output = output * amp
            else:
                output = net(img)
            
            if args.curriculum == 'W':
                delta = (output - target)**2
                k_w = 2e-3 * args.value_factor * args.downsample**2
                b_w = 5e-3 * args.value_factor * args.downsample**2
                T = torch.ones_like(target,dtype=torch.float32) * epoch * k_w + b_w
                W = T / torch.max(T,output)
                delta = delta * W
                mse_loss = torch.mean(delta)
            else:
                mse_loss = mse_criterion(output, target)
            
            if args.loss == 'mse+lc':
                loss = mse_loss + 1e2 * cal_lc_loss(output, target) * args.downsample
            elif args.loss == 'ssim':
                loss = 1 - ssim_loss(output, target)
            elif args.loss == 'mse+ssim':
                loss = 100 * mse_loss + 1e-2*(1-ssim_loss(output,target))
            elif args.loss == 'mse+la':
                loss = mse_loss + cal_spatial_abstraction_loss(output, target)
            elif args.loss == 'la':
                loss = cal_spatial_abstraction_loss(output, target)
            elif args.loss == 'ms-ssim':
                #to do
                pass
            elif args.loss == 'adversial':
                # to do 
                pass
            elif args.loss == 'bayes':
                prob_list = post_prob(points, st_sizes)
                loss = bayes_criterion(prob_list, targets, output)
            else:
                loss = mse_loss
            
            # add the cross entropy loss for attention map
            if args.objective == 'dmp+amp':
                cross_entropy = (amp_gt * torch.log(amp) + (1 - amp_gt) * torch.log(1 - amp)) * -1
                cross_entropy_loss = torch.mean(cross_entropy)
                loss = loss + cross_entropy_loss * args.amp_k
            
            loss.backward()
            optimizer.step()
            data_loss = loss.item()
            train_loss += data_loss
            if args.loss=='bayes':
                N = inputs.size(0)
                pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
            
            if args.loss!='bayes' and it%freq==0:
                print('[ep:{}], [it:{}], [loss:{:.8f}], [output:{:.2f}, target:{:.2f}]'.format(epoch+1, it, data_loss, output[0].sum().item(), target[0].sum().item()))
        if args.val_patch:
            mae, rmse = val_patch(net, test_loader, value_factor)
        elif args.loss=='bayes':
            mae, rmse = val_bayes(net, test_loader, value_factor)
        else:
            mae, rmse = val(net, test_loader, value_factor)
        if not (args.warm_up and epoch<args.warm_up_steps):
            if args.scheduler == 'plt':
                scheduler.step(best_mae)
            elif args.scheduler != 'None':
                scheduler.step()
        
        if mae + 0.1 * rmse < best_mae + 0.1 * best_rmse:
            best_mae, best_rmse = mae, rmse
            torch.save(net.state_dict(), save_path)
        
        if args.loss=='bayes':
            logger.info('{} Epoch {}/{} Loss:{:.8f},MAE:{:.2f},RMSE:{:.2f} lr:{:.8f}, [CUR]:{mae:.1f}, {rmse:.1f}, [Best]:{b_mae:.1f}, {b_rmse:.1f}'.format(model_name, epoch+1, args.epochs, train_loss/len(train_loader),epoch_mae.get_avg(), np.sqrt(epoch_mse.get_avg()),optimizer.param_groups[0]['lr'], mae=mae, rmse=rmse, b_mae=best_mae, b_rmse=best_rmse))
        else:
            logger.info('{} Epoch {}/{} Loss:{:.8f}, lr:{:.8f}, [CUR]:{mae:.1f}, {rmse:.1f}, [Best]:{b_mae:.1f}, {b_rmse:.1f}'.format(model_name, epoch+1, args.epochs, train_loss/len(train_loader), optimizer.param_groups[0]['lr'], mae=mae, rmse=rmse, b_mae=best_mae, b_rmse=best_rmse))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crowd Counting')
    parser.add_argument('--model', metavar='model name', default='VGG16_13', choices=['VGG16_13', 'DefCcNet', 'InceptionV3', 'CAN', 'Res50_back3'], type=str)
    parser.add_argument('--downsample', metavar='downsample ratio', default=1, choices=[1, 2, 4, 8], type=int)
    parser.add_argument('--dataset', metavar='dataset name', default='sha', choices=['sha','shb','qnrf'], type=str)
    parser.add_argument('--resume', metavar='resume model if exists', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--gpu', default='0', help='assign device')
    parser.add_argument('--scheduler', default='plt', help='lr scheduler', choices=['plt', 'cos', 'step', 'cyclic', 'exp', 'None'], type=str)
    parser.add_argument('--optimizer', default='Adam', help='optimizer', choices=['Adam','SGD'], type=str)
    parser.add_argument('--decay', default=1e-4, help='weight decay', type=float)
    parser.add_argument('--epochs', default=200, type=int)
    
    parser.add_argument('--loss', default='mse', choices=['mse','mse+lc','ssim','mse+ssim','mse+la','la','ms-ssim','bayes','adversial'])
    parser.add_argument('--val_patch', metavar='val on patch if set to True', default=0, choices=[0,1], type=int)
    
    parser.add_argument('--crop_mode', default='random', choices=['random', 'one', 'fixed+random', 'fixed', 'mixed', 'curriculum'], type=str)
    parser.add_argument('--crop_scale', metavar='patch scale, work when not using batch norm or bayes', default=0.5, type=float)
    parser.add_argument('--crop_size', default=256, help='the size of cropping from original images. Work when using batch norm or bayes', type=int)
    
    parser.add_argument('--warm_up', default=0, help='warm up from 0.1*lr to lr by warm up steps', type=int)
    parser.add_argument('--warm_up_steps', default=10, help='warm up steps', type=int)
    
    parser.add_argument('--curriculum', default='None', metavar='curriculum learning', choices=['None','W'])
    parser.add_argument('--value_factor', default=1.0, metavar='value factor * gt', type=float)
    parser.add_argument('--objective', default='dmp', choices=['dmp', 'dmp+amp'], type=str)
    parser.add_argument('--amp_k', default=0.1, help="only work when objective is 'dmp+amp'. loss = loss + k * cross_entropy_loss", type=float)
    parser.add_argument('--use_bg', default=0, help='if using background images(without any person) in training', choices=[0,1], type=int)
    
    parser.add_argument('--bn', default=0, help='if using batch normalization', type=int)
    parser.add_argument('--bs', default=4, help='batch size if using bn', type=int)
    parser.add_argument('--random_crop_n', default=4, metavar='random crop number for each image, only work when using bn', type=int)
    
    parser.add_argument('--sp', default=0, help='spatial pyramid module', type=int)
    parser.add_argument('--se', default=0, help='squeeze excitation module', type=int)
    parser.add_argument('--nl', default='relu', help='non-linear layer', choices=['relu', 'prelu', 'swish'], type=str)
    args = parser.parse_args()
    
    main(args)
    
