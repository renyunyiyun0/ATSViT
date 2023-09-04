import argparse
import sys
import os
import os.path as osp
import glob
from multiprocessing import Pool
import cv2
import numpy as np
from math import floor as floor
from random import shuffle
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path
    out_full_path = osp.join('./cvpr_test/', vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    vr = cv2.VideoCapture(full_path)
    w = int(vr.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videolen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    #cvpr
    scale = 6
    interval = 2

    n = floor(videolen//(scale*interval))#

    j=0

    m =0
    for i in range(videolen):
        ret, frame = vr.read()
        img = resize(frame[:, :, ::-1], 256, False, True, True, cv2)
        img = np.array(img)

        if i == 0:
            h, w = img.shape[0], img.shape[1]
            imgs_ndarry = np.zeros([videolen//interval, h, w, 3], dtype=np.int8)
        if (i+1) % interval ==0:
            imgs_ndarry[m, :, :, :] = img
            m +=1
        if (i+1) % (scale*interval) == 0:
            if j < n:
                j +=1
                k = (j-1)*scale
                img = get_fragment(imgs_ndarry[k:k+scale,:,:,:], 2, 4)
                if ret == False:
                    continue
                img = img.astype(np.uint8)
                if img is not None:

                    img = np.array(img)
                    img = img[:, :, ::-1]

                    cv2.imwrite('{}/img_{:05d}_360p.jpg'.format(out_full_path, j), img)

                else:
                    print('[Warning] length inconsistent!'
                          'Early stop with {} out of {} frames'.format(j, videolen))
                    break
    print('full_path={} vid_name={} num_frames={} dump_frames done'.format(full_path, vid_name, n))
    sys.stdout.flush()
    return True
def resize(img,short_size,fixed_ratio,keep_ratio,do_round,backend):

            if isinstance(img, np.ndarray):
                h, w, _ = img.shape
            elif isinstance(img, Image.Image):
                w, h = img.size
            else:
                raise NotImplementedError

            if w <= h:
                ow = short_size
                if fixed_ratio:
                    oh = int(short_size * 4.0 / 3.0)
                elif not keep_ratio:  # no
                    oh = short_size
                else:
                    scale_factor = short_size / w
                    oh = int(h * float(scale_factor) +
                             0.5) if do_round else int(h *
                                                            short_size / w)
                    ow = int(w * float(scale_factor) +
                             0.5) if do_round else int(w *
                                                            short_size / h)
            else:
                oh = short_size
                if fixed_ratio:
                    ow = int(short_size * 4.0 / 3.0)
                elif not keep_ratio:  # no
                    ow = short_size
                else:
                    scale_factor = short_size / h
                    oh = int(h * float(scale_factor) +
                             0.5) if do_round else int(h *
                                                            short_size / w)
                    ow = int(w * float(scale_factor) +
                             0.5) if do_round else int(w *
                                                            short_size / h)
            if ow > oh:
                scale_size = floor(floor(ow//8)//4)*4
                ow = scale_size*8


            else:
                # if (oh % 8) != 0:
                scale_size = floor(floor(oh // 8) // 4) * 4
                oh = scale_size * 8
            if backend == 'pillow':
                img = img.resize((ow, oh), Image.BILINEAR)
            elif backend == 'cv2' and (keep_ratio is not None):

                   img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            else:

                   img = Image.fromarray(
                        cv2.resize(np.asarray(img), (ow, oh),
                                   interpolation=cv2.INTER_LINEAR))

            return img
def get_fragment(img,g_num,fragment_num):

    T,H,W,C = img.shape

    gh_size = H // g_num
    gw_size = W // g_num
    fragment_size_h = gh_size // fragment_num
    fragment_size_w = gw_size // fragment_num

    target_video = np.zeros([H, W, C])
    img = img.reshape([T, g_num,gh_size, g_num, gw_size,C]).transpose([0,1,3,2,4,5])

    target_video = target_video.reshape([H // gh_size, gh_size, W // gw_size, gw_size, C]).transpose([0, 2, 1, 3, 4])
    T, Hn,Wn,hgs,wgs,C = img.shape
    img = img.reshape([T,Hn,Wn,fragment_num,fragment_size_h,fragment_num, fragment_size_w,C]).transpose([0,1,2,3,5,4,6,7]) #T Hgn wgn hfn wfn fs fs c
    target_video = target_video.reshape(
        [Hn,Wn,hgs//fragment_size_h,fragment_size_h,wgs//fragment_size_w, fragment_size_w,C]).transpose(
        [0,1,2,4,3,5,6])
    T,hgn,wgn,hfn,wfn,fs_h,fs_w,c = img.shape


    for k1 in range(hgn):
        for k2 in range(wgn):
            for k3 in range(hfn):
                for k4 in range(wfn):
                    data_ssim = np.zeros([T, T])
                    imgs_ndarry = img[:,k1,k2,k3,k4,:,:,:]

                    T_f, H_f, W_f, C_f = imgs_ndarry.shape
                    for i in range(T_f):
                        k = T - i
                        n = T - 1
                        for j in range(k):
                            data_ssim[i, n] = ssim(imgs_ndarry[i, :, :, :], imgs_ndarry[n, :, :, :], multichannel=True, win_size=7)
                            n = n - 1
                    for i in range(T_f):
                        for j in range(T_f):
                            data_ssim[j, i] = data_ssim[i, j]
                    idx = sorted(enumerate(data_ssim.mean(axis=0)), key=lambda x: x[1])[0][0] #modify min
        
                    target_video[k1,k2,k3,k4,:,:,:] = img[idx,k1,k2,k3,k4,:,:,:]
    target_video = target_video.transpose([0,1,2,4,3,5,6]).reshape([Hn,Wn,hgs,wgs,C])
    target_video = target_video.transpose([0,2,1,3,4]).reshape([H,W,C])
    return target_video
def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir', type=str,default='./test/')
    parser.add_argument('--out_dir', type=str,default='./cvpr_test/')
    parser.add_argument('--level', type=int,
                        choices=[1, 2],
                        default=1)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument("--out_format", type=str, default='dir',
                        choices=['dir', 'zip'], help='output format')
    parser.add_argument("--ext", type=str, default='mp4',
                        choices=['avi', 'mp4'], help='video file extensions')
    parser.add_argument("--resume", action='store_true', default=0,
                        help='resume optical flow extraction '
                        'instead of overwriting')
    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()
    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    print("args.src_dir:",args.src_dir)
    if args.level == 2:
        fullpath_list = glob.glob(args.src_dir + '/*/*')
        done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*' )
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
        done_list = list(map(lambda p: p.split('/')[-1], done_fullpath_list))

        vid_list = set(vid_list).difference(set(done_list))
        vid_list = list(vid_list)
        fullpath_list = []
        for i in range(len(vid_list)):
            fullpath_list.append(osp.join(args.src_dir, vid_list[i]))
        print('Resuming. number of videos to be done: ', len(vid_list))
    shuffle(fullpath_list)
    n=int(0.6*len(fullpath_list))
    fullpath_list_train=fullpath_list[0:n]
    fullpath_list_val=fullpath_list[n: ]
    if args.level == 2:
        vid_list = list(map(lambda p: osp.join(
            '/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
        vid_list_train = list(map(lambda p: p.split('/')[-1], fullpath_list_train))
        vid_list_val = list(map(lambda p: p.split('/')[-1], fullpath_list_val))
    list = []
    for i in range(len(vid_list)):
        list.append(i)
    pool = Pool(args.num_worker)
    pool.map(dump_frames, zip(
        fullpath_list, vid_list, list))

