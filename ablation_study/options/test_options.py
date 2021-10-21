import argparse
import os
import time
from pathlib import Path

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--dataset', type=str, default='paris_streetview',
                                 help='dataset of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--data_file', type=str, default='', help='the file storing testing file paths')
        self.parser.add_argument('--test_dir', type=str, default='./test_results', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='./checkpoints', help='pretrained models are given here')
        self.parser.add_argument('--model_prefix', type=str, default='snap', help='models are saved here')
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')

        # for setting inputs
        self.parser.add_argument('--model', type=str, default='srn')
        self.parser.add_argument('--random_crop', type=int, default=1,
                                 help='using random crop to process input image when '
                                      'the required size is smaller than the given size')
        self.parser.add_argument('--random_mask', type=int, default=1,
                                 help='using random mask')
        self.parser.add_argument('--use_cn', type=bool, default=True)
        self.parser.add_argument('--feat_expansion_op', type=str, default='subpixel')

        self.parser.add_argument('--random_seed', type=bool, default=False)

        self.parser.add_argument('--img_shapes', type=str, default='256,256,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128',
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--test_num', type=int, default=-1)
        self.parser.add_argument('--fa_alpha', type=float, default=0.5)

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=64,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')

        # result_img
        self.parser.add_argument('--result_img_shapes', type=str, default='160,320,3',
                                 help='given shape parameters: h,w,c or h,w')

        # Mask
        self.parser.add_argument('--after_FPN_img_shapes', type=str, default='256,512,3',
                                 help='given shape parameters: h,w,c or h,w')

        # Save model
        self.parser.add_argument('--save_model_dir', type=str, default='', help='')

        # Save only basename (iter)
        self.parser.add_argument('--save_only_basename', type=int, default=0, help='')

        # inpainting
        self.parser.add_argument('--inpainting', type=int, default=0)

        # celebA_testmask
        self.parser.add_argument('--celebahq_testmask', type=int, default=0)

        # beach_testmask & beach_rearrange
        self.parser.add_argument('--beach_testmask', type=int, default=0)
        self.parser.add_argument('--beach_rearrange', type=int, default=0)

        # only for cub200
        self.parser.add_argument('--rgb_correction', type=int, default=0)
        self.parser.add_argument('--correction_outside', type=int, default=0)

        # flip_all
        self.parser.add_argument('--flip_all', type=int, default=0)

        # feature
        self.parser.add_argument('--feature', type=int, default=0)

        # random size
        self.parser.add_argument('--random_size', type=int, default=0)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.dataset_path = self.opt.data_file

        Path(self.opt.test_dir).mkdir(parents=True, exist_ok=True)

        assert self.opt.random_crop == 0 or self.opt.random_crop == 1
        self.opt.random_crop = True if self.opt.random_crop == 1 else False

        assert self.opt.random_mask == 0 or self.opt.random_mask == 1
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.use_cn == 0 or self.opt.use_cn == 1
        self.opt.use_cn = True if self.opt.use_cn == 1 else False

        assert self.opt.model in ['srn', 'srn-hr']
        assert self.opt.feat_expansion_op in ['subpixel', 'deconv', 'bilinear-conv', 'unfold']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_after_FPN_img_shapes = self.opt.after_FPN_img_shapes.split(',')
        self.opt.after_FPN_img_shapes = [int(x) for x in str_after_FPN_img_shapes]

        str_result_img_shapes = self.opt.result_img_shapes.split(',')
        self.opt.result_img_shapes = [int(x) for x in str_result_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        # model name and date
        self.opt.date_str = 'test_'+time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.dataset

        self.opt.model_folder += '_' + self.opt.model

        if self.opt.model == 'srn' or self.opt.model == 'srn-hr':
            self.opt.model_folder += '_' + self.opt.feat_expansion_op
            self.opt.model_folder += '_wo-cn' if self.opt.use_cn is False else ''
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_rand-mask' if self.opt.random_mask else ''
        if self.opt.random_mask:
            self.opt.model_folder += '_seed-' + str(self.opt.seed)
        self.opt.saving_path = os.path.join(self.opt.test_dir, self.opt.model_folder)

        if self.opt.save_only_basename:
            self.opt.saving_path2 = os.path.join(self.opt.test_dir, os.path.basename(self.opt.load_model_dir))

        else:
            self.opt.saving_path2 = self.opt.saving_path

        print("save to " + self.opt.saving_path2)
        Path(self.opt.saving_path2).mkdir(parents=True, exist_ok=True)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
