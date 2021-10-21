import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
import random
import glob

# using mask images
class DataLoader4_ldr:
    def __init__(self, filename, im_size, batch_size, exact_decay, decay_steps, more_turns_to_more_masks,
                 specific_mask_number, mask_range, minimum_mask_number, curriculum_range, out2in,
                 grid_number):
        self.filename = filename
        self.filelist = open(filename, 'rt').read().splitlines()

        if not self.filelist:
            print(len(self.filelist))
            exit('\nError: file list is empty\n')

        if "CUB_200" in self.filename:
            self.len_files = len(self.filelist)
            self.ldr_shape = (256, 256, 3)

        self.im_size = im_size
        self.batch_size = batch_size
        self.data_queue = None

        # self.masks_16 = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in glob.glob("./mask_images_16/mask16_*.png")]

        self.grid_number = grid_number
        self.grid_number_square = grid_number * grid_number
        mask_imgs = glob.glob("./mask_images_{}/*.png".format(str(self.grid_number_square)))
        self.masks_NxN = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in mask_imgs]

        if exact_decay:
            self.number_of_masks_decay_steps = self.len_files
        else:
            self.number_of_masks_decay_steps = decay_steps

        self.more_turns_to_more_masks = more_turns_to_more_masks
        self.term_index = 0

        self.specific_mask_number = specific_mask_number
        self.mask_range = mask_range

        self.minimum_mask_number = minimum_mask_number

        self.curriculum_range = curriculum_range

        self.out2in = out2in

    def next(self, global_step, mask_curriculum):
        with tf.variable_scope('feed'):
            filelist_tensor = tf.convert_to_tensor(self.filelist, dtype=tf.string)
            self.data_queue = tf.train.slice_input_producer([filelist_tensor], shuffle=True)

            im_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            im_gt = tf.cast(im_gt, tf.float32)
            im_gt.set_shape(self.ldr_shape)

            def get_mask_tensor(global_step):
                if len(self.mask_range) == 2:
                    number_of_masks = random.randrange(int(self.mask_range[0]), int(self.mask_range[1]) + 1)
                elif self.specific_mask_number > 0:
                    number_of_masks = self.specific_mask_number
                elif mask_curriculum:
                    if self.more_turns_to_more_masks and self.grid_number == 4:
                        if global_step > 100 * (self.term_index + 1) * (self.term_index + 1) + 900 * (
                                self.term_index + 1):
                            self.term_index = self.term_index + 1
                        number_of_masks = max(15 - self.term_index, self.minimum_mask_number)
                    else:
                        # temporary
                        if self.more_turns_to_more_masks and self.grid_number != 4:
                            print(
                                "Warning: Changed parameter \"more_turns_to_more_masks\" to False. Yet still you can implement it in data.py")

                        if self.out2in:
                            number_of_masks = min(
                                self.minimum_mask_number + global_step // self.number_of_masks_decay_steps,
                                self.grid_number_square - 1)
                        else:
                            number_of_masks = max(self.grid_number_square - 1 - global_step // self.number_of_masks_decay_steps,
                                                  self.minimum_mask_number)

                    if self.curriculum_range:
                        number_of_masks = random.randrange(number_of_masks, self.grid_number_square)
                else:
                    number_of_masks = 1
                random_regions = random.sample(self.masks_NxN, number_of_masks)
                # mask들 미리 만들면 다음 코드들이 필요 없어짐
                random_region = np.maximum.reduce(random_regions)
                random_region = random_region[None, :, :, None]
                random_region = random_region.astype(np.float32)
                return random_region

            # mask_tensor 1. pick a smaller number as the epoch goes on
            # mask_tensor 2. purely random(get_mask_tensor needs to be fixed to do so.)
            mask_tensor = tf.py_func(get_mask_tensor, [global_step], tf.float32)
            mask_tensor.set_shape([1, 256, 256, 1])

            mask_tensor = tf.image.resize(mask_tensor, [self.im_size[0], self.im_size[1]])
            mask_tensor.set_shape([1, self.im_size[0], self.im_size[1], 1])

            batch_gt = tf.train.batch([im_gt], batch_size=self.batch_size, num_threads=4)

        return batch_gt, mask_tensor
