import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
import random
import glob


# using pre-defined mask images
class DataLoader4_ldr_beach:
    def __init__(self, filename, im_size, batch_size, exact_decay, decay_steps, more_turns_to_more_masks,
                 specific_mask_number, mask_range, minimum_mask_number, curriculum_range, out2in,
                 beach_center_mask, comp80000):
        self.filename = filename
        self.filelist = open(filename, 'rt').read().splitlines()

        if not self.filelist:
            print(len(self.filelist))
            exit('\nError: file list is empty\n')

        self.len_files = len(self.filelist)
        self.ldr_shape = (256, 256, 3)
        self.im_size = im_size
        self.batch_size = batch_size
        self.data_queue = None

        self.masks_16 = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in glob.glob("./mask_images_16/mask16_*.png")]

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

        self.beach_center_mask = beach_center_mask
        self.comp80000 = comp80000

    def next(self, global_step, mask_curriculum):
        with tf.variable_scope('feed'):
            filelist_tensor = tf.convert_to_tensor(self.filelist, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([filelist_tensor], shuffle=True)

            im_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            im_gt = tf.cast(im_gt, tf.float32)
            im_gt.set_shape(self.ldr_shape)

            def rearrange(im):
                new_im = np.zeros_like(im)
                h, w, c = im.shape
                new_im[:, w // 2:w, :] = im[:, 0:w // 2, :]
                new_im[:, 0:w // 2, :] = im[:, w // 2:w, :]
                return new_im

            # tf.slice(im_gt, [ldr_shape[0]//2, ldr_shape[1]//2, 0], [ldr_shape[0], ldr_shape[1], 3])
            # tf.slice(im_gt, [0, 0, 0], [ldr_shape[0]//2, ldr_shape[1]//2, 3])
            rearranged_gt = tf.py_func(rearrange, [im_gt], tf.float32)
            rearranged_gt.set_shape(self.ldr_shape)

            batch_beach = tf.train.batch([im_gt, rearranged_gt], batch_size=self.batch_size, num_threads=4)

            def get_mask_tensor(global_step):
                if self.beach_center_mask == True:
                    h, w = self.ldr_shape[0], self.ldr_shape[1]
                    mask = np.zeros((h, w)).astype(np.float32)
                    mask[:, w // 4: 3 * w // 4] = 255
                    mask = mask[None, :, :, None]
                    return mask

                if len(self.mask_range) == 2:
                    number_of_masks = random.randrange(int(self.mask_range[0]), int(self.mask_range[1]) + 1)
                elif self.specific_mask_number > 0:
                    number_of_masks = self.specific_mask_number
                elif mask_curriculum:
                    if self.more_turns_to_more_masks:
                        if global_step > 100 * (self.term_index + 1) * (self.term_index + 1) + 900 * (
                                self.term_index + 1):
                            self.term_index = self.term_index + 1
                        number_of_masks = max(15 - self.term_index, self.minimum_mask_number)
                    else:
                        if self.out2in:
                            number_of_masks = min(
                                self.minimum_mask_number + global_step // self.number_of_masks_decay_steps, 15)
                        else:
                            number_of_masks = max(15 - global_step // self.number_of_masks_decay_steps,
                                                  self.minimum_mask_number)

                    if self.curriculum_range:
                        number_of_masks = random.randrange(number_of_masks, 16)
                else:
                    number_of_masks = 1
                random_regions = random.sample(self.masks_16, number_of_masks)
                # mask들 미리 만들면 다음 코드들이 필요 없어짐
                random_region = np.maximum.reduce(random_regions)
                random_region = random_region[None, :, :, None]
                random_region = random_region.astype(np.float32)
                return random_region

            def get_comp_mask(im, global_step):
                # 32 steps
                # 3.125% -> 50%
                # 40000 / 32 = 1250
                # 80000 / 32 = 2500
                if self.comp80000:
                    decay = 2500
                else:
                    decay = 1250

                h, w, c = im.shape
                mask_percentage = 3.125 + (global_step // decay) * ((50 - 3.125) / 31)
                new_mask = np.zeros((h, w), dtype=np.float32)
                new_mask[:, int((w // 2) * (1 - mask_percentage / 100)):int((w // 2) * (1 + mask_percentage / 100))] = 255
                new_mask = new_mask[None, :, :, None]
                return new_mask

            mask_tensor = tf.py_func(get_mask_tensor, [global_step], tf.float32)
            mask_tensor.set_shape([1, self.im_size[0], self.im_size[1], 1])

            comp_mask_tensor = tf.py_func(get_comp_mask, [im_gt, global_step], tf.float32)
            comp_mask_tensor.set_shape([1, self.im_size[0], self.im_size[1], 1])

        return batch_beach, mask_tensor, comp_mask_tensor
