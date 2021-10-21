import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
import random
import glob

# using pre-defined mask images
class DataLoader4_ldr_celebahq:
    def __init__(self, filename, im_size, batch_size, exact_decay, decay_steps, more_turns_to_more_masks,
                 specific_mask_number, mask_range, minimum_mask_number, curriculum_range, out2in):
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

        # self.masklist = [n.replace("/data", "/mask") for n in self.filelist]
        self.masklist = []
        for n in self.filelist:
            if n.count("/data") > 1:
                print("please check data.py")
                exit(-1)
            self.masklist.append(n.replace("/data", "/mask").replace('.jpg', '.png'))

    def next(self, global_step, mask_curriculum):
        with tf.variable_scope('feed'):
            filelist_tensor = tf.convert_to_tensor(self.filelist, dtype=tf.string)
            masklist_tensor = tf.convert_to_tensor(self.masklist, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([filelist_tensor, masklist_tensor], shuffle=True)

            im_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            im_gt = tf.cast(im_gt, tf.float32)
            im_gt.set_shape(self.ldr_shape)

            celebahq_mask_tensor = tf.image.decode_image(tf.read_file(self.data_queue[1]), channels=1)
            celebahq_mask_tensor = tf.cast(celebahq_mask_tensor, tf.float32)
            celebahq_mask_tensor.set_shape(list(self.ldr_shape[:-1]) + [1])

            batch_celebahq = tf.train.batch([im_gt, celebahq_mask_tensor], batch_size=self.batch_size, num_threads=4)

            def get_mask_tensor(global_step):
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
                            number_of_masks = min(self.minimum_mask_number + global_step // self.number_of_masks_decay_steps, 15)
                        else:
                            number_of_masks = max(15 - global_step // self.number_of_masks_decay_steps, self.minimum_mask_number)

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

            mask_tensor = tf.py_func(get_mask_tensor, [global_step], tf.float32)
            mask_tensor.set_shape([1, self.im_size[0], self.im_size[1], 1])

        return batch_celebahq, mask_tensor