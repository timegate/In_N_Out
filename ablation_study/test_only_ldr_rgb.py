import os
import tensorflow as tf
from net.network import SemanticRegenerationNet

from options.test_options import TestOptions
import subprocess
import numpy as np
import cv2
import time
import random
import glob

import scipy

# K.set_session()
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend as K

# This code only affects batch norm (we don't use dropout).
# Double checked that this does not affect to the conclusion of our paper (The metrics can get very slightly worse. but the tendency between baseline and In-N-Out method remains the same).
K.set_learning_phase(1)

"""
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
                                                   ))
"""

def generate_mask_without_margin(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        if im_shapes[1] == 512 or im_shapes[1] == 1024:
            of0 = 0
            of1 = (im_shapes[1] - mask_shapes[1]) // 2
        elif im_shapes[1] == 128:
            of0 = 0
            of1 = 0
        else:
            of0 = (im_shapes[0]-mask_shapes[0])//2
            of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask

def rearrange(im):
    dim = len(im.shape)

    if dim == 4:
        im = np.squeeze(im)

    new_im = np.zeros_like(im)
    h, w, c = im.shape
    new_im[:, w // 2:w, :] = im[:, 0:w // 2, :]
    new_im[:, 0:w // 2, :] = im[:, w // 2:w, :]

    if dim == 4:
        return new_im[None, :, :, :]
    else:
        return new_im

# from https://stackoverflow.com/questions/34047874/scipy-ndimage-interpolation-zoom-uses-nearest-neighbor-like-algorithm-for-scalin
def zoomArray(inArray, finalShape, sameSum=False,
              zoomFunction=scipy.ndimage.zoom, **zoomKwargs):
    """

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------

    inArray: n-dimensional numpy array (1D also works)
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    inArray = np.asarray(inArray, dtype=np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled

# need to be checked
# Assumption: batch_size = 1
def feature_to_weight(feature, power=1):
    feature = feature[0]

    # need to deepcopy?
    powered_feature = feature
    for i in range(power-1):
        powered_feature = powered_feature * feature
    feature = np.sum(powered_feature, axis=-1)
    feature = zoomArray(feature, (config.img_shapes[1], config.img_shapes[0]))
    return feature

def normalize_feature(feature):
    feature_max, feature_min = np.max(feature), np.min(feature)
    feature = np.clip((feature - feature_min) / (feature_max - feature_min), 0, 1)
    return feature

def histogram_equalization(layer):
    layer_dtype = layer.dtype
    layer = 255.0 * layer
    layer = layer.astype(np.uint8)
    layer = cv2.equalizeHist(layer)
    layer = layer.astype(layer_dtype)
    layer /= 255.0
    return layer[:, :, np.newaxis]

config = TestOptions().parse()
config.max_delta_shapes = [0, 0]

result_images = glob.glob(config.saving_path2 + "/*.png")
if "/CUB_200_2011/" in config.saving_path2 and len(result_images) == 8940:
    print("already fullfilled")
    exit(0)

if (("/celebA/" in config.saving_path2) or ("/celebA_128mask/" in config.saving_path2)) and len(result_images) == 14965:
    print("already fullfilled")
    exit(0)

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png')) + glob.glob(
        os.path.join(config.dataset_path, '*.PNG'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)

total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

if config.celebahq_testmask:
    # testmask = [n.replace("/data", "/mask") for n in pathfile]
    testmask = []
    for n in pathfile:
        if n.count("/data") > 1:
            print("please check train_only_ldr.py")
            exit(-1)
        testmask.append(n.replace("/data", "/mask").replace('.jpg', '.png'))
    # testmask = natsorted(testmask)

if config.model == 'srn':
    model = SemanticRegenerationNet()
elif config.model == 'srn-hr':
    model = HRSemanticRegenerationNet()
else:
    print('unknown model types.')
    exit(1)

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
session = tf.Session(config=sess_config)
graph = tf.get_default_graph()
set_session(session)

# 이거 체크하기
# with tf.Session(config=sess_config) as sess:
with graph.as_default():
    # Model
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    input_with_noise, input_with_pad, output, after_FPN, in_FPN, in_CPN = model.evaluate3(
        input_image_tf, mask_tf, config=config, reuse=reuse)

    # casting?

    # Load Model
    # Need to check these variables carefully.
    vars_list = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    """
    print(len(vars_list))
    print(len(tf.contrib.framework.list_variables(config.load_model_dir)))
    print("\n".join(sorted([str(v) for v in vars_list])))
    print("\n".join([str(v) for v in tf.contrib.framework.list_variables(config.load_model_dir)]))
    """

    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    session.run(assign_ops)
    print('Model loaded.')

    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    # Save Model
    if config.save_model_dir != "":
        inputs = {
            "input_image_tf": input_image_tf,
            "mask_tf": mask_tf,
        }
        outputs = {
            "pred_ldr": output,
        }
        tf.saved_model.simple_save(session, config.save_model_dir, inputs, outputs)
        print('Model saved.')

    # Test
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)

    for i in range(test_num):
        image = cv2.imread(pathfile[i])
        image = image[:,:,::-1]

        if config.beach_rearrange:
            image = rearrange(image)

        image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))

        if config.random_size:
            random_mask_shapes = [np.random.randint(1, config.img_shapes[0]), np.random.randint(1, config.img_shapes[1])]
            mask = generate_mask_without_margin(config.img_shapes, random_mask_shapes, config.random_mask)
            mask = mask * 255

        elif config.celebahq_testmask:
            mask = cv2.imread(testmask[i], cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=2)
        elif "beach" in config.dataset:
            h, w = config.img_shapes[0], config.img_shapes[1]
            mask = np.zeros((h, w)).astype(np.float32)
            mask[:, w // 4: 3 * w // 4] = 255
            mask = np.expand_dims(mask, axis=2)
            if config.beach_rearrange:
                mask = 255 - mask
        else:
            mask = generate_mask_without_margin(config.img_shapes, config.mask_shapes, config.random_mask)
            mask = mask * 255

        image = np.expand_dims(image, 0).astype(np.float32)
        mask = np.expand_dims(mask, 0).astype(np.float32)

        print('{} / {}'.format(i, test_num))
        start_t = time.time()
        result = session.run([input_with_noise, input_with_pad, output, after_FPN, in_FPN, in_CPN],
                             feed_dict={input_image_tf: image, mask_tf: mask})
        duration_t = time.time() - start_t
        total_time += duration_t

        noisy_input, padded_input, pred_ldr, feature_after_FPN, features_in_FPN, features_in_CPN = result

        noisy_input = np.clip((noisy_input + 1) * 127.5, 0, 255)
        padded_input = np.clip((padded_input + 1) * 127.5, 0, 255)
        pred_ldr = np.clip((pred_ldr + 1) * 127.5, 0, 255)

        # feature_to_weight: channel과 batch를 없애고 사이즈를 256, 256에 맞춤
        feature_after_FPN = feature_to_weight(feature_after_FPN)

        # features = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x, conv7_x, conv8_x, conv9_x, conv10_x, conv11_x, conv12_x, conv13_x, conv14_x, conv15_x]
        # features = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x, conv7_x, conv8_x, conv9_x, conv10_x, conv11_x, cn_x, conv12_x, conv13_x, conv14_x, conv15_x, conv16_x, conv17_x]
        conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15 = features_in_FPN
        xconv1, xconv2, xconv3, xconv4, xconv5, xconv6, xconv7, xconv8, xconv9, xconv10, xconv11, cn_x, xconv12, xconv13, xconv14, xconv15, xconv16, xconv17 = features_in_CPN

        """
        conv1 = feature_to_weight(conv1)
        # conv2 = feature_to_weight(conv2, power=2)
        # conv3 = feature_to_weight(conv3, power=4)
        conv3 = feature_to_weight(conv3, power=2)
        conv5 = feature_to_weight(conv5, power=4)
        # layer135 = normalize_feature(conv1 + conv3 + conv5)
        layer1 = normalize_feature(conv1)
        layer3 = normalize_feature(conv3)
        layer5 = normalize_feature(conv5)
        """
        layers = []
        xlayers = []
        for conv in features_in_FPN:
            layers.append(normalize_feature(feature_to_weight(conv)))

        for xconv in features_in_CPN:
            xlayers.append(normalize_feature(feature_to_weight(xconv)))

        # print(feature_after_FPN.max(), feature_after_FPN.min())
        # 0.81903756 -0.53696895

        # print(feature_after_FPN.shape)
        # (1, 128, 256, 64)

        # proper CAM needed to see this.
        # feature_after_FPN = np.clip((feature_after_FPN + 1) * 127.5, 0, 255)

        save_name = os.path.basename(pathfile[i])
        print(os.path.join(config.saving_path2, save_name + '(ldr).png'))

        m = mask[0].astype(np.uint8)
        n = noisy_input[0].astype(np.uint8)

        if config.beach_rearrange:
            m = 255 - m
            n = rearrange(n)
            image = rearrange(image)
            padded_input = rearrange(padded_input)
            pred_ldr = rearrange(pred_ldr)

        cv2.imwrite(os.path.join(config.saving_path2, save_name + '(mask).png'), m)

        if config.rgb_correction:
            bbox = np.where(m == 255)
            bbox = np.min(bbox[0]), np.max(bbox[0]), np.min(bbox[1]), np.max(bbox[1])

            new_n = n.copy()
            if config.flip_all:
                new_n = new_n[:, :, ::-1]
            else:
                n_crop_rgb = n[bbox[0]:bbox[1], bbox[2]:bbox[3]][:, :, ::-1]
                new_n[bbox[0]:bbox[1], bbox[2]:bbox[3]] = n_crop_rgb
                if config.correction_outside:
                    new_n = new_n[:, :, ::-1]

            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(input_with_noise).png'),
                        new_n)
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(input_with_pad).png'),
                        padded_input[0].astype(np.uint8)[:, :, ::-1])
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(ldr).png'),
                        image[0].astype(np.uint8)[:, :, ::-1])
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(pred_ldr).png'), pred_ldr[0].astype(np.uint8)[:,:,::-1])

            if config.feature:
                LDR = image[0].astype(np.uint8)[:, :, ::-1]
                # FEATURE = np.repeat(feature_after_FPN[:, :, np.newaxis], 3, axis=2)
                # FEATURE = np.repeat(layer135[:, :, np.newaxis], 3, axis=2)
                # cv2.imwrite(os.path.join(config.saving_path2, save_name + '(feature).png'), LDR * FEATURE)

                """
                l1_feature = np.repeat(histogram_equalization(layer1), 3, axis=2)
                l3_feature = np.repeat(histogram_equalization(layer3), 3, axis=2)
                l5_feature = np.repeat(histogram_equalization(layer5), 3, axis=2)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer1).png'), LDR * l1_feature)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer3).png'), LDR * l3_feature)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer5).png'), LDR * l5_feature)
                """
                for layer_index, l in enumerate(layers):
                    l_feature = np.repeat(l[:, :, np.newaxis], 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer{}).png'.format(layer_index + 1)), LDR * l_feature)

                for layer_index, l in enumerate(layers):
                    l_feature = np.repeat(histogram_equalization(l), 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(equalized_layer{}).png'.format(layer_index + 1)), LDR * l_feature)

                for layer_index, l in enumerate(xlayers):
                    l_feature = np.repeat(l[:, :, np.newaxis], 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(xlayer{}).png'.format(layer_index + 1)), LDR * l_feature)

                for layer_index, l in enumerate(xlayers):
                    l_feature = np.repeat(histogram_equalization(l), 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(equalized_xlayer{}).png'.format(layer_index + 1)), LDR * l_feature)
        else:
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(input_with_noise).png'), n)
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(input_with_pad).png'),
                        padded_input[0].astype(np.uint8))
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(ldr).png'),
                        image[0].astype(np.uint8))
            cv2.imwrite(os.path.join(config.saving_path2, save_name + '(pred_ldr).png'), pred_ldr[0].astype(np.uint8))
            if config.feature:
                LDR = image[0].astype(np.uint8)
                # FEATURE = np.repeat(feature_after_FPN[:, :, np.newaxis], 3, axis=2)
                # FEATURE = np.repeat(layer135[:, :, np.newaxis], 3, axis=2)
                # cv2.imwrite(os.path.join(config.saving_path2, save_name + '(feature).png'), LDR * FEATURE)

                """
                l1_feature = np.repeat(histogram_equalization(layer1), 3, axis=2)
                l3_feature = np.repeat(histogram_equalization(layer3), 3, axis=2)
                l5_feature = np.repeat(histogram_equalization(layer5), 3, axis=2)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer1).png'), LDR * l1_feature)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer3).png'), LDR * l3_feature)
                cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer5).png'), LDR * l5_feature)
                """
                for layer_index, l in enumerate(layers):
                    l_feature = np.repeat(l, 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(layer{}).png'.format(layer_index + 1)),
                                LDR * l_feature)

                for layer_index, l in enumerate(layers):
                    l_feature = np.repeat(histogram_equalization(l), 3, axis=2)
                    cv2.imwrite(
                        os.path.join(config.saving_path2, save_name + '(layer{}_he).png'.format(layer_index + 1)),
                        LDR * l_feature)

                for layer_index, l in enumerate(xlayers):
                    l_feature = np.repeat(l, 3, axis=2)
                    cv2.imwrite(os.path.join(config.saving_path2, save_name + '(xlayer{}).png'.format(layer_index + 1)),
                                LDR * l_feature)

                for layer_index, l in enumerate(xlayers):
                    l_feature = np.repeat(histogram_equalization(l), 3, axis=2)
                    cv2.imwrite(
                        os.path.join(config.saving_path2, save_name + '(xlayer{}_he).png'.format(layer_index + 1)),
                        LDR * l_feature)

        if reuse is False:
            reuse = True
    print('total time > {}s, average time > {}s'.format(total_time, total_time / test_num))
