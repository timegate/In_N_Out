import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import add_arg_scope
from functools import partial

from net.ops import random_sqaure, Margin, fixed_bbox_withMargin, center_bbox_withMargin, bbox2mask
from net.ops import confidence_driven_mask, relative_spatial_variant_mask, deconv_frac_strided
from net.ops import flatten, gan_wgan_loss, gradients_penalty, random_interpolates
from net.ops import subpixel_conv, bilinear_conv, context_normalization, max_downsampling, unfold_conv
from net.ops import id_mrf_reg
from util.util import f2uint

import numpy as np

# Mix
import random

class SemanticRegenerationNet:
    def __init__(self):
        self.name = 'SemanticRegenerationNet'
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.d_unit = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    @add_arg_scope
    def _deconv(self, x, filters, name='deconv', reuse=False):
        h, w = x.get_shape().as_list()[1:3]
        x = tf.image.resize_nearest_neighbor(x, [h * 2, w * 2], align_corners=True)
        with tf.variable_scope(name, reuse=reuse):
            x = self.conv3(inputs=x, filters=filters, strides=1, name=name + '_conv')
        return x

    @add_arg_scope
    def FPN(self, x, cnum):
        conv3, conv5, deconv = self.conv3, self.conv5, self._deconv
        conv1_x = conv5(inputs=x, filters=cnum, strides=1, name='conv1')
        conv2_x = conv3(inputs=conv1_x, filters=cnum * 2, strides=2, name='conv2_downsample')
        conv3_x = conv3(inputs=conv2_x, filters=cnum * 2, strides=1, name='conv3')
        conv4_x = conv3(inputs=conv3_x, filters=cnum * 4, strides=2, name='conv4_downsample')
        conv5_x = conv3(inputs=conv4_x, filters=cnum * 4, strides=1, name='conv5')
        conv6_x = conv3(inputs=conv5_x, filters=cnum * 4, strides=1, name='conv6')

        conv7_x = conv3(inputs=conv6_x, filters=cnum * 4, strides=1, dilation_rate=2, name='conv7_atrous')
        conv8_x = conv3(inputs=conv7_x, filters=cnum * 4, strides=1, dilation_rate=4, name='conv8_atrous')
        conv9_x = conv3(inputs=conv8_x, filters=cnum * 4, strides=1, dilation_rate=8, name='conv9_atrous')
        conv10_x = conv3(inputs=conv9_x, filters=cnum * 4, strides=1, dilation_rate=16, name='conv10_atrous')

        conv11_x = conv3(inputs=conv10_x, filters=cnum * 4, strides=1, name='conv11')
        conv12_x = conv3(inputs=conv11_x, filters=cnum * 4, strides=1, name='conv12')
        conv13_x = deconv(conv12_x, filters=cnum * 2, name='conv13_upsample')
        conv14_x = conv3(inputs=conv13_x, filters=cnum * 2, strides=1, name='conv14')
        conv15_x = deconv(conv14_x, filters=cnum, name='conv15_upsample')
        features = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x, conv7_x, conv8_x, conv9_x, conv10_x, conv11_x,
                    conv12_x, conv13_x, conv14_x, conv15_x]
        return conv15_x, features

    @add_arg_scope
    def CPN(self, x_fe, x_in, mask, cnum, use_cn=True, alpha=0.5):
        conv3, conv5, deconv = self.conv3, self.conv5, self._deconv
        ones_x = tf.ones_like(x_in)[:, :, :, 0:1]
        xnow = tf.concat([x_fe, x_in, mask * ones_x], axis=3)

        conv1_x = conv5(inputs=xnow, filters=cnum, strides=1, name='xconv1')
        conv2_x = conv3(inputs=conv1_x, filters=cnum, strides=2, name='xconv2_downsample')
        conv3_x = conv3(inputs=conv2_x, filters=cnum * 2, strides=1, name='xconv3')
        conv4_x = conv3(inputs=conv3_x, filters=cnum * 2, strides=2, name='xconv4_downsample')
        conv5_x = conv3(inputs=conv4_x, filters=cnum * 4, strides=1, name='xconv5')
        conv6_x = conv3(inputs=conv5_x, filters=cnum * 4, strides=1, name='xconv6')

        conv7_x = conv3(inputs=conv6_x, filters=cnum * 4, strides=1, dilation_rate=2, name='xconv7_atrous')
        conv8_x = conv3(inputs=conv7_x, filters=cnum * 4, strides=1, dilation_rate=4, name='xconv8_atrous')
        conv9_x = conv3(inputs=conv8_x, filters=cnum * 4, strides=1, dilation_rate=8, name='xconv9_atrous')
        conv10_x = conv3(inputs=conv9_x, filters=cnum * 4, strides=1, dilation_rate=16, name='xconv10_atrous')

        conv11_x = conv3(inputs=conv10_x, filters=cnum * 4, strides=1, name='allconv11')
        if use_cn:
            cn_x = context_normalization(conv11_x, mask, alpha=alpha)
        conv12_x = conv3(inputs=cn_x, filters=cnum * 4, strides=1, name='allconv12')
        conv13_x = deconv(conv12_x, filters=cnum * 2, name='allconv13_upsample')
        conv14_x = conv3(inputs=conv13_x, filters=cnum * 2, strides=1, name='allconv14')
        conv15_x = deconv(conv14_x, filters=cnum, name='allconv15_upsample')
        conv16_x = conv3(inputs=conv15_x, filters=cnum // 2, strides=1, name='allconv16')
        conv17_x = tf.layers.conv2d(inputs=conv16_x, kernel_size=3, filters=3, strides=1, activation=None,
                                    padding='SAME',
                                    name='allconv17')
        x = tf.clip_by_value(conv17_x, -1, 1)
        features = [conv1_x, conv2_x, conv3_x, conv4_x, conv5_x, conv6_x, conv7_x, conv8_x, conv9_x, conv10_x, conv11_x,
                    cn_x, conv12_x, conv13_x, conv14_x, conv15_x, conv16_x, conv17_x]
        return x, features

    def build_generator2(self, x, xin_expanded, mask, config=None, reuse=False, name='inpaint_net'):
        if config is not None:
            use_cn = config.use_cn
            assert config.feat_expansion_op in ['subpixel', 'deconv', 'bilinear-conv', 'unfold']
            if config.feat_expansion_op == 'subpixel':
                feature_expansion_op = subpixel_conv
            elif config.feat_expansion_op == 'deconv':
                feature_expansion_op = deconv_frac_strided
            elif config.feat_expansion_op == 'unfold':
                feature_expansion_op = unfold_conv
            else:
                feature_expansion_op = bilinear_conv
        else:
            use_cn = True
            feature_expansion_op = subpixel_conv

        target_shape = mask.get_shape().as_list()[1:3]
        # xin_expanded = tf.pad(x, [[0, 0], [margin.top, margin.bottom], [margin.left, margin.right], [0, 0]])
        xin_expanded.set_shape((x.get_shape().as_list()[0], target_shape[0], target_shape[1], 3))
        expand_scale_ratio = int(np.prod(mask.get_shape().as_list()[1:3]) / np.prod(x.get_shape().as_list()[1:3]))

        """
        print("target")
        print(target_shape, x, xin_expanded, expand_scale_ratio)
        print("MM")
        print(margin, mask)
        print("ESR")
        print(expand_scale_ratio)
        """

        # two stage network
        cnum = config.g_cnum
        with tf.variable_scope(name, reuse=reuse):
            x, features_FPN = self.FPN(x, cnum)

            # mask
            if x.shape[1] != config.after_FPN_img_shapes[0] or x.shape[2] != config.after_FPN_img_shapes[1]:
                print("Warning: FPN output shape is different from input shape.")
                print("x: " + str(x.shape))
                print("config: " + str(config.after_FPN_img_shapes))
                x = tf.image.resize(x, [config.after_FPN_img_shapes[0], config.after_FPN_img_shapes[1]])

            # subpixel module, ensure the output channel the same as the input
            # subpixel 이외의 것을 쓰려면, channel 수를 바꿔야 할 수 있음.
            x_fe = feature_expansion_op(x, cnum * expand_scale_ratio, 3, target_shape,
                                        name='feat_expansion_' + config.feat_expansion_op)

            x, features_CPN = self.CPN(x_fe, xin_expanded, mask, cnum, use_cn, config.fa_alpha)
        return x, x_fe, features_FPN, features_CPN

    def build_wgan_contextual_discriminator(self, x, mask, config, reuse=False):
        if mask.shape[0] != 1:
            assert (x.shape[0] % mask.shape[0] == 0)
            tensor_for_tiling = tf.constant([x.shape[0] // mask.shape[0], 1, 1, 1], tf.int32)
            # gradient can be doubled.
            mask = tf.tile(mask, tensor_for_tiling)
        cnum = config.d_cnum
        dis_conv = self.d_unit
        with tf.variable_scope('D_context', reuse=reuse):
            h, w = x.get_shape().as_list()[1:3]
            x = dis_conv(x, cnum, name='dconv1')
            x = dis_conv(x, cnum * 2, name='dconv2')
            x = dis_conv(x, cnum * 4, name='dconv3')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                 name='dconv4')
            mask = max_downsampling(mask, ratio=8)
            x = x * mask
            x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
            mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
            return x, mask_local[0: config.batch_size]

    def build_wgan_global_discriminator(self, x, config, reuse=False):
        cnum = config.d_cnum
        dis_conv = self.d_unit
        with tf.variable_scope('D_global', reuse=reuse):
            x = dis_conv(x, cnum, name='conv1')
            x = dis_conv(x, cnum * 2, name='conv2')
            x = dis_conv(x, cnum * 4, name='conv3')
            x = dis_conv(x, cnum * 2, name='conv4')
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_global, config, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.build_wgan_global_discriminator(
                batch_global, config=config, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_global

    def build_contextual_wgan_discriminator(self, batch_global, mask, config, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.build_wgan_global_discriminator(batch_global, config=config, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.build_wgan_contextual_discriminator(batch_global, mask,
                                                                              config=config, reuse=reuse)
            return dout_local, dout_global, mask_local

    # mask
    def build_net2_ldr(self, batch_data, mask, global_step, config, summary=True, reuse=False):
        batch_pos = batch_data / 127.5 - 1.

        if 0 < config.mix_ratio and config.mix_ratio < 1:
            def given_mask(mask):
                mask = mask / 255.
                if not config.reverse_given_mask:
                    mask = 1. - mask  # we need to predict context
                return mask

            def get_random_mask():
                bbox, _ = random_sqaure(config)
                mask = bbox2mask(bbox, config)
                if not config.train_inpainting:
                    mask = 1. - mask
                return mask

            # use random_mask with percentage of mix_ratio * 100
            use_given_mask = tf.random_uniform([]) > config.mix_ratio
            mask = tf.cond(use_given_mask, lambda: given_mask(mask), lambda: get_random_mask())
        else:
            if config.use_random_mask:
                bbox, _ = random_sqaure(config)
                mask = bbox2mask(bbox, config)
                if not config.train_inpainting:
                    mask = 1. - mask
            else:
                mask = mask / 255.
                if not config.reverse_given_mask:
                    mask = 1. - mask  # we need to predict context

        # 원래 mask region에는 batch_data를, mask_region이 아닌곳에는 random noise를

        if config.noise_curriculum:
            # Augmentation4. noise
            float_global_step = tf.cast(global_step, tf.float32)
            # 42 / 15 = 2.8
            noise_ratio = 0.0 + 5.6 * (float_global_step / config.train_spe) / 255.0
            noise_ratio = tf.cond(noise_ratio > 84.0 / 255.0, lambda: 1.0, lambda: noise_ratio)

            batch_incomplete = batch_pos * (1. - mask) + (
                    noise_ratio * tf.random.normal(shape=batch_pos.get_shape(), mean=0.0, stddev=1.0,
                                                   dtype=tf.float32) + (1 - noise_ratio) * batch_pos) * mask
        else:
            batch_incomplete = batch_pos * (1. - mask) + tf.random.uniform(shape=batch_pos.get_shape(), maxval=1.,
                                                                           minval=-1., dtype=tf.float32) * mask
        xin_expanded = batch_pos * (1. - mask) + tf.zeros_like(batch_pos, dtype=tf.float32) * mask

        if config.l1_type == 0:
            mask_priority = relative_spatial_variant_mask(mask)
        elif config.l1_type == 1:
            mask_priority = confidence_driven_mask(mask)
        else:
            mask_priority = mask

        # mask_priority가 어떻게 생겼지? 그냥 만들어진 priority mask+(threshold)를 넣어줘도 되지않나?
        x, x_fe, features_FPN, features_CPN = self.build_generator2(batch_incomplete, xin_expanded, mask, config=config, reuse=reuse)
        batch_predicted = x

        # apply mask and complete image
        batch_complete = batch_predicted * mask + batch_pos * (1. - mask)

        losses = {}
        losses['g_loss'] = 0

        if config.use_L1:
            losses['l1_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * mask_priority)

        if config.use_AE:
            losses['ae_loss'] = config.pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x) * (1. - mask))
            losses['ae_loss'] /= tf.reduce_mean(1. - mask)

        if config.use_id_mrf:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            losses['id_mrf_loss'] = id_mrf_reg(batch_predicted, batch_pos, config)

        if config.use_WGAN:
            losses['d_loss'] = 0

            # gan
            batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

            # wgan with gradient penalty
            build_critics = self.build_contextual_wgan_discriminator
            # seperate gan
            global_wgan_loss_alpha = 1.0

            pos_neg_local, pos_neg_global, mask_local = build_critics(batch_pos_neg, mask, config=config, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['wgan_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = build_critics(interpolates_global, mask, config=config, reuse=True)
            # apply penalty
            # print("gp")
            # Tensor("Reshape_6:0", shape=(8, 256, 320, 3), dtype=float32) Tensor("discriminator_1/D_context/truediv:0", shape=(8,), dtype=float32) Tensor("discriminator/D_context/ResizeNearestNeighbor:0", shape=(16, 256, 320, 1), dtype=float32)
            # print(interpolates_local, dout_local, mask_local)
            # 앞이 8 뒤가 16
            # tf.gradients(y, x)[0]==8 * mask_local == 16
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask_local)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']

        if summary:
            if config.use_L1:
                ldr_l1 = tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
                tf.add_to_collection('summary_op', ldr_l1)
            if config.use_AE:
                ldr_ae = tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
                tf.add_to_collection('summary_op', ldr_ae)
            if config.use_id_mrf:
                ldr_im = tf.summary.scalar('losses/id_mrf_loss', losses['id_mrf_loss'])
                tf.add_to_collection('summary_op', ldr_im)
            if config.use_WGAN:
                ldr_d = tf.summary.scalar('losses/d_loss', losses['d_loss'])
                ldr_wg = tf.summary.scalar('losses/wgan_gp_loss', losses['gp_loss'])
                tf.add_to_collection('summary_op', ldr_d)
                tf.add_to_collection('summary_op', ldr_wg)

            viz_batch_pos = tf.image.resize(batch_pos[0:1], [config.result_img_shapes[0], config.result_img_shapes[1]])
            # viz_mask = tf.image.grayscale_to_rgb(tf.image.resize(mask, [config.result_img_shapes[0], config.result_img_shapes[1]]) * 255.)
            viz_batch_incomplete = tf.image.resize(batch_incomplete[0:1],
                                                   [config.result_img_shapes[0], config.result_img_shapes[1]])
            # viz_xin_expanded = tf.image.resize(xin_expanded[0:1], [config.result_img_shapes[0], config.result_img_shapes[1]])
            viz_batch_complete = tf.image.resize(batch_complete[0:1],
                                                 [config.result_img_shapes[0], config.result_img_shapes[1]])

            # pre_viz_img = tf.concat([batch_pos, batch_incomplete, xin_expanded, batch_complete], axis=2)
            viz_img = tf.concat([viz_batch_pos, viz_batch_incomplete, viz_batch_complete], axis=2)
            # tf.summary.image('gt__input w padding__prediction', f2uint(pre_viz_img))
            ldr_gt_and_pred = tf.summary.image('ldr/ldr_gt_and_pred', f2uint(viz_img))
            tf.add_to_collection('image_summary_op', ldr_gt_and_pred)

        if config.use_L1:
            losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        if config.use_AE:
            losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        if config.use_id_mrf:
            losses['g_loss'] += config.mrf_alpha * losses['id_mrf_loss']
        if config.use_WGAN:
            losses['g_loss'] += config.gan_loss_alpha * losses['wgan_loss']

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        return g_vars, d_vars, losses

    def evaluate3(self, image, mask, config, reuse=False):
        batch_pos = image / 127.5 - 1.
        mask = mask / 255.
        if not config.inpainting:
            mask = 1. - mask  # we need to predict context

        batch_incomplete = batch_pos * (1. - mask) + tf.random.uniform(shape=batch_pos.get_shape(), maxval=1.,
                                                                       minval=-1., dtype=tf.float32) * mask
        xin_expanded = batch_pos * (1. - mask) + tf.zeros_like(batch_pos, dtype=tf.float32) * mask

        x, x_fe, features_FPN, features_CPN = self.build_generator2(batch_incomplete, xin_expanded, mask, config=config, reuse=reuse)
        batch_predict = x

        # apply mask and reconstruct
        batch_complete = batch_predict * mask + batch_pos * (1 - mask)
        print("batch_complete!")
        print(batch_complete)

        return batch_incomplete, xin_expanded, batch_complete, x_fe, features_FPN, features_CPN
