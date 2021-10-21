import os
import tensorflow as tf
from net.network import SemanticRegenerationNet
from data.data import DataLoader4_ldr
from data.data_celebahq import DataLoader4_ldr_celebahq
from data.data_beach import DataLoader4_ldr_beach
from options.train_options import TrainOptions

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend as K

config = TrainOptions().parse()

tf.keras.backend.clear_session()
K.set_learning_phase(1)

# tfconfig = tf.ConfigProto(device_count = {'GPU': config.proto_gpu})
tfconfig = tf.ConfigProto()
if config.gpu_allow_growth:
    tfconfig.gpu_options.allow_growth = True
else:
    tfconfig.gpu_options.allow_growth = False
sess = tf.Session(config=tfconfig)
graph = tf.get_default_graph()
set_session(sess)

model = SemanticRegenerationNet()

# control_dependency?
# https://stackoverflow.com/questions/60882387/how-to-get-current-global-step-in-data-pipeline

# DATASET
# not accurate global_step because of prefetching in the "next" function
# but it dosen't matter much cause we use global_step in a form of (global_step/train_spe).
# usually train_spe = 1000.
global_step = tf.Variable(1, name='global_step', trainable=False)
decay_iter = config.train_spe
if config.decay_iter > 0:
    decay_iter = config.decay_iter

if "celebahq" in config.dataset:
    dataLoader4 = DataLoader4_ldr_celebahq(filename=config.dataset_path, im_size=config.img_shapes,
                                           batch_size=config.batch_size, exact_decay=config.exact_spe,
                                           decay_steps=decay_iter,
                                           more_turns_to_more_masks=config.more_turns_to_more_masks,
                                           specific_mask_number=config.specific_mask_number,
                                           mask_range=config.mask_range,
                                           minimum_mask_number=config.minimum_mask_number,
                                           curriculum_range=config.curriculum_range, out2in=config.out2in)
    images_and_celebahq_mask, mask = dataLoader4.next(global_step, config.mask_curriculum)
    images = images_and_celebahq_mask[0]
elif "beach" in config.dataset:
    dataLoader4 = DataLoader4_ldr_beach(filename=config.dataset_path, im_size=config.img_shapes,
                                        batch_size=config.batch_size, exact_decay=config.exact_spe,
                                        decay_steps=decay_iter,
                                        more_turns_to_more_masks=config.more_turns_to_more_masks,
                                        specific_mask_number=config.specific_mask_number,
                                        mask_range=config.mask_range,
                                        minimum_mask_number=config.minimum_mask_number,
                                        curriculum_range=config.curriculum_range, out2in=config.out2in,
                                        beach_center_mask=config.beach_center_mask, comp80000=config.comp80000)
    images_and_comp_images, mask, comp_mask = dataLoader4.next(global_step, config.mask_curriculum)
    if config.training_comp:
        images = images_and_comp_images[1]
        mask = comp_mask
    else:
        images = images_and_comp_images[0]
        mask = mask
else:
    dataLoader4 = DataLoader4_ldr(filename=config.dataset_path, im_size=config.img_shapes,
                                  batch_size=config.batch_size, exact_decay=config.exact_spe,
                                  decay_steps=decay_iter, more_turns_to_more_masks=config.more_turns_to_more_masks,
                                  specific_mask_number=config.specific_mask_number, mask_range=config.mask_range,
                                  minimum_mask_number=config.minimum_mask_number,
                                  curriculum_range=config.curriculum_range, out2in=config.out2in, grid_number=config.grid_number)
    images, mask = dataLoader4.next(global_step, config.mask_curriculum)
increment_global_step_op = tf.assign_add(global_step, 1, name='increment')
# images, images2, mask = images[:, :, :, ::-1], images2[:, :, :, ::-1], mask[:, :, :, ::-1]

import math

print("1 epoch: {}".format(math.ceil(dataLoader4.len_files / config.batch_size)))

# TRAIN
if config.exact_spe:
    config.train_spe = math.ceil(dataLoader4.len_files / config.batch_size)
    config.max_iters = config.train_spe * 20

g_vars, d_vars, losses = model.build_net2_ldr(images, mask, global_step, config=config)

# GRADIENT DESCENT
lr = tf.get_variable(
    'lr', shape=[], trainable=False,
    initializer=tf.constant_initializer(config.lr))

g_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
g_train_op = g_optimizer.minimize(losses['g_loss'], var_list=g_vars)

if config.use_WGAN:
    # 같은 객체?
    # d_optimizer = g_optimizer
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    d_train_op = d_optimizer.minimize(losses['d_loss'], var_list=d_vars)

if config.use_L1:
    c_l1 = tf.Variable(0.0)
    tf.summary.scalar('combined/l1_loss', c_l1, collections=['combined_summary_op'])

if config.use_AE:
    c_ae = tf.Variable(0.0)
    tf.summary.scalar('combined/ae_loss', c_ae, collections=['combined_summary_op'])

if config.use_id_mrf:
    c_im = tf.Variable(0.0)
    tf.summary.scalar('combined/im_loss', c_im, collections=['combined_summary_op'])

if config.use_WGAN:
    c_d = tf.Variable(0.0)
    tf.summary.scalar('combined/d_loss', c_d, collections=['combined_summary_op'])

    c_wg = tf.Variable(0.0)
    tf.summary.scalar('combined/wg_loss', c_wg, collections=['combined_summary_op'])

saver = tf.train.Saver(max_to_keep=100000, keep_checkpoint_every_n_hours=0.1)
summary_op = tf.summary.merge_all('summary_op')
image_summary_op = tf.summary.merge_all('image_summary_op')
combined_summary_op = tf.summary.merge_all('combined_summary_op')

# with tf.Session() as sess:
with graph.as_default():
    set_session(sess)
    # sess.run(tf.global_variables_initializer())
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    sess.run(tf.initialize_variables(all_variables))
    if config.load_model_dir != '':
        print('[-] Loading the pretrained model from: {}'.format(config.load_model_dir))

        if config.load_model_dir.endswith("/"):
            config.load_model_dir = config.load_model_dir[:-1]

        if "snap" in os.path.basename(config.load_model_dir):
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.load_model_dir))
        else:
            ckpt = tf.train.get_checkpoint_state(config.load_model_dir)

        if ckpt:
            print(config.load_model_dir)
            if config.inherit_d:
                print("Brings g_vars and d_vars.")
                assign_ops = list(
                    map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                        g_vars + d_vars))
            else:
                print("Brings only g_vars.")
                assign_ops = list(
                    map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                        g_vars))

            sess.run(assign_ops)
            print("[*] Loading SUCCESS.")
        else:
            print("[x] Loading ERROR.")

    if config.specific_model_folder == '':
        summary_writer = tf.summary.FileWriter(config.model_folder, sess.graph, flush_secs=30)
    else:
        summary_writer = tf.summary.FileWriter(config.specific_model_folder, sess.graph, flush_secs=30)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(1, config.max_iters + 1):
        if config.use_WGAN:
            for _ in range(5):
                _ = sess.run([d_train_op])

        if step % 20 != 0:
            _, _ = sess.run([g_train_op, increment_global_step_op])

        else:
            _, _, g_loss, train_losses, train_images_summary = sess.run(
                [g_train_op, increment_global_step_op, losses['g_loss'], losses,
                 image_summary_op])

            print(
                '({:04d}, {:04d}), G_loss > {}'.format(step // config.train_spe, step % config.train_spe, g_loss))
            summary_writer.add_summary(train_images_summary, global_step=step)

            # combined_summary (frame for combining summaries)
            if config.use_L1 and config.use_AE and config.use_id_mrf and config.use_WGAN:
                _, _, _, _, _, combined_summary = sess.run(
                    [c_ae, c_d, c_im, c_l1, c_wg, combined_summary_op],
                    feed_dict={c_ae: train_losses['ae_loss'], c_d: train_losses['d_loss'],
                               c_im: train_losses['id_mrf_loss'],
                               c_l1: train_losses['l1_loss'], c_wg: train_losses['gp_loss']})
            else:
                if config.use_L1 and config.use_AE and (not config.use_id_mrf) and (not config.use_WGAN):
                    _, _, combined_summary = sess.run([c_l1, c_ae, combined_summary_op],
                                                      feed_dict={c_l1: train_losses['l1_loss'],
                                                                 c_ae: train_losses['ae_loss']})
                else:
                    raise

            summary_writer.add_summary(combined_summary, global_step=step)

        # save_iter
        if step % config.save_iter == 0:
            if config.specific_model_folder == '':
                saver.save(sess, os.path.join(config.model_folder, config.model_prefix), step)
            else:
                saver.save(sess, os.path.join(config.specific_model_folder, config.model_prefix), step)

    coord.request_stop()
    coord.join(thread)
