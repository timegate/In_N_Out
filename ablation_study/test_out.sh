model_dir=$1
test_dir=$2
log_dir=$3
is_inpainting=$4
psnr_dir=$5
iter=$6
seed=$7

mf=${model_dir}/snap-${iter}.meta
  
current_model_path="${mf%.*}"

echo test_only_ldr_rgb.py --dataset celebahq_256 --data_file "/root/In_N_Out/ablation_study/celebA/partition_256/val.txt" --load_model_dir "${current_model_path}" --test_dir "${test_dir}" --img_shapes 256,256 --after_FPN_img_shapes 256,256 --result_img_shapes 256,256 --mask_shapes 128,128 --g_cnum 64 --seed ${seed} --save_only_basename 1 --inpainting ${is_inpainting} --feature 0 --random_size 0 --celebahq_testmask 1 --rgb_correction 1 --correction_outside 0
python test_only_ldr_rgb.py --dataset celebahq_256 --data_file "/root/In_N_Out/ablation_study/celebA/partition_256/val.txt" --load_model_dir "${current_model_path}" --test_dir "${test_dir}" --img_shapes 256,256 --after_FPN_img_shapes 256,256 --result_img_shapes 256,256 --mask_shapes 128,128 --g_cnum 64 --seed ${seed} --save_only_basename 1 --inpainting ${is_inpainting} --feature 0 --random_size 0 --celebahq_testmask 1 --rgb_correction 1 --correction_outside 0

# metrics
# model_basename=$(basename -- "$current_model_path")
# test_dir_basename=$(basename -- "$test_dir")
# echo "det_cub200.py" --saved_path "${test_dir}/${model_basename}" --log_folder ${log_dir} --log_prefix "${model_basename}"
# python "det_cub200.py" --saved_path "${test_dir}/${model_basename}" --log_folder "${log_dir}" --log_prefix "${model_basename}"

# echo ./psnr_ssim.py "${test_dir}/${model_basename}" "${test_dir}/${model_basename}" --log_file "${psnr_dir}/${model_basename}.txt"
# python ./psnr_ssim.py "${test_dir}/${model_basename}" "${test_dir}/${model_basename}" --log_file "${psnr_dir}/${model_basename}.txt"

