for name in baseline_batch8 out_batch8
do
for ((i=500;i<=40000;i+=500))
do
./test_in.sh "/root/In_N_Out/ablation_study/checkpoints/checkpoint_ldr/${name}" "/root/In_N_Out/ablation_study/in_test_results/${name}" "/root/In_N_Out/ablation_study/in_logs/${name}" 1 "/root/In_N_Out/ablation_study/in_psnr_ssim/${name}" ${i}
done
done

