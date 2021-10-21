for name in baseline_batch8 in_batch8
do
for ((i=500;i<=40000;i+=500))
do
./test_out.sh "/root/In_N_Out/ablation_study/checkpoints/checkpoint_ldr_out/${name}" "/root/In_N_Out/ablation_study/out_test_results/${name}" "/root/In_N_Out/ablation_study/out_logs/${name}" 0 "/root/In_N_Out/ablation_study/out_psnr_ssim/${name}" ${i} 3
done
done

