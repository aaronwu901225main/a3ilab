# ##  CEDUE  ###
# for seed in 2 211 1212
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP/$seed/ckpt.pt

#     CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# #     # CUDA_VISIBLE_DEVICES=0 python bay_CEdue.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed 

# #     # CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# #     # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# done

# for seed in 211 1212 2
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP/$seed/ckpt.pt
#     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/$seed/ckpt.pt


#     #CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 

#     #CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

#     CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

# done

# for seed in 211 1212 2
# do 
#     ckpt1=/root/notebooks/nfs/work/Kelly.Lin/GP/old/$seed/ckpt.pt
#     ckpt2=/root/notebooks/nfs/work/Kelly.Lin/GP/new/$seed/ckpt.pt
#     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/$seed/ckpt.pt


#     #CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 

#     #CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

#     CUDA_VISIBLE_DEVICES=0 python inference.py --relabel --checkpoint_path1 $ckpt1 --checkpoint_path2 $ckpt2 --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

# done

# for seed in 238 #589 #211 1212 2
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP/$seed/ckpt.pt
#     mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/MLP/$seed/ckpt.pt
#     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/GP/$seed/ckpt.pt
    
    
#     # CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 
    
#     CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

    # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --mlp_checkpoint_path $mlp_ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

# done

for seed in 2 211 1212  #0 1 2 3 4 5 6 7 8 9 10 11 12 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/baseline/$seed/ckpt.pt
# #     mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/MLP/$seed/ckpt.pt
# #     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/GP/$seed/ckpt.pt
    
    
# #     CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 
    
# #     CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

    CUDA_VISIBLE_DEVICES=0 python inference_CE_six.py --relabel --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CE_rd_$seed

done
