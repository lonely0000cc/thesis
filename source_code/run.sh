#test
#python test.py --model_name UnFlowSRGNet --epoch_to_load 250 --test_lr True --n_resblocks 20 --im_crop_H 128 --im_crop_W 128 \
#               --checkpoint_dir /local/home/yuanhao/scratch/tem/EDSR_PWC_GAN/2020-05-13-03:23:02 --use_pretrained_model True \
#               --batch_size 1 --have_gt True
               #--batch_size 1 --have_gt True

#tessttrain
python test.py --model_name TestSRNet --test_train True --is_training True --n_resblocks 16 --im_crop_H 32 --im_crop_W 32 \
               --lr 1e-4 --lr_step 50 --freq_visual 1 --batch_size 1 --end_epoch 200 --have_gt True

#run locally

#python main.py --model_name UnFusionFlowSRNet --lr 1e-4 --im_crop_H 32 --im_crop_W 32 --n_resblocks 20 \
#--data_dir /local/home/yuanhao/thesis/super_resolution_image/dataset/satellite/training/ \
#--checkpoint_dir /local/home/yuanhao/thesis/super_resolution_image/tem/EDSR_PWC_GAN \
#--model_dir /local/home/yuanhao/scratch/tem/pretrained_models \
#--end_epoch 1 --freq_visual 1 --batch_size 2
#--use_pretrained_model True --epoch_to_load pretrained

#run on cluster
#bsub -o 1000_new_satellite_SRNet_texture_loss_batch_DRLN.out -W 120:00 -n 5 \
#     -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#     python main.py --model_name FlowSRNet --lr 1e-5 --lr_step 500 --im_crop_H 320 --im_crop_W 320 \
#                    --end_epoch 1000 --batch_size 2 \
#                    --description test_new_satellite_images_SRNet_DRLN
