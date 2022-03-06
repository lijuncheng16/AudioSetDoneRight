# CUDA_VISIBLE_DEVICES=2 python train_multimodal_late_fusion.py --model_type TAL-trans --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 20 --additional_outname 20ep --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay
# CUDA_VISIBLE_DEVICES=2 python train_multimodal_late_fusion.py --model_type MMTLF --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 20 --additional_outname 20ep --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay  
# CUDA_VISIBLE_DEVICES=2 python train_multimodal_late_fusion.py --model_type TAL-trans --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 40 --additional_outname TALtrans_mgpu --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay --multi_gpu True
#CUDA_VISIBLE_DEVICES=1 python train_multimodal_video_only.py --model_type VM --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 30 --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.25 --additional_outname dropout0.25 --scheduler warmup-decay
CUDA_VISIBLE_DEVICES=2 python train_multimodal_late_fusion.py --model_type MMT --fusion_module 0 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 20 --additional_outname early --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay
CUDA_VISIBLE_DEVICES=2 python train_multimodal_late_fusion.py --model_type MMT --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 20 --additional_outname mid1 --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay
CUDA_VISIBLE_DEVICES=1 python train_multimodal_late_fusion.py --model_type wide_resnet50_2 --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 4e-4 --pooling att --max_ckpt 20 --additional_outname wideresnet --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay
CUDA_VISIBLE_DEVICES=1 python train_multimodal_late_fusion.py --model_type AST --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 1e-5 --pooling att --max_ckpt 80 --additional_outname AST_debug --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay --multi_gpu True
CUDA_VISIBLE_DEVICES=0,1 python train_multimodal_late_fusion.py --model_type AST --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 1e-5 --pooling att --max_ckpt 80 --additional_outname AST_divide_8_64 --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler multistepLR --multi_gpu True
python train_multimodal_late_fusion.py --model_type resnet --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 3000 --init_lr 4e-4 --pooling att --max_ckpt 30 --additional_outname res50mgpu --n_trans_layers 2 --gradient_accumulation 3 --transformer_dropout 0.75 --optimizer adam --scheduler warmup-decay --multi_gpu True
CUDA_VISIBLE_DEVICES=1,2,4,5 python train_multimodal_late_fusion.py --model_type AST --fusion_module 1 --embedding_size 1024 --batch_size 100 --ckpt_size 2500 --init_lr 1e-5 --pooling att --max_ckpt 80 --additional_outname AST_divide_8_64 --n_trans_layers 2 --gradient_accumulation 0 --transformer_dropout 0.75 --optimizer adam --scheduler multistepLR --multi_gpu True
CUDA_VISIBLE_DEVICES=1,2,4,5 python train_multimodal_late_fusion.py --model_type AST  --batch_size 100 --ckpt_size 2500 --init_lr 1e-5 --pooling att --max_ckpt 80 --additional_outname AST_divide_8_64 --gradient_accumulation 1  --optimizer adam --scheduler multistepLR --multi_gpu True --beta1 0.95 --beta2 0.999 --weight_decay 5e-7 --normalize_scale 16
CUDA_VISIBLE_DEVICES=1,2,4,5 python train_multimodal_late_fusion.py --model_type AST  --batch_size 100 --ckpt_size 2500 --init_lr 4e-5 --pooling att --max_ckpt 80 --additional_outname AST_divide_8_64 --gradient_accumulation 1  --optimizer adam --scheduler reduce --multi_gpu True --beta1 0.95 --beta2 0.999 --weight_decay 5e-7 --normalize_scale 8
python train_multimodal_late_fusion.py --model_type AST  --batch_size 100 --ckpt_size 2500 --init_lr 1e-4 --pooling att --max_ckpt 80 --additional_outname AST_64 --gradient_accumulation 1  --optimizer adam --scheduler reduce --multi_gpu True --beta1 0.95 --beta2 0.999 --weight_decay 5e-7 --normalize_scale 4