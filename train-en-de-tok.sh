export CUDA_VISIBLE_DEVICES=5


python train.py -data ~/NMT/pytorch/New-NMT/data/wmt14-en-de_tok_joint_bpe32_len_256 -save_model model_wmt14-en-de_tok_joint_bpe32_len256_4096_8_8000_0.98_2/wmt14-en-de_tok_joint_bpe32_len_256_4096_8_8000_0.98_2 -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -train_steps 300000 -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 8 -optim adam -adam_beta2 0.98 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps 2000 -save_checkpoint_steps 2000 -world_size 1 -gpu_ranks 0 -keep_checkpoint 1 -position_encoding -tensorboard -tensorboard_log_dir run/wmt14-en-de_tok_joint_bpe32_len_256_4096_4_8000_0.98_2

