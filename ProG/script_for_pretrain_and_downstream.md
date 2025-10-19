python pre_train.py --pretrain_task Edgepred_GPPT --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0 --data_path 'prompt_graph/pretrain/cora_modified_095.npz'
——————Edgepred_GPPT [Pretrain] Epoch 116/1000 | Train Loss 0.04742 | AUC 0.9997 | AP 0.9997 | Accuracy 0.9854 | Cost Time 2.715s—————————说明训练的没啥问题啊，挺符合扰动图的。
不对，这里还没写好脚本，参数传递应该还没打通，主要是在Edgepred_GPPT中，需要传递data_path
好像写好了？等会检查一下再跑



python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model_self/Cora/Edgepred_GPPT.GCN.128hidden_dim.pth' --downstream_task LinkTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GPF-plus' --shot_num 1 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0