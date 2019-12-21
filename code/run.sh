#!/bin/bash
i=2020
#---crcan
nohup python crcan.py --methodname crcan --save_dir ../output_disk/crcan_cora_lp_seed$i/ --dataset_name cora --dataset_fea_pkl ../data/uu/pkl/cora_fea_seed$i.pkl --out_emb_fpath ../output_disk/crcan_cora_lp_seed$i/crcan.emb --dropout 0.0 --b1 0.9 --b2 0.999 --hid1_dim 32 --hid2_dim 32   --lr 0.005 --n_epochs 200 --batch_size 3000 --recall_k 50 --net_type uu --do_nclu 0 > log/crcan_cora_lp_seed$i 2>&1 &

#---rgcn
nohup python rgcn.py --methodname rgcn --save_dir ../output_disk/rgcn_cora_lp_seed$i/ --dataset_name cora --dataset_fea_pkl ../data/uu/pkl/cora_fea_seed$i.pkl --out_emb_fpath ../output_disk/rgcn_cora_lp_seed$i/rgcn.emb --dropout 0.0 --b1 0.9 --b2 0.999 --recon_adj_hid1_dim 32 --recon_adj_hid2_dim 32 --recon_adj_hid3_dim 16 --recon_attr_hid1_dim 32 --recon_attr_hid2_dim 32 --recon_attr_hid3_dim 16 --std 0.1 --lr 0.005 --n_epochs 200 --batch_size 3000 --tao 0.001 --recall_k 50 --net_type uu --do_nclu 0 > log/rgcn_cora_lp_seed$i 2>&1 &


#---crcan-ui
nohup python crcan_ui.py --methodname crcan_ui --save_dir ../output_disk/crcan_ui_citeulike_s10_mc_fm/ --dataset_name citeulike_s10_mc_fm --dataset_fea_pkl ../data/useritem/citeulike_s10_mc_fm/input.pkl --out_emb_fpath ../output_disk/crcan_ui_citeulike_s10_mc_fm/crcan_ui.emb --dropout 0.0 --b1 0.9 --b2 0.999 --hid1_dim 32 --hid2_dim 32  --lr 0.005 --n_epochs 600 --batch_size 3000 --recall_k 10 --net_type ui --do_nclu 0 > log/crcan_ui_citeulike 2>&1 &



#---rgcn-ui
nohup python rgcn_ui.py --methodname rgcn_ui --save_dir ../output_disk/rgcn_ui_citeulike_s10_mc_fm/ --dataset_name citeulike_s10_mc_fm --dataset_fea_pkl ../data/useritem/citeulike_s10_mc_fm/input.pkl --out_emb_fpath ../output_disk/rgcn_ui_citeulike_s10_mc_fm/rgcn_ui.emb --dropout 0.0 --b1 0.9 --b2 0.999 --recon_adj_hid1_dim 32 --recon_adj_hid2_dim 32 --recon_adj_hid3_dim 32 --recon_attr_hid1_dim 32 --recon_attr_hid2_dim 32 --recon_attr_hid3_dim 32 --std 0.1 --lr 0.005 --n_epochs 600 --batch_size 3000 --tao 0.001 --recall_k 10 --net_type ui > log/rgcn_ui_citeulike_citeulike  2>&1 &