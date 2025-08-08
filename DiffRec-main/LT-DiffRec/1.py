import os

cmd = (
    "python -u main.py --cuda --dataset=ml-1m_clean "
    "--data_path=../datasets/amazon-book_clean/ --emb_path=../datasets/ "
    "--lr1=5e-4 --lr2=1e-4 --wd1=0 --wd2=0 --batch_size=400 --n_cate=2 "
    "--in_dims=[300] --out_dims=[] --lamda=0.03 --mlp_dims=[300] "
    "--emb_size=10 --mean_type=x0 --steps=5 --noise_scale=0.7 "
    "--noise_min=0.001 --noise_max=0.005 --sampling_steps=0 "
    "--reweight=1 --w_min=0.1 --w_max=1.0 --log_name=log --round=1 --gpu=0"
)

os.system(cmd)
