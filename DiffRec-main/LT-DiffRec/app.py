from flask import Flask, request, jsonify
import torch
import numpy as np
import argparse

import models.gaussian_diffusion as gd
from models.Autoencoder import AutoEncoder as AE
from models.DNN import DNN
import data_utils

app = Flask(__name__)

# --- Model Loading (similar to inference.py) ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--emb_path', type=str, default='./datasets/')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=10, help='steps for sampling/denoising')

args = parser.parse_args()

args.data_path = args.data_path + args.dataset + '/'

if args.dataset == 'yelp_clean':
    args.steps = 5
    args.noise_scale = 0.01
    args.noise_min = 0.005
    args.noise_max = 0.01
else:
    raise ValueError("Dataset not configured for Flask app")

device = torch.device("cuda:0" if args.cuda else "cpu")

# Load pre-trained models
model_path = "../checkpoints/LT-DiffRec/"
if args.dataset == "yelp_clean":
    model_name = "yelp_clean_0.0005lr1_5e-05lr2_0.0wd1_0.0wd2_bs200_cate2_in[300]_out[]_lam0.03_dims[300]_emb10_x0_steps5_scale0.01_min0.005_max0.01_sample0_reweight1_wmin0.5_wmax1.0_log.pth"
    AE_name = "yelp_clean_0.0005lr1_5e-05lr2_0.0wd1_0.0wd2_bs200_cate2_in[300]_out[]_lam0.03_dims[300]_emb10_x0_steps5_scale0.01_min0.005_max0.01_sample0_reweight1_wmin0.5_wmax1.0_log_AE.pth"

model = torch.load(model_path + model_name, map_location=device).to(device)
Autoencoder = torch.load(model_path + AE_name, map_location=device).to(device)
model.eval()
Autoencoder.eval()


if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

# Load data for n_items
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'
train_data, _, _, _, _, n_item = data_utils.data_load(train_path, valid_path, test_path, 0.1, 1.0)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id')
    top_n = data.get('top_n', 10)

    if user_id is None:
        return jsonify({'error': 'user_id is required'}), 400

    
    user_history = torch.FloatTensor(train_data[user_id].A).to(device)

    with torch.no_grad():
        _, user_latent, _ = Autoencoder.Encode(user_history)
        
        
        user_latent_recon = diffusion.p_sample(model, user_latent, args.sampling_steps, args.sampling_noise)
        
        
        prediction = Autoencoder.Decode(user_latent_recon)

    
    prediction[user_history.nonzero()] = -np.inf

    
    _, recommended_items = torch.topk(prediction, top_n)
    
    
    return jsonify({'recommended_items': recommended_items.cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(debug=True)
