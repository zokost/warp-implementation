from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from prepare_data import create_prompts, PromptDataset
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def slerp(initial_model, model_list, interpolation_factor):
    initial_state_dict = initial_model.state_dict()
    for key in initial_state_dict.keys():
        initial_vector = initial_state_dict[key]
        vector_0 = model_list[0].state_dict()[key]
        vector_1 = model_list[1].state_dict()[key]

        delta_0 = vector_0 - initial_vector
        delta_1 = vector_1 - initial_vector
        angle = torch.acos((delta_0 * delta_1).sum() / (delta_0.norm() * delta_1.norm()))
        sin_angle = torch.sin(angle)

        interpolated_delta = (torch.sin((1.0 - interpolation_factor) * angle) / sin_angle) * delta_0 + (torch.sin(interpolation_factor * angle) / sin_angle) * delta_1
        initial_state_dict[key] += interpolated_delta

    initial_model.load_state_dict(initial_state_dict)
    return initial_model

def liti(theta_init, theta_slerp, nu):
    averaged_state_dict = {key: (1 - nu) * theta_init.state_dict()[key] + nu * theta_slerp.state_dict()[key]
                           for key in theta_init.state_dict().keys()}
    return GPT2LMHeadModel.from_pretrained('lvwerra/gpt2-imdb', state_dict=averaged_state_dict)

def train_warp(config):
    I = config['training_params']['iterations']
    T = config['training_params']['training_steps']
    M = config['training_params']['rl_runs']
    mu = float(config['training_params']['ema_update_rate'])
    nu = float(config['training_params']['interpolation_factor'])
    beta = float(config['training_params']['kl_coefficient'])
    batch_size = int(config['training_params']['batch_size'])
    learning_rate = float(config['training_params']['learning_rate'])
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    reward_model = DistilBertForSequenceClassification.from_pretrained(config['model_paths']['reward_model']).to(device)
    reward_tokenizer = DistilBertTokenizer.from_pretrained(config['model_paths']['reward_tokenizer'])
    
    prompts = create_prompts(reward_tokenizer, config)
    prompt_dataset = PromptDataset(prompts)
    
    sft_model = GPT2LMHeadModel.from_pretrained(config['model_paths']['sft_model']).to(device)
    sft_model_tokenizer = GPT2Tokenizer.from_pretrained(config['model_paths']['sft_model'])
    sft_model_tokenizer.pad_token = sft_model_tokenizer.eos_token
    
    theta_init = copy.deepcopy(sft_model)
    sft_model_tokenizer.padding_side = 'left'
    reward_tokenizer.padding_side = 'left'
    if sft_model_tokenizer.pad_token_id is None:
        sft_model_tokenizer.pad_token_id = sft_model_tokenizer.eos_token_id

    for i in range(I):
        theta_list = []
        for m in range(M):
            theta_m = copy.deepcopy(theta_init)
            theta_m_ema = copy.deepcopy(theta_init)
            optimizer = torch.optim.AdamW(theta_m.parameters(), lr=learning_rate)
            prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=True)
            losses = []
            for t in tqdm(range(T), desc=f'iteration: {m + i * M} of {M * I}'):
                for batch in prompt_loader:
                    optimizer.zero_grad()
                    
                    inputs = sft_model_tokenizer(batch, return_tensors="pt", truncation=True, padding=True,).to(device)
                    outputs = theta_m.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=sft_model_tokenizer.eos_token_id, max_length=25)
                    generated_texts_warp = [sft_model_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                    
                    logits = theta_m(outputs).logits
                    logits = F.log_softmax(logits, dim=-1)            
                
                    outputs = theta_m_ema.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=sft_model_tokenizer.eos_token_id, max_length=25)
                    logits_ema = theta_m_ema(outputs).logits
                    logits_ema = F.log_softmax(logits_ema, dim=-1)
                    
                    reward_inputs = reward_tokenizer(generated_texts_warp, return_tensors="pt", padding=True, truncation=True).to(device)
                    rewards = torch.sigmoid(reward_model(**reward_inputs).logits)
                    
                    kl_reward = torch.mean(logits - logits_ema)
                    kl_reward = torch.mean(rewards - beta * kl_reward)
                    
                    loss = -torch.mean(kl_reward * logits)
                    losses.append(-loss.item())
                    
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        for param, ema_param in zip(theta_m.parameters(), theta_m_ema.parameters()):
                            ema_param.data.mul_(1 - mu).add_(mu, param.data)
                
            theta_list.append(theta_m)
        theta_slerp = slerp(theta_init, theta_list, 1 / M)
        theta_init = liti(theta_init, theta_slerp, nu)
    sft_model = liti(sft_model, theta_slerp, nu)
    sft_model.save_pretrained(config['model_paths']['output_model'])
    return sft_model


if __name__ == "__main__":
    config = load_config()
    train_warp(config)
