import torch
import yaml
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from prepare_data import create_prompts

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate(sft_model, warp_model, sft_model_tokenizer, reward_model, reward_tokenizer, device, num_iter=3):
    sft_model_tokenizer.pad_token = sft_model_tokenizer.eos_token
    sft_model_tokenizer.padding_side = 'left'
    warp_sum = 0
    sft_sum = 0
    kl_div_sum = 0
    
    for _ in range(num_iter):
        with torch.no_grad():
            test_prompts = create_prompts(sft_model_tokenizer, config, test=True)
            inputs = sft_model_tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            warp_outputs = warp_model.generate(**inputs, max_length=25)
            sft_outputs = sft_model.generate(**inputs, max_length=25)
            
            warp_texts = [sft_model_tokenizer.decode(output, skip_special_tokens=True) for output in warp_outputs]
            sft_texts = [sft_model_tokenizer.decode(output, skip_special_tokens=True) for output in sft_outputs]
            
            warp_inputs = reward_tokenizer(warp_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            sft_inputs = reward_tokenizer(sft_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            warp_rewards = reward_model(**warp_inputs).logits.mean().item()
            sft_rewards = reward_model(**sft_inputs).logits.mean().item()
            
            warp_logits = warp_model(**inputs).logits
            sft_logits = sft_model(**inputs).logits
            kl_div = F.kl_div(F.log_softmax(warp_logits, dim=-1), F.softmax(sft_logits, dim=-1), reduction='batchmean').item()
            
            warp_sum += warp_rewards
            sft_sum += sft_rewards
            kl_div_sum += kl_div
    
    warp_rewards = warp_sum / num_iter
    sft_rewards = sft_sum / num_iter
    kl_div = kl_div_sum / num_iter
    
    print(f'warp_rewards: {warp_rewards}, sft_rewards: {sft_rewards}, kl_div: {kl_div}')
    return warp_rewards, sft_rewards, kl_div

if __name__ == '__main__':
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    warp_model = GPT2LMHeadModel.from_pretrained(config['model_paths']['output_model']).to(device)
    sft_model = GPT2LMHeadModel.from_pretrained(config['model_paths']['sft_model']).to(device)
    reward_model = DistilBertForSequenceClassification.from_pretrained(config['model_paths']['reward_model']).to(device)
    reward_tokenizer = DistilBertTokenizer.from_pretrained(config['model_paths']['reward_tokenizer'])
    sft_model_tokenizer = GPT2Tokenizer.from_pretrained(config['model_paths']['sft_model'])
    sft_model_tokenizer.pad_token = sft_model_tokenizer.eos_token
    
    evaluate(sft_model, warp_model, sft_model_tokenizer, reward_model, reward_tokenizer, device)
