import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import time

from data_loader import train_test_generate_mob_time_series_dataloader
from model import MobilityBERT, MobilityBERTMoE
from predict import mobility_generation_evaluation

def load_model(model, model_path, device):
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    return model

def train_step(model, optimizer, criterion, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device):
    model.train()
    optimizer.zero_grad()
    
    input_seq_feature, historical_locations, predict_seq_feature, future_locations = [b.to(device) for b in [input_seq_feature, historical_locations, predict_seq_feature, future_locations]]
    logits = model(input_seq_feature, historical_locations, predict_seq_feature)
    loss = criterion(logits.view(-1, logits.size(-1)), future_locations.view(-1))
    loss.backward()
    optimizer.step()
    
    return loss.item()

def configure_optimizer(model, base_lr, location_embedding_lr):
    # Group parameters to apply different learning rates
    base_params = [p for n, p in model.named_parameters() if "location_embedding" not in n]
    location_embedding_params = [p for n, p in model.named_parameters() if "location_embedding" in n]
    if location_embedding_lr is None:
        location_embedding_lr = base_lr  # Use the base learning rate if none is provided for location embedding
    optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': location_embedding_params, 'lr': location_embedding_lr}
    ], lr=base_lr, weight_decay=0.01)
    return optimizer

def configure_model(model_name, num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length, device):
    if model_name == 'MobilityBERT':
        model = MobilityBERT(num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length)
    elif model_name == 'MobilityBERTMoE':
        model = MobilityBERTMoE(num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length)
    model.to(device)
    return model

def train(model, optimizer, train_loader, num_epochs, device, test_df, input_seq_length, predict_seq_length):
    criterion = nn.CrossEntropyLoss()
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps // 10, total_steps=num_training_steps)

    best_geo_bleu = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            loss = train_step(model, optimizer, criterion, *batch, device)
            total_loss += loss
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        geobleu_loss, dtw_loss, accuracy = test(model, test_df, device, input_seq_length, predict_seq_length)
        print(f"GEO-BLEU: {geobleu_loss:.4f}, DTW: {dtw_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        
        if geobleu_loss > best_geo_bleu:
            best_geo_bleu = geobleu_loss
            model_path = f'your_path_here/bert_{time.time()}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")

def test(model, test_df, device, input_seq_length, predict_seq_length):
    model.eval()
    with torch.no_grad():
        geobleu_loss, dtw_loss, accuracy = mobility_generation_evaluation(model, test_df, device, input_seq_length, predict_seq_length)
    
    return geobleu_loss, dtw_loss, accuracy

def prepare_data_loader(city, input_seq_length, predict_seq_length, batch_size, subsample, random_seed, subsample_number, test_size, look_back_len):
    return train_test_generate_mob_time_series_dataloader(city, input_seq_length, predict_seq_length, subsample, random_seed, subsample_number, test_size, batch_size, look_back_len)
