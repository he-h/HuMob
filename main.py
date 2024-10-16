import argparse
import torch
from train import configure_model, load_model, train, prepare_data_loader, configure_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on mobility data with full parameter flexibility.')
    
    # Required arguments
    parser.add_argument('--model_name', type=str, required=True, choices=['MobilityBERT', 'MobilityBERTMoE'], help='Model to train.')
    
    # model parameters
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model layers.')
    parser.add_argument('--hidden_layers', type=int, default=12, help='Number of hidden layers in the model.')
    parser.add_argument('--attention_heads', type=int, default=16, help='Number of attention heads in the model.')
    parser.add_argument('--day_embedding_size', type=int, default=64, help='Size of the day embedding.')
    parser.add_argument('--time_embedding_size', type=int, default=64, help='Size of the time embedding.')
    parser.add_argument('--day_of_week_embedding_size', type=int, default=64, help='Size of the day of week embedding.')
    parser.add_argument('--weekday_embedding_size', type=int, default=32, help='Size of the weekday embedding.')
    parser.add_argument('--location_embedding_size', type=int, default=256, help='Size of the location embedding.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model.')
    parser.add_argument('--max_seq_length', type=int, default=75*48, help='Maximum sequence length for the model.')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate for the optimizer.')
    parser.add_argument('--location_embedding_lr', type=float, help='Specific learning rate for location embeddings, if different from the base rate.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pretrained model file.')
    
    # data parameters
    parser.add_argument('--city', type=str, default='A', help='City to train on.')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training.')
    parser.add_argument('--subsample', default=False, help='Subsample the dataset for faster training.')
    parser.add_argument('--subsample_number', type=int, default=1000, help='Number of users to subsample.')
    parser.add_argument('--input_seq_length', type=int, default=240, help='Length of the input sequence.')
    parser.add_argument('--predict_seq_length', type=int, default=30, help='Length of the prediction sequence.')
    parser.add_argument('--look_back_len', type=int, default=50, help='Length of the look back window.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device(args.device)
    model = configure_model(model_name=args.model_name,
                            num_location_ids=40000,
                            hidden_size=args.hidden_size,
                            hidden_layers=args.hidden_layers,
                            attention_heads=args.attention_heads,
                            day_embedding_size=args.day_embedding_size,
                            time_embedding_size=args.time_embedding_size,
                            day_of_week_embedding_size=args.day_of_week_embedding_size,
                            weekday_embedding_size=args.weekday_embedding_size,
                            location_embedding_size=args.location_embedding_size,
                            dropout=args.dropout,
                            max_seq_length=args.max_seq_length,
                            device=device)
    
    if args.model_path is not None:
        model = load_model(model, args.model_path, device)
    optimizer = configure_optimizer(model, args.lr, args.location_embedding_lr)
    train_loader, test_df, _ = prepare_data_loader(city=args.city,
                                       input_seq_length=args.input_seq_length,
                                       predict_seq_length=args.predict_seq_length,
                                       batch_size=args.batch_size,
                                       subsample=args.subsample,
                                       random_seed=42,
                                       subsample_number=args.subsample_number,
                                       test_size=0.1,
                                       look_back_len=args.look_back_len)

    train(model, optimizer, train_loader, args.num_epochs, device, test_df, args.input_seq_length, args.predict_seq_length)

if __name__ == "__main__":
    main()
