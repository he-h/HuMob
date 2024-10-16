import torch
from torch import nn
from transformers import BertModel, BertConfig
from utils import *

class SelfAttention(nn.Module):
    """
    A custom self-attention layer for sequence processing, which scales to multiple attention heads.

    Attributes:
        num_attention_heads (int): The number of separate attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads combined.
        query, key, value (nn.Linear): Linear layers for transforming input into query, key, and value vectors.
        dense (nn.Linear): A dense layer for transforming the concatenated outputs of the attention mechanism.
        layer_norm (nn.LayerNorm): Layer normalization to stabilize the neural network's training.
    """
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        """
        Helper function to transpose the dimensions of the input tensor for the multi-head attention scores computation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transposed tensor.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Defines the computation performed at every call of the self-attention layer.

        Args:
            hidden_states (torch.Tensor): The input sequences to the layer.

        Returns:
            torch.Tensor: The output of the self-attention layer.
        """
        # Apply linear layers to compute query, key, and value tensors
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Transpose the output for multi-head attention operation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores and normalize them
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Weighted sum of value vectors based on the attention probabilities
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final projection and layer normalization
        attention_output = self.dense(context_layer)
        attention_output = self.layer_norm(attention_output + hidden_states)
        return attention_output


class MobilityTransformer(nn.Module):
    """
    A simplified transformer model tailored for mobility prediction with fewer parameters than MobilityBERT.

    Attributes:
        num_location_ids (int): Number of distinct location identifiers.
        hidden_size, hidden_layers, attention_heads (int): Configuration parameters for the transformer.
        day_embedding, time_embedding, day_of_week_embedding, weekday_embedding, location_embedding (nn.Embedding): Embeddings for different input features.
        input_projection (nn.Linear): Projects concatenated embeddings to the transformer's input size.
        output_projection (nn.Linear): Projects transformer output to predict locations.
    """
    def __init__(self, num_location_ids, hidden_size, hidden_layers, attention_heads,
                 day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size,
                 max_seq_length):
        super().__init__()
        self.config = BertConfig(
            vocab_size=1,  # Not using traditional text tokenization
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attention_heads,
            intermediate_size=hidden_size * 2,  # Reduced intermediate layer size for efficiency
            max_position_embeddings=max_seq_length
        )
        self.bert = BertModel(self.config)

        # Embedding layers for various input features
        self.day_embedding = nn.Embedding(75, day_embedding_size)
        self.time_embedding = nn.Embedding(48, time_embedding_size)
        self.day_of_week_embedding = nn.Embedding(7, day_of_week_embedding_size)
        self.weekday_embedding = nn.Embedding(2, weekday_embedding_size)
        self.location_embedding = nn.Embedding(num_location_ids, location_embedding_size)

        # Project combined embeddings to hidden size for transformer input
        input_size = day_embedding_size + time_embedding_size + day_of_week_embedding_size + weekday_embedding_size + location_embedding_size + 1
        self.input_projection = nn.Linear(input_size, self.config.hidden_size)
        self.output_projection = nn.Linear(self.config.hidden_size, num_location_ids)

    def forward(self, input_seq_feature, historical_locations, predict_seq_feature):
        """
        Processes both historical and future input sequences for mobility prediction using an embedded transformer.

        Args:
            input_seq_feature (torch.Tensor): Historical input features.
            historical_locations (torch.Tensor): Locations corresponding to historical inputs.
            predict_seq_feature (torch.Tensor): Future sequence features for prediction.

        Returns:
            torch.Tensor: Predicted location probabilities.
        """
        # Extract features and embed each
        historical_days, historical_times, historical_day_of_weeks, hist_weekday, hist_delta = map(
            lambda x: input_seq_feature[:, :, x], range(5))
        future_days, future_times, future_day_of_weeks, future_weekday, future_delta = map(
            lambda x: predict_seq_feature[:, :, x], range(5))

        # Embed and concatenate features for input
        hist_embedded = torch.cat([
            self.day_embedding(historical_days),
            self.time_embedding(historical_times),
            self.day_of_week_embedding(historical_day_of_weeks),
            hist_delta.unsqueeze(-1).float(),
            self.location_embedding(historical_locations)
        ], dim=-1)
        
        future_embedded = torch.cat([
            self.day_embedding(future_days),
            self.time_embedding(future_times),
            self.day_of_week_embedding(future_day_of_weeks),
            future_delta.unsqueeze(-1).float(),
            torch.zeros_like(hist_embedded)[:, :future_days.size(1)]
        ], dim=-1)

        combined_input = torch.cat([hist_embedded, future_embedded], dim=1)
        transformer_input = self.input_projection(combined_input)

        # Feed into transformer and project output to location space
        transformer_output = self.bert(inputs_embeds=transformer_input)
        logits = self.output_projection(transformer_output.last_hidden_state[:, -future_days.size(1):])
        return logits
    
def hf_predict_location(df, start_day=60, end_day=74):
    '''
    Historical frequency (HF) predicts future locations using historical visit patterns based on time and weekday.
    '''
    # Create a column to combine x and y as a tuple for easier manipulation
    df['location'] = list(zip(df['x'], df['y']))

    # Pre-calculate most frequent locations for each filter condition
    # Condition 1: Most frequent location same weekday and time
    df['weekday'] = df['d'] % 7
    weekday_time_mode = df[df['d'] < start_day].groupby(['uid', 'weekday', 't'])['location'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Condition 2: Most frequent location same time
    time_mode = df[df['d'] < start_day].groupby(['uid', 't'])['location'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Condition 3: Most frequent location nearby time
    def get_nearby_time_mode(df, uid, time, window=1):
        subset = df[(df['uid'] == uid) & (df['d'] < start_day) & (df['t'].between(time - window, time + window)) & (df['location'] != (999, 999))]
        if subset.shape[0] > 0:
            return subset['location'].mode().iloc[0]
        return None

    # Iterating over the rows to predict
    prediction_days = df[(df['d'] >= start_day) & (df['d'] <= end_day)]

    tqdm.pandas()
    def predict_row(row):
        uid = row['uid']
        day = row['d']
        time = row['t']
        weekday = day % 7

        # Condition 1: Check most frequent location for same weekday and time
        if (uid, weekday, time) in weekday_time_mode.index:
            prediction = weekday_time_mode.loc[(uid, weekday, time)]
            if prediction is not None:
                return prediction

        # Condition 2: Check most frequent location for same time
        if (uid, time) in time_mode.index:
            prediction = time_mode.loc[(uid, time)]
            if prediction is not None:
                return prediction

        # Condition 3: Check most frequent location for nearby times
        prediction = get_nearby_time_mode(df, uid, time, window=1)
        if prediction is not None:
            return prediction

        # Condition 4: User’s overall most frequent location
        user_data = df[(df['uid'] == uid) & (df['d'] < start_day) & (df['location'] != (999, 999))]
        if not user_data.empty:
            return user_data['location'].mode().iloc[0]

        return (999, 999)

    # Apply prediction
    df.loc[prediction_days.index, 'location'] = prediction_days.progress_apply(predict_row, axis=1)
    df[['predicted_x', 'predicted_y']] = pd.DataFrame(df['location'].tolist(), index=df.index)

    # Drop auxiliary columns
    df.drop(columns=['location', 'weekday'], inplace=True)
    return df
    
class MobilityBERT(nn.Module):
    """
    A BERT-based model adapted for mobility prediction, integrating multiple time-specific and location-specific embeddings.

    Attributes:
        num_location_ids (int): The number of unique location identifiers.
        hidden_size, hidden_layers, attention_heads (int): Configuration parameters for the BERT model.
        day_embedding, time_embedding, day_of_week_embedding, weekday_embedding, location_embedding (nn.Embedding): Embeddings for different types of input features.
        dropout (float): Dropout rate for regularization.
        max_seq_length (int): Maximum length of the input sequences.
        bert (BertModel): The adapted BERT model for processing embeddings.
        input_projection (nn.Linear): Projects concatenated embeddings to the BERT input size.
        output_projection (nn.Linear): Projects BERT output to location prediction space.
        layer_norm (nn.LayerNorm): Normalizes input data before feeding into BERT.
        self_attention (SelfAttention): Custom self-attention layer applied after BERT.
        residual_fc (nn.Linear): A fully connected layer to create a residual connection around the BERT inputs.
    """
    def __init__(self, num_location_ids, hidden_size, hidden_layers, attention_heads,
                 day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size,
                 location_embedding_size, dropout,
                 max_seq_length):
        super().__init__()
        self.config = BertConfig(
            vocab_size=1,  # No vocabulary since not using traditional text inputs
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_length,
            hidden_act='gelu',
            hidden_dropout_prob=.1,
            attention_probs_dropout_prob=.1,
            initializer_range=.02,
            layer_norm_eps=1e-12
        )
        
        # Embeddings for different input features
        self.day_embedding = nn.Embedding(75, day_embedding_size)
        self.time_embedding = nn.Embedding(48, time_embedding_size)
        self.day_of_week_embedding = nn.Embedding(7, day_of_week_embedding_size)
        self.weekday_embedding = nn.Embedding(2, weekday_embedding_size)
        self.location_embedding = nn.Embedding(num_location_ids, location_embedding_size)

        # Input concatenation and projection
        input_size = day_embedding_size + time_embedding_size + day_of_week_embedding_size + location_embedding_size + weekday_embedding_size + 1
        self.input_projection = nn.Linear(input_size, self.config.hidden_size)
        self.output_projection = nn.Linear(self.config.hidden_size, num_location_ids)
        
        # Additional layers for processing and normalization
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttention(self.config.hidden_size, attention_heads)
        self.residual_fc = nn.Linear(input_size, self.config.hidden_size)

    def forward(self, input_seq_feature, historical_locations, predict_seq_feature):
        """
        The forward method processes both historical and future input features through embeddings, BERT, and a custom attention layer.

        Args:
            input_seq_feature (torch.Tensor): Input features for historical data points.
            historical_locations (torch.Tensor): Tensor of location indices for historical data.
            predict_seq_feature (torch.Tensor): Input features for prediction targets.

        Returns:
            torch.Tensor: The logits representing the likelihood of each location.
        """
        # Process input features through embeddings and concatenate them
        historical_days, historical_times, historical_day_of_weeks, hist_weekday, hist_delta = map(
            lambda x: input_seq_feature[:, :, x], range(5))
        future_days, future_times, future_day_of_weeks, future_weekday, future_delta = map(
            lambda x: predict_seq_feature[:, :, x], range(5))
        
        # Embedding layers for different features
        hist_day_emb = self.day_embedding(historical_days)
        hist_time_emb = self.time_embedding(historical_times)
        hist_dow_emb = self.day_of_week_embedding(historical_day_of_weeks)
        hist_weekday_emb = self.weekday_embedding(hist_weekday)
        hist_delta = hist_delta.unsqueeze(-1).float()  # Delta feature for time interval
        hist_loc_emb = self.location_embedding(historical_locations)
        
        # Normalize concatenated historical inputs
        historical_input = self.layer_norm(torch.cat([hist_day_emb, hist_time_emb, hist_dow_emb, hist_weekday_emb, hist_delta, hist_loc_emb], dim=-1))
        
        # Similar process for future inputs, using zeros for location embeddings (indicating predictions)
        future_input = self.layer_norm(torch.cat([
            self.day_embedding(future_days),
            self.time_embedding(future_times),
            self.day_of_week_embedding(future_day_of_weeks),
            self.weekday_embedding(future_weekday),
            future_delta.unsqueeze(-1).float(),
            torch.zeros_like(hist_loc_emb)[:, :future_days.size(1)]], dim=-1))
        
        # Combine and process through BERT and self-attention
        combined_input = torch.cat([historical_input, future_input], dim=1)
        residual = self.residual_fc(combined_input)
        projected_input = self.input_projection(combined_input) + residual
        projected_input = self.dropout(projected_input)
        attention_output = self.self_attention(projected_input)
        outputs = self.bert(inputs_embeds=attention_output)
        
        # Output projection for location prediction
        logits = self.output_projection(outputs.last_hidden_state[:, -future_days.size(1):])
        return logits
    
    
class ExpertLayer(nn.Module):
    """
    Defines a single expert layer as part of a mixture of experts, using GELU activation and linear transformation.

    Attributes:
        fc (nn.Linear): Fully connected layer that transforms input to output size.
        activation (nn.GELU): Gaussian Error Linear Unit activation function.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass of the expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output tensor after the linear transformation.
        """
        return self.activation(self.fc(x))

class MixtureOfExperts(nn.Module):
    """
    Implements a mixture of experts layer, where each input is dynamically routed to multiple expert layers based on learned gating.

    Attributes:
        experts (nn.ModuleList): A list of expert layers.
        gate (nn.Linear): Gating mechanism to determine the contribution of each expert to the output.
    """
    def __init__(self, num_experts, input_size, output_size):
        super().__init__()
        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        """
        Forward pass through the mixture of experts. Computes a weighted sum of expert outputs based on gating.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The combined output of the experts, weighted by the gating mechanism.
        """
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.view(batch_size * seq_len, -1)  # Flatten input for processing by experts

        # Process input through all experts
        expert_outputs = torch.stack([expert(x_reshaped) for expert in self.experts], dim=1)
        
        # Compute gating probabilities and apply to expert outputs
        gate_logits = self.gate(x_reshaped)
        gate_probs = nn.functional.softmax(gate_logits, dim=1)
        output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)

        output = output.view(batch_size, seq_len, -1)  # Reshape output to match input dimensions
        return output


class MobilityBERTMoE(nn.Module):
    """
    A BERT-based model that integrates a mixture of experts for predicting mobility data. This model is designed to handle sequence prediction tasks by incorporating expert layers for more dynamic output adjustments based on input features.

    Attributes:
        num_location_ids (int): Number of distinct location identifiers.
        hidden_size, hidden_layers, attention_heads (int): Configurations for the underlying BERT model.
        day_embedding, time_embedding, day_of_week_embedding, weekday_embedding, location_embedding (nn.Embedding): Embeddings for various input features.
        dropout (float): Dropout rate for regularization.
        max_seq_length (int): Maximum length of the input sequences.
        num_experts (int): Number of expert layers in the mixture of experts.
        bert (BertModel): Adapted BERT model used for deep contextual embeddings.
        input_projection (nn.Linear): Linear layer to project concatenated inputs into an appropriate size for BERT.
        moe (MixtureOfExperts): Mixture of experts layer to refine output based on learned input features.
        output_projection (nn.Linear): Final projection to output space, predicting location IDs.
        layer_norm (nn.LayerNorm): Normalizes inputs to improve the training stability.
        self_attention (SelfAttention): Additional self-attention mechanism applied to inputs.
        residual_fc (nn.Linear): Fully connected layer for creating a residual connection.
    """
    def __init__(self, num_location_ids=40000, hidden_size=256, hidden_layers=24, attention_heads=16,
                 day_embedding_size=64, time_embedding_size=64, day_of_week_embedding_size=64, weekday_embedding_size=32,
                 location_embedding_size=256, dropout=0.2, max_seq_length=7*48 + 48, num_experts=8):
        super().__init__()
        self.config = BertConfig(
            vocab_size=1,
            max_position_embeddings=max_seq_length,
        )

        self.bert = BertModel(self.config)


        self.day_embedding = nn.Embedding(75, day_embedding_size)
        self.time_embedding = nn.Embedding(48, time_embedding_size)
        self.day_of_week_embedding = nn.Embedding(7, day_of_week_embedding_size)
        self.weekday_embedding = nn.Embedding(2, weekday_embedding_size)
        self.location_embedding = nn.Embedding(num_location_ids, location_embedding_size)

        input_size = day_embedding_size + time_embedding_size + day_of_week_embedding_size + location_embedding_size + weekday_embedding_size + 1
        self.input_projection = nn.Linear(input_size, self.config.hidden_size)
        
        self.moe = MixtureOfExperts(num_experts, self.config.hidden_size, self.config.hidden_size)
        self.output_projection = nn.Linear(self.config.hidden_size, num_location_ids)

        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttention(self.config.hidden_size, attention_heads)
        self.residual_fc = nn.Linear(input_size, self.config.hidden_size)

    def forward(self, input_seq_feature, historical_locations, predict_seq_feature):
        """
        Processes input sequences, embedding them, and routing through a BERT model integrated with a Mixture of Experts for dynamic output customization.

        Args:
            input_seq_feature (torch.Tensor): Feature tensor for historical input sequences.
            historical_locations (torch.Tensor): Locations corresponding to historical sequences.
            predict_seq_feature (torch.Tensor): Feature tensor for future prediction sequences.

        Returns:
            torch.Tensor: Logits predicting future locations based on the input sequences.
        """
        historical_days, historical_times, historical_day_of_weeks, hist_weekday, hist_delta = input_seq_feature[:, :, 0], input_seq_feature[:, :, 1], input_seq_feature[:, :, 3], input_seq_feature[:, :, 4], input_seq_feature[:, :, 5]
        future_days, future_times, future_day_of_weeks, future_weekday, future_delta = predict_seq_feature[:, :, 0], predict_seq_feature[:, :, 1], predict_seq_feature[:, :, 3], predict_seq_feature[:, :, 4], predict_seq_feature[:, :, 5]
        
        hist_day_emb = self.day_embedding(historical_days)
        hist_time_emb = self.time_embedding(historical_times)
        hist_dow_emb = self.day_of_week_embedding(historical_day_of_weeks)
        hist_weekday_emb = self.weekday_embedding(hist_weekday)
        hist_delta = hist_delta.unsqueeze(-1).float()
        hist_loc_emb = self.location_embedding(historical_locations)
        historical_input = self.layer_norm(torch.cat([hist_day_emb, hist_time_emb, hist_dow_emb, hist_weekday_emb, hist_delta, hist_loc_emb], dim=-1))
        
        future_day_emb = self.day_embedding(future_days)
        future_time_emb = self.time_embedding(future_times)
        future_dow_emb = self.day_of_week_embedding(future_day_of_weeks)
        future_weekday_emb = self.weekday_embedding(future_weekday)
        future_delta = future_delta.unsqueeze(-1).float()
        future_seq_length = future_days.size(1)
        future_input = self.layer_norm(torch.cat([future_day_emb, future_time_emb, future_dow_emb, future_weekday_emb, future_delta, torch.zeros_like(hist_loc_emb)[:, :future_seq_length]], dim=-1))
        
        combined_input = torch.cat([historical_input, future_input], dim=1)
        residual = self.residual_fc(combined_input)
        projected_input = self.input_projection(combined_input) + residual
        projected_input = self.dropout(projected_input)
        
        attention_output = self.self_attention(projected_input)
        
        outputs = self.bert(inputs_embeds=attention_output)
        
        moe_output = self.moe(outputs.last_hidden_state[:, -future_days.size(1):])
        logits = self.output_projection(moe_output)
        return logits