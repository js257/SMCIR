
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mutual_info_score
import scipy
from SMCIR.models.singleTask.way.all_loss import SIMSE

# Linear projection layer: Project all modalities to the same dimension
class FeatureProjector(nn.Module):
    def __init__(self, text_dim, visual_dim, audio_dim, common_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, common_dim) if text_dim != common_dim else nn.Identity()
        self.visual_proj = nn.Linear(visual_dim, common_dim) if visual_dim != common_dim else nn.Identity()
        self.audio_proj = nn.Linear(audio_dim, common_dim) if audio_dim != common_dim else nn.Identity()

    def forward(self, text_feats, visual_feats, audio_feats):
        text_feats = self.text_proj(text_feats)  # [batch, seq_len_t, common_dim]
        visual_feats = self.visual_proj(visual_feats)  # [batch, seq_len_v, common_dim]
        audio_feats = self.audio_proj(audio_feats)  # [batch, seq_len_a, common_dim]
        return text_feats, visual_feats, audio_feats


# ---------------------------
# DMFD (Dynamic Multi-feature Fusion Detector)
# ---------------------------

# Modality dropout detection module (learnable threshold)
class DMFD(nn.Module):
    def __init__(self, args, text_dim, visual_dim, audio_dim):
        super(DMFD, self).__init__()
        self.F_v = [0] * 64
        self.F_a = [0] * 64
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.lbda = args.lbda
        # Initialize projection layer
        self.projector = FeatureProjector(text_dim, visual_dim, audio_dim, args.hidden1)

    def compute_entropy(self, batch_x, bins=30):
        """Compute entropy for each sample in the batch, returns [batch_size]"""
        batch_size = batch_x.shape[0]
        entropies = np.zeros(batch_size)

        for i in range(batch_size):
            x = batch_x[i].flatten()
            hist = np.histogram(x, bins=bins, density=True)[0]
            hist = hist[hist > 0]  # Avoid log(0)
            entropies[i] = entropy(hist) if len(hist) > 0 else 0

        return entropies

    def compute_mutual_info(self, batch_feat1, batch_feat2, bins=30):
        """
        Compute mutual information at batch level.
        batch_feat1: [batch_size, seq_len, feature_dim]
        batch_feat2: [batch_size, seq_len, feature_dim]
        Returns: [batch_size]
        """
        batch_size = batch_feat1.shape[0]
        mutual_infos = np.zeros(batch_size)

        for i in range(batch_size):
            feat1 = batch_feat1[i].flatten()
            feat2 = batch_feat2[i].flatten()

            # Histogram discretization
            hist1, _ = np.histogram(feat1, bins=bins)
            hist2, _ = np.histogram(feat2, bins=bins)

            # Compute mutual information
            mutual_infos[i] = mutual_info_score(hist1, hist2)

        return mutual_infos

    def calculate_sequence_distance(self, real_sequences, fake_sequences, feature_extractor):
        """
        Compute the distribution distance between real and generated sequences.
        :param real_sequences: Real sequences, shape: [batch, seq_len, dim]
        :param fake_sequences: Generated sequences, shape: [batch, seq_len, dim]
        :param feature_extractor: Feature extraction model
        :return: Distribution distance
        """
        # Extract features
        real_features = feature_extractor(real_sequences).detach().cpu().numpy()
        fake_features = feature_extractor(fake_sequences).detach().cpu().numpy()

        # Compute mean and covariance
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

        # Compute Fréchet distance
        diff = mu_real - mu_fake
        covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        distance = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return distance

    def compute_modal_similarity(self, batch_feat1, batch_feat2, target_seq_len=50):
        """
        Compute batch-level modality similarity.
        batch_feat1: [batch_size, seq_len1, feature_dim]
        batch_feat2: [batch_size, seq_len2, feature_dim]
        Returns: [batch_size]
        """
        batch_size = batch_feat1.shape[0]
        similarities = np.zeros(batch_size)

        for i in range(batch_size):
            feat1 = batch_feat1[i]
            feat2 = batch_feat2[i]

            # Handle different lengths using adaptive_avg_pool1d
            feat1 = torch.nn.functional.adaptive_avg_pool1d(feat1.T, target_seq_len).T
            feat2 = torch.nn.functional.adaptive_avg_pool1d(feat2.T, target_seq_len).T

            # Compute cosine similarity
            sim = cosine_similarity(feat1.cpu().detach().numpy(), feat2.cpu().detach().numpy())
            similarities[i] = np.mean(sim)

        return similarities

    def forward(self, visual_feats, audio_feats, text_feats):
        batch_size = visual_feats.size(0)
        modality_status = torch.zeros(batch_size, 3).to(visual_feats.device)  # 0: Normal, 1: Missing, 2: Highly Missing

        # Project to the same dimension
        text_feats, visual_feats, audio_feats = self.projector(text_feats, visual_feats, audio_feats)

        # Convert to NumPy for calculation
        visual_feats_np = visual_feats.cpu().detach().numpy()
        audio_feats_np = audio_feats.cpu().detach().numpy()
        text_feats_np = text_feats.cpu().detach().numpy()

        # Compute entropy (batch level)
        H_v = self.compute_entropy(visual_feats_np)  # [batch_size]
        H_a = self.compute_entropy(audio_feats_np)  # [batch_size]

        # Compute mutual information with text modality
        MI_vt = self.compute_mutual_info(visual_feats_np, text_feats_np, bins=30)  # [batch_size]
        MI_at = self.compute_mutual_info(audio_feats_np, text_feats_np, bins=30)  # [batch_size]

        # Compute modality correlation (batch level)
        S_vt = self.compute_modal_similarity(visual_feats, text_feats, target_seq_len=50)  # [batch_size]
        S_at = self.compute_modal_similarity(audio_feats, text_feats, target_seq_len=50)  # [batch_size]

        for i in range(batch_size):
            # Compute weighted information fusion score
            self.F_v[i] = self.alpha * H_v[i] + self.beta * MI_vt[i] + self.gamma * S_vt[i]
            self.F_a[i] = self.alpha * H_a[i] + self.beta * MI_at[i] + self.gamma * S_at[i]

        # Check modality status for each sample
        for i in range(batch_size):
            if S_vt[i] > 0:
                visual_dropout = False
                modality_status[i, 1] = 0
            else:
                visual_dropout = (self.F_v[i] < torch.tensor(np.mean(self.F_v) - self.lbda * (np.std(self.F_v) / (1 + self.F_v[i]))))
                modality_status[i, 1] = visual_dropout

            # Check if audio modality is missing
            if S_at[i] > 0:
                audio_dropout = False
                modality_status[i, 2] = 0
            else:
                audio_dropout = (self.F_a[i] < torch.tensor(np.mean(self.F_a) - self.lbda * (np.std(self.F_a) / (1 + self.F_a[i]))))
                modality_status[i, 2] = audio_dropout

            # Check if both modalities are missing
            if visual_dropout and audio_dropout:
                modality_status[i, 1] = 2
                modality_status[i, 2] = 2

        return modality_status


class DualPoolConcat(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        max_out = self.maxpool(x)
        avg_out = self.avgpool(x)
        out = torch.cat([max_out, avg_out], dim=1)  # [B, 2C, 1]
        return out


# ---------------------------
# MSSE (Multi-Scale Attention Enhancement Module)
# ---------------------------

class MSSE(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # Multi-scale convolution
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)

        # Fusion layer after convolution
        self.fusion = nn.Conv1d(in_channels * 3, in_channels, kernel_size=1)

        # Attention generation (SE module)
        self.attention = nn.Sequential(
                        DualPoolConcat(in_channels),                      # [B, 2C, 1]
                        nn.Flatten(),                                      # [B, 2C]
                        nn.Linear(in_channels * 2, in_channels // reduction),
                        nn.LayerNorm(in_channels // reduction),
                        nn.ReLU(),                                        # Or nn.ReLU()
                        nn.Linear(in_channels // reduction, in_channels),
                        nn.Sigmoid(),
                        nn.Unflatten(-1, (in_channels, 1))                # [B, C, 1]
                    )

    def forward(self, x):  # x: [B, C, T]
        out1 = self.conv1(x)
        out2 = self.conv3(x)
        out3 = self.conv5(x)

        out = torch.cat([out1, out2, out3], dim=1)  # [B, 3C, T]
        out = self.fusion(out)  # [B, C, T]

        # SE-like attention
        w = self.attention(out)

        # Weighted enhancement
        return x * w + x


# ---------------------------
# Generator: ComplexNovelGenerator
# ---------------------------

class ComplexNovelGenerator(nn.Module):
    def __init__(self, args, available_dim, target_dim, num_heads=1, dropout=0.1):
        """
        Parameters:
          available_dim: Feature dimension of available modalities after fusion.
          target_dim: Output target modality feature dimension.
          hidden_dim: LSTM hidden layer dimension.
          num_layers: Number of LSTM layers.
          num_heads: Number of attention heads.
          transformer_layers: Number of Transformer Encoder layers.
          dropout: Dropout probability.
          eta: Scaling factor for residual connection.
        """
        super(ComplexNovelGenerator, self).__init__()

        self.hidden_dim = args.hidden2
        self.num_layers = args.num_layers
        self.transformer_layers = args.transformer_layers
        self.eta = args.eta

        # LSTM layer to process the available modalities
        self.lstm = nn.LSTM(input_size=available_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True)

        # Transformer Encoder to further capture contextual relationships
        self.transformer_encoder = TransformerEncoder(embed_dim=self.hidden_dim * 2,
                                                      num_heads=8,
                                                      layers=self.transformer_layers,
                                                      res_dropout=dropout,
                                                      attn_mask=True)

        # Fully connected layers to map to target modality
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, target_dim)

        # Multihead attention to fine-tune feature fusion
        self.attention = nn.MultiheadAttention(embed_dim=target_dim, num_heads=num_heads, batch_first=True)
        self.activation = nn.ReLU()

        # Layer normalization and dropout for stability
        self.norm1 = nn.LayerNorm(self.hidden_dim * 2)
        self.norm2 = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-Scale Spatial Encoding (MSSE) for feature enhancement
        self.MLSE = MSSE(available_dim)

    def forward(self, x, condition):
        """
        Parameters:
          x: [seq_len_x, missing_dim] Missing modality features.
          condition: [seq_len_condition, available_dim] Available modality fusion features.

        Returns:
          [batch, target_seq_len, target_dim] Generated target modality features.
        """
        seq_len_x, _ = x.shape
        seq_len_condition, _ = condition.shape

        # Apply MSSE to the condition features for enhancement
        condition = condition.unsqueeze(0)  # [batch, seq_len_condition, available_dim]
        condition = self.MLSE(condition.permute(0, 2, 1)).permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(condition)  # [batch, seq_len_x, hidden_dim*2]
        lstm_out = self.norm1(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Transformer Encoder for global contextual modeling
        trans_out = self.transformer_encoder(lstm_out)  # [batch, seq_len_x, hidden_dim*2]

        # Residual connection to fuse LSTM and Transformer outputs
        feat = lstm_out + trans_out  # [batch, seq_len_x, hidden_dim*2]

        # Fully connected layers for mapping, followed by activation
        fc1_out = self.activation(self.fc1(feat))  # [batch, seq_len_x, hidden_dim]
        fc2_out = self.fc2(fc1_out)  # [batch, seq_len_x, target_dim]

        # Self-attention for fine-grained feature fusion
        attn_out, _ = self.attention(fc2_out, fc2_out, fc2_out)
        out = fc2_out + attn_out
        out = self.norm2(out)  # [batch, seq_len_x, target_dim]

        # If the condition sequence length does not match x, perform interpolation
        if seq_len_condition != seq_len_x:
            out = F.interpolate(out.permute(0, 2, 1), size=seq_len_x, mode='linear', align_corners=True)
            out = out.permute(0, 2, 1)  # Restore to [batch, seq_len_x, available_dim]

        # Residual connection to the original missing modality
        out = self.eta * out + x    # mosi: 0.60, mosei: 0.4
        return out  # [batch, target_seq_len, target_dim]


# ---------------------------
# Fusion Network
# ---------------------------

class FusionNetwork(nn.Module):
    def __init__(self, args, text_dim, model_dim, hidden_dim, output_dim):
        super(FusionNetwork, self).__init__()

        # Define linear layers for dimensionality reduction
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.model_fc = nn.Linear(model_dim, hidden_dim)

        # Define fusion layer using Transformer Encoder
        self.transformer_encoder = TransformerEncoder(embed_dim=hidden_dim * 2,
                                                      num_heads=8,
                                                      layers=args.transformer_layers,
                                                      res_dropout=0.1,
                                                      attn_mask=True)

        # Fully connected layer for the final output
        self.fusion_fc = nn.Linear(hidden_dim * 2, output_dim)  # Output dimension: 128

    def forward(self, text_output, model_output):
        """
        Parameters:
          text_output: Text modality output features.
          model_output: Fusion of visual and audio modality features.

        Returns:
          Output of fused modality features.
        """
        # Dimensionality reduction for both text and model outputs
        text_embedding = self.text_fc(text_output).unsqueeze(0)  # [64, 50, hidden_dim]
        model_embedding = self.model_fc(model_output).unsqueeze(0)  # [64, 500, hidden_dim]

        # Interpolate model embedding to match the time steps of text modality
        model_embedding = torch.nn.functional.interpolate(
            model_embedding.permute(0, 2, 1),  # Adjust dimensions to [batch, feature, seq]
            size=text_embedding.size(1),  # Interpolate to text modality time steps
            mode='linear'
        ).permute(0, 2, 1)  # Restore dimensions to [batch, seq, feature]

        # Concatenate text and model features
        fused_embedding = torch.cat([text_embedding, model_embedding],
                                    dim=-1)  # [64, 50, hidden_dim * 2]

        # Apply Transformer Encoder for fusion
        fused_embedding = self.transformer_encoder(fused_embedding)

        # Final fully connected layer for output
        output = self.fusion_fc(fused_embedding).squeeze(0)  # [64, 50, 128]

        return output

# ---------------------------
# CMCG (Context-aware Multi-modal Completion Generator)
# ---------------------------

class CMCG(nn.Module):
    def __init__(self, args, text_dim, visual_dim, audio_dim):
        super(CMCG, self).__init__()

        # Define input dimensions
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim

        # Instantiate the generators for visual and audio modalities
        self.visual_generator = ComplexNovelGenerator(args, self.text_dim, self.visual_dim)
        self.audio_generator = ComplexNovelGenerator(args, self.text_dim, self.audio_dim)

        # Define Fusion Networks for visual and audio modalities
        self.fusion_v = FusionNetwork(args, self.text_dim, self.audio_dim, args.hidden2, self.text_dim)
        self.fusion_a = FusionNetwork(args, self.text_dim, self.visual_dim, args.hidden2, self.text_dim)

    def calculate_loss_and_update(self, condition, miss_feature, generator):
        """
        Generates the missing features using the specified generator.

        Parameters:
          condition: [batch, seq_len, dim] Fusion of available modality features.
          miss_feature: [batch, seq_len, dim] Missing modality features.
          generator: Generator model for the missing modality.

        Returns:
          Generated missing modality features.
        """
        generated_features = generator(miss_feature, condition)  # [batch, seq_len, dim]
        return generated_features

    def forward(self, text_input, visual_input, audio_input, modality_status):
        """
        Forward pass for CMCG. Handles modality fusion and generation for missing modalities.

        Parameters:
          text_input: [batch_size, seq_len, text_dim] Text modality input.
          visual_input: [batch_size, seq_len, visual_dim] Visual modality input.
          audio_input: [batch_size, seq_len, audio_dim] Audio modality input.
          modality_status: [batch_size, 3] Status of the modalities: 0 -> available, 1 -> missing, 2 -> both missing.

        Returns:
          visual_output: [batch_size, seq_len, visual_dim] Generated or original visual modality features.
          audio_output: [batch_size, seq_len, audio_dim] Generated or original audio modality features.
        """
        batch_size = text_input.size(0)
        text_features = text_input  # The text features to guide the generation of missing modalities.

        # Initialize the generated modality outputs with zeros
        visual_output = torch.zeros_like(visual_input).to(text_input.device)
        audio_output = torch.zeros_like(audio_input).to(text_input.device)

        # Process each sample in the batch
        for i in range(batch_size):
            # Handle missing visual modality (audio available)
            if modality_status[i, 1] == 1 and modality_status[i, 2] == 0:  # Only visual modality is missing
                fusion_model = self.fusion_v(text_input[i], audio_input[i])
                # Generate visual modality using the visual generator
                visual_output[i] = self.calculate_loss_and_update(fusion_model, visual_input[i], self.visual_generator)
                audio_output[i] = audio_input[i]  # Keep original audio input

            # Handle missing audio modality (visual available)
            elif modality_status[i, 2] == 1 and modality_status[i, 1] == 0:  # Only audio modality is missing
                fusion_model = self.fusion_a(text_input[i], visual_input[i])
                audio_output[i] = self.calculate_loss_and_update(fusion_model, audio_input[i], self.audio_generator)
                visual_output[i] = visual_input[i]  # Keep original visual input

            # Handle both visual and audio modalities being missing
            elif modality_status[i, 1] == 2 and modality_status[i, 2] == 2:  # Both modalities are missing
                visual_output[i] = self.calculate_loss_and_update(text_features[i], visual_input[i], self.visual_generator)
                audio_output[i] = self.calculate_loss_and_update(text_features[i], audio_input[i], self.audio_generator)
            else:  # No modality is missing
                visual_output[i] = visual_input[i]  # Use original visual modality
                audio_output[i] = audio_input[i]  # Use original audio modality

        return visual_output, audio_output


class MultimodalFusionNetwork(nn.Module):
    def __init__(self, text_dim, visual_dim, audio_dim, hidden_dim, output_dim):
        super(MultimodalFusionNetwork, self).__init__()

        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 降维层
        self.text_fc = nn.Linear(self.text_dim, self.hidden_dim)
        self.visual_fc = nn.Linear(self.visual_dim, self.hidden_dim)
        self.audio_fc = nn.Linear(self.audio_dim, self.hidden_dim)

        # 融合层
        self.fusion_fc = nn.Linear(self.hidden_dim * 3, self.output_dim)

    def _align_temporal(self, x, target_length):
        """动态时间对齐"""
        if x.size(1) == target_length:
            return x
        return F.interpolate(
            x.permute(0, 2, 1),
            size=target_length,
            mode='linear'
        ).permute(0, 2, 1)

    def forward(self, text_output, visual_output, audio_output):

        # 降维
        text_embedding = self.text_fc(text_output)         # [B, T, hidden_dim]
        visual_embedding = self.visual_fc(visual_output)   # [B, T', hidden_dim]
        audio_embedding = self.audio_fc(audio_output)      # [B, T'', hidden_dim]

        # 时间对齐（以文本为基准）
        target_len = text_embedding.size(1)
        visual_embedding = self._align_temporal(visual_embedding, target_len)
        audio_embedding = self._align_temporal(audio_embedding, target_len)

        # 拼接特征
        fused_embedding = torch.cat([text_embedding, visual_embedding, audio_embedding], dim=-1)  # [B, T, hidden_dim*3]
        output = self.fusion_fc(fused_embedding)  # [B, T, output_dim]

        return output


class MultiModalAttentionFusion(nn.Module):
    def __init__(self, args):
        """
        Parameters:
          args: Configuration arguments containing the necessary parameters like feature dimensions and flag settings.
        """
        super(MultiModalAttentionFusion, self).__init__()

        # Flag to control which modality (visual/audio) to generate
        self.Flag_S = args.Flag_S

        # Similarity loss calculation module
        self.sim = SIMSE()
        self.sim_loss = 0.0  # Initialize similarity loss to zero

        # Feature dimensions for text, visual, and audio inputs
        self.text_in, self.audio_in, self.visual_in = args.feature_dims

        # Modality dropout detector for identifying missing modalities
        self.modality_dropout_detector = DMFD(args, self.text_in, self.visual_in, self.audio_in)

        # Modality completion module to generate missing modalities
        self.modality_completion_module = CMCG(args, self.text_in, self.visual_in, self.audio_in)

        # Multimodal Fusion Network for emotion recognition
        self.MultimodalFusionNetwork = MultimodalFusionNetwork(self.text_in, self.visual_in, self.audio_in,
                                                               args.hidden2, args.out_dim)

    def multimodal_emotion_analysis(self, text_input, visual_input, audio_input):
        """
        This function handles the emotion analysis by identifying missing modalities
        and performing modality completion.

        Parameters:
          text_input: Text modality input [batch_size, seq_len, text_dim]
          visual_input: Visual modality input [batch_size, seq_len, visual_dim]
          audio_input: Audio modality input [batch_size, seq_len, audio_dim]

        Returns:
          text_input: Text modality input (unchanged)
          visual_output: Completed or original visual modality
          audio_output: Completed or original audio modality
          modality_status: A tensor indicating the status of each modality
        """
        # Detect missing modalities with the modality dropout detector
        with torch.no_grad():
            modality_status = self.modality_dropout_detector(visual_input, audio_input, text_input)

        # Perform modality completion (if needed) based on the detected modality status
        visual_output, audio_output = self.modality_completion_module(
            text_input, visual_input, audio_input, modality_status
        )

        return text_input, visual_output, audio_output, modality_status

    def forward(self, text_embedding, visual, acoustic):
        """
        The forward pass of the model, performing the emotion recognition task with modality completion.

        Parameters:
          text_embedding: Text modality embedding [batch_size, seq_len, text_dim]
          visual: Visual modality input [batch_size, seq_len, visual_dim]
          acoustic: Audio modality input [batch_size, seq_len, audio_dim]

        Returns:
          out_put: Final fused output from the multimodal fusion network
          sim_loss: Similarity loss between generated and original modalities (if Flag_S indicates generation)
        """
        # Perform multimodal emotion analysis (modality detection and completion)
        text_output, visual_output, audio_output, modality_status = self.multimodal_emotion_analysis(text_embedding,
                                                                                                     visual, acoustic)

        # Perform multimodal fusion for emotion recognition
        out_put = self.MultimodalFusionNetwork(text_output, visual_output, audio_output)

        # Calculate similarity loss if required based on the Flag_S parameter
        if self.Flag_S == 1:  # Generate visual modality
            self.sim_loss = self.sim(visual_output, visual)
        elif self.Flag_S == 2:  # Generate audio modality
            self.sim_loss = self.sim(audio_output, acoustic)
        elif self.Flag_S == 3:  # Generate both visual and audio modalities
            self.sim_loss = self.sim(visual_output, visual)
            self.sim_loss += self.sim(audio_output, acoustic)
        else:  # No generation (only fusion)
            self.sim_loss = 0.0

        return out_put, self.sim_loss
