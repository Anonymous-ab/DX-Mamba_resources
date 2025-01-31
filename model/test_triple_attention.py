import torch
import torch.nn as nn

class AlignmentLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pooled_size, num_heads):
        super(AlignmentLayer, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.pool = nn.AdaptiveAvgPool1d(pooled_size)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

    def forward(self, image_features, text_indices):
        # Step 1: Use the embedding layer for text features
        text_embeddings = self.embedding_layer(text_indices)  # Shape: (2, 240, 768)

        # Step 2: Apply Adaptive Average Pooling to reduce to [2, 49, 768]
        pooled_text_features = self.pool(text_embeddings.permute(0, 2, 1))  # Shape: (2, 768, 240)
        pooled_text_features = pooled_text_features.permute(0, 2, 1)  # Back to [2, 49, 768]

        # Step 3: Prepare for multi-head attention
        # We need to transpose the dimensions to [seq_len, batch, embed_dim]
        image_features = image_features.permute(1, 0, 2)  # Shape: (49, 2, 768)
        pooled_text_features = pooled_text_features.permute(1, 0, 2)  # Shape: (49, 2, 768)

        # Step 4: Apply multi-head attention
        attn_output, attn_weights = self.multihead_attention(image_features, pooled_text_features, pooled_text_features)

        # Step 5: Transpose back to [batch, seq_len, embed_dim]
        aligned_features = attn_output.permute(1, 0, 2)  # Shape: (2, 49, 768)

        return aligned_features

# Example usage
# Step 1: Generate example tensors
image_features = torch.randn(2, 49, 768)  # Features from images
text_indices = torch.randint(0, 240, (2, 240))  # Integer indices for text features

# Step 2: Create an instance of the alignment layer
num_embeddings = 240
embedding_dim = 768
pooled_size = 49
num_heads = 8  # Number of attention heads

alignment_layer = AlignmentLayer(num_embeddings, embedding_dim, pooled_size, num_heads)

# Step 3: Process the features
aligned_output = alignment_layer(image_features, text_indices)

print("Aligned output shape:", aligned_output.shape)  # Should print: torch.Size([2, 49, 768])
