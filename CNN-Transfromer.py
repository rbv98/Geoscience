import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.models import Model

def create_conv_transformer(num_features, d_model=64, num_heads=4, ff_dim=128, num_transformer_blocks=2):
    """
    Hybrid Conv-Transformer architecture:
    1. CNN cleans up the signal and extracts local patterns
    2. Transformer learns global feature relationships
    """
    inputs = Input(shape=(num_features, 1))
    
    # === CNN FEATURE EXTRACTION STAGE ===
    print(f"Building Conv-Transformer with {num_features} features, {d_model} model dim, {num_heads} heads")
    
    # First Conv Block - Signal cleaning
    x = Conv1D(32, kernel_size=2, padding='same', activation='relu',
               kernel_regularizer=regularizers.L2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second Conv Block - Pattern extraction
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=regularizers.L2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Project to transformer dimension
    x = Conv1D(d_model, kernel_size=1, padding='same')(x)  # 1x1 conv for dimension matching
    
    # === TRANSFORMER STAGE ===
    # Positional encoding (optional for well logs, but can help)
    positions = tf.range(num_features, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
    pos_encoding = Dense(d_model, use_bias=False)(positions)
    x = x + pos_encoding
    
    # Transformer blocks
    for i in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output)
        
        # Add & Norm
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # === OUTPUT STAGE ===
    # Custom layer for global average pooling
    class GlobalAveragePooling1D(tf.keras.layers.Layer):
        def call(self, x):
            return tf.reduce_mean(x, axis=1)
    
    # Global average pooling to aggregate feature representations
    x = GlobalAveragePooling1D()(x)  # Shape: (batch_size, d_model)
    
    # Final dense layers with regularization
    x = Dense(128, activation='relu', 
              kernel_regularizer=regularizers.L2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, activation='relu',
              kernel_regularizer=regularizers.L2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs, name='ConvTransformer')
    return model

# Create and compile the model
print("Creating Conv-Transformer model...")
conv_transformer = create_conv_transformer(
    num_features=len(features),
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2
)

conv_transformer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Display model architecture
print("\nModel Architecture:")
conv_transformer.summary()

# Train the model
print("\nTraining Conv-Transformer...")
history_ct = conv_transformer.fit(
    X_train_cnn, y_train,  # Using CNN-shaped data (batch_size, features, 1)
    validation_split=0.2,
    epochs=150, 
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

# Save the model
conv_transformer.save('conv_transformer_model.h5')

# Evaluate on test set
y_pred_ct = conv_transformer.predict(X_test_cnn).flatten()
print("\n Conv-Transformer Test Results:")
evaluate("Conv-Transformer", y_test, y_pred_ct)
