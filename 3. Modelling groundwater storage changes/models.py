from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Concatenate, UpSampling2D, Lambda, Conv3D, MaxPooling3D, UpSampling3D, SpatialDropout3D

import tensorflow as tf

def Unet2D(input_shape, dropout_rate=0.1):
    """
    Arguments:
      input_shape
    
      Return:
        model
    """
    inputs = Input(input_shape)
    # Derive a binary validity mask from channel 0 (assumes -1 marks ocean).
    mask = Lambda(lambda t: tf.cast(tf.not_equal(t[..., 0:1], -1.0), t.dtype),
            name='mask')(inputs)
    x = Concatenate(name='inputs_with_mask')([inputs, mask])

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    if dropout_rate: c1 = Dropout(dropout_rate, name='drop_c1')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # Down to 12x12
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    if dropout_rate: c2 = Dropout(dropout_rate, name='drop_c2')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # Down to 6x6
    
    # Bottleneck
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    if dropout_rate: c3 = Dropout(dropout_rate, name='drop_c32d')(c3)

    # Decoder
    u4 = UpSampling2D((2, 2))(c3)  # Back to 12x12
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D((2, 2))(c4)  # Back to 24x24
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(c5)
    
    # Output layer for regression
    outputs = Conv2D(1, (1, 1), activation='linear')(c5)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def UNet3D(
    input_shape,
    base=16,
    dropout_rate=0.1,
    spatial_dropout=False
):
    Drop = SpatialDropout3D if spatial_dropout else Dropout

    inputs = Input(input_shape)

    # ---- Encoder ----
    c1 = Conv3D(base, (3,3,3), padding='same', activation='relu')(inputs)
    c1 = Conv3D(base, (3,3,3), padding='same', activation='relu')(c1)
    if dropout_rate: c1 = Drop(dropout_rate, name='drop_c1')(c1)
    p1 = MaxPooling3D(pool_size=(1,2,2))(c1)

    c2 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(p1)
    c2 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(c2)
    if dropout_rate: c2 = Drop(dropout_rate, name='drop_c2')(c2)
    p2 = MaxPooling3D(pool_size=(1,2,2))(c2)

    # ---- Bottleneck ----
    b  = Conv3D(base*4, (3,3,3), padding='same', activation='relu')(p2)
    b  = Conv3D(base*4, (3,3,3), padding='same', activation='relu')(b)
    if dropout_rate: b = Drop(dropout_rate, name='drop_b')(b)

    # ---- Decoder ----
    u4 = UpSampling3D((1,2,2))(b)
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)
    u4 = Concatenate()([u4, c2])
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)

    u5 = UpSampling3D((1,2,2))(u4)
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)
    u5 = Concatenate()([u5, c1])
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)

    outputs = Conv3D(1, (1,1,1), padding='same', activation='linear')(u5)
    output = Lambda(lambda t: t[:, -1], name='last')(outputs)

    return Model(inputs, output, name='UNet3D_reg')



def _resize_bool_mask(mask):
    """Reshape a [H,W] bool mask to [1,H,W,1] for broadcasting over (batch, H, W, C)."""
    mask = tf.convert_to_tensor(mask)
    if mask.shape.rank == 2:
        mask = mask[..., tf.newaxis]     # [H,W,1]
    if mask.shape.rank == 3:
        mask = mask[tf.newaxis, ...]     # [1,H,W,1]
    return mask


def make_masked_mse_loss(fixed_mask_hw):
    """Return a loss(y_true, y_pred) that averages MSE over True pixels only."""
    mask = _resize_bool_mask(fixed_mask_hw)        # [1,H,W,1], bool
    mask_f = tf.cast(mask, tf.float32)

    @tf.function
    def loss_fn(y_true, y_pred):
        per_pix = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1, keepdims=True)  # [B,H,W,1]
        per_pix = tf.where(mask, per_pix, 0.0)
        denom = tf.reduce_sum(mask_f)  # number of valid pixels (same every step)
        return tf.reduce_sum(per_pix) / tf.maximum(denom, 1e-12)
    return loss_fn


def masked_r2(y_true, y_pred):
    """R² over valid (non-fill) cells only, for the masked unet losses.

    The masked losses zero out the ``FILL_VALUE`` cells (ocean/nodata, -1) so those
    cells carry no training signal; this mirrors that by excluding them from R².
    """
    fill_value = -1.0
    mask = tf.cast(tf.not_equal(y_true, fill_value), y_true.dtype)
    n = tf.reduce_sum(mask)
    y_mean = tf.reduce_sum(y_true * mask) / (n + tf.keras.backend.epsilon())
    sst = tf.reduce_sum(mask * tf.square(y_true - y_mean))
    ssr = tf.reduce_sum(mask * tf.square(y_true - y_pred))
    return 1.0 - ssr / (sst + tf.keras.backend.epsilon())


def make_masked_mse_with_trend(
    fixed_mask_hw,
    w_main=1.0,     # weight of standard masked MSE
    w_trend=0.3,    # weight of low-frequency (trend) loss
    w_mean=0.1,     # weight of domain mean loss
    ksize=9         # low-pass kernel size (odd, e.g., 7/9/11)
):
    """
    Returns loss(y_true, y_pred) = w_main*MSE_masked + w_trend*MSE_lowpass_masked + w_mean*MSE_mean_masked.
    - fixed_mask_hw: [H, W] or [H, W, 1] bool array indicating valid pixels.
    - Low-pass uses normalized average pooling to respect missing data.
    """

    mask = _resize_bool_mask(fixed_mask_hw)            # [1, H, W, 1], bool
    mask_f = tf.cast(mask, tf.float32)                 # same shape
    denom_spatial = tf.reduce_sum(mask_f)              # scalar (constant across steps)

    k = int(ksize)
    if k % 2 == 0:
        raise ValueError("ksize must be odd")

    def lowpass_normalized(x):
        """
        x: [B, H, W, 1]
        Returns normalized avg_pool2d(x * mask) / avg_pool2d(mask), avoiding bias near gaps.
        """
        num = tf.nn.avg_pool2d(x * mask_f, ksize=k, strides=1, padding='SAME')
        den = tf.nn.avg_pool2d(mask_f,     ksize=k, strides=1, padding='SAME')
        return num / tf.maximum(den, 1e-6)

    @tf.function
    def loss_fn(y_true, y_pred):
        # ---- 1) Standard masked per-pixel MSE ----
        per_pix = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1, keepdims=True)  # [B,H,W,1]
        per_pix = per_pix * mask_f
        main_loss = tf.reduce_sum(per_pix) / tf.maximum(denom_spatial, 1e-12)

        # ---- 2) Low-frequency (trend) masked MSE ----
        y_true_lp = lowpass_normalized(y_true)   # [B,H,W,1]
        y_pred_lp = lowpass_normalized(y_pred)
        lp_err = tf.math.squared_difference(y_pred_lp, y_true_lp) * mask_f
        trend_loss = tf.reduce_sum(lp_err) / tf.maximum(denom_spatial, 1e-12)

        # ---- 3) Masked spatial-mean MSE ----
        # compute per-batch means over valid pixels
        sum_true = tf.reduce_sum(y_true * mask_f, axis=[1, 2, 3], keepdims=True)  # [B,1,1,1]
        sum_pred = tf.reduce_sum(y_pred * mask_f, axis=[1, 2, 3], keepdims=True)
        mean_true = sum_true / tf.maximum(denom_spatial, 1e-12)
        mean_pred = sum_pred / tf.maximum(denom_spatial, 1e-12)
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))  # average over batch

        return w_main * main_loss + w_trend * trend_loss + w_mean * mean_loss

    return loss_fn
