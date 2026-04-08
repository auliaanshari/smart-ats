import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "category"
FEATURE_KEY = "resume_text"

def transformed_name(key):
    """Menambahkan suffix _xf pada fitur yang sudah ditransformasi"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Melakukan preprocessing pada raw features.
    """
    outputs = {}
    
    # 1. Mengubah teks resume menjadi lowercase
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    # 2. Mengubah label string (misal: "HR", "IT") menjadi index integer (0, 1, 2...)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
