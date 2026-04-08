import tensorflow as tf
import tensorflow_transform as tft
import os
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "category"
FEATURE_KEY = "resume_text"
NUM_CLASSES = 24 

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
        
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_set = input_fn(fn_args.train_files, tf_transform_output, 20) 
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 20)
    
    hparams = fn_args.hyperparameters.get('values')
    embedding_dim = hparams['embedding_dim']
    conv_filters = hparams['conv_filters']
    lstm_units = hparams['lstm_units']
    dense_units = hparams['dense_units']
    learning_rate = hparams['learning_rate']
    l2_rate = hparams.get('l2_rate', 1e-4)

    VOCAB_SIZE = 4000
    SEQUENCE_LENGTH = 300
    
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=VOCAB_SIZE,
        ngrams=2, 
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)
        
    train_texts = []
    for features, labels in train_set:
        text_batch = features[transformed_name(FEATURE_KEY)]
        for text_tensor in tf.reshape(text_batch, [-1]):
            raw_bytes = text_tensor.numpy()
            cleaned_text = raw_bytes.decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii')
            train_texts.append(cleaned_text)
            
    vectorize_layer.adapt(train_texts)
    vocab = vectorize_layer.get_vocabulary()

    model_vectorize_layer = tf.keras.layers.TextVectorization(
        vocabulary=vocab,
        standardize="lower_and_strip_punctuation",
        max_tokens=VOCAB_SIZE,
        ngrams=2, 
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = model_vectorize_layer(reshaped_narrative)
    
    x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)
    
    x = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))(x)
    
    x = tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.9)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['sparse_categorical_accuracy']
    )
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=5, restore_best_weights=True, verbose=1)
    
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es],
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=20
    )
    
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
