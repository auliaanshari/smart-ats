import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
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

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', kt.Tuner), ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)

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

    def model_builder(hp):
        embedding_dim = hp.Choice('embedding_dim', values=[64, 128])
        conv_filters = hp.Choice('conv_filters', values=[64, 128])
        lstm_units = hp.Choice('lstm_units', values=[32]) 
        dense_units = hp.Int('dense_units', min_value=64, max_value=128, step=64)
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4])
        l2_rate = hp.Choice('l2_rate', values=[1e-4, 1e-5])

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
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['sparse_categorical_accuracy']
        )
        return model

    # Bayesian Optimization
    tuner = kt.BayesianOptimization(
        model_builder,
        objective='val_sparse_categorical_accuracy',
        max_trials=10, # Coba 10 trial cerdas
        num_initial_points=2, # 2 trial pertama acak, sisanya dianalisis secara matematis
        directory=fn_args.working_dir,
        project_name='resume_kt_bayesian'
    )

    fit_kwargs = {
        "x": train_set,
        "validation_data": val_set,
        "steps_per_epoch": fn_args.train_steps,
        "validation_steps": fn_args.eval_steps,
        "epochs": 5 
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
