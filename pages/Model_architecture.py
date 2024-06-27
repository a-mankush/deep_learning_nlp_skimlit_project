import streamlit as st

st.header("MediScan Model Architecture")
st.markdown(
    """
This model is a **multi-modal** text classification model that leverages both **word-level** and
**character-level** information, along with **contextual features** like **line number** and **total line count**,
to **predict** the **category** of a text snippet.
"""
)
st.image(image="images/model.png")

"""
### 0. Data Loading
```python
train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((
                                train_sentences, train_chars,
                                train_line_numbers_one_hot, train_total_lines_one_hot
))

train_char_token_pos_label = tf.data.Dataset.from_tensor_slices(train_labels_ohe)

train_char_token_pos_dataset = tf.data.Dataset.zip((
                train_char_token_pos_data,
                train_char_token_pos_label)
).batch(32).prefetch(tf.data.AUTOTUNE)
```
- For loading the data as efficiently as possible I use the tf.data API
- `tf.data.Dataset.from_tensor_slices`: This function creates TensorFlow datasets from in-memory tensors 
- `tf.data.Dataset.zip`: This combines the two datasets
- `batch(32)`: This batches the data into chunks of 32 elements each. Batching helps improve efficiency during training as the model can process multiple examples simultaneously.
- `prefetch(tf.data.AUTOTUNE)`: This instructs TensorFlow to pre-fetch data from the disk or memory in the background while the model is training. This can significantly improve training speed by overlapping data loading with model training computations.
"""

st.markdown(
    """
### 1. Token inputs/Model
```python
token_input = keras.Input(shape=(1,), dtype="string", name='token_input')
text_vectorizer_layer = text_vectorizer(token_input)
text_embedding = embedding(text_vectorizer_layer)
bi_lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(64))(text_embedding)
custom_token_model = keras.Model(inputs=token_input, outputs=bi_lstm_layer)
```
- **Input Layer (token_input):** This is where tokenized text data is fed into the model. The input shape is set to (1,), indicating a single string (one tokenized sentence).
- **Text Vectorization (text_vectorizer_layer):** The Keras TextVectorization layer converts the input text into a numerical format suitable for deep learning.
- **Embedding (text_embedding):** The Embedding layer transforms the numerical representations obtained from text vectorization into dense vectors with fixed dimensions. It learns relationships and meanings between words.
- **Bidirectional LSTM (bi_lstm_layer):** A Bidirectional LSTM layer processes the embedded tokens in both forward and backward directions. This helps capture context information effectively.
"""
)

"""
### 2. Character inputs/Model
```python
char_input = keras.Input(shape=(1,), dtype="string", name='char_input')
char_vectorizer_layer = char_vectorizer(char_input)
char_embeding_layer = char_embeding(char_vectorizer_layer)
bi_char_lstm = keras.layers.Bidirectional(keras.layers.LSTM(64))(char_embeding_layer)
char_model = keras.Model(char_input, bi_char_lstm)
```
- **Input Layer (char_input):** Similar to token input, this layer receives character-level data in the form of strings.
- **Char Vectorization (char_vectorizer_layer):** TextVectorization layer for characters, converting characters into numerical representations.
- **Embedding (char_embeding_layer):** Embedding layer for character-level data, transforming characters into dense vectors.
- **Bidirectional LSTM (bi_char_lstm):** Bidirectional LSTM layer processes the embedded characters bidirectionally, capturing information from both ends.
"""

"""
### 2.1 Concat Layer
```python
char_token_embedding = keras.layers.Concatenate(name="char_token_embedding")([custom_token_model.output, char_model.output])
```
- **Concatenation (char_token_embedding):** The outputs of the token and character models are concatenated. This allows the model to consider information from both token and character levels simultaneously.
"""

"""
### 3. Line Number Inputs/Model
```python
line_number_input = keras.Input(shape=(16,), name="line_number_input")
line_number_output = keras.layers.Dense(128, activation='relu')(line_number_input)
line_number_model = keras.Model(line_number_input, line_number_output)
```
- **Input Layer (line_number_input):** Represents the one-hot-encoded line number information.
- **Dense Layer (line_number_output):** A dense layer with ReLU activation processes the line number input.

"""

"""
### 4. Total Line Inputs/Model
```python
total_line_input = keras.Input(shape=(20,), name='total_line_input')
total_line_output = keras.layers.Dense(128, activation="relu")(total_line_input)
total_line_model = keras.Model(total_line_input, total_line_output)
```
- **Input Layer (total_line_input):** Represents the one-hot-encoded total lines information.
- **Dense Layer (total_line_output):** A dense layer with ReLU activation processes the total lines input.
"""

"""
### 5. Concatenating all layers
```python
concat_layer = keras.layers.Concatenate(name="concat_of_token_char_total_no")([line_number_model.output, total_line_model. output,z])
```
- **Concatenation (concat_layer):** Concatenates the outputs from the line number model, total line model, and the concatenated token and char layers.
"""

"""
### 6. Dropout and Output layers
```python
hidden_layer = keras.layers.Dense(256, activation='relu')(concat_layer)
dropout_layer_2 = keras.layers.Dropout(0.5)(hidden_layer)
outputs = keras.layers.Dense(5, activation='softmax')(dropout_layer_2)
```
- **Dense Layer (hidden_layer):** A dense layer with ReLU activation processes the concatenated information.
- **Dropout (dropout_layer_2):** Dropout layer helps prevent overfitting by randomly dropping a fraction of units during training.
- **Output Layer (outputs):** The final dense layer with softmax activation produces probabilities for each class.
"""

"""
### 7. Building the model
```python
model_8b = keras.Model(
        inputs=[
                token_input,
                char_input,
                line_number_input,
                total_line_input
                ], 
        outputs=outputs
)
```
- The **Model** class is used to define the complete MediScan model, taking token, char, line number, and total line inputs and producing the final output.
"""


col1, col2, col3 = st.columns(3)

col3.page_link("pages/dataset.py", label="Next: Dataset ➡️")

