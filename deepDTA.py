## Importing dataset
import pandas as pd
dataset = pd.read_csv('Finalized_dataset.csv', low_memory=False)

## Setting targets
X = dataset.drop(columns=['Affinity'])
y = dataset['Affinity']

## Splitting the dataframe into Numpy Arrays
print('Splitting dataframe into numpy arrays...')
canonical_prefix = 'canonical_'
isomeric_prefix = 'isomeric_'
sequence_prefix =  'sequence_'

print('Converting strings values to numerics...')
import numpy as np
import ast

def parse_and_convert(df, prefix):
    cols = df.filter(regex=f'^{prefix}').columns
    print(f"Parsing {len(cols)} columns with prefix '{prefix}'")

    parsed_cols = []
    for col in cols:
        def safe_parse(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, (int, float)):
                return [x]
            try:
                obj = ast.literal_eval(x)
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, (int, float)):
                    return [obj]
                return []
            except Exception as e:
                print(f"Warning: failed parsing in column {col}: {x}. Error: {e}")
                return []

        parsed_col = df[col].apply(safe_parse)

        max_len = max(parsed_col.apply(len).max(), 1)

        def pad_list(lst):
            return [float(i) for i in lst] + [0.0] * (max_len - len(lst))

        padded_lists = parsed_col.apply(pad_list).tolist()
        arr = np.array(padded_lists, dtype=np.float32)
        parsed_cols.append(arr)

    combined_array = np.hstack(parsed_cols)
    return combined_array

X_canonical_parsed = parse_and_convert(X, 'canonical_')
X_isomeric_parsed = parse_and_convert(X, 'isomeric_')
X_sequence_parsed = parse_and_convert(X, 'sequence_')

print(f"Parsed canonical features shape: {X_canonical_parsed.shape}")
print(f"Parsed isomeric features shape: {X_isomeric_parsed.shape}")
print(f"Parsed sequence features shape: {X_sequence_parsed.shape}")

print('Converting inputs into float32 type...')
X_canonical = X_canonical_parsed.astype(np.float32)
X_isomeric = X_isomeric_parsed.astype(np.float32)
X_sequence = X_sequence_parsed.astype(np.float32)

X_canonical = np.array(X_canonical, dtype=np.float32)
X_isomeric = np.array(X_isomeric, dtype=np.float32)
X_sequence = np.array(X_sequence, dtype=np.float32)

print('Splitting data for train and test...')
## Splitting Data for Train and Test
from sklearn.model_selection import train_test_split

(X_canonical_train, X_canonical_test,
    X_isomeric_train, X_isomeric_test,
    X_sequence_train, X_sequence_test,
    y_train, y_test) = train_test_split(X_canonical, X_isomeric, X_sequence,y, test_size=0.2, random_state=42)

print('Building Model...')
## Defining the Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def multi_input_model(input_shapes):
    canonical_shape, isomeric_shape, sequence_shape = input_shapes

    input_canonical = Input(shape=(canonical_shape,), name='canonical_input')
    input_isomeric = Input(shape=(isomeric_shape,), name='isomeric_input')
    input_sequence = Input(shape=(sequence_shape,), name='sequence_input')

    def branch(input_layer):
        x = Dense(256, activation='relu')(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        return x
    
    branch_canonical = branch(input_canonical)
    branch_isomeric = branch(input_isomeric)
    branch_sequence = branch(input_sequence)

    concatenated = Concatenate()([branch_canonical, branch_isomeric, branch_sequence])

    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[input_canonical,input_isomeric,input_sequence], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = multi_input_model(
    (X_canonical.shape[1], X_isomeric.shape[1], X_sequence.shape[1])
)

print(model.summary())

## Fitting and Evaluating
history = model.fit([X_canonical_train,X_isomeric_train,X_sequence_train], y_train, validation_split=0.1,epochs=20,batch_size=20,verbose=1)

test_loss, test_mae = model.evaluate(
    [X_canonical_test, X_isomeric_test, X_sequence_test],
    y_test
)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")