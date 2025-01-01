from dependencies import *


app = Flask(__name__)

progress = ""
weights=[]

@app.route('/', methods=['GET'])
def home():
    return "Server is live!", 200

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify({'progress': progress}), 200

def train_local_model(tensor,label):
    global progress
    global weights
    
    # Convert tensors and labels to numpy arrays
    X = np.array(tensor)
    y = np.array(label)
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.set_weights(weights)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    

    history = model.fit(X, y, epochs=5)
    progress = f"Finished training in Server 1 Number of epoch 5 , Accuracy: {history.history['accuracy'][-1]}, Loss :{history.history['loss'][-1]}"

    # # Get the updated weights
    updated_weights = model.get_weights()

    # Send the updated weights back to the master server
    send_data_to_master(updated_weights)



@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        tensor_label_list = json.loads(request.get_json())
        tensors = [np.array(d["tensor"]) for d in tensor_label_list]
        labels = [int(d["label"]) for d in tensor_label_list]
        
        
        # # Process the entire batch
        train_local_model(tensors, labels)
        return jsonify({'message': 'Image received'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload_model', methods=['POST'])
def get_global_model():
    global weights
    data = request.get_json()
    weights_list=json.loads(data.get('weights'))
    n_weights = [np.array(w) for w in weights_list]
    weights=n_weights
    
    return 'Child Server 1 Recieved Global Model', 200


def send_data_to_master(data):
    weights_list = [w.tolist() for w in data]
    weights_json = json.dumps(weights_list)
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:8000/receive_data', json={"weights":weights_json}, headers=headers)
    return response.text, response.status_code

if __name__ == "__main__":
    # Start the Flask app
    app.run(port=8001)
