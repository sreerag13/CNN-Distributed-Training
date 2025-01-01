from dependencies import *


app = Flask(__name__)

data_from_servers = []
num_child_servers = 2
model = load_model('model.h5')
logs = []

@app.route('/', methods=['GET'])
def home():
    return "Server is live!", 200

@app.route('/logs', methods=['GET'])
def get_logs():
    global logs
    return jsonify({'logs': logs}), 200

@app.route('/initiateModel', methods=['POST'])
def initiateModel():
    model = load_model('model.h5')
    # Get the model weights
    weights = model.get_weights()
    weights_list = [w.tolist() for w in weights]
    weights_json = json.dumps(weights_list)
    headers = {'Content-Type': 'application/json'}

    response1 = requests.post('http://localhost:8001/upload_model', json={"weights":weights_json}, headers=headers)
    response2 = requests.post('http://localhost:8002/upload_model', json={"weights":weights_json}, headers=headers)
      
    return jsonify({"Server 1 Response":response1.text,
                    "Server 1 Status":response1.status_code,
                    "Server 2 Response":response2.text,
                    "Server 2 Status":response2.status_code,

                })


@app.route('/receive_data', methods=['POST'])
def receive_data():
    global data_from_servers
    data = request.get_json()
    weights_list=json.loads(data.get('weights'))
    n_weights = [np.array(w) for w in weights_list]

    data_from_servers.append(n_weights)

    # Add the received data to the list
    aggregate_data(data_from_servers)

    return 'New weights recieved from Child servers', 200


def aggregate_data(weights):
    global logs
    iteration=len(weights)
    average_weights = [np.mean([model_weights[i] for model_weights in weights], axis=0) 
                   for i in range(len(weights[0]))]
    model = load_model('model.h5')
    model.set_weights(average_weights)
    model.save('model.h5')
    
    logs.append(f"Logs From Master Server: Iteration {iteration}")
    

if __name__ == "__main__":
    # Start the Flask app
    app.run(port=8000)