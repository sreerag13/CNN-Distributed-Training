
from dependencies import *
fig = plt.figure()
st.title('Distributed Training of Convolutional Neural Networks')

def convert_to_tensor_batch(files):
    # Initialize an empty list to store dictionaries (each containing tensor and label)
    tensor_label_list = []
    
    # Process each file in the batch
    for file in files:
        image = Image.open(file)
        # Resize the image to 100x100 pixels
        image = image.resize((100, 100))
        # Convert the image to a numpy array
        image_array = np.array(image)
        # Normalize the image array and convert it to a tensor
        tensor = tf.convert_to_tensor(image_array, dtype=tf.float32) / 255.0
        # Get the label from the file name (assuming file name format: "label_image.png")
        label = get_labels(file.name)
        # Create a dictionary with tensor and label
        tensor_label_dict = {"tensor": tensor.numpy().tolist(), "label": label}
        # Append the dictionary to the list
        tensor_label_list.append(tensor_label_dict)
    
    # Create a JSON object containing the list of tensor-label dictionaries
    json_object = json.dumps(tensor_label_list)
    return json_object,tensor.shape

def get_labels(name):
    filename = name
    # Generate label
    if 'cat' in filename:
        label = 0
    elif 'dog' in filename:
        label = 1
    else:
        label = None  
    return label

#@st.cache_data
def check_server_status(url,name):
    
    server_url =url

    if server_url:
        try:
            response = requests.get(server_url)
            if response.status_code == 200:
                
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            return False
    
master_status=check_server_status('http://localhost:8000','Master')
child1_status=check_server_status('http://localhost:8001','Child 1')
child2_status=check_server_status('http://localhost:8002','Child 2')

st.markdown('**Server Status Checker**')


    

if master_status and child1_status and child2_status:
    st.success(f'All servers are live!')
    resp = requests.post('http://localhost:8000/initiateModel')
    st.markdown("**Global Model Weights sent to child servers:**")

    model = load_model('model.h5')
    weights = model.get_weights()
    flattened_weights = np.concatenate([w.flatten() for w in weights])

    
        
    # Print the summary statistics
    st.write('*Global model summary statistics of Layers before training*')
    summary_stats_before = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max'],
                        'Value': [np.mean(flattened_weights), np.median(flattened_weights), np.std(flattened_weights), np.min(flattened_weights), np.max(flattened_weights)]
                        })
    st.table(summary_stats_before)
    for server, response in json.loads(resp.text).items():
        st.markdown(f"{server}: {response}")
else:
    st.error(f'Some servers are not live!')



if __name__ == "__main__":
    files1=None
    files2=None
    
    st.subheader('Image Upload Section')
    col1, col2 = st.columns(2)
    

    # Create a file uploader in each column
    files1 = col1.file_uploader("Upload Image to Server 1", type=["png","jpg","jpeg"], accept_multiple_files=True, key="uploader1")
    files2 = col2.file_uploader("Upload Image to Server 2", type=["png","jpg","jpeg"], accept_multiple_files=True, key="uploader2")
    
    

    if files1 and len(files1) >= 1:
        col1.markdown("Server 1 Uploaded Images:")
        for i, file in enumerate(files1):
            col1.image(file, caption=f"Image {i+1}",width =100, use_column_width=True)
    else:
        col1.error("Upload minimum 1 images at a time")
    if files2 and len(files2) >= 3:
        col2.markdown("Server 2 Uploaded Images:")
        for i, file in enumerate(files2):
            col2.image(file, caption=f"Image {i+1}",width =100, use_column_width=True)
    else:
        col2.error("Upload minimum 1 images at a time")

    

    
    class_btn = st.button("Process", key="process_button")
    headers = {'Content-Type': 'application/json'}
    st.subheader('Logs')
    if class_btn:
        if not master_status or not child1_status :
            st.error("All the servers are not live.Please check the servers are setup correctly")
        
        if files1 is None or files2 is None or not 1 <= len(files2) <= 3 or not 1 <= len(files1) <= 3:
            st.error("Invalid Input, please upload an image to each of the servers")
        else:
            
            if files1 and len(files1) >= 1:

                upload1Data,size=convert_to_tensor_batch(files1)
                response = requests.post('http://localhost:8001/upload', json =upload1Data, headers=headers)
                st.write(f'Response from server 1: {response.text}',f'Input image size : {size}')
            
            if files2 and  len(files2) >= 3:

                upload2Data,size=convert_to_tensor_batch(files2)
                response = requests.post('http://localhost:8002/upload', json =upload2Data, headers=headers)
                st.write(f'Response from server 2: {response.text}',f'Input image size : {size}')

            progress_placeholder = st.empty()
            progress_url = 'http://localhost:8001/progress'

            # Send a GET request to the API
            try:
                response = requests.get(progress_url)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                data = response.json()
                progress_placeholder.write("Training logs From Server 1:")
                progress_placeholder.write(data.get('progress'))
            except requests.exceptions.RequestException as e:
                print(str(e))
                #progress_placeholder.error(f"Error fetching data: {str(e)}")

            progress_placeholder2 = st.empty()
            progress_url2 = 'http://localhost:8002/progress'

            # Send a GET request to the API
            try:
                response = requests.get(progress_url2)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                data = response.json()
                progress_placeholder2.write("Training logs From Server 1:")
                progress_placeholder2.write(data.get('progress'))
            except requests.exceptions.RequestException as e:
                print(str(e))

            logs_placeholder = st.empty()
            logs_url = 'http://localhost:8000/logs'

            # Send a GET request to the API
            try:
                response = requests.get(logs_url)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                data = response.json()
                st.write("Logs From Master Server:")
                st.write('Global model summary statistics of Layers after training')
                model = load_model('model.h5')
                weights = model.get_weights()
                flattened_weights = np.concatenate([w.flatten() for w in weights])

                
                    
                summary_stats_after = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max'],
                        'Value': [np.mean(flattened_weights), np.median(flattened_weights), np.std(flattened_weights), np.min(flattened_weights), np.max(flattened_weights)]
                        })
                st.table(summary_stats_after)
            except requests.exceptions.RequestException as e:
                print(str(e))
    


    
    
