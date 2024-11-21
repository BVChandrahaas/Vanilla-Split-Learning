# Vanilla-Split-Learning
A small implementation guide on how simple vanilla configuration for split learning works using Auto-Encoders.
Split Learning is a concept of Distributed Machine Learning created by MIT Media Labs where Client expects Data privacy and doesn't want to expose the data, sometimes the attributes too while training a machine learning model for their tasks. It can significantly increase the models performance by reducing the losses while the clients can leverage data privacy.

For more info regarding this topic you can follow the link :-  https://www.media.mit.edu/search/?q=split%20learning
and go through their publications for more AI content.

## Project Structure
<pre>
Vanilla_Split
├── Client
│   ├── client.py
│   └── Templates
│       └── index.html
├── Server
│   └── server.py
├── requirements.txt
└── README.md
</pre>
## Features

### Key Functionality

* **Image Upload**: User-friendly web interface for uploading images
* **Feature Extraction**: Client-side Convolutional Neural Network (CNN) for feature extraction
* **Classification**: Server-side dense neural network model for accurate image classification
* **Web Application**: Built with Flask for seamless user interaction

## Requirements

### Dependencies

* Flask
* TensorFlow
* Pandas
* NumPy
* Pillow

### Installation
<pre>
pip install -r requirements.txt
</pre>
## Execution Guide

### Step 1: Clone Repository

<pre>
git clone https://github.com/BVChandrahaas/Vanilla_Split.git
cd Vanilla_Split
</pre>
### Step 2: Set Up Virtual Environment (Optional)
<pre>python -m venv venv</pre>
### Activate Virtual Environment:

#### Windows: <pre>venv\Scripts\activate</pre>

#### macOS/Linux: <pre>source venv/bin/activate</pre>

### Step 3: Install Requirements
<pre>pip install -r requirements.txt</pre>

### Step 4: Start Server
<pre>cd Server
python server.py
</pre>
### Step 5: Start Client in another Terminal
<pre>cd Client
python client.py
</pre>
### Step 6: Open your web browser and navigate to http://127.0.0.1:5000.

## Usage

1. **Upload Image**: Use the "Upload" button to upload an image.
2. **Client Preprocessing**: The client extracts features from the uploaded image.
3. **Server Classification**: The server classifies the features and returns the results.
4. **View Results**: Results are displayed on the web page.

## Contributing

We welcome contributions! Feel free to:

* Fork this repository
* Submit pull requests
* Report issues

Your contributions will help improve Vanilla Split Learning.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
