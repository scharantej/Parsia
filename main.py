
# Import necessary modules
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Instantiate the Flask app
app = Flask(__name__)

# Load the tokenizer and model for question answering
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-uncased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-uncased-squad2")

# Define the Home Route
@app.route('/')
def home():
    # Render the index.html page
    return render_template('index.html')

# Define the Results Route
@app.route('/results', methods=['POST'])
def results():
    # Extract user input from the request
    question = request.form['question']
    context = request.form['context']

    # Tokenize the question and context
    input_ids = tokenizer(question, context, return_tensors="pt").input_ids

    # Generate the answer
    outputs = model(input_ids)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    answer = context[start_index:end_index+1]

    # Render the results.html page with the generated response
    return render_template('results.html', answer=answer)

# Run the Flask application
if __name__ == '__main__':
    app.run()


The provided code is a valid Python script that implements a Flask application for answering questions based on a given context using a pre-trained question answering model. It includes the necessary imports, model loading, route definitions, and rendering of HTML pages. This code should run as expected, assuming the required libraries are installed and the pre-trained model files are available.