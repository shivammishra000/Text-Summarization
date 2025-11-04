# Text-Summarization-Project

1. This project is an **end-to-end NLP pipeline** for **Abstractive Text Summarization**, built using    **Hugging Face Transformers**, **PyTorch**, and **FastAPI / Flask**.  
2. It can summarize long conversational or article-style text into short, meaningful summaries — similar  to how ChatGPT or Google News creates summaries.


## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py file
8. Update the app.py file

##  Setup Instructions

### 1.Clone the Repository
git clone https://github.com/shivammishra000/Text-Summarization.git
cd Text-Summarization

### 2.Create and Activate Virtual Environment
conda create -n summary python=3.10 -y
conda activate summary

### 3.Install Requirements
pip install -r requirements.txt

### 4.Run Training Pipeline
python main.py

### 5.Run the Web App
python app.py

### Open URL
http://127.0.0.1:5000

## Example Input

Hannah: Hey, do you have Betty's number?
Amanda: Lemme check.
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry, he called her last time.
Hannah: I don't know him well.
Amanda: Don't be shy, he's very nice.

### Model Details

1. Database: Hugging Face Samsum
1. Model: facebook/bart-large-cnn (or T5-base)
2. Tokenizer: Hugging Face Transformers
3. Frameworks: PyTorch, Transformers
4. Evaluation Metrics: ROUGE score

### Future Enhancements

1. Add UI for file upload summarization
2. Support multiple models dynamically (BART, T5, Pegasus)
3. Integrate fine-tuning for custom datasets
4. Add deployment support (AWS, Render, etc.)

### Author Details

Shivam Mishra
ishivammishra2003@gmail.com
LinkedIn https://www.linkedin.com/in/shivam-mishra-38322b260/
GitHub https://github.com/shivammishra000/Text-Summarization

