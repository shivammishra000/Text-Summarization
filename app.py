from flask import Flask, render_template, request
from textSummarizer.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    input_text = ""
    
    if request.method == "POST":
        input_text = request.form["input_text"]
        obj = PredictionPipeline()
        summary = obj.predict(input_text)
    
    return render_template("index.html", summary=summary, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
