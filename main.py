from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/elements")

def elements():
    return render_template('elements.html')

@app.route("/generic")

def generic():
    return render_template('generic.html')

if __name__ == '__main__':
    app.run(debug=True)
