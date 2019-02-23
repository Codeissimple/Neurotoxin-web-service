from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route('/results')
def results():
    return render_template("results.html")

@app.route('/results', methods=['POST'])
def results():
    return (request.form['demoform'])

if __name__ == '__main__':
    app.run()
