from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/results.html")
def results():
    return render_template("results.html")

@app.route('/results.html', methods=['POST'])
def result():
    return render_template("results.html", form36=request.form['form36'], form38=request.form['form38'], form39=request.form['form39'], form40=request.form['form40'], form42=request.form['form42'])

if __name__ == '__main__':
    app.run()
