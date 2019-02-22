from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/demo")

def demo():
    return render_template("demo.html")

@app.route('/demo', methods=['POST'])
def result():
    return 'Received !'

if __name__ == '__main__':
    app.run()
