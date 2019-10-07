import logging

from flask import Flask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def index():
    return 'Flask Machine Learning Service is running!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)

