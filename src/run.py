import os

from ml_server import app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(os.environ.get('PORT') or 5000))
