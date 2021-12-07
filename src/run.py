from ml_server import app
import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(os.environ.get('PORT') or 5000), debug=True)
