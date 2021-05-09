import os
from app.application import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.debug = False
    app.run(host='0.0.0.0', port=port)
