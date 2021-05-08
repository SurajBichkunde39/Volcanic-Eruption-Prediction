import sys
from app.application import app

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=int(sys.argv[1]))
