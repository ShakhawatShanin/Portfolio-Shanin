from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 2025))  # Use Render's PORT or default to 8080 locally
    app.run(host="0.0.0.0", port=port, debug=False)  # Disable debug for production