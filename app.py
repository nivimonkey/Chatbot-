from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    transcription = data.get('transcription', '')
    # Process the transcription as needed
    print('Transcription:', transcription)
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True)