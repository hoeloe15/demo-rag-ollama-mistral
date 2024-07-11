import sys
from pathlib import Path

# Add the parent directory to the sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.models import initialize, chain
from flask import Flask, request, jsonify

app = Flask(__name__)

print("Initializing model...")
initialize()
print("Model initialized:", chain is not None)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        if chain is None:
            print("Chain is None at request handling")
            return jsonify({"error": "Model not initialized"}), 500

        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"Received question: {question}")
        response = chain.invoke(question)
        print(f"Response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
