from flask import Flask, render_template, request
from gensim.models.word2vec import Word2Vec

app = Flask(__name__)
loaded_model = Word2Vec.load('./iwjoeModel')

@app.route("/", methods=['GET', 'POST'])
def Similarity():
    similarity = None

    if request.method == 'POST':
        keyword = request.form.get('keyword', '')
        vocab = loaded_model.wv
        similarity = vocab.similarity('프로그램', keyword)


    return render_template('index.html', similarity=similarity)

if __name__ == '__main__':
    app.run(debug=True)