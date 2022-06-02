import textstat, nltk, wikipedia, neattext.functions as nfx
from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = '7e4142ec382bec8b7728491b3853bd7f'
app.config['UPLOAD_FOLDER'] = 'uploads/'
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words("english"))
abstractive_pipeline = pipeline("summarization")


def text_cleaner(text):
    newstr = text.lower()
    newstr = nfx.clean_text(newstr)
    return newstr.strip()


def remove_puncts(text):
    punctuations = """!()-[]{};:'"\,<>/?@#$%^&*_~="""
    string = ''
    for char in text:
        if char not in punctuations:
            string += char
    return string


def extractive_summarization(text):
    result = ""
    if text != '':
        words = word_tokenize(text)
        freqTable = dict()
        for word in words:
            word = word.lower()
            if word in stopwords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        average = int(sumValues / len(sentenceValue))
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.1 * average)):
                result += sentence + "\n"
    return result


def abstract_summarization(text):
    text = text_cleaner(text)
    abstractive_summary = abstractive_pipeline(text)
    return abstractive_summary


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/textsum', methods=['GET', 'POST'])
def textsum():
    if request.method == 'POST':
        text = request.form['ta']
    else:
        text = ''
    text = str(text)
    n = len(word_tokenize(text))
    extractive_summary = extractive_summarization(text)
    abstractive_summary = abstract_summarization(text)[0]['summary_text']
    return render_template('about.html', original_length=n, ari_original=textstat.automated_readability_index(text),
                           ari_ext=textstat.automated_readability_index(extractive_summary),
                           ari_abs=textstat.automated_readability_index(abstractive_summary),
                           original_sent_len=len(sent_tokenize(text)),
                           ext_sent_len=len(sent_tokenize(extractive_summary)),
                           abs_sent_len=len(sent_tokenize(abstractive_summary)),
                           ext_word_len=len(word_tokenize(extractive_summary)),
                           abs_word_len=len(word_tokenize(abstractive_summary)), extractive_summary=extractive_summary,
                           abstractive_summary=abstractive_summary)


@app.route('/docsum', methods=['GET', 'POST'])
def docsum():
    title = ''
    text = ''
    extractive_summary, abstractive_summary = '', ''
    if request.method == 'POST':
        url = request.form['wikiped']
        url = url.strip()
        possibilities = wikipedia.search(url)
        li = []
        for p in possibilities:
            li.append(p.lower())
        extractive_summary, abstract_summary = '', ''
        if url in li:
            page = wikipedia.page(url)
            title = page.title
            text = text_cleaner(page.content)
            text = remove_puncts(text)
            extractive_summary = extractive_summarization(text)
            extractive_summary = title + "\n " + extractive_summary
            abstractive_summary = page.summary
        else:
            url = li[0]
            page = wikipedia.page(url)
            title = page.title
            text = text_cleaner(page.content)
            text = remove_puncts(text)
            extractive_summary = extractive_summarization(text)
            extractive_summary = title + "\n " + extractive_summary
            abstractive_summary = page.summary

    return render_template('portfolio.html', title=title, original_length=len(word_tokenize(text)),
                           ari_original=textstat.automated_readability_index(text),
                           ari_ext=textstat.automated_readability_index(extractive_summary),
                           ari_abs=textstat.automated_readability_index(abstractive_summary),
                           original_sent_len=len(sent_tokenize(text)),
                           ext_sent_len=len(sent_tokenize(extractive_summary)),
                           abs_sent_len=len(sent_tokenize(abstractive_summary)),
                           ext_word_len=len(word_tokenize(extractive_summary)),
                           abs_word_len=len(word_tokenize(abstractive_summary)), extractive_summary=extractive_summary,
                           abstractive_summary=abstractive_summary)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
