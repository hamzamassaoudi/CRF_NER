
import pycrfsuite
from format_data import read_text_labeled_sentences
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import random
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

#%% Utils functions
def word2features(train_sample, i):
    token = train_sample[i]
    word = token.text
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.pos='+token.pos_,
        'word.dep='+token.dep_,
        'word.is_stop=%s' %token.is_stop,
        'word.lemma=' + token.lemma_,
        'word.tag=' + token.tag_,
        'word.shape=' + token.shape_,
        'word.is_alpha=%s' %token.is_alpha,        
    ]
    if i > 0:
        token1 = train_sample[i-1]
        word1 = token1.text
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.pos='+token1.pos_,
            '-1:word.dep='+token1.dep_,
            '-1:word.is_stop=%s' %token1.is_stop,
            '-1:word.lemma=' + token1.lemma_,
            '-1:word.tag=' + token1.tag_,
            '-1:word.shape=' + token1.shape_,
            '-1:word.is_alpha=%s' %token1.is_alpha,    
        ])
    else:
        features.append('BOS')
        
    if i < len(train_sample)-1:
        token1 = train_sample[i+1]
        word1 = token1.text
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.pos='+token1.pos_,
            '+1:word.dep='+token1.dep_,
            '+1:word.is_stop=%s' %token1.is_stop,
            '+1:word.lemma=' + token1.lemma_,
            '+1:word.tag=' + token1.tag_,
            '+1:word.shape=' + token1.shape_,
            '+1:word.is_alpha=%s' %token1.is_alpha,   
        ])
    else:
        features.append('EOS')       
    return features

def sent2features(train_sample):
    return [word2features(train_sample, i) for i in range(len(train_sample))]

def encode_labels(labels):
    return ["DISEASE" if label==1 else "O" for label in labels]
#%%
def train(X_train, y_train, model_path="models/crf_model.crfsuite"):
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.44,   # coefficient for L1 penalty
        'c2': 1e-4,  # coefficient for L2 penalty
        'max_iterations': 60,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    print("model's parameters : {}".format(trainer.params()))
    trainer.train(model_path)
    print("Last iteration log {}".format(trainer.logparser.last_iteration))

def main(disease_data):
    # Devide disease dataset into train and test sets
    training_data = disease_data[int(0.3*len(disease_data)):]
    test_data = disease_data[:int(0.3*len(disease_data))]
    # Calcule features for both training and test datasets
    X_train = [sent2features(s[0]) for s in training_data]
    y_train = [encode_labels(s[1]) for s in training_data]

    X_test = [sent2features(s[0]) for s in test_data]
    y_test = [s[1] for s in test_data]
    # Train the model
    train(X_train, y_train)

    # Predict labels for a given sentence example
    tagger = pycrfsuite.Tagger()
    tagger.open("models/crf_model.crfsuite")
    i=0
    print("sentence: {}".format(test_data[i][0]))
    print("predicted labels: {}". format(tagger.tag(X_test[i])))
    print("real labels {}".format(encode_labels(y_test[i])))
    # Calculate test metrics
    outputs = []
    for i in range(len(X_test)):
        outputs.append(tagger.tag(X_test[i]))

    targets = sum(y_test, [])
    outputs = sum(outputs, [])
    outputs = [0 if output=="O" else 1 for output in outputs]

    print("conf_matrix: \n", confusion_matrix(targets, outputs))
    print("precision score:\n", precision_score(targets, outputs))
    print("recall score:\n", recall_score(targets, outputs))
    print("F1 score:\n", f1_score(targets, outputs))

if __name__=="__main__":
    nlp = English()
    # Create a blank Tokenizer with just the English vocab
    tokenizer = Tokenizer(nlp.vocab)
    filepath = 'data/sentences_with_roles_and_relations.txt'
    disease_data = read_text_labeled_sentences(filepath, tokenizer)
    random.shuffle(disease_data)
    main(disease_data)
