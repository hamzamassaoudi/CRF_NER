from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
#%%
def format_sentence(sentence, tokenizer):
    tokens = tokenizer(sentence.lstrip())
    labels =[]
    toks = []
    i=0
    while i<len(tokens):
        if tokens[i].text== "<DISONLY>":
            i+=1
            while tokens[i].text!="</DISONLY>":
                labels.append(1)
                toks.append(tokens[i])
                i+=1
            i+=1
        else:
            labels.append(0)
            toks.append(tokens[i])
            i+=1
    return toks,labels

def read_text_labeled_sentences(filepath, tokenizer):
    training_data = []
    with open(filepath, "r") as fp:
        line = fp.readline()
        while line:
            sentence, label = line.split("||")
            if label[:-1]=="DISONLY":
                tokens, labels = format_sentence(sentence, tokenizer)
                training_data.append((tokens, labels))
            line = fp.readline()
    return training_data

if __name__=="__main__":
    nlp = English()
    # Create a blank Tokenizer with just the English vocab
    tokenizer = Tokenizer(nlp.vocab)
    filepath = 'data/sentences_with_roles_and_relations.txt'
    training_data = read_text_labeled_sentences(filepath, tokenizer)
    sample_data_point = random.choice(training_data)
    print("Tokens:\n {}".format(sample_data_point[0]))
    print("Labels:\n {}".format(sample_data_point[1]))