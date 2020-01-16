# Anonymization of medical reports
 In this project, we are looking to anonymize medical diagnoses reports by identifying disease names.
We are using CRFs for Named-Entity Recognition task.
The idea is to classsify words in each sentence and detect disease names, which are considered as sensitive information.
> Example : **Testicular cancer ** and **endometriosis**  have increased in incidence during the last decades.

# Reading data

```
from format_data import read_text_labeled_sentences
import random
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)
filepath = 'data/sentences_with_roles_and_relations.txt'
disease_data = read_text_labeled_sentences(filepath, tokenizer)
sample_data_point = random.choice(disease_data)
print("Tokens:\n {}".format(sample_data_point[0]))
print("Labels:\n {}".format(sample_data_point[1]))
```

# Apply CRFs model

```
from CRF_classifier import main
random.shuffle(disease_data)
main(disease_data)
```
