import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
total = pd.read_csv('../data/train.csv').fillna(' ')

train_df, validate_df, test_df = np.split(total.sample(frac=1, random_state=42), [int(.7*len(total)), int(.8*len(total))])

print("Dataset sizes")
print("Train: {}; Validate: {}; Test: {}".format(len(train_df), len(validate_df), len(test_df)))

def reverse_sigmoid(x):
    x = np.array(x)
    y = np.log(x/(1-x))
    return list(y)

def getdatasetstate(args):
    pass

def train(args, labeled, resume_from, ckpt_file):
    train_subset = train_df.iloc[labeled]
    
    print("Training with {} records".format(len(train_subset)))
    train_text = train_subset['comment_text']
    validate_text = validate_df['comment_text']
    test_text = test_df['comment_text']

    all_text = pd.concat([train_text, validate_text, test_text])

    print("Creating character vectorizer...")
    char_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char',
                                      stop_words='english', ngram_range=(2, 6), max_features=50000)

    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    validate_char_features = char_vectorizer.transform(validate_text)

    print("Creating word vectorizer...")
    word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', 
                                      token_pattern=r'\w{1,}', stop_words='english', 
                                      ngram_range=(1, 1), max_features=10000)

    word_vectorizer.fit(train_text)
    train_word_features = word_vectorizer.transform(train_text)
    validate_word_features = word_vectorizer.transform(validate_text)
    
    train_features = hstack([train_char_features, train_word_features])
    validate_features = hstack([validate_char_features, validate_word_features])

    print("Training Logistic Regression model")
        
    output, train_output, target_df, val_df, val_target = {}, pd.DataFrame.from_dict({}), pd.DataFrame.from_dict({}), \
                                                          pd.DataFrame.from_dict({}), pd.DataFrame.from_dict({})
    
    classifiers = {}
    for class_name in class_names:
        train_target = train_subset[class_name]
        val_target[class_name] = validate_df[class_name]
        classifiers[class_name] = LogisticRegression(C=0.1, solver='sag')

        ind = train_target.index[0]
        train_target.at[ind] = 1 if sum(train_target) == 0 else 0
        classifiers[class_name].fit(train_features, train_target)

        val_df[class_name] = classifiers[class_name].predict(validate_features)            
        train_output[class_name] = classifiers[class_name].predict(train_features)
        target_df[class_name] = train_target
        
    val_preds, val_targets = val_df.to_dict('split')['data'], val_target.to_dict('split')['data']
    print("Validation accuracy score: {}".format(accuracy_score(val_targets, val_preds)))
    
    output['predictions'] = train_output.to_dict('split')['data']
    output['labels'] = target_df.to_dict('split')['data']

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    ckpt = {"classifiers": classifiers, "word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer}
    pickle.dump(ckpt, open(os.path.join(args["EXPT_DIR"], ckpt_file), "wb"))
    
    return output

def test(args, ckpt_file):
    ckpt = pickle.load(open(os.path.join(args["EXPT_DIR"], ckpt_file), "rb"))
    test_text = test_df['comment_text']
    test_char_features = ckpt["char_vectorizer"].transform(test_text)
    test_word_features = ckpt["word_vectorizer"].transform(test_text)
    test_features = hstack([test_char_features, test_word_features])
    
    test_df_, target = pd.DataFrame.from_dict({}), pd.DataFrame.from_dict({})
    for class_name in class_names:
        target[class_name] = test_df[class_name]
        test_df_[class_name] = ckpt["classifiers"][class_name].predict(test_features)
    
    test_preds, test_targets = test_df_.to_dict('split')['data'], target.to_dict('split')['data']
    
    return {'predictions': test_preds, 'labels': test_targets}

def infer(args, unlabeled, ckpt_file):
    infer_subset = train_df.iloc[unlabeled]
    infer_text = infer_subset['comment_text']

    ckpt = pickle.load(open(os.path.join(args["EXPT_DIR"], ckpt_file), "rb"))
    
    infer_char_features = ckpt["char_vectorizer"].transform(infer_text)
    infer_word_features = ckpt["word_vectorizer"].transform(infer_text)
    infer_features = hstack([infer_char_features, infer_word_features])
    
    infer_df = pd.DataFrame.from_dict({})
    for class_name in class_names:
        infer_df[class_name] = ckpt["classifiers"][class_name].predict_proba(infer_features)[:, 1]
        
    infer_preds = infer_df.to_dict('split')['data']
    infer_preds = {k: {'logits': reverse_sigmoid(val)} for k, val in enumerate(infer_preds)}
    
    return {"outputs": infer_preds}

if __name__ == "__main__":
    # Do your testing here.
    args = {"EXPT_DIR": '.'}
    labeled = list(range(5000))
    resume_from = None
    ckpt_file = 'ckpt_0'
    unlabeled = list(range(10000, 15000))

    train(args, labeled, resume_from, ckpt_file)
    test(args, ckpt_file)
    infer(args, unlabeled, ckpt_file)