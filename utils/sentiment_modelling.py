from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class modelling(object):
    def cnt_vct(self, X_train, X_test):
        count_vectorizer = CountVectorizer(min_df=2)

        # Fit and transform X_train
        X_train_count = count_vectorizer.fit_transform(X_train)

        # Transform X_test using the fitted count_vectorizer
        X_test_count = count_vectorizer.transform(X_test)

        return X_train_count, X_test_count, count_vectorizer
    
    def model(self, df):
        def remove_stop_words(text):
            words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
            if len(words) == 0:
                print("Empty document after stop words removal:", text)
            return ' '.join(words)
        
        X = df["text"]
        y = df["target"]

        X = X.apply(remove_stop_words)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train_count, X_test_count, cnt_vect = self.cnt_vct(X_train, X_test)

        log_reg_model = LogisticRegression()
        log_reg_model.fit(X_train_count, y_train)

        y_pred_log_reg = log_reg_model.predict(X_test_count)

        accuracy = accuracy_score(y_test, y_pred_log_reg)

        return log_reg_model, accuracy, cnt_vect
    
    def pred(self, model, new_input, cnt_vct):
        def remove_stop_words(text):
            words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
            if len(words) == 0:
                print("Empty document after stop words removal:", text)
            return ' '.join(words)
            
        new_input_processed = remove_stop_words(new_input)

        # Transform the new input using the fitted CountVectorizer
        new_input_count = cnt_vct.transform([new_input_processed])

        predicted_target = model.predict(new_input_count)

        return predicted_target
