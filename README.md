Multinomial Logistic Regression for News Topic Classification 

CSC2042S assignment: a from-scratch PyTorch multinomial logistic regression model for news topic classification, trained and evaluated on three MasakhaNEWS languages English (eng), isiXhosa (xho), and chiShona (sna).

Project structure

Run the notebook from the same folder as the dataset, as received in the reop
news-dataset/eng/train.tsv and dev and test and its the same for the other languages

Each .tsv has columns category, headline, text, url. 


The system is built around a core class, MultinomialLogisticRegression(nn.Module). It 
implemented 3 functions. Forward, which returns raw logits, kept unnormalized so that 
nn.CrossEntropyLoss's internal softmax is not applied in double. Compute_probabilities, 
which applies softmax to forward's output and predict, which returns argmax over those 
probabilities as the final class prediction. 
The rest of the system uses separate functions for each part of the overall model. 
load_language loads the train, development, and test files and creates a full_text column by 
combining the headline and article body. clean_text and add_clean_column lowercase the text 
and remove punctuation. 
build_vectoriser and vectorise_splits create either a CountVectorizer or TfidfVectorizer. The 
vectoriser is fitted only on the training data, and the same fitted vocabulary is then used to 
transform the development and test sets. This prevents information from the development or 
test sets from leaking into training. 
train_model performs mini-batch stochastic gradient descent and can stop early when 
validation accuracy stops improving. evaluate calculates accuracy, micro and macro 
precision, recall and F1, as well as a full classification report. Finally, run_training combines 
these functions into one reusable training route. 
Reproducibility is enforced by calling a set_seed() function as the first line of every training 
function, ensuring identical results across repeated runs of the same configuration checked by 
rerunning a specific hyperparameter configuration twice and getting the same output. 


See the accompanying PDF report for the full explaination, and the notebook for all hyperparameter tables and training curves.
