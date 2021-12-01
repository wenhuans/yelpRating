# yelpRating
Yelp Rating Prediction with Global Vectors for WordRepresentation (GloVe) Embedding, Bidirectional Long Short-term Memory (LSTM) and Gated Recurrent Units(GRU) Layers

Sentiment analysis of business review texts and subsequent star rating prediction
are trivial tasks for humans, yet remain more difficult for machines. 

In this project,
we analyzed a provided Yelp dataset and identified features that influence the
polarity of the reviews. Then, we treated the task of star rating prediction based on
review texts and selected features as multi-class classification problem and built
several traditional classification algorithms as baseline models. The performance
of each baseline model was analyzed using metrics including accuracy, confusion
matrix, precision and recall matrix. A Recurrent Neural Network (RNN) based
classifier with LSTM and GRU layers were designed and implemented. 

Using the same dataset, this model achieved significant improvement in all metrics when
compared with baseline models and achieved 89.4% test accuracy. The same model
structure was trained and tested using a larger dataset (1 million reviews) from
Yelp.com, and higher test accuracy was observed with increased data size.
