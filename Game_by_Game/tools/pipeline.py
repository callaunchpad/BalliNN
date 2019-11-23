import pickle

class Model():
  def __init__(self, model, filename=None):
    if (filename is not None):
      # load the model from disk
      self.model = pickle.load(open(filename, 'rb'))
    else:
      self.model = model
  
  def train(self, features, labels):
    self.model.fit(features, labels)

  def predict(self, data):
    return self.model.predict(data)

  def save(self, filename):
    # save the model to disk
    pickle.dump(self.model, open(filename, 'wb'))

  def getScore(self, test_data, truths):
    return self.model.score(test_data, truths)

  def get_params(self, deep=True):
    return self.model.get_params(deep)

  def get_coefs(self):
    return self.model.coef_
  
  def get_intercept(self):
    return self.model.intercept_



  
