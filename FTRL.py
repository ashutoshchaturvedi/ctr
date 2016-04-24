from math import exp, log, sqrt
from csv import DictReader
from datetime import datetime
import sys

# Paths to data
train = 'train.csv'               # path to training file, remove .csv extension if running on Mac
test = 'test.csv'                 # path to testing file, remove .csv extension if running on Mac

# Model parameters
alpha = 0.1  # learning rate
beta = 1.0   # smoothing parameter for adaptive learning rate
L1 = 1.0     # L1 regularization
L2 = 1.0     # L2 regularization

# Feature/hash trick
D = 2 ** 24     # Use 16 MB memory

# Training/Validation
iterations = 1       # learn training data for N passes
holdafter = 29       # data after 29th (exclusive) are used as validation

class ftrl_proximal(object):
    '''
        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D        

        # model                       
        self.n = [0.0] * D   # n: squared sum of past gradients
        self.z = [0.0] * D   # z: weights as given in paper
        self.w = {}          # w: lazy weights

    def _indices(self, x):
        #first term of each instance is bias term
        yield 0

        for index in x:
            yield index        

    def predict(self, x):
       
        # parameters
        L1 = self.L1
        L2 = self.L2
        alpha = self.alpha
        beta = self.beta        

        # model
        n = self.n  # n: squared sum of past gradients
        z = self.z  # z: weights
        w = {}

        # wdotx is the inner product of w and x
        wdotx = 0.
        for i in self._indices(x):
            sign = -1.0 if z[i] < 0 else 1.0  # get sign of z[i]

            # w needs to be calculated for each feature, we calculate it dynamically using z and n as per algorithm in the paper            
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.0
            else:
                # apply L1, L2 regularization to z and w (formula given in paper)
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            #calculate dot product, x is assumed to be 1 (present) as we are running the loop only for features in x
            wdotx += w[i]

        # store the weight w to be used in update function
        self.w = w

        # sigmoid function to calculate probability of y for this row
        return 1. / (1. + exp(-max(min(wdotx, 35.0), -35.0)))

    def update(self, x, p, y):        
        
        alpha = self.alpha

        n = self.n
        z = self.z
        w = self.w
        
        g = p - y
        
        #update model parameters
        for i in self._indices(x):
            numerator = float(sqrt(n[i] + g * g) - sqrt(n[i]))
            sigma = numerator / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):   
    p = max(min(p, 1.0 - 10e-15), 10e-15)
    return -log(p) if y == 1.0 else -log(1.0 - p)


def data(path, D):
    # Function to prepare data by applying hash-trick with one-hot-encode
    for i, row in enumerate(DictReader(open(path))):        
        ID = row['id']
        del row['id'] #delete ID as unique values are not useful features
        
        y = 0.0
        if 'click' in row:
            if row['click'] == '1':
                y = 1.0
            del row['click'] #response variable is not a feature, hence deleted

        # extract date
        date = int(row['hour'][4:6])

        #We need only hour, as data belongs to same month of a given year. Date is given in format YYMMDDHH
        row['hour'] = row['hour'][6:]

        # build feature x
        x = []
        for col_name in row:
            col_value = row[col_name]

            # one-hot encoding with hash trick
            index = abs(hash(col_name + '_' + col_value)) % D
            x.append(index)

        yield i, date, ID, x, y
        

#Main Function start
start = datetime.now()

# initialize model
model = ftrl_proximal(alpha, beta, L1, L2, D)

# start training
for itr in xrange(iterations):
    loss = 0.0
    count = 0

    for i, date, ID, x, y in data(train, D):
        #    i: counter        
        #    ID: id provided in original data
        #    x: features
        #    y: label [1(click)/0(no click)]
        
        p = model.predict(x)

        if (holdafter and date > holdafter):
            # Validate for instances after "holdafter" date to get evaluate the model            
            loss += logloss(p, y)
            count += 1
        else:
            # update model with new information since it's an online algorithm
            model.update(x, p, y)

    print('Iteration %d finished, Validation logloss: %f, Time taken: %s' % (
        itr, loss/count, str(datetime.now() - start)))