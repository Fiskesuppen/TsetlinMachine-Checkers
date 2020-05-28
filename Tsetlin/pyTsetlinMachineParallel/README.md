# pyTsetlinMachineParallel

Multi-threaded implementation of the Tsetlin Machine (https://arxiv.org/abs/1804.01508), Convolutional Tsetlin Machine (https://arxiv.org/abs/1905.09688), Regression Tsetlin Machine (https://arxiv.org/abs/1905.04206, https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165, https://link.springer.com/chapter/10.1007/978-3-030-30244-3_23), and Weighted Tsetlin Machines (https://arxiv.org/abs/1911.12607, https://arxiv.org/abs/2002.01245), with support for continuous features (https://arxiv.org/abs/1905.04199, https://link.springer.com/chapter/10.1007%2F978-3-030-22999-3_49) and multigranular clauses (https://arxiv.org/abs/1909.07310, https://link.springer.com/chapter/10.1007/978-3-030-34885-4_11).

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/Weighted_Tsetlin_Machine_Example_Configuration_Full.png">
</p>

## Installation

```bash
pip install pyTsetlinMachineParallel

export OMP_NUM_THREADS=10
```

## Documentation

Documentation coming soon at https://pytsetlinmachineparallel.readthedocs.io/en/latest/

## Tutorials

Convolutional Tsetlin Machine tutorial, https://github.com/cair/convolutional-tsetlin-machine-tutorial

## Examples

### Multiclass Demo

#### Code: NoisyXORDemo.py

```python
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np 

train_data = np.loadtxt("NoisyXORTrainingData.txt")
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("NoisyXORTestData.txt")
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))))
```

#### Output

```bash
python3 ./NoisyXORDemo.py 

Accuracy: 100.00%

Prediction: x1 = 1, x2 = 0, ... -> y = 1
Prediction: x1 = 0, x2 = 1, ... -> y = 1
Prediction: x1 = 0, x2 = 0, ... -> y = 0
Prediction: x1 = 1, x2 = 1, ... -> y = 0
```

### Interpretability Demo

#### Code: InterpretabilityDemo.py

```python
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np 

number_of_features = 20
noise = 0.1

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

tm = MultiClassTsetlinMachine(10, 15, 3.0, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("\nClass 0 Positive Clauses:\n")
for j in range(0, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.ta_action(0, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))

print("\nClass 0 Negative Clauses:\n")
for j in range(1, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.ta_action(0, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")
for j in range(0, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.ta_action(1, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))

print("\nClass 1 Negative Clauses:\n")
for j in range(1, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm.ta_action(1, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))
```

#### Output

```bash
python3 ./InterpretabilityDemo.py

Accuracy: 100.0

Class 0 Positive Clauses:

Clause #0:  ¬x0 ∧ ¬x1
Clause #2:   x0 ∧  x1
Clause #4:   x0 ∧  x1
Clause #6:  ¬x0 ∧ ¬x1
Clause #8:  ¬x0 ∧ ¬x1

Class 0 Negative Clauses:

Clause #1:   x0 ∧ ¬x1
Clause #3:   x0 ∧ ¬x1
Clause #5:   x1 ∧ ¬x0
Clause #7:   x1 ∧ ¬x0
Clause #9:   x0 ∧ ¬x1

Class 1 Positive Clauses:

Clause #0:   x1 ∧ ¬x0
Clause #2:   x1 ∧ ¬x0
Clause #4:   x0 ∧ ¬x1
Clause #6:   x0 ∧ ¬x1
Clause #8:   x0 ∧ ¬x1

Class 1 Negative Clauses:

Clause #1:   x0 ∧  x1
Clause #3:  ¬x0 ∧ ¬x1
Clause #5:  ¬x0 ∧ ¬x1
Clause #7:  ¬x0 ∧ ¬x1
Clause #9:   x0 ∧  x1
```

### 2D Convolution Demo

#### Code: 2DNoisyXORDemo.py

```python
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np 

train_data = np.loadtxt("2DNoisyXORTrainingData.txt")
X_train = train_data[:,0:-1].reshape(train_data.shape[0], 4, 4)
Y_train = train_data[:,-1]

test_data = np.loadtxt("2DNoisyXORTestData.txt")
X_test = test_data[:,0:-1].reshape(test_data.shape[0], 4, 4)
Y_test = test_data[:,-1]

ctm = MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)

ctm.fit(X_train, Y_train, epochs=5000)

print("Accuracy:", 100*(ctm.predict(X_test) == Y_test).mean())

Xi = np.array([[[0,1,1,0],
		[1,1,0,1],
		[1,0,1,1],
		[0,0,0,1]]])

print("\nInput Image:\n")
print(Xi)
print("\nPrediction: %d" % (ctm.predict(Xi)))
```

#### Output

```bash
python3 ./2DNoisyXORDemo.py 

Accuracy: 99.71%

Input Image:

[[0 1 1 0]
 [1 1 0 1]
 [1 0 1 1]
 [0 0 0 1]]

Prediction: 1
```

### Continuous Input Demo

#### Code: BreastCancerDemo.py

```python
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from pyTsetlinMachineParallel.tools import Binarizer
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

b = Binarizer(max_bits_per_feature = 10)
b.fit(X)
X_transformed = b.transform(X)

tm = MultiClassTsetlinMachine(800, 40, 5.0)

print("\nMean accuracy over 100 runs:\n")
tm_results = np.empty(0)
for i in range(100):
	X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size=0.2)

	tm.fit(X_train, Y_train, epochs=25)
	tm_results = np.append(tm_results, np.array(100*(tm.predict(X_test) == Y_test).mean()))
	print("#%d Average Accuracy: %.2f%% +/- %.2f" % (i+1, tm_results.mean(), 1.96*tm_results.std()/np.sqrt(i+1)))
```
#### Output

```bash
python3 ./BreastCancerDemo.py 

Mean accuracy over 100 runs:

#1 Average Accuracy: 97.37% +/- 0.00
#2 Average Accuracy: 97.37% +/- 0.00
...
#99 Average Accuracy: 97.52% +/- 0.29
#100 Average Accuracy: 97.54% +/- 0.29
```

### MNIST Demo

#### Code: MNISTDemo.py

```python
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50, 10.0)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python3 ./MNISTDemo.py 

Accuracy over 250 epochs:

#1 Accuracy: 94.91% Training: 4.43s Testing: 1.02s
#2 Accuracy: 96.06% Training: 3.66s Testing: 1.03s
#3 Accuracy: 96.46% Training: 3.24s Testing: 1.07s
...

#248 Accuracy: 98.19% Training: 1.77s Testing: 1.06s
#249 Accuracy: 98.19% Training: 1.90s Testing: 1.05s
#250 Accuracy: 98.21% Training: 1.70s Testing: 1.06s
```
### MNIST Demo w/Weighted Clauses

#### Code: MNISTDemoWeightedClauses.py

```python
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50*100, 10.0, weighted_clauses=True)

print("\nAccuracy over 100 epochs:\n")
for i in range(100):
        start_training = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python3 ./MNISTDemoWeightedClauses.py

Accuracy over 100 epochs:

#1 Accuracy: 93.58% Training: 6.01s Testing: 1.53s
#2 Accuracy: 95.22% Training: 3.09s Testing: 1.44s
#3 Accuracy: 95.98% Training: 3.02s Testing: 1.08s
...

#98 Accuracy: 98.05% Training: 1.78s Testing: 1.06s
#99 Accuracy: 98.16% Training: 1.67s Testing: 1.06s
#100 Accuracy: 98.14% Training: 1.70s Testing: 1.08s
```

### MNIST 2D Convolution Demo w/Weighted Clauses

#### Code: MNISTDemo2DConvolutionWeightedClauses.py

```python
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = MultiClassConvolutionalTsetlinMachine2D(2000, 50*100, 5.0, (10, 10), weighted_clauses=True)

print("\nAccuracy over 30 epochs:\n")
for i in range(30):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
```

#### Output

```bash
python3 ./MNISTDemo2DConvolutionWeightedClauses.py 

Accuracy over 30 epochs:

#1 Accuracy: 97.78% (26.13s)
#2 Accuracy: 98.05% (26.81s)
#3 Accuracy: 98.25% (27.27s)
...

#28 Accuracy: 99.07% (30.84s)
#29 Accuracy: 99.09% (30.21s)
#30 Accuracy: 99.12% (29.52s)
```
### Fashion MNIST 2D Convolution Demo w/Weighted Clauses

#### Code: FashionMNISTDemo2DConvolutionWeightedClauses.py

```python
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import cv2
from keras.datasets import fashion_mnist

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

tm = MultiClassConvolutionalTsetlinMachine2D(8000, 100*100, 10.0, (10, 10), weighted_clauses=True)

print("\nAccuracy over 20 epochs:\n")
for i in range(20):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
```

#### Output

```bash
python3 ./FashionMNISTDemo2DConvolutionWeightedClauses.py 

Accuracy over 20 epochs:

#1 Accuracy: 84.98% (72.75s)
#2 Accuracy: 86.76% (67.73s)
#3 Accuracy: 87.74% (69.19s)
...

#18 Accuracy: 90.50% (63.15s)
#19 Accuracy: 90.59% (76.57s)
#20 Accuracy: 90.77% (69.71s)
```

### IMDb Text Categorization Demo

#### Code: IMDbTextCategorizationDemo.py

```python
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])
	
	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			
			if phrase in vocabulary:
				vocabulary[phrase] += 1
			else:
				vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
	if vocabulary[phrase] < 10:
		continue

	phrase_bit_nr[phrase] = bit_nr
	bit_nr_phrase[bit_nr] = phrase
	bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_train[i,phrase_bit_nr[phrase]] = 1

	Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_test[i,phrase_bit_nr[phrase]] = 1				

	Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

tm = MultiClassTsetlinMachine(10000, 80*100, 10.0, weighted_clauses=True)

print("\nAccuracy over 30 epochs:\n")
for i in range(30):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output:

```bash
python ./IMDbTextCategorizationDemo.py

Downloading dataset...
Producing bit representation...
Selecting features...

Accuracy over 30 epochs:

#1 Accuracy: 85.61% Training: 86.99s Testing: 21.02s
#2 Accuracy: 86.92% Training: 76.44s Testing: 20.19s
#3 Accuracy: 86.53% Training: 68.72s Testing: 22.44s
...

#28 Accuracy: 89.12% Training: 40.38s Testing: 27.77s
#29 Accuracy: 89.04% Training: 42.99s Testing: 26.28s
#30 Accuracy: 89.08% Training: 39.81s Testing: 26.38s
```

### Regression Demo

#### Code: RegressionDemo.py

```python
from pyTsetlinMachineParallel.tm import RegressionTsetlinMachine
from pyTsetlinMachineParallel.tools import Binarizer
import numpy as np
from time import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

california_housing = datasets.fetch_california_housing()
X = california_housing.data
Y = california_housing.target

b = Binarizer(max_bits_per_feature = 10)
b.fit(X)
X_transformed = b.transform(X)

tm = RegressionTsetlinMachine(1000, 500*10, 2.75, weighted_clauses=True)

print("\nRMSD over 25 runs:\n")
tm_results = np.empty(0)
for i in range(25):
	X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y)

	start = time()
	tm.fit(X_train, Y_train, epochs=30)
	stop = time()
	tm_results = np.append(tm_results, np.sqrt(((tm.predict(X_test) - Y_test)**2).mean()))

	print("#%d RMSD: %.2f +/- %.2f (%.2fs)" % (i+1, tm_results.mean(), 1.96*tm_results.std()/np.sqrt(i+1), stop-start))
```

#### Output

```bash
python3 ./RegressionDemo.py 

RMSD over 25 runs:

#1 RMSD: 0.61 +/- 0.00 (1.16s)
#2 RMSD: 0.62 +/- 0.01 (0.99s)
#3 RMSD: 0.61 +/- 0.01 (0.96s)
...

#23 RMSD: 0.61 +/- 0.00 (0.93s)
#24 RMSD: 0.61 +/- 0.00 (1.06s)
#25 RMSD: 0.61 +/- 0.00 (1.02s)
```

## Further Work

* Multilayer Tsetlin Machine
* Recurrent Tsetlin Machine
* GPU support
* Optimize convolution code
* More extensive hyper-parameter search for the demos

## Requirements

- Python 3.7.x, https://www.python.org/downloads/
- Numpy, http://www.numpy.org/
- OpenMP 5.0, https://www.openmp.org/
- Ubuntu or macOS

## Acknowledgements

I thank my colleagues from the Centre for Artificial Intelligence Research (CAIR), Lei Jiao, Xuan Zhang, Geir Thore Berge, Darshana Abeyrathna, Saeed Rahimi Gorji, Sondre Glimsdal, Rupsa Saha, Bimal Bhattarai, Rohan K. Yadev, Bernt Viggo Matheussen, Morten Goodwin, Christian Omlin, Vladimir Zadorozhny (University of Pittsburgh), Jivitesh Sharma, and Ahmed Abouzeid, for their contributions to the development of the Tsetlin machine family of techniques. I would also like to thank our House of CAIR partners, Alex Yakovlev, Rishad Shafik, Adrian Wheeldon, Jie Lei, Tousif Rahman (Newcastle University), Jonny Edwards (Temporal Computing), Marco Wiering (University of Groningen), Adrian Phoulady, Anders Refsdal Olsen, Halvor Smørvik and Erik Mathisen for their many contributions.

## Tsetlin Machine Papers

```bash
@InProceedings{gorji2020indexing,
  title="{Increasing the Inference and Learning Speed of Tsetlin Machines with Clause Indexing}",
  author={Saeed {Gorji} and Ole Christoffer {Granmo} and Sondre {Glimsdal} and Jonathan {Edwards} and Morten {Goodwin}},
  booktitle={Accepted to International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{abeyrathna2020integer,
  title="{A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation,}",
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle={Accepted to International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{phoulady2020weighted, 
  author={Adrian {Phoulady} and Ole-Christoffer {Granmo} and Saeed Rahimi {Gorji} and Hady Ahmady {Phoulady}}, 
  booktitle={Proceedings of the Ninth International Workshop on Statistical Relational AI (StarAI 2020)}, 
  title="{The Weighted Tsetlin Machine: Compressed Representations with Clause Weighting}",
  year={2020}
}
```

```bash
@InProceedings{wheeldon2020pervasive, 
  author={Adrian {Wheeldon} and Rishad {Shafik} and Alex {Yakovlev} and Jonathan {Edwards} and Ibrahim {Haddadi} and Ole-Christoffer {Granmo}}, 
  booktitle={SCONA Workshop at Design, Automation and Test in Europe (DATE 2020)}, 
  title="{Tsetlin Machine: A New Paradigm for Pervasive AI}",
  year={2020}
}
```

```bash
@article{abeyrathna2019nonlinear, 
  author={K. Darshana {Abeyrathna} and Ole-Christoffer {Granmo} and Xuan {Zhang} and Lei {Jiao} and Morten {Goodwin}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{The Regression Tsetlin Machine - A Novel Approach to Interpretable Non-Linear Regression}",
  volume={378}, issue={2164},
  year={2019}
}
```

```bash
@InProceedings{gorji2019multigranular,
  author = {Saeed Rahimi {Gorji} and Ole-Christoffer {Granmo} and Adrian {Phoulady} and Morten {Goodwin}},
  title = "{A Tsetlin Machine with Multigranular Clauses}",
  booktitle="Lecture Notes in Computer Science: Proceedings of the Thirty-ninth International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2019)", year="2019",
  volume = {11927},
  publisher="Springer International Publishing"
}
```

```bash
@article{berge2019text, 
  author={Geir Thore {Berge} and Ole-Christoffer {Granmo} and Tor Oddbjørn {Tveit} and Morten {Goodwin} and Lei {Jiao} and Bernt Viggo {Matheussen}}, 
  journal={IEEE Access}, 
  title="{Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications}",
  volume={7},
  pages={115134-115146}, 
  year={2019}, 
  doi={10.1109/ACCESS.2019.2935416}, 
  ISSN={2169-3536}
}
```

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal = {arXiv preprint arXiv:1905.09688}, year = {2019}
}
```

```bash
@InProceedings{abeyrathna2019regressiontsetlin,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Jiao}, Lei and {Goodwin}, Morten},
  title = "{The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems}",
  editor="Moura Oliveira, Paulo and Novais, Paulo and Reis, Lu{\'i}s Paulo ",
  booktitle="Progress in Artificial Intelligence", year="2019",
  publisher="Springer International Publishing",
  pages="268--280"
}
```

```bash
@InProceedings{abeyrathna2019continuousinput,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Zhang}, Xuan and {Goodwin}, Morten},
  title = "{A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks}",
  booktitle = "{Advances and Trends in Artificial Intelligence. From Theory to Practice}", year = "2019",
  editor = "Wotawa, Franz and Friedrich, Gerhard and Pill, Ingo and Koitz-Hristov, Roxane and Ali, Moonis",
  publisher = "Springer International Publishing",
  pages = "564--578"
}
```

```bash
@article{granmo2018tsetlin,
  author = {{Granmo}, Ole-Christoffer},
  title = "{The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}",
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018}
}
```

## Licence

Copyright (c) 2020 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
