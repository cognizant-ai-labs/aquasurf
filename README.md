# aquasurf
The paper Efficient Activation Function Optimization through Surrogate Modeling is available [here](https://arxiv.org/abs/2301.05785).

---

## Installation

Install the package with
```bash
pip install git+https://github.com/cognizant-ai-labs/aquasurf.git
```
The code has been tested with `python==3.10.6` and `tensorflow==2.8.0`.

---

## Overview

Optimizing activation functions for your architecture and task requires the following steps:
* Modify your `Model` to use a custom `ActivationFunction`
* Subclass an `ActivationFunctionDatabase`
* Populate the database with the desired search space
* Calculate FIM eigenvalues
* Search for better activation functions

Each of the steps is discussed in detail below.

---

## Modify your `Model` to use a custom `ActivationFunction`
The `ActivationFunction` class in `activation.py` is used for creating activation functions to use with TensorFlow models.  The computation graph operators can be adjusted by modifying the dictionaries `N_ARY_FUNCTIONS`, `BINARY_FUNCTIONS`, and `UNARY_FUNCTIONS`.  

First, import the class
```python
from aquasurf.activation import ActivationFunction
```

An activation function can then be created by specifying the `fn_name` parameter.  It is used as a typical `Layer` in a TensorFlow `Model`.  For example:

```python
def build_model(fn_name):
    ...
    x = Dense(100)(x)
    x = ActivationFunction(fn_name=fn_name)(x)
    x = Dense(100)(x)
    ...
    return model
```
The model can then be instantiated with different activation functions, such as
```python
model = build_model(fn_name='max(relu(x),cosh(elu(x)))')
```
or
```python
model = build_model(fn_name='sum_n(abs(x),swish(x),sigmoid(x))')
```

---

## Subclass an `ActivationFunctionDatabase`

The `ActivationFunctionDatabase` class in `database.py` manages a `sqlite3` database instance that stores information about the activation functions in the search space.  In order to run your own experiment, you need to first create a subclass and override a few parameters.

As an example, an activation function database for All-CNN-C on CIFAR-100 would look something like the following:

```python
import tensorflow as tf
from aquasurf.database import ActivationFunctionDatabase
# Define these yourself.  They will be specific to your scripts.
from my_script import load_batch, build_model  

class My_AllCNNC_CIFAR100_AFD(ActivationFunctionDatabase):
    def __init__(self, db_path):
        super().__init__(db_path)

    # One batch and the corresponding labels.  
    # If you encounter OOM, you may need to use fewer samples here,
    # but you can still use the regular batch size during training.
    self.samples, self.labels = load_batch()

    # Loss function used in training
    self.loss = tf.keras.losses.CategoricalCrossentropy()

    # The number of weights in each layer of All-CNN-C
    self.weights_per_layer = [
        2592+96,
        82944+96,
        82944+96,
        165888+192,
        331776+192,
        331776+192,
        331776+192,
        36864+192,
        1920+10
    ]
    
    # Functions you want to begin the search with
    self.baseline_fns = [
        'elu(x)',
        'relu(x)',
        'selu(x)',
        'sigmoid(x)',
        'softplus(x)',
        'softsign(x)',
        'swish(x)',
        'tanh(x)',
    ]

    # Manually insert baseline functions if they don't exist
    self.cursor.execute(
        'SELECT fn_name FROM activation_functions WHERE fn_name IN ({})'.format(
            ','.join(['?'] * len(self.baseline_fns))
        ),
        self.baseline_fns
    )
    baseline_fns_in_db = [row[0] for row in self.cursor.fetchall()]
    missing_baseline_fns = list(set(self.baseline_fns) - set(baseline_fns_in_db))
    if len(missing_baseline_fns) > 0:
        self.populate_database(fn_names_list=missing_baseline_fns)
        self.calculate_fisher_eigs(fn_names_list=missing_baseline_fns)

    # This method must be overridden and should return a TensorFlow Model
    # that uses the activation function specified by fn_name.
    def build_model(self, fn_name):
        model = build_model(fn_name=fn_name)
        return model
```

The default regression model used for predicting activation function performance is `self.regression_model = KNeighborsRegressor(n_neighbors=3)`.  You are free to override this if desired.

---

## Populate the database with the desired search space

Next, the search space needs to be defined.  To do so, instantiate the database in a terminal window like so:

```python
$ python
>>> from my_script.database import My_AllCNNC_CIFAR100_AFD
>>> db_path = './databases/my_allcnnc_cifar100.db'
>>> afd = My_AllCNNC_CIFAR100_AFD(db_path)
```

If your database includes `self.baseline_fns`, their function outputs and FIM eigenvalues will be calculated the first time the database is instantiated.  This may take a moment.

Next, populate the database.  The following commands will insert all activation functions of the form `'unary(unary(x))'` and `'binary(unary(x),unary(x))'` into the database.  Any such schema can be used, and `'n-ary'` operators can be utilized as well.  Additionally, specific activation functions can be inserted by specifying a list of their names.

```python
>>> afd.populate_database(schema='unary(unary(x))')
>>> afd.populate_database(schema='binary(unary(x),unary(x))')
>>> afd.populate_database(fn_names_list=['relu(x)', 'swish(tanh(x))'])
```

Populating the database in this way will automatically calculate and store the activation function output features for each of the functions.  This may take a moment, depending on the size of the schema.  However, the FIM eigenvalues are not calculated at this point, as this often takes several seconds or a few minutes for each function, depending on the architecture.

---

## Calculate FIM eigenvalues

To calculate FIM eigenvalues, use the following command.

```python
>>> afd.calculate_fisher_eigs()
```

To speed up the calculation of the eigenvalues, it may be useful to open one terminal window per available GPU and execute the command in each of them.  Once the number of functions with FIM eigenvalue features calculated is high enough (a few thousand was used in the paper), the jobs can be manually stopped.  The `afd.summary()` command will print this and other information about the database.

```
>>> afd.summary()
Number of activation functions: 1023524
Number of unique activation functions: 425896
Number of unique functions with eigenvalues calculated: 5000
Number of evaluated activation functions: 100
Number of running activation functions: 0
Best so far: prod_n(sigmoid(x),negative(x),hard_sigmoid(x)) with validation accuracy 0.6396
```

The eigenvalue calculation is handled by the `FIM` class in `fisher.py`.  There are a few requirements your model must adhere to in order to calculate the eigenvalues correctly, and the class will print error messages if they are not satisfied. The main things to be aware of are:

* The only layers with weights that are currently supported are `Conv2D`, `DepthwiseConv2D`, and `Dense`.  If your model has other types of layers with weights, eigenvalues corresponding to those weights will not be calculated.
* Nested TensorFlow models are currently not supported.
* Activation functions must be implemented in separate layers:
```python
# Replace this
outputs = Dense(100, activation='softmax')(x)

# with this
x = Dense(100)(x)
outputs = Activation('softmax')(x)
```

---

## Search for better activation functions

Searching for better activation functions requires a few small modifications to your training script.  First, import and instantiate the database you subclassed previously.

```python
from my_script.database import My_AllCNNC_CIFAR100_AFD
db_path = './databases/my_allcnnc_cifar100.db'
afd = My_AllCNNC_CIFAR100_AFD(db_path)
```

Next, the `afd.suggest_fn()` method will return the name of the activation function with the highest predicted performance.  This step involves fitting a `UMAP` model to the activation function outputs and FIM eigenvalues for all of the activation functions evaluated so far, and then using a regression model to predict performance.  It may take a moment.  Use the suggested activation function to instantiate your model.

```python
fn_name = afd.suggest_fn()
model = build_model(fn_name=fn_name)
```

After this, set the status of the activation function in the database to `'running'`.  This command will prevent other jobs from evaluating this function, so you are free to search for an activation function with multiple parallel workers.  The command will update the status for all functionally equivalent activation functions.  If the suggested function was `add(tanh(x),selu(x))`, another worker will not simultaneously evaluate `add(selu(x),tanh(x))`, since they are functionally equivalent.

```python
afd.update_for_all_equivalent_fns(fn_name, 'status', 'running')
```

Now, execute your training script as normal.  After training completes, the results in the database need to be updated to inform future performance prediction.  The most important updates to make are setting the `status` to `done` and updating the `val_acc`, since this is what performance prediction is based off of.  If desired, additional information can be supplied as well.

```python
afd.update_for_all_equivalent_fns(fn_name, 'status', 'done')
afd.update_for_all_equivalent_fns(fn_name, 'train_acc', train_acc)
afd.update_for_all_equivalent_fns(fn_name, 'train_loss', train_loss)
afd.update_for_all_equivalent_fns(fn_name, 'val_acc', val_acc)
afd.update_for_all_equivalent_fns(fn_name, 'val_loss', val_loss)
afd.update_for_all_equivalent_fns(fn_name, 'test_acc', test_acc)
afd.update_for_all_equivalent_fns(fn_name, 'test_loss', test_loss)
afd.update_for_all_equivalent_fns(fn_name, 'runtime', runtime)
```


