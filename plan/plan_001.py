

# Preparing the input data
    # TODO: get that data from dataframe
    # https://www.tensorflow.org/tutorials/load_data/csv
    # probably experiment on: tff.simulation.datasets.shakespeare or ff.simulation.datasets.stackoverflow
    # emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    # TODO: divide it

# TODO: Preprocessing the data

# TODO: construct federated dataset
# def make_federated_data(client_data, client_ids):
#   return [
#       preprocess(client_data.create_tf_dataset_for_client(x))
#       for x in client_ids

# TODO: choose clients
# sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
#
# federated_train_data = make_federated_data(emnist_train, sample_clients)
#
# print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
# print('First dataset: {d}'.format(d=federated_train_data[0]))

# TODO: create a model
# def create_keras_model():
#   return tf.keras.models.Sequential([
#       tf.keras.layers.Input(shape=(784,)),
#       tf.keras.layers.Dense(10, kernel_initializer='zeros'),
#       tf.keras.layers.Softmax(),
#   ])

# TODO: wrap it under tff.learning.Model
# def model_fn():
#   # We _must_ create a new model here, and _not_ capture it from an external
#   # scope. TFF will call this within different graph contexts.
#   keras_model = create_keras_model()
#   return tff.learning.from_keras_model(
#       keras_model,
#       input_spec=preprocessed_example_dataset.element_spec,
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# TODO: make TFF construct a Federated Averaging algorithm by invoking the helper function
#  tff.learning.build_federated_averaging_process
# iterative_process = tff.learning.build_federated_averaging_process(
#     model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
#     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# NUM_ROUNDS = 11
# for round_num in range(2, NUM_ROUNDS):
#   state, metrics = iterative_process.next(state, federated_train_data)
#   print('round {:2d}, metrics={}'.format(round_num, metrics))

# initialize and next

# TODO: Defining model variables,
# MnistVariables = collections.namedtuple(
#     'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')
#
# def create_mnist_variables():
#   return MnistVariables(
#       weights=tf.Variable(
#           lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
#           name='weights',
#           trainable=True),
#       bias=tf.Variable(
#           lambda: tf.zeros(dtype=tf.float32, shape=(10)),
#           name='bias',
#           trainable=True),
#       num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
#       loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
#       accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))

# TODO: Define forward pass, and metrics
# def mnist_forward_pass(variables, batch):
#   y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
#   predictions = tf.cast(tf.argmax(y, 1), tf.int32)
#
#   flat_labels = tf.reshape(batch['y'], [-1])
#   loss = -tf.reduce_mean(
#       tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
#   accuracy = tf.reduce_mean(
#       tf.cast(tf.equal(predictions, flat_labels), tf.float32))
#
#   num_examples = tf.cast(tf.size(batch['y']), tf.float32)
#
#   variables.num_examples.assign_add(num_examples)
#   variables.loss_sum.assign_add(loss * num_examples)
#   variables.accuracy_sum.assign_add(accuracy * num_examples)
#
#   return loss, predictions

# def get_local_mnist_metrics(variables):
#   return collections.OrderedDict(
#       num_examples=variables.num_examples,
#       loss=variables.loss_sum / variables.num_examples,
#       accuracy=variables.accuracy_sum / variables.num_examples)

# TODO: aggregate the local metrics
# @tff.federated_computation
# def aggregate_mnist_metrics_across_clients(metrics):
#   return collections.OrderedDict(
#       num_examples=tff.federated_sum(metrics.num_examples),
#       loss=tff.federated_mean(metrics.loss, metrics.num_examples),
#       accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))

# class MnistModel(tff.learning.Model):
#
#   def __init__(self):
#     self._variables = create_mnist_variables()
#
#   @property
#   def trainable_variables(self):
#     return [self._variables.weights, self._variables.bias]
#
#   @property
#   def non_trainable_variables(self):
#     return []
#
#   @property
#   def local_variables(self):
#     return [
#         self._variables.num_examples, self._variables.loss_sum,
#         self._variables.accuracy_sum
#     ]
#
#   @property
#   def input_spec(self):
#     return collections.OrderedDict(
#         x=tf.TensorSpec([None, 784], tf.float32),
#         y=tf.TensorSpec([None, 1], tf.int32))
#
#   @tf.function
#   def forward_pass(self, batch, training=True):
#     del training
#     loss, predictions = mnist_forward_pass(self._variables, batch)
#     num_exmaples = tf.shape(batch['x'])[0]
#     return tff.learning.BatchOutput(
#         loss=loss, predictions=predictions, num_examples=num_exmaples)
#
#   @tf.function
#   def report_local_outputs(self):
#     return get_local_mnist_metrics(self._variables)
#
#   @property
#   def federated_output_computation(self):
#     return aggregate_mnist_metrics_across_clients

# USE TF variables, describe type and rap every function as tf.function

# RUN:
iterative_process = tff.learning.build_federated_averaging_process(
    MnistModel,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))
state = iterative_process.initialize()
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

for round_num in range(2, 11):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

evaluation = tff.learning.build_federated_evaluation(MnistModel)
str(evaluation.type_signature)

