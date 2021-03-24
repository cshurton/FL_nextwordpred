
class Client:

    def __init__(self, name):
        self.name = name
        self.sheet = []
        self.message_box = []
        print("client named:", self.name, "created.")


    def compute_gradient(self): # + attributes in the future: learninng_rate
        """It computes gradient for one client

        Attributes:
            learning_rate (float): learning rate
        Returns:

        """
        pass

    def compute_local_w(self):
        """It computes local weights. At some point compute_gradient() will be invoked from here.

        Returns:
            w: the local weights
        """
        pass

    def learn(self): # + attributes in the future: epoch, batchsize, learning_rate
        """Govern the learning process.

        Check message box, if there is an update of averaged weights coming from the server, saves it in its own calculation
        and learn on new data.

        Attributes:
            epoch (int): the number of training passes each client makes over its local dataset on each round
            batchsize (int): local minibatch size
            learning_rate (float): learning rate
        Returns:
            w - the updated weights
        """
        print(self.name, "is learning.")
