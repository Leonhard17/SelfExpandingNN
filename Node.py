"""
This file contains the class which defines the architecture
# TODO: Add more details
"""

# imports
import jax
import jax.numpy as jnp
import numpy as np

class Node:
    """
    A class representing a node in a neural network. It's used as a building block for the Cluster class.

    This class simulates a simple node with basic processing capabilities.
    The node can dynamically change its connections and weights during training.

    This is part of a research project and is still in development.

    Current features:
        - Simple processing
        - Topology functions
    To be added:
        - Decay time
        - Different processing (embedding, attention)
        - Signal delay


    Attributes:
        bias (float): The bias of the node.
        in_weights (jnp.array): The weights of the inputs.
        out_weights (jnp.array): The weights of the outputs.
        in_connections (list): List of node IDs that feed into this node.
        out_connections (list): List of node IDs that this node feeds into.
        activation (float): The current activation value of the node.
        outputs (jnp.array): The current outputs of the node.
        inputs (jnp.array): The current inputs to the node.
        id (int): The unique identifier of the node.
    """

    def __init__(self, bias, in_weights, out_weights, id):
        """
        Initializes the Node with the given parameters.

        Args:
            bias (float): The bias of the node.
            in_weights (jnp.array): The weights of the inputs.
            out_weights (jnp.array): The weights of the outputs.
            id (int): The unique identifier of the node.
        """
        self.bias = bias
        self.in_weights = in_weights
        self.out_weights = out_weights
        self.in_connections = []
        self.out_connections = []
        self.activation = 0
        self.outputs = jnp.array([])
        self.inputs = jnp.array([])
        self.id = id

# run functions
    def run(self):
        """
        Runs one step of the node's processing.

        Processes the inputs and computes the next activation and outputs.
        """
        # TODO: Save last activation so it's not lost
        self.outputs = jnp.dot(self.out_weights, self.activation)
        self.activation = jnp.dot(self.in_weights, self.inputs) + self.bias #TODO: Look if activation function is needed

    def set_inputs(self, inputs):
        """
        Sets the inputs for the node.

        Args:
            inputs (jnp.array): The inputs for the node, must match the size of in_weights.

        Raises:
            ValueError: If the size of inputs does not match the size of in_weights.
        """
        if inputs.size != self.in_weights.size:
            raise ValueError("Input size does not match in_weights size")
        self.inputs = inputs

    def get_inputNodes(self):
        """
        Returns a list of the IDs of the nodes which feed into this node.

        Returns:
            list: List of node IDs.
        """
        return self.in_connections

    def get_outputs(self):
        """
        Returns the outputs of the node.

        If the node is run for the first time, returns a tensor of zeros.

        Returns:
            jnp.array: The outputs of the node.
        """
        if self.outputs.size == 0:
            return jnp.zeros(self.out_weights.size)
        return self.outputs

    def get_outputNodes(self):
        """
        Returns a list of the IDs of the nodes which this node feeds into.

        Returns:
            list: List of node IDs.
        """
        return self.out_connections

# Topology functions
    # TODO: Create a function that automatically sorts connections
    # TODO: Change so that it's connection pair wise, do in Cluster
    # Maybe rewrite function for better functionality with cluster

    def add_input(self, weight, nodeId):
        """
        Adds a new input to the node.

        Args:
            weight (float): The weight of the input.
            nodeId (int): The ID of the node to connect to.
        """
        self.in_weights = jnp.append(self.in_weights, weight)
        self.in_connections.append(nodeId)

    def add_output(self, weight, nodeId):
        """
        Adds a new output to the node.

        Args:
            weight (float): The weight of the output.
            nodeId (int): The ID of the node to connect to.
        """
        self.out_weights = jnp.append(self.out_weights, weight)
        self.out_connections.append(nodeId)

    def remove_input(self, index):
        """
        Removes an input from the node.

        Args:
            index (int): The index of the input to remove.
        """
        self.in_weights = jnp.delete(self.in_weights, index)
        self.in_connections.pop(index)

    def remove_output(self, index):
        """
        Removes an output from the node.

        Args:
            index (int): The index of the output to remove.
        """
        self.out_weights = jnp.delete(self.out_weights, index)
        self.out_connections.pop(index)

# Helper functions
    def get_id(self):
        """
        Returns the ID of the node.

        Returns:
            int: The ID of the node.
        """
        return self.id

    def get_activation(self):
        """
        Returns the activation of the node.

        Returns:
            float: The activation value.
        """
        return self.activation

    def get_bias(self):
        """
        Returns the bias of the node.

        Returns:
            float: The bias value.
        """
        return self.bias

    def get_in_weights(self):
        """
        Returns the input weights of the node.

        Returns:
            jnp.array: The input weights.
        """
        return self.in_weights

    def get_out_weights(self):
        """
        Returns the output weights of the node.

        Returns:
            jnp.array: The output weights.
        """
        return self.out_weights

    def get_input_size(self):
        """
        Returns the size of the input weights.

        Returns:
            int: The size of the input weights.
        """
        return self.in_weights.size

    def get_output_size(self):
        """
        Returns the size of the output weights.

        Returns:
            int: The size of the output weights.
        """
        return self.out_weights.size

# Debugging functions
    def print_node(self):
        """
        Prints the node information.
        """
        print("Node id: ", self.id)
        print("Activation: ", self.activation)
        print("Bias: ", self.bias)
        print("Inputs: ", self.in_weights)
        print("Outputs: ", self.out_weights)


    """
    Simple node class, used as building block for the cluster class

    We get inputs from other nodes and outputs to other nodes
    The nummber and connection nodes in training should dynamically change
    The processing inside is not fixed and will be either an embedding or attention
    To close mimic real neurons the activation will also have a deacay time

    This architecture is still in development only some features will be added for now
    so debugging is simpler
    """