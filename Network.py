"""
This file contains the class which defines the network architecture
# TODO: Add more details
"""

# imports
import jax
import jax.numpy as jnp
import numpy as np
from Node import Node

class Cluster:
    def __init__(self, num_inputs: int, num_outputs, num_nodes: int = 0, nodes : list[Node] = None, nodeModel: Node = Node, init_net: bool = True): # uses basic node model, can be changed in the future
        """
        This class saves and edits the computation tree of the nodes
        Meaning it saves the connections between the nodes
        Additionally it displays neighboring nodes needed for training
        """
        """
        This creates a cluster of nodes
        The cluster will have an input and output
        these are either connected to other clusters or inference for the user

Planned:
- Different types of nodes
- Different types of connections
- Propagation delay
- Spacial embedding used for topology
        """

        # TODO : Code when no nodes are given
        # TODO : Make compatible with node ids
        # NOTE : Changes that either nodes are given or parameter for init
        """
        Overview

        Init:
            1) Model parameters
                - nodes init
                - connections
                - io
            2) Model setup
                - node creation
                - connection setup

        Functions:


        """


        # nodes
        self.nodeModel = nodeModel

        if len(nodes) == 0 and num_nodes == 0:
            raise ValueError("No nodes given, provide either a list of nodes or the number of nodes to be created")
        # model parameters
        # TODO: assignation of input and output nodes
        if nodes:
            self.nodes = nodes
            self.num_nodes = len(nodes)
        else:
            self.nodes = []
            self.num_nodes = num_nodes

        # io
        self.input = num_inputs
        self.outputs = num_outputs

        # cluster size
        if self.num_nodes <= 1:
            raise ValueError("Number of nodes must be greater than 1, or provide list of nodes")
        self.connections = jnp.zeros((self.num_nodes, self.num_nodes)) # directional adjecency matrix, row 
        # io
        self.input_nodes = []
        self.output_nodes = []

        # NOTE Magic number for now
        self.input_nodes.append(int (nodes[0].id))
        self.output_nodes.append(int (nodes[-1].id))

        if init_net:
            # out and input for testing, later done in setupNetwork
            self.input_nodes.append(int (nodes[0].id))
            self.output_nodes.append(int (nodes[-1].id))

            # setup
            for i in range(num_nodes):
                # create disconnected nodes
                self.nodes.append(self.nodeModel(0, jnp.array([]), jnp.array([]), i))
                
            # setup network
            self.setupNetwork()
            return

# Setup functions

    def setupNetwork(self, initialization: str = "simple"):
        """ 
        This function uses diffrent types of algorithms to setup the network
        For now it creates a simple network with a topology similar to normal nns
    Planned:
    - Random connections
    - Evolutionary algorithm
    - Load own nodes
        """
        # TODO: IO setup
        # NOTE : This function should be outlaid to dedicated classes
            
        # TODO
        
        return None


# Topology functions

    def add_node(self, node: Node):
        """
        Takes a node as input and adds it to the adjacency matrix
        There it won't be connected to any other node 

        Args:
            node (Node): node of nodeModel used to create the cluster
        """
        # add node to list of nodes
        self.nodes.append(node)
        # adds node to adjacency matrix
        self.connections = jnp.append(self.connections, jnp.zeros((len(self.connections), 1)), axis=1)
        self.connections = jnp.append(self.connections, jnp.zeros((1, len(self.connections[0]))), axis=0)
        return None
    
    def add_connection(self, node1: Node, node2: Node):
        """
        Takes two nodes and adds a connection between them, node1 -> node2
        The change is made to both the adjacency matrix and the weights of the nodes

        Args:
            node1 (Node): node from which the connection originates 
            node2 (Node): node to which the connection goes
                for both also the classes nodeModel is used
        """
        # get ids, these are the adresses of the nodes
        node1_id = int(node1.get_id())
        node2_id = int(node2.get_id())
        # add connection to adjacency matrix
        self.connections = self.connections.at[node1_id, node2_id].set(1)
        # add connections to nodes
        # TODO: variable initialisation
        node1.add_output(0.5, node2_id)
        node2.add_input(0.5, node1_id)
        return None
    
    def remove_node(self, index: int):
        """
        Takes the index of a node and removes it from the node-list and adjacency matrix

        Args:
            index (int): index of node to be removed # NOTE: Could be later changed to node id
        """
        # NOTE: Implement that the node ids are shifted
        # remove node from list of nodes
        self.nodes.delete(index)
        # remove node from adjacency matrix
        self.connections = jnp.delete(self.connections, index, axis=0)
        self.connections = jnp.delete(self.connections, index, axis=1)
        return None
    
    #TODO: Add add-input and add-output functions

# Helper functions

    def get_neighbors(self, node: Node, degree: int = 1) -> jnp.ndarray:
        """
        This function returns the neighbors of a ceratin degree of a node
        Example: degree 1: direct neighbors, degree 2: neighbors of neighbors, ...
    
        This function uses the recursive collect_neighbors function to get the neighbors

        Args:
            node (Node): node for which the neighbors are searched
            degree (int): degree of neighbors that should be searched
        """
        neighbors = self.collect_neighbors(node, node, degree)
        neighbors = jnp.array(neighbors)
        neighbors = jnp.unique(neighbors)
        return neighbors
        
    def collect_neighbors(self, node: Node, prevNode: Node, degree: int = 1) -> jnp.ndarray:
        """
        This function returns the neighbors of a certain degree of a node
        Example: degree 1: direct neighbors, degree 2: neighbors of neighbors, ...
    
        Args:
            node (Node): node for which the neighbors are searched
            degree (int): degree of neighbors that should be searched
        """
        # init empty list
        neighbors = []
        if degree <= 0:
            raise ValueError("Degree must be at least 1")
        # recursive search
        if degree <= 1:
            # end of search
            neighbors = list(jnp.where(self.connections[node.get_id()] == 1)[0])
            # remove previous node if directed connection exists
            if self.connections[node.get_id()][prevNode.get_id()] == 1:
               neighbors.remove(prevNode.get_id())
        else:
            # get next neighbors
            cur_neighbors = list(jnp.where(self.connections[node.get_id()] == 1)[0])
            # remove previous node if directed connection exists
            if self.connections[node.get_id()][prevNode.get_id()] == 1:
                neighbors.remove(prevNode.get_id())
            # go over neighbors
            for cur_neighbor in cur_neighbors:
                #TODO: optimize efficiency
                # get new values
                neighbors.extend(self.collect_neighbors(self.nodes[cur_neighbor], node, degree - 1))
        
        return neighbors
       
# Run functions
    def run(self, u_inputs: jnp.ndarray) -> jnp.ndarray:
        """
        This function runs the cluster for one iteration with the given input
        It might take multiple iterations to even reach the output and maybe longer for a stable output
        The main complexity is in sorting in remapping the inputs and outputs
        Since all nodes are connected to each other the order of the nodes is important
        This approach will be changed later to be more efficient for now nodes are sorted

        Args:
            u_inputs (jnp.ndarray): user inputs for the cluster
        
        Returns:
            jnp.ndarray: output of the cluster
        """
        # This first parts collects the current network values and uses them to run the network

        # create input for each node from the adjacency matrix
        net_inputs = []
        output_connections = []
        # get new inputs
        input_connections = []
        for node in self.nodes:
            net_inputs.append(node.get_outputs())
            output_connections.append(node.get_outputNodes())
            input_connections.append(node.get_inputNodes())

        print("inputs: ", net_inputs)
        print("out_conn: ", output_connections)
        print("in_conn: ", input_connections)
        # map the inputs to the nodes
        node_inputs = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in input_connections[i]:
                list_pos = output_connections[j].index(i)
                node_inputs[i].append(net_inputs[j][list_pos])
        print("node_inputs: ", node_inputs)
        # run nodes
        # here also the external input is added
        for i in range(self.num_nodes):
            self.nodes[i].set_inputs(jnp.array(node_inputs[i]))
            self.nodes[i].run()
            if self.nodes[i].id in self.input_nodes:
                temp_activation = self.nodes[i].get_activation()
                self.nodes[i].activation = temp_activation + u_inputs[self.input_nodes.index(i)]
                print("input: ", u_inputs[self.input_nodes.index(i)])
        # get output
        for i in range(len(self.output_nodes)):
            output = jnp.append(output, self.nodes[self.output_nodes[i]].get_activation())
        return output

# Debugging functions
    def print_connections(self):
        """
        Prints out the adjacency matrix of the cluster
        Useful for debugging
        """
        print(self.connections)
        return None
    
    def print_activations(self):
        """
        Prints out the activations of the nodes
        Useful for debugging
        """
        for node in self.nodes:
            print("Node: ", node.get_id(), "Activation: ", node.get_activation())
        return None
    
    """
    def expand(self):
        
        Expands the Cluster depending on information density
        and the surrounding nodes
        
        pass

    def connections(self):
        pass
    
    def get_nodes(self):
        pass
"""