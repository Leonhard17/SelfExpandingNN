# imports
import jax
import jax.numpy as jnp
import numpy as np
# Node and Network
from Node import Node
from Network import Cluster

class DataHandler:
    def __init__(self, cluster: Cluster, data: jnp.ndarray = jnp.array([]), target: jnp.ndarray = jnp.array([]), encoding : str = "spike_amp"): # TODO, encoding type, decoding type, start iterations, number of iterations to get the network started, batch size, sampling steps
        """
        This class handles input and output data.
        For now it only offer a simple way to input and output data trough a network. 

        Data encoding uses algorithms similar to SNNs

        # NOTE: Optimization is handled by the Cluster class, maybe change to runner class
        # TODO: Diffrerent learning type depending on network running type, pretraining, running...
        # TODO: Create subclasses which read in diffrent data formats, like csv, images, ...
        # TODO: Batch size and sampling steps
        # TODO: Dependency on network type, like pretrained or functional
        # TODO: Put some features in sperate network runner
        # TODO: Error handling
Planned:
- Encoding types (rate dependent and continous spike dependent)
- Output decoding (takes the average, median, ...)
- Start iterations, number of iterations to get the network started

        Args:
            cluster (Cluster): Cluster to which the data is fed
            data (jnp.ndarray): Input data
            target (jnp.ndarray): Target data
        """
        # size definitions from the cluster
        self.cluster = cluster
        self.input_size = len(cluster.input_nodes)
        self.output_size = len(cluster.output_nodes)
        # data
        self.data = data
        self.target = target
        self.encoding = encoding
        # size check
        # TODO : Make same error type as used in other files
        assert len(data) == len(target), "Data and target size mismatch"
        assert len(data[0]) == self.input_size, "Data input size mismatch"
        assert len(target[0]) == self.output_size, "Data output size mismatch"
        # data info
        self.index = 0
        self.data_size = len(data)

    # TODO: Create subclasses, which take images, csv, ...
    def load_data(self, data):
        """
        Takes in Data and stores it in the class to be later accessible by the runner
        # TODO: Implement functions to load data from diffrent formats

        Args:
            data (jnp.ndarray): Data to be loaded

        """
        raise NotImplementedError("Implemented trough a subclass")

    """
    # TODO : Check working
    Idea for Csv read in
    @classmethod
    def from_csv(cls, path):
        # convert csv to jnp array
        data = jnp.loadtxt(csv_file_path, delimiter=',')
        target = jnp.zeros(data.shape)
        return cls(cluster, data, target, encoding_type, decoding_type, batch_size, sampling_steps) # call cunstructor
    """
    # TODO: other formats and images
    
# Signal processing
    def encode_spike_amp(self, cur_data):
        """
        Encodes the data into a spike amplitude format.
        
        This format creates an array of the diffrent signal timesteps used for running the network
        """
        # give out the current data
        spike_amp_data = jnp.array([cur_data]) # TODO: Check if necessary
        # create zeros for the next spike
        zeros = jnp.zeros(spike_amp_data.shape)
        # merge to for two amplitudes for a spike
        cur_data = jnp.vstack([cur_data, zeros])
        return cur_data
    
    # TODO: Implement rate encoding

# Iterator
    """
    The iterator includes has batch size elements
    The elements have 2 features one is temporal and the other is the data for each input
    The temporal features should be switched in a circle for each iteration of the network
    """
    def  __iter__(self):
        return self
    
    def __next__(self):
        # TODO get batch
        # Return one element for now

        # select encoding mode and create elements
        if self.encoding == "spike_amp":
            cur_data = self.encode_spike_amp(self.data[self.index])
            cur_target = self.target[self.index]
        else:
            raise NotImplementedError("Encoding not implemented, try spike_amp")
        # increment index
        self.index += 1
        # check if index is out of bounds
        if self.index == self.data_size + 1: # TODO: One bigger sice data is accesed before
            self.index = 0
            raise StopIteration
        return cur_data, cur_target

# Helper functions
    def reset(self):
        """
        Resets the index of the data to 0
        """
        self.index = 0
    
    def convert_data(self, data, encoder):
        """ 
        Converts the data into the encoding format

        Args:
            data (jnp.ndarray): Data to be converted
            encoder (str): Encoding type
        """
        if encoder == "spike_amp":
            return self.encode_spike_amp(data)
        else:
            raise NotImplementedError("Encoding not implemented, try spike_amp")