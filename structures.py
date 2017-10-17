import numpy as np
import random
import math
class FullyConnectedNetwork:
	def __init__(self, layer_heights = [9,3,9]):
		def CreateLayers():
			self.layers = []

			### Create and append input layer
			self.input_layer = [Node.__init__() for i in range(layer_heights[0])]
			self.layers.append(self.input_layer)

			### Create and append hidden layers
			self.hidden_layers = []
			for depth in range(1,len(layer_heights)-1):
				self.hidden_layers.append([Node.__init__() for i in range(layer_heights[depth])])
			self.layers+=self.hidden_layers
			
			### Create and append output layer
			self.output_layer = [Node.__init__() for i in range(layer_heights[0])]
			self.layers.append(self.output_layer)


		def CreateFullConnections():
			for i in range(len(self.layers)-1):
				for parent in self.layers[i]:
					for child in self.layers[i+1]:
						parent.AddChild(child)


		def Train(data):
			if len(data) != len(self.input_layer):
				print 'Error: Input has wrong dimension'
				return
			for inp in self.input_layer: inp.input


		CreateLayers()

		CreateFullConnections()

		


		

class Node():
	def __init__(self, purpose = 'H'):
		if purpose == 'H':
			self.inputs = []
			self.outputs = []
		elif purpose = 'I':
			self.inputs = [0]
			self.outputs = []
		elif purpose = 'O':
			self.inputs = []
			self.outputs = [0]
		else:
			print 'Error: Wrong node purpose key'
			return

	def AddChild(self,child):
		new_connection = Connection(self,child)
		self.outputs.append(new_connection)
		child.inputs.append(new_connection)

	def forward(self):

		### Calculate total input
		total_input = 0
		for inp in self.inputs:
			total_input += inp.charge

		### Reset Charge of output
		for out in self.outputs:
			out.Update(total_input)

class Connection():
	def __init__(self, inp, out, initial_weight = None):
		self.weight = random.random() if initial_weight == None else initial_weight
		self.mother = inp
		self.child = out
		self.charge = None
		self.activation = self.sigmoid

	def Update(self, total_in):
		self.charge = self.activation( total_in * self.weight )

	def sigmoid(self, x):
		return 1/(1+math.e(-x))
	

