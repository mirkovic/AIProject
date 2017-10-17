import numpy as np
import random
import math
from scipy import misc
from matplotlib import pyplot as plt


def sigmoid(x):
	try:
		return 1/(1+math.exp(-x))
	except:
		return 0
	
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))

def relu(x):
	if x >= 0: return x
	else: return 0

def relu_prime(x):
	if x >= 0: return 1
	else: return 0

def distance(data,centroid):
	return np.sum(np.square(np.subtract(data,centroid)))
	
def OpenImage(file_name):
	pixels = misc.imread(file_name)
	grids = []
	for i in range(1,len(pixels)-1):
		for j in range(1,len(pixels[0])-1):
			grid = []
			for k in [-1,0,1]:
				for l in [-1,0,1]:
					grid.append(sum(pixels[i+k][j+l])/3)
			grids.append(grid)
	return grids

def CreateImage(pixels,name='random'):
	big = []
	for i in range(174):
		new_row = []
		for j in range(281):
			if i*174+j < 174*281:
				new_row.append(pixels[i*281+j])
			else:
				new_row.append([255,255,255])
		big.append(new_row)
	img = misc.toimage(big)		
	img.save(name+'.bmp')

def kmeans(data,nclusters):

	dimension = len(data[0])
	centroids = []
	oldcentroids = []
	for i in range(nclusters):
		this_cent = []
		for j in range(dimension):
			this_cent.append(random.random())
		centroids.append(this_cent)
	print
	print
	#print centroids
	total_error = 0
	for i in range(100):
		if np.array_equal(oldcentroids, centroids):
			break
		groups = [[] for i in range(nclusters)]
		for d in data:
			errors = map(lambda x: distance(d,x),centroids)
			#print errors
			best_group =  errors.index(min(errors))
			#print len(groups[best_group])
			groups[best_group].append(d)
			#print map(len,groups)
		oldcentroids = centroids
		centroids = np.zeros((len(groups),dimension)) 
		for j in range(len(groups)):
			centroids[j] = np.divide(np.sum(groups[j],axis=0),len(groups[j]))
		prev_error = total_error
		total_error = sum(map(lambda x: min(map(lambda y: distance(x,y), centroids)), data))
		print total_error
		if float(abs(prev_error-total_error))/float(total_error) < .001:
			return (centroids,total_error)
		
	return (centroids,total_error)



class FullyConnectedNetwork:
	### layer_heights[i] gives the number of nodes to have at layer of depth i
	def __init__(self, rate = .01, layer_heights = [9,12,9,3,3]):
	
		self.learning_rate = rate
		self.derivative = relu_prime
		self.layers = []

		### Create and append input layer nodes
		self.input_layer = [Node(self) for i in range(layer_heights[0])]
		self.input_layer.append(Node( self, bias=True))
		self.layers.append(self.input_layer)

		### Create and append hidden layers nodes
		self.hidden_layers = []
		for depth in range(1,len(layer_heights)-1):
			new_hidden = [ Node(self) for i in range(layer_heights[depth]) ]
			new_hidden.append( Node( self, bias=True ) )
			self.hidden_layers.append(new_hidden)
		self.layers+=self.hidden_layers
		
		### Create and append output layer nodes
		self.output_layer = [Node(self) for i in range(layer_heights[len(layer_heights)-1])]
		self.layers.append(self.output_layer)

		### create connections between nodes
		for i in range(len(self.input_layer)-1):
			self.input_layer[i].CreateInput()
			
		for i in range(len(self.layers)-1):
			for parent in self.layers[i]:
				for child in self.layers[i+1]:
					if i != len(self.layers)-2 and child == self.layers[i+1][-1]:
						continue
					parent.AddChild(child)
		for node in self.output_layer:
			node.CreateOutput()

			
	def Train(self,input_data,output_data):
		self.TakeInput(input_data)
		self.BackPropogation(output_data)
		
		
	def TakeInput(self, input_data, should_print = False):
		if len(input_data) != len(self.input_layer) - 1 :
			print( 'Error: Input has wrong dimension' )
			return
			
		### Adjust Inputs
		input_node = iter(self.input_layer)
		for inp in input_data: 
			next(input_node).inputs[0].charge = inp
			
		### send inputs to next layer
		for layer in self.layers:
			if should_print:
				print
				print 'next layer'
			for node in layer:
				node.forward(should_print = should_print)
			
			
	def BackPropogation(self,data):
		if len(data) != len(self.output_layer):
			print( 'Error: Output has wrong dimension' )
			return
		
		### calculate errors of output nodes
		for i in range(len(self.output_layer)):
			self.output_layer[i].error = self.derivative(self.output_layer[i].total_input) * ( self.output_layer[i].outputs[0].charge - data[i] )
		
		### now begin propogating errors back through the network
		nlayers = len(self.layers)
		for i in range( nlayers - 1 ):
			for node in self.layers[nlayers - i - 2]:
				node.backward()
	
	
	def PrintWeights(self):
		for i in range(len(self.layers)):
			layer = self.layers[i]
			print( 'layer: ' + str(i) )
			for node in layer:
				print( [j.weight for j in node.inputs] )
			print()
			
	def PrintErrors(self):
		#print( [j.inputs[0].charge for j in self.input_layer] )
		for i in range(len(self.layers)):
			#print( [j.charge for j in self.input_layer] )
			layer = self.layers[i]
			print( 'layer: ' + str(i) )
			print( [j.error for j in layer] )
			print()
			
	def PrintOutput(self):
		print( [i.outputs[0].charge for i in self.output_layer] )
	
	def GetOutput(self):
		return [i.outputs[0].charge for i in self.output_layer]

		

class Node():
	def __init__(self, net, bias = False):
		self.net = net
		self.inputs = []
		self.outputs = []
		self.error = 0
		self.total_input = 0
		self.derivative = relu_prime
		if bias:
			self.inputs.append(Connection(None,self,initial_weight=1))
			self.inputs[0].charge = 1

	### adds a connection between self and child
	def AddChild(self,child):
		new_connection = Connection(self,child)
		self.outputs.append(new_connection)
		child.inputs.append(new_connection)
	
	### This method is used to create an input connection for input nodes
	def CreateInput(self):
		self.inputs.append(Connection(None,self,initial_weight=1))
		
	### This method is used to create an output connection for output nodes
	def CreateOutput(self):
		self.outputs.append(Connection(self,None,initial_weight=1))

	### This method is used to propgate input values through a node in a network
	def forward(self,should_print = False):
		### Calculate total input
		self.total_input = sum(inp.charge for inp in self.inputs)

		### Reset Charge of outputs
		if should_print:
			print 'forwarding node'
		for out in self.outputs:
			out.Update(self.total_input,should_print=should_print)
	
	### This method is used to propogate errors back through the network, then change weights
	def backward(self):
		error = 0
		for output in self.outputs:
			error += output.child.error * output.weight
		self.error = error * self.derivative(self.total_input)
		for output in self.outputs:
			output.weight = output.weight - self.net.learning_rate * self.total_input * output.child.error
			if output.weight < 0:
				output.weight = 0
		
### these are connections betweeen nodes
class Connection():
	def __init__(self, inp, out, initial_weight = None):
		self.weight = random.uniform(-1,1) if initial_weight == None else initial_weight
		self.mother = inp
		self.child = out
		self.charge = None
		self.activation = relu

	def Update(self, total_charge,should_print = False):
		self.charge = self.activation( total_charge * self.weight )
		if should_print:
			print 'input: ' + str(total_charge) + ',   weight: ' + str(self.weight), ',   output: ' + str(self.charge)


	
#OpenImage('wood1.jpg')
#exit()
#Network = FullyConnectedNetwork()
#Network.PrintWeights()
input_data = []
output_data = []
data = []
with open('input.csv') as f:
	for line in f.readlines():
		line = map(float,line.replace('\n','').split(','))
		input_data.append(np.divide(line,256))
with open('colors.csv') as f:
	for line in f.readlines():
		line = map( float,line.replace('\n','').split(','))
		output_data.append(np.divide(line,256))
with open('data.csv') as f:
	for line in f.readlines():
		line = map( float,line.replace('\n','').split(','))
		data.append(np.divide(line,256))
#print len(input_data)		
#print len(output_data)
CreateImage(output_data)
kmeans_errors = []
clusters = {}
for i in range(1,2):
	results = kmeans(input_data,nclusters=i)
	clusters[i] = results[0]
	kmeans_errors.append(results[1])
plt.plot(kmeans_errors)
plt.savefig('kmeanserrors.png')
plt.close()
print kmeans_errors
for i in range(1,len(kmeans_errors)):
	if float(kmeans_errors[i]-kmeans_errors[i-1])/kmeans_errors[i] < .15:
		nclusters = i 
#######
nclusters = 1
#######
selected_clusters = clusters[nclusters]
training_data = []
testing_data = []
for c in range(nclusters):
	training_data.append([])
	testing_data.append([])
num_outputs = 3 
for i in range(len(input_data)):
	data_errors = map(lambda x: distance(x,input_data[i]),selected_clusters)
	c = data_errors.index(min(data_errors))
	if random.random() < .8:
		training_data[c].append(i)
	else:
		testing_data[c].append(i)

### train networks
Networks = []
for c in range(len(training_data)):
	Networks.append(FullyConnectedNetwork())
	for i in training_data[c]:
		Networks[c].Train(input_data[i],output_data[i][:num_outputs])
	Networks[c].PrintWeights()



#### test networks on testing data
names= ['red','green','blue']
all_errors = []
red_error = []
green_error = []
blue_error = []
for c in range(len(testing_data)):
	all_errors.append( [[],[],[]])
	for i in testing_data[c]:
		Networks[c].TakeInput(input_data[i])
		errors = np.subtract(Networks[c].GetOutput(),output_data[i][:num_outputs])
		for j in range(num_outputs):
			all_errors[c][j].append(errors[j])

print map(lambda x: np.divide( np.sum(x,axis=1),float(len(x[0]) )),all_errors )
new_all = []
for c in range(len(all_errors[0])):
	new_list = []
	for i in range(len(all_errors)):
		new_list += all_errors[i][c]
	new_all.append(new_list)
all_errors = new_all
#### create training image
colors = np.zeros((len(input_data),3))
for c in range(len(testing_data)):
	for i in testing_data[c]+training_data[c]:
		Networks[c].TakeInput(input_data[i])
		colors[i] = np.multiply(Networks[c].GetOutput(),256)
shifts = map(lambda x: float(sum(x))/len(x),all_errors)
colors = np.subtract(colors,np.multiply(shifts,256))
CreateImage(colors,name='reconstructed')


for i in range(len(shifts)):
	all_errors[i] =  np.subtract(all_errors[i],shifts[i])
print 'plotting'
for j in range(num_outputs):
	plt.hist(all_errors[j],bins = 50)
	plt.savefig(names[j]+'.png')
	plt.close()
#Network.PrintWeights()


