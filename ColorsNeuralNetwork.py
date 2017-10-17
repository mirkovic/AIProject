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

def linear(x):
	return x

def linear_prime(x):
	return 1

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

def CreateImage(pixels,name='random',x=281,y=174):
	big = []
	for i in range(y):
		new_row = []
		for j in range(x):
			if i*x+j < y*x:
				new_row.append(pixels[i*x+j])
			else:
				new_row.append([0,0,0])
		big.append(new_row)
	img = misc.toimage(big)		
	img.save(name+'.bmp')

def kmeans(data,nclusters,tolerance = .001):

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
		if total_error < 2000:
			print centroids
		if float(abs(prev_error-total_error))/float(total_error) < tolerance:
			return (centroids,total_error)
		
	return (centroids,total_error)

def BestKmeans(data,rang):
	error = {}
	for i in range(rang[0],rang[1]):
		error[i] = []
		error[i].append(kmeans(data,nclusters = i, tolerance = .1))[1]
	for i in range(rang[0],rang[1]-1):
		if float(error[i+1] - error[i] ) / error[i+1] < .2:
			return i+1

def BestCluster(data,clusters):
	### I am a wizard
	return map(lambda x: x.index(min(x)),[map(lambda x: distance(x,data),clusters)])[0]

def Dif(inp,out):
	return inp - out
		
def Class(inp, out, clusters):
	return BestCluster(inp,clusters) == BestCluster(out,clusters)


class FullyConnectedNetwork:
	### layer_heights[i] gives the number of nodes to have at layer of depth i
	def __init__(self, rate = .01, layer_heights = [9,3,3,3], error_function=Dif,clusters=[]):
	
		self.learning_rate = rate
		self.function = relu 
		self.derivative = relu_prime
		self.Error = error_function
		self.clusters = clusters
		self.layers = []

		### Create and append input layer nodes
		self.input_layer = [Node(self, self.function, self.derivative) for i in range(layer_heights[0])]
		self.input_layer.append(Node( self, self.function, self.derivative, bias=True))
		self.layers.append(self.input_layer)

		### Create and append hidden layers nodes
		self.hidden_layers = []
		for depth in range(1,len(layer_heights)-1):
			new_hidden = [ Node(self, self.function, self.derivative ) for i in range(layer_heights[depth]) ]
			new_hidden.append( Node( self, self.function, self.derivative, bias=True ) )
			self.hidden_layers.append(new_hidden)
		self.layers+=self.hidden_layers
		
		### Create and append output layer nodes
		self.output_layer = [Node(self,self.function, self.derivative) for i in range(layer_heights[len(layer_heights)-1])]
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

		
	def TrainAll(self,inp, out, fraction=.8):
		train_data = []
		test_data = []
		for i in range(len(inp)):
			if random.random() < fraction:
				train_data.append(i)
			else:
				test_data.append(i)

		random.shuffle(train_data)
		#count = 0
		for i in train_data:
			#for node in self.output_layer:
			#	print [j.weight for j in node.inputs]
			self.Train(inp[i],out[i])
			#print
			#if count > 10:
			#	exit()
			#else:
			#	c6ount+=1
			#self.PrintWeights()
		return train_data, test_data

	def ClusterOnError(self, clusters, datapoints, inp, out):
		clusterpoints = []
		for c in range(len(clusters)):
			clusterpoints.append([])
		for i in datapoints:
			self.TakeInput(inp[i])
			output = self.GetOutput()
			error = np.subtract(output,out[i])
			cluster_errors = map(lambda x: distance(x,error),clusters)
			cluster = cluster_errors.index(min(cluster_errors))
			clusterpoints[cluster].append(i)
		
		return clusterpoints


	def Analyze(self, datapoints, inp, out, section = ''):
		errors = []
		section = str(section)
		for i in datapoints:
			Network.TakeInput(inp[i])
			output = Network.GetOutput()
			error = np.subtract(output,out[i])
			errors.append(error)
		
		colors = {0:'red',1:'green',2:'blue'}
		for i in range(len(errors[0])):
			output_error = []
			for j in range(len(errors)):
				output_error.append(errors[j][i])
			plt.hist(output_error,bins = 50)
			plt.savefig(section+colors[i]+'.png')
			plt.close()

		return errors
			

	
		

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
			#if i == 0:
			#	print self.output_layer[i].outputs[0].chargeo
			errorvars = [self.output_layer[i].outputs[0].charge, data[i]]
			if self.Error == Class:	
				errorvars.append(self.clusters)
			self.output_layer[i].error = self.derivative(self.output_layer[i].total_input) * self.Error(*errorvars)
		
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
		for node in self.output_layer:
			print [j.weight for j in node.outputs]
			
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
	def __init__(self, net, function, derivative, bias = False):
		self.net = net
		self.inputs = []
		self.outputs = []
		self.error = 0
		self.total_input = 0
		self.derivative = derivative
		self.function = function
		if bias:
			self.inputs.append(Connection(None,self,self.function,initial_weight=0))
			self.inputs[0].charge = .01

	### adds a connection between self and child
	def AddChild(self,child):
		new_connection = Connection(self, child, self.function)
		self.outputs.append(new_connection)
		child.inputs.append(new_connection)
	
	### This method is used to create an input connection for input nodes
	def CreateInput(self):
		self.inputs.append(Connection(None,self,self.function, initial_weight=1))
		
	### This method is used to create an output connection for output nodes
	def CreateOutput(self):
		self.outputs.append(Connection(self,None,self.function, initial_weight=1))

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
			if self.function == relu and output.weight < 0:
				output.weight = 0
		
### these are connections betweeen nodes
class Connection():
	def __init__(self, inp, out, function, initial_weight = None):
		self.weight = random.uniform(-1,1) if initial_weight == None else initial_weight
		self.mother = inp
		self.child = out
		self.charge = None
		self.activation = function

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
datagrey = []
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
		datagrey.append([line[4],line[4],line[4]])
print len(data)
CreateImage(output_data)
CreateImage(datagrey, name='other',x=641,y=361)
Network = FullyConnectedNetwork()
training_data,leftover_data = Network.TrainAll(input_data,output_data, fraction=.8)
Network.PrintWeights()
#for i in range(10):
#	for i in range(len(input_data)):
#		Network.Train(input_data[i],output_data[i])
all_errors = Network.Analyze(leftover_data,input_data,output_data)

colors = np.zeros((len(input_data),3))
for i in range(len(input_data)):
	Network.TakeInput(input_data[i])
	colors[i] = np.multiply(Network.GetOutput(),256)
reds = []
greens = []
blues = []
for i in colors:
	reds.append(i[0])
	greens.append(i[1])
	blues.append(i[2])
plt.hist(reds,bins=50)
plt.savefig('test.png')
plt.close()
plt.hist(greens,bins=50)
plt.savefig('testgreen.png')
plt.close()
plt.hist(blues,bins=50)
plt.savefig('testblue.png')
plt.close()
CreateImage(colors,name='noshift')
shifts = np.sum(all_errors,axis=0)
shifts = np.divide(shifts,float(len(all_errors))/256)
prev_colors = colors
colors = np.subtract(colors,shifts)
CreateImage(colors,name='reconstructed')

data_color = []
csv = ''
for i in data:
	Network.TakeInput(i)
	new_data = np.multiply(Network.GetOutput(),256)
	data_color.append(new_data)
	line = ''
	for point in new_data:
		line+=str(point)+','
	print line[:-1]
	csv += str(line[:-1])+'\n'
fil = open('final.csv','w')
fil.write(csv)
fil.close()
	
CreateImage(data_color,name='data',x=641,y=361)

best = BestKmeans(output_data,[3,7])
colors = kmeans(output_data,nclusters=best)[0]

ncolors = len(colors)
Networks = FullyConnectedNetwork(error_function=Class,clusters =colors)

for i in range(len(input_data)):
	Networks.Train(input_data[i],output_data[i])

data_color = []
for i in data:
	Networks.TakeInput(i)
	print Networks.PrintOutput()
	data_color.append(np.multiply(colors[BestCluster(Networks.GetOutput(),colors)],256))
CreateImage(data_color,name='data_other',x=641,y=361)
exit()









clusters = kmeans(map(lambda x: input_data[x],training_data),nclusters = 4)[0]
cluster_data = []
for c in clusters:
	cluster_data.append([])

#cluster_indices = Network.ClusterOnError(clusters,training_data,input_data,output_data)
Networks = []
for c in range(len(cluster_indices)):
	Networks.append(FullyConnectedNetwork())
	for i in cluster_indices[c]:
		Networks[c].Train(input_data[i],output_data[i])
	Networks[c].PrintWeights()

cluster_indices = Network.ClusterOnError(clusters,leftover_data,input_data,output_data)
for c in cluster_indices:
	Networks[c].Analyze(cluster_indices[c],input_data,output_data,section=c)





#print len(input_data)		
#print len(output_data)
#CreateImage(output_data)
#kmeans_errors = []
#clusters = {}
#for i in range(1,2):
#	results = kmeans(input_data,nclusters=i)
#	clusters[i] = results[0]
#	kmeans_errors.append(results[1])
#plt.plot(kmeans_errors)
#plt.savefig('kmeanserrors.png')
#plt.close()
#print kmeans_errors
#for i in range(1,len(kmeans_errors)):
#	if float(kmeans_errors[i]-kmeans_errors[i-1])/kmeans_errors[i] < .15:
#		nclusters = i 
########
#nclusters = 1
########
#selected_clusters = clusters[nclusters]
#training_data = []
#testing_data = []
#for c in range(nclusters):
#	training_data.append([])
#	testing_data.append([])
#num_outputs = 3 
#for i in range(len(input_data)):
#	data_errors = map(lambda x: distance(x,input_data[i]),selected_clusters)
#	c = data_errors.index(min(data_errors))
#	if random.random() < .8:
#		training_data[c].append(i)
#	else:
#		testing_data[c].append(i)
#
#### train networks
#Networks = []
#for c in range(len(training_data)):
#	Networks.append(FullyConnectedNetwork())
#	for i in training_data[c]:
#		Networks[c].Train(input_data[i],output_data[i][:num_outputs])
#	Networks[c].PrintWeights()
#
#
#
##### test networks on testing data
#names= ['red','green','blue']
#all_errors = []
#red_error = []
#green_error = []
#blue_error = []
#for c in range(len(testing_data)):
#	all_errors.append( [[],[],[]])
#	for i in testing_data[c]:
#		Networks[c].TakeInput(input_data[i])
#		errors = np.subtract(Networks[c].GetOutput(),output_data[i][:num_outputs])
#		for j in range(num_outputs):
#			all_errors[c][j].append(errors[j])
#
#print map(lambda x: np.divide( np.sum(x,axis=1),float(len(x[0]) )),all_errors )
#new_all = []
#for c in range(len(all_errors[0])):
#	new_list = []
#	for i in range(len(all_errors)):
#		new_list += all_errors[i][c]
#	new_all.append(new_list)
#all_errors = new_all
#clusters = kmeans(all_errors,nclusters = 4)[0]
#SecondIterationNetworks = []
#for cluster in clusters:
#	SecondIterationNetworks.append(FullyConnectedNetwork())
#
#for c in range(len(testing_data)):
#	for i in testing_data[c]:
#		
#
##### create training image
#colors = np.zeros((len(input_data),3))
#for c in range(len(testing_data)):
#	for i in testing_data[c]+training_data[c]:
#		Networks[c].TakeInput(input_data[i])
#		colors[i] = np.multiply(Networks[c].GetOutput(),256)
#shifts = map(lambda x: float(sum(x))/len(x),all_errors)
#colors = np.subtract(colors,np.multiply(shifts,256))
#CreateImage(colors,name='reconstructed')
#
#
#for i in range(len(shifts)):
#	all_errors[i] =  np.subtract(all_errors[i],shifts[i])
#print 'plotting'
#for j in range(num_outputs):
#	plt.hist(all_errors[j],bins = 50)
#	plt.savefig(names[j]+'.png')
#	plt.close()
##Network.PrintWeights()


