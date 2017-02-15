fptr = open("mnist_large.txt", 'r')

collst = []

line = fptr.readline()
collst = line.split(" ")

print len(collst)
	
fptr.close()
