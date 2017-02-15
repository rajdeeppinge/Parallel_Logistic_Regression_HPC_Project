fptr = open("mnist_large.dat", 'r')

fnew = open("mnist_large.txt", 'w')
#fobj = fptr.read()

for line in fptr:
	line = " ".join(line.split(","))
	fnew.write("1 " + line)
	
fptr.close()
fnew.close()
