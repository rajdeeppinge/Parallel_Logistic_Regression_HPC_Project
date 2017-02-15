fxptr = open("X_bin_update.dat", 'r')
fyptr = open("y_bin.dat", 'r')

fxnew = open("X_bin_input.dat", 'w')
fxtest = open("X_bin_test.dat", 'w')

fynew = open("y_bin_input.dat", 'w')
fytest = open("y_bin_test.dat", 'w')

#fobj = fptr.read()
counter = 1

for line in fxptr:
	if counter <= 800 or counter >= 1100 :
		fxnew.write(line)
	else:
		fxtest.write(line)
	counter += 1
		
counter = 1

for line in fyptr:
	if counter <= 800 or counter >= 1100 :
		fynew.write(line)
	else:
		fytest.write(line)
	counter += 1
	
fxptr.close()
fxnew.close()
fxtest.close()

fyptr.close()
fynew.close()
fytest.close()
