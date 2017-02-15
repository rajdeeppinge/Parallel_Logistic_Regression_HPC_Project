fxptr = open("X_multiclass.dat", 'r')
fyptr = open("y.dat", 'r')

fxnew = open("X_multi_input.dat", 'w')
fxtest = open("X_multi_test.dat", 'w')

fynew = open("y_multi_input.dat", 'w')
fytest = open("y_multi_test.dat", 'w')

#fobj = fptr.read()
counter = 1

for line in fxptr:
	if counter <= 800 or (counter > 1100 and counter <= 1900) or (counter > 2100 and counter <= 2900) or (counter > 3100 and counter <= 3900) or (counter > 4100 and counter <= 4900):
		fxnew.write(line)
	else:
		fxtest.write(line)
	counter += 1
		
counter = 1

for line in fyptr:
	if counter <= 800 or (counter > 1100 and counter <= 1900) or (counter > 2100 and counter <= 2900) or (counter > 3100 and counter <= 3900) or (counter > 4100 and counter <= 4900) :
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
