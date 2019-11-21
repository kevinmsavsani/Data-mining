def parse_file(filepath):

    data = []  # create an empty list to collect the data
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        i = 0
        file1 = open("myfile.txt","w")#append mode 
        while line:

            number = line.strip().split(' ')
            print(number)
            listToStr = ' '.join([str(elem) for elem in number]) 
            file1.write(listToStr) 

            line = file_object.readline()
            i += 1
            if i > 100:
            	return 0

    file1.close()


if __name__ == '__main__':
    filepath = 'train.txt'
    parse_file(filepath)
