with open('./data_files/mxm_dataset_test.txt') as f:
    for line in f:
        l = line.strip()
        if l[0] == '%':
            tokens = l.split(',')
            print(len(tokens))
            exit()