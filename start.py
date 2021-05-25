from JM_TMB_AGE import Run_1

import sys;

if __name__ == "__main__":
    #get file name
    data_file = "melanoma.xlsx"
    N = 10
    second = False
    for i in range(1, len(sys.argv)):
        args = sys.argv[i].split("=", 1)
        if args[0] == "data":
            data_file = args[1]
        elif args[0] == "N":
            N = int(args[1])
        elif args[0] == "second":
            if args[1] == "true":
                second = True
    
    print("Data file from: " + data_file)
    print("Dimension of U and A: " + str(N))
    Run_1("data/" + data_file, N, second)
