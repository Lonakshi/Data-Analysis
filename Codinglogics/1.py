def pypart(n):

    for i in range(0, n):

        for j in range(0, (n-1)-i):

            print(" ", end = "")

        for j in range(0, i+1):

            print("*", end="")

        print("\r")


n=5
pypart(n)

            
