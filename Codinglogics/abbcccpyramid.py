def pypart(n):
    num=65
    for i in range(0,n):

        for j in range(0, i+1):

          print(chr(num), end= " ")

        print("\r")
        num+=1


n=5
pypart(n)
            
