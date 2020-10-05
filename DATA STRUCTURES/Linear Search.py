#Linear search

def linear_search(arr,x):
     for i in range(len(arr)):
         if arr[i]==x:
            result =i
            print("Element found at index  ", str(result))
     else:
       print("Element not found!")
    
 

#Test array

arr= [4,2,4,1,19,18]
x=0

#Driver function
result = linear_search(arr,x)
