def binary_search(arr, x):

   low=0
   high= len(arr)
   mid=0  

   while low <high:
       mid = (high+low)//2

#if x is greater than mid, then we can ignore left subarray can be ignored.

       if arr[mid] < x:
           low= mid+1

       elif arr[mid]> x:
          high = mid-1

       else:
         return mid


   else:
      return -1

#Test array

arr= [1,4,7,10,14]

x=14

#Function call

result = binary_search(arr,x)

if result!=-1:
    print("Element is found at index ",str(result))
else:
    print("Element not found")

    
    
