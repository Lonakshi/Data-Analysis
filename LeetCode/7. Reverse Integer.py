#Given a 32-bit signed integer, reverse digits of an integer.


#Note:
#Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.

 

7. Reverse Integer






class Solution:
    def reverse(self, x: int) -> int:
            
            #if x is negative first we will make it positive
            flag = False
            if(x<0):
                x=0-x
                flag = True
            
             
            rev_num=0
            while(x>0):
              
                rev_num = rev_num*10 + x %10
                x= x//10
                    
            #we will make the result negative if x was negative.
            if(flag):
                rev_num = 0- rev_num
            
            #for the overflow cases as it should only reverse integers not long
            if rev_num >= 2** 31 -1 or rev_num <= -2** 31: 
               
               rev_num= 0 
            
            
            
            return rev_num
        
