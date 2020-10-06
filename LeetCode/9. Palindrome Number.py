#Determine whether an integer is a palindrome.
#An integer is a palindrome when it reads the same backward as forward.

#Follow up: Could you solve it without converting the integer to a string


#9. Palindrome Number

class Solution:
    def isPalindrome(self, x: int) -> bool:
         
        pal=0
        y=x 
        while x > 0 :
            
            pal = (pal*10) + (x%10)
            x=x//10
             
       
       
        
        
        if pal==y:
            return True
        
        else: 
            return False
