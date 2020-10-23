

##Given the array candies and the integer extraCandies, where candies[i]
#represents the number of candies that the ith kid has.

#For each kid check if there is a way to distribute extraCandies among the kids such that he or she can have the greatest number of candie
#s among them. Notice that multiple kids can have the greatest number of candies.





class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        result= []
        n=len(candies)
        max_number = max(candies)
        sum=0
        k= extraCandies

        for i in range(n):
        
            if candies[i]+k >= max_number:
                result.append(True)
            else:
                result.append(False)
                
        return result
    
