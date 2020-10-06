
#13. Roman to Integer





class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        dic = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
        n = len(s)
        i=0
        total = dic[s[n-1]]
       
        for i in range(0,n-1):
            current = dic[s[i+1]]
            prev = dic[s[i]]
            if prev >= current:
                  total += prev 
                
            else:
                total-= prev
        
        return total
