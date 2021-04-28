#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 将给定数组排序
# @param arr int整型一维数组 待排序的数组
# @return int整型一维数组
#
class Solution:
    def MySort(self , arr ):
        # write code here
        self.quicksort(arr,0,len(arr)-1)
        return arr
    def quicksort(self,arr,l,r):
        if l<r:
            point = self.partition(arr,l,r)
            self.quicksort(arr,l,point-1)
            self.quicksort(arr,point+1,r)
    def partition(self,arr,l,r):
        start = l
        first = arr[l]
        while l<r:
            while l<r and arr[r]>=first:
                r-=1
            while l<r and arr[l]<=first:
                l+=1
            arr[l],arr[r]=arr[r],arr[l]
        arr[l],arr[start]=arr[start],arr[l]
        return l