# 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。
#
# 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
#
# 你可以按任意顺序返回答案。
#
# 只会存在一个有效答案

def twoSum( nums, target) :
    for i in range(len(nums)):
        for j in range(len(nums)):
            # 同一个位置的数字不能使用两次，且和要等于target
            if i != j and nums[i] + nums[j] == target:
                return [i, j]

if __name__ == '__main__ ':
    # 输入字符串
    nums_str=input()
    # 转为int型列表
    nums = [int(x) for x in nums_str[1:-1].split(',')]
    # 输入target
    target = int(input())
    # 调用类方法
    result = twoSum(nums,target)
    # 输出结果
    print(result)