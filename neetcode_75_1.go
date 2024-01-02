package main

import (
    "sort"
    "strings"
    "unicode"
    "math"
)

// 217. Contains Duplicate
func containsDuplicate(nums []int) bool {
    // hm := make(map[int]int)
    // for _, n := range nums {
    //     if hm[n] > 0 {
    //         return true
    //     }
    //     hm[n] += 1
    // }
    // return false

    hs := make(map[int]bool)
    for _, n := range nums {
        if hs[n] {
            return true
        }
        hs[n] = true
    }
    return false
}

// 242. Valid Anagram
func isAnagram(s string, t string) bool {
    if len(s) != len(t) {
        return false 
    }
    ctr := make([]int, 26)
    for i, v := range s {
        // ctr[s[i] - 'a']++
        ctr[v - 'a']++
        ctr[t[i] - 'a']-- 
    }

    for _, c := range ctr {
        if c != 0 {
            return false
        }
    }
    return true
}

// 1. Two Sum
func twoSum(nums []int, target int) []int {
    hm := make(map[int]int) 
    for i, v := range nums {
        if value, ok := hm[target - v]; ok {
            return []int{value, i}
        }
        hm[v] = i
    }
    return []int{-1}
}

// 49. Group Anagrams
func groupAnagrams(strs []string) [][]string {
    m := make(map[string][]string)
    for _, s := range strs {
        runes := []rune(s)
        sort.Slice(runes, func(i, j int) bool { 
            return runes[i] < runes[j]
        })
        key := string(runes)
        m[key] = append(m[key], s)
    }

    ans := make([][]string, 0)
    for _, v := range m {
        ans = append(ans, v)
    }
    
    return ans
}

// 347. Top K Frequent Elements
func topKFrequent(nums []int, k int) []int {
    ctr := make(map[int]int)
    for _, n := range nums {
        ctr[n]++
    }
    
    vec := make([][2]int, len(ctr))
    i := 0
    for k, v := range ctr {
        vec[i][0] = k 
        vec[i][1] = v 
        i++
    }
    sort.Slice(vec, func(i int, j int) bool {
        return vec[i][1] > vec[j][1]
    })

    ans := make([]int, k)
    for i := 0; i < k; i++ {
        ans[i] = vec[i][0]
    }
    return ans
}

// 238. Product of Array Except Self
func productExceptSelf(nums []int) []int {
    n := len(nums)
    ans := make([]int, n)
    for i := 0; i < n; i++ {
        ans[i] = 1
    }

    prefix, suffix := 1, 1
    for i := 0; i < n; i++ {
        ans[i] *= prefix
        prefix *= nums[i]
    }

    for i := n - 1; i > -1; i-- {
        ans[i] *= suffix
        suffix *= nums[i]
    }
    return ans
}

// 271. Encode and Decode Strings
type Codec struct {
    
}

// Encodes a list of strings to a single string.
func (codec *Codec) Encode(strs []string) string {
    return strings.Join(strs, "&$$$$$&")
}

// Decodes a single string to a list of strings.
func (codec *Codec) Decode(strs string) []string {
    return strings.Split(strs, "&$$$$$&")
}

// 128. Longest Consecutive Sequence
func longestConsecutive(nums []int) int {
    hs := make(map[int]bool, len(nums)) // a map with size argument can improve efficiency
    for _, n := range nums {
        hs[n] = true
    }
    mx := 0
    for n := range hs {
        if !hs[n - 1] {
            ln := 1
            for hs[n + ln] {
                ln += 1
            }
            mx = max(mx, ln)
        }
    }
    return mx
}

// 125. Valid Palindrome
func isPalindrome(s string) bool {
    s = strings.Map(func(r rune) rune {
        if !unicode.IsLetter(r) && !unicode.IsNumber(r) {
            return -1
        }
        return unicode.ToLower(r)
    }, s)
    l, r := 0, len(s) - 1
    for l < r {
        if s[l] != s[r] {
            return false
        }
        l++
        r--
    } 
    return true
}

// 15. 3Sum
func threeSum(nums []int) [][]int {
    sort.Slice(nums, func(i, j int) bool {
        return nums[i] < nums[j]
    })
    ans := make([][]int, 0)
    ln := len(nums)

    for i, v := range nums {
        if i > 0 && nums[i - 1] == v {
            continue
        }
        l := i + 1
        r := ln - 1

        for l < r {
            sm := v + nums[l] + nums[r]
            if sm > 0 {
                r -= 1
            } else if sm < 0 {
                l += 1
            } else {
                ans = append(ans, []int{v, nums[l], nums[r]})
                l += 1
                for l < r && nums[l - 1] == nums[l] {
                    l += 1
                }
            }
        }
    }
    return ans
}

// 11. Container With Most Water
func maxArea(height []int) int {
    l, r, mx := 0, len(height) - 1, 0
    for l < r {
        hl, hr := height[l], height[r]
        mx = max(mx, min(hl, hr) * (r - l))
        if hl < hr {
            l++
        } else {
            r--
        }
    }
    return mx
}

// 121. Best Time to Buy and Sell Stock
func maxProfit(prices []int) int {
    l, mx := 0, 0
    for r := range prices {
        pl, pr := prices[l], prices[r]
        if pl >= pr {
            l = r
        }
        mx = max(mx, pr - pl)
    }
    return mx
}

// 3. Longest Substring Without Repeating Characters
func lengthOfLongestSubstring(s string) int {
    // l, mx, hm := 0, 0, make(map[byte]int)
    // for r := range s {
    //     for hm[s[r]] > 0 {
    //         hm[s[l]] -= 1
    //         l += 1
    //     }
    //     hm[s[r]] += 1
    //     mx = max(mx, r - l + 1)
    // }
    // return mx
    l, mx, hm := 0, 0, make(map[rune]int)
    for r, v := range s {
        if value, ok := hm[v]; ok {
            l = max(l, value + 1)
        }
        hm[v] = r
        mx = max(mx, r - l + 1)
    }
    return mx
}

// 424. Longest Repeating Character Replacement
func characterReplacement(s string, k int) int {
    l, mx, hm, n := 0, 0, make(map[uint8]int, len(s)), len(s)
    for r := range s {
        hm[s[r]]++
        mx = max(mx, hm[s[r]])
        if r - l + 1 - mx > k {
            hm[s[l]]--
            l++
        }
    }
    return n - l
}


// Byte is equivalent to uint8. Rune is equivalent to int32. 
// It has to be declared explicitly. It is the default type of a character.

// 76. Minimum Window Substring
func minWindow(s string, t string) string {
    cs, ct := make(map[byte]int), make(map[byte]int)
    for i := range t {
        ct[t[i]]++
    }
    have, need := 0, len(ct)
    mn := math.MaxUint32
    start, end := 0, -1
    l := 0

    for r := range s {
        cs[s[r]]++
        if cs[s[r]] == ct[s[r]] {
            have++
        }

        for have == need {
            if mn > r - l  + 1 {
                mn = r - l + 1
                start = l
                end = r
            }

            cs[s[l]]--
            if cs[s[l]] < ct[s[l]] {
                have--
            }
            l++
        }
    }
    return s[start: end + 1]
}

// 20. Valid Parentheses
func isValid(s string) bool {
    hm := map[rune]rune{
        ')': '(',
        ']': '[',
        '}': '{',
    }
    stack := make([]rune, 0)
    for _, c := range s {
        ln := len(stack)
        if ln > 0 && stack[ln - 1] == hm[c] {
            stack = stack[:ln - 1]
        } else {
            stack = append(stack, c)
        }
    }
    return len(stack) == 0
}

// 🚀 153. Find Minimum in Rotated Sorted Array
func findMin(nums []int) int {
    l, r := 0, len(nums) - 1

    // if nums[l] < nums[r] {
    //     return nums[l]
    // }

    for l < r {
        m := (l + r) / 2
        // if nums[m] < nums[r] then it would be find max 
        if nums[m] > nums[r] {
            l = m + 1
        } else {
            r = m
        }
    }
    return nums[l]
}

// 🚀 33. Search in Rotated Sorted Array
func search(nums []int, target int) int {
    l, r := 0, len(nums) - 1
    for l <= r {
        m := (l + r) / 2
        if target < nums[m] {
            if nums[l] > nums[m] || target >= nums[l] {
                r = m - 1
            } else {
                l = m + 1
            }
        } else if target > nums[m] {
            if nums[m] > nums[r] || target <= nums[r] {
                l = m + 1
            } else {
                r = m - 1
            }
        } else {
            return m
        }
    }
    return -1
}

// 206. Reverse Linked List

type ListNode struct {
    Val int
    Next *ListNode
}
 
 
 func reverseList(head *ListNode) *ListNode {
    crt := head
    var prev *ListNode

    for crt != nil {
        next := crt.Next
        crt.Next = prev
        prev = crt
        crt = next
    }
    return prev
}

// 21. Merge Two Sorted Lists
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummy := &ListNode{}
    crt := dummy
    l, r := list1, list2
    for l != nil && r != nil {
        if l.Val < r.Val {
            crt.Next = l
            l = l.Next
        } else {
            crt.Next = r
            r = r.Next
        }
        crt = crt.Next
    }

    if l != nil {
        crt.Next = l
    }
    if r != nil {
        crt.Next = r
    }
    return dummy.Next
}

// 143. Reorder List
func reorderList(head *ListNode)  {
    s, f := head, head.Next
    for f != nil && f.Next != nil {
        s = s.Next
        f = f.Next.Next
    }
    
    var prev *ListNode
    crt := s.Next
    s.Next = nil
    for crt != nil {
        next := crt.Next
        crt.Next = prev
        prev = crt
        crt = next
    }

    dummy := &ListNode{}
    crt = dummy
    l, r := head, prev
    for l != nil && r != nil {
        crt.Next = l
        l = l.Next
        crt = crt.Next

        crt.Next = r
        r = r.Next
        crt = crt.Next
    }   

    if l != nil {
        crt.Next = l
    }
}

// 19. Remove Nth Node From End of List
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Next: head}
    crt := head
    for n > 0 {
        crt = crt.Next
        n--
    }
    l, r := dummy, crt
    for r != nil {
        l = l.Next
        r = r.Next
    }
    l.Next = l.Next.Next
    return dummy.Next
}

// 141. Linked List Cycle
func hasCycle(head *ListNode) bool {
    s, f := head, head
    for f != nil && f.Next != nil {
        s = s.Next
        f = f.Next.Next
        if s == f {
            return true
        }
    }
    return false
}

// 23. Merge k Sorted Lists
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }

    for len(lists) > 1 {
        merged := make([]*ListNode, 0)
        for i := 0; i < len(lists); i += 2 {
            l := lists[i]
            r := &ListNode{}
            if i + 1 < len(lists) {
                r = lists[i + 1]
            } else {
                r = nil
            }
            merged = append(merged, merge(l, r))
        }
        lists = merged
    }
    return lists[0]
}

func merge(l *ListNode, r *ListNode) *ListNode {
    dummy := &ListNode{}
    crt := dummy
    for l != nil && r != nil {
        if l.Val < r.Val {
            crt.Next = l
            l = l.Next
        } else {
            crt.Next = r
            r = r.Next
        }
        crt = crt.Next
    }

    if l != nil {
        crt.Next = l
    }
    if r != nil {
        crt.Next = r
    }
    return dummy.Next
}


// 226. Invert Binary Tree
type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}

func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    left := root.Left
    root.Left = root.Right
    root.Right = left
    invertTree(root.Left)
    invertTree(root.Right)
    return root
}



