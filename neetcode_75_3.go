package main

import (
    "sort"
    "math"
)

// 213. House Robber II
func rob(nums []int) int {
    ln := len(nums)
    if ln == 1 {
        return nums[0]
    }    
    var rob func(nums []int) int
    rob = func(nums []int) int {
        rob, notRob := 0, 0
        for _, n := range nums {
            rob, notRob = notRob + n, max(rob, notRob) 
        }
        return max(rob, notRob)
    }
    return max(rob(nums[1:]), rob(nums[:ln - 1]))
}

// 5. Longest Palindromic Substring
func longestPalindrome(s string) string {
    start, end := 0, 0
    n := len(s)

    m := make([][]int, n)
    for i := 0; i < n; i++ {
        m[i] = make([]int, n)
    }

    for d := 0; d < n; d++ {
        for r := 0; r < n - d; r++ {
            c := r + d
            if r == c {
                m[r][c] = 1
                start, end = r, c
            } else if s[r] == s[c] && c - r == 1 {
                m[r][c] = 1
                start, end = r, c
            } else if s[r] == s[c] && m[r + 1][c - 1] == 1 {
                m[r][c] = 1
                start, end = r, c
            }
        }
    }
    return s[start: end + 1]
}

// 647. Palindromic Substrings
func countSubstrings(s string) int {
    ans := 0
    n := len(s)

    m := make([][]int, n)
    for i := 0; i < n; i++ {
        m[i] = make([]int, n)
    }

    for d := 0; d < n; d++ {
        for r := 0; r < n - d; r++ {
            c := r + d
            if r == c {
                m[r][c] = 1
                ans += 1
            } else if s[r] == s[c] && c - r == 1 {
                m[r][c] = 1
                ans += 1
            } else if s[r] == s[c] && m[r + 1][c - 1] == 1 {
                m[r][c] = 1
                ans += 1
            }
        }
    }
    return ans
}

// 91. Decode Ways
func numDecodings(s string) int {
    if s[0] == '0' {
        return 0
    }
    n := len(s) + 1
    t := make([]int, n)
    t[0], t[1] = 1, 1
    for i := 2; i < n; i++ {
        if s[i - 1] != '0' {
            t[i] += t[i - 1]
        }
        s2 := s[i - 2: i]
        if "10" <= s2 && s2 <= "26" {
            t[i] += t[i - 2]
        }
    }
    return t[n - 1]
}

// 322. Coin Change
func coinChange(coins []int, amount int) int {
    n := amount + 1
    t := make([]int, n)
    for i := range t {
        t[i] = n
    }
    t[0] = 0
    for i := range t {
        for _, c := range coins {
            if i >= c {
                t[i] = min(t[i], 1 + t[i - c])
            }
        }
    }
    if t[n - 1] == n {
        return -1
    }
    return t[n - 1]
}

// 152. Maximum Product Subarray
func maxProduct(nums []int) int {
    mx, prevMn, prevMx := math.MinInt, 1, 1
    for _, n := range nums {
        crtMn := min(n, n * prevMn, n * prevMx)
        crtMx := max(n, n * prevMn, n * prevMx)
        mx = max(mx, crtMx)

        prevMn = crtMn
        prevMx = crtMx
    }
    return mx
}

// 139. Word Break
func wordBreak(s string, wordDict []string) bool {
    n := len(s) + 1
    t := make([]bool, n)
    t[0] = true

    for i := 0; i < n; i++ {
        if t[i] {
            for _, word := range wordDict {
                ln := len(word)
                if i + ln < n && s[i: i + ln] == word {
                    t[i + ln] = true
                    if t[n - 1] {
                        return true
                    }
                }
            }
        }
    }
    return false
}

// 300. Longest Increasing Subsequence
func lengthOfLIS(nums []int) int {
    ans := make([]int, 0)
		for _, n := range nums {
			ln := len(ans)
			if ln == 0 || ans[ln - 1] < n {
				ans = append(ans, n)
			} else {
				l, r := 0, ln - 1
				for l < r {
					m := (l + r) / 2
					if ans[m] < n {
						l = m + 1
					} else {
						r = m
					}
				}
				ans[r] = n
			}
		}
		return len(ans)
}

// 62. Unique Paths
func uniquePaths(m int, n int) int {
    mat := make([][]int, m)
    for r := 0; r < m; r++ {
        s := make([]int, n)
        s[0] = 1
        mat[r] = s
    }
    for c := 0; c < n; c++ {
        mat[0][c] = 1
    }

    for r := 1; r < m; r++ {
        for c := 1; c < n; c++ {
            mat[r][c] = mat[r - 1][c] + mat[r][c - 1]
        }
    }
    
    return mat[m - 1][n - 1]
}

// 1143. Longest Common Subsequence
func longestCommonSubsequence(text1 string, text2 string) int {
    rows, cols := len(text1) + 1, len(text2) + 1
    m := make([][]int, rows)
    for i := 0; i < rows; i++ {
        m[i] = make([]int, cols)
    }
    for r := 1; r < rows; r++ {
        for c := 1; c < cols; c++ {
            if text1[r - 1] == text2[c - 1] {
                m[r][c] = 1 + m[r - 1][c - 1]
            } else {
                m[r][c] = max(m[r - 1][c], m[r][c - 1])
            }
        }
    }
    return m[rows - 1][cols - 1]
}

// 53. Maximum Subarray
func maxSubArray(nums []int) int {
    mx := nums[0]
    for i := 1; i < len(nums); i++ {
        nums[i] = max(nums[i], nums[i] + nums[i - 1])
        mx = max(mx, nums[i])
    }
    return mx;
}

// 55. Jump Game
func canJump(nums []int) bool {
    n := len(nums)
    ptr := n - 1
    for i := n - 2; i > -1; i-- {
        if i + nums[i] >= ptr {
            ptr = i 
        }
    }
    return ptr == 0
}

// 57. Insert Interval
func insert(intervals [][]int, newInterval []int) [][]int {
    ans := make([][]int, 0)
    for i := 0; i < len(intervals); i++ {
        if newInterval[1] < intervals[i][0] {
            ans = append(ans, newInterval)
            ans = append(ans, intervals[i:]...)
            return ans
        } else if intervals[i][1] < newInterval[0] {
            ans = append(ans, intervals[i])
        } else {
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
        }
    }
    ans = append(ans, newInterval)
    return ans
}

// 56. Merge Intervals
func mergeIntervals(intervals [][]int) [][]int {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    ans := intervals[:1]
    for _, itvl := range intervals {
        last := ans[len(ans) - 1]
        if last[1] >= itvl[0] {
            last[1] = max(last[1], itvl[1])
        } else {
            ans = append(ans, itvl)
        }
    }
    return ans
}

// 435. Non-overlapping Intervals
func eraseOverlapIntervals(intervals [][]int) int {
    sort.Slice(intervals, func(i int, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    ans := 0
    lastEnd := intervals[0][1]
    for _, itvl := range intervals[1:] {
        if lastEnd > itvl[0] {
            lastEnd = min(lastEnd, itvl[1])
            ans++
        } else {
            lastEnd = itvl[1]
        }
    }
    return ans
}

// 252. Meeting Rooms
func canAttendMeetings(intervals [][]int) bool {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    for i := 1; i < len(intervals); i++ {
        if intervals[i - 1][1] > intervals[i][0] {
            return false
        }
    }
    return true
}

// 253. Meeting Rooms II
func minMeetingRooms(intervals [][]int) int {
    times := make([][]int, 0)
    for _, itvl := range intervals {
        times = append(times, []int{itvl[0], 1})
        times = append(times, []int{itvl[1], -1})
    }
    sort.Slice(times, func(i int, j int) bool {
        if times[i][0] == times[j][0] {
            return times[i][1] < times[j][1]
        }
        return times[i][0] < times[j][0]
    })
    mx, cnt := 0, 0
    for _, t := range times {
        cnt += t[1]
        mx = max(mx, cnt)
    }
    return mx
}

// 48. Rotate Image
func rotate(matrix [][]int)  {
    n := len(matrix)
    for r := 0; r < n; r++ {
        for c := 0; c < r; c++ {
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        }
    }

    for r := 0; r < n; r++ {
        for c := 0; c < n / 2; c++ {
            matrix[r][c], matrix[r][n - c - 1] = matrix[r][n - c - 1], matrix[r][c]
        }
    }
}

// 54. Spiral Matrix
func spiralOrder(matrix [][]int) []int {
    rT, rB := 0, len(matrix)
    cL, cR := 0, len(matrix[0])
    ans := make([]int, 0)

    for rT < rB && cL < cR {
        for c := cL; c < cR; c++ {
            ans = append(ans, matrix[rT][c])
        }
        rT += 1

        for r:= rT; r < rB; r++ {
            ans = append(ans, matrix[r][cR - 1])
        }
        cR -= 1

        if rT < rB {
            for c := cR - 1; c > cL - 1; c-- {
                ans = append(ans, matrix[rB - 1][c])
            }
        }
        rB -= 1

        if cL < cR {
            for r := rB - 1; r > rT - 1; r-- {
                ans = append(ans, matrix[r][cL])
            }
        }
        cL += 1
    }
    return ans
}

// 73. Set Matrix Zeroes
func setZeroes(matrix [][]int)  {
    zeroes := make([][]int, 0)
    rows, cols := len(matrix), len(matrix[0])
    for r := 0; r < rows; r++ {
        for c := 0; c < cols; c++ {
            if matrix[r][c] == 0 {
                zeroes = append(zeroes, []int{r, c})
            }
        }
    }
    zeroRows, zeroCols := make([]int, 0), make([]int, 0)
    for _, zero := range zeroes {
        zeroRows = append(zeroRows, zero[0])
        zeroCols = append(zeroCols, zero[1])
    }

    for r := 0; r < rows; r++ {
        for _, c := range zeroCols {
            matrix[r][c] = 0
        }
    }

    for _, r := range zeroRows {
        for c := 0; c < cols; c++ {
            matrix[r][c] = 0
        }
    }
}

// 191. Number of 1 Bits
func hammingWeight(num uint32) int {
    cnt := 0
    for num != 0 {
        num &= num - 1
        cnt += 1
    }
    return cnt
}

// 338. Counting Bits
func countBits(n int) []int {
    t := make([]int, n + 1)
    bit := 1
    for i := 1; i < n + 1; i++ {
        if bit * 2 == i {
            bit = i
        }
        t[i] = 1 + t[i - bit]
    }
    return t
}

// 190. Reverse Bits
func reverseBits(num uint32) uint32 {
    var ans uint32
    for i := 0; i < 32; i++ {
        ans <<= 1
        ans ^= num & 1
        num >>= 1
    }
    return ans
}

// 268. Missing Number
func missingNumber(nums []int) int {
    ans := len(nums)
    for i, v := range nums {
        ans ^= i ^ v
    }
    return ans
}

// 371. Sum of Two Integers
func getSum(a int, b int) int {
    for b != 0 {
        carry := (a & b) << 1
        a ^= b
        b = carry 
    }
    return a
}