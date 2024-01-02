package main

import (
	"fmt"
	"sort"
	"strings"
)

func main() {
	// 🚀 Beats 100.00% of users with Go
	fmt.Println(canJump([]int{3, 2, 1, 0, 4}))
	unsorted_str := "ghfisaw"
	// chars := []rune(unsorted_str)
	chars := strings.Split(unsorted_str, "")
	fmt.Println(chars)
	sort.Slice(chars, func(i, j int) bool { 
	   return chars[i] < chars[j]
	})
	fmt.Println(chars)

	for _, c := range unsorted_str {
		fmt.Println(c - 'a')
		break
	}

	// hm := make(map[int]bool)
	// hm[1] = true
	// fmt.Println(hm[1])
	// fmt.Println(hm[0])

	// hm1 := make(map[int]int)
	// hm1[1] = 0
	// hm1[2] = 1
	// fmt.Println(hm1[1])
	// fmt.Println(hm1[2])
	// fmt.Println(hm1[3])

	// s := "01234"
	// fmt.Println(s[0])
	// fmt.Println(s[0] == '0', s[0] == 48)
	// fmt.Println(s[:4])

	// s := []int{2, 4, 0, 1, 9}
	// sort.Slice(s, func(i int, j int) bool {
	// 	return s[i] < s[j]
	// })
	// fmt.Println(s)

	str := "love"
	runes := []rune(str)
	fmt.Println(runes)
	fmt.Println(string(runes))

}




