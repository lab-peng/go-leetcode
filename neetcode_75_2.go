package main

import (
    "math"
    "Slices"
)

// 104. Maximum Depth of Binary Tree
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    return 1 + max(maxDepth(root.Left), maxDepth(root.Right))
}

// 100. Same Tree
func isSameTree(p *TreeNode, q *TreeNode) bool {
    if p == nil && q == nil {
        return true
    }
    return (p != nil && q != nil && p.Val == q.Val) && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

// 572. Subtree of Another Tree
func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
    if root == nil {
        return false
    }
    if isSametree(root, subRoot) {
        return true
    }
    return isSubtree(root.Left, subRoot) || isSubtree(root.Right, subRoot)
}

func isSametree(p *TreeNode, q *TreeNode) bool {
    if p == nil && q == nil {
        return true
    }
    return (p != nil && q != nil && p.Val == q.Val) && isSametree(p.Left, q.Left) && isSametree(p.Right, q.Right)
}

// 235. Lowest Common Ancestor of a Binary Search Tree
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if p.Val < root.Val && q.Val < root.Val {
        return lowestCommonAncestor(root.Left, p, q)
    }
    if p.Val > root.Val && q.Val > root.Val {
        return lowestCommonAncestor(root.Right, p, q)
    }
    return root
}

// 102. Binary Tree Level Order Traversal
func levelOrder(root *TreeNode) [][]int {
    ans := make([][]int, 0)

    var q []*TreeNode
    if root != nil {
        q = append(q, root)
    }
    for len(q) > 0 {
        lvl := make([]int, 0)
        l := len(q)
        for i := 0; i < l; i++ {
            crt := q[0]    
            q = q[1:]
            if crt != nil {
                lvl = append(lvl, crt.Val)
                if crt.Left != nil {
                    q = append(q, crt.Left)
                }
                if crt.Right != nil {
                    q = append(q, crt.Right)
                }
            }
        }
        ans = append(ans, lvl)
    }
    return ans
}

// 98. Validate Binary Search Tree
func isValidBST(root *TreeNode) bool {
    mn := math.MinInt64
    mx := math.MaxInt64
    return isValidBSTHelper(mn, root, mx)
}


func isValidBSTHelper(lower int, root *TreeNode, upper int) bool {
    if root == nil {
        return true
    }
    if !(lower < root.Val && root.Val < upper) {
        return false
    }
    return isValidBSTHelper(lower, root.Left, root.Val) && isValidBSTHelper(root.Val, root.Right, upper)
}

// 230. Kth Smallest Element in a BST
func walk(root *TreeNode, c chan int) {
	if root == nil {
		return
	}

	walk(root.Left, c)
	c <- root.Val
	walk(root.Right, c)
}


func kthSmallest(root *TreeNode, k int) int {
	c := make(chan int)
	go walk(root, c)

	for i := 0; i < k-1; i++ {
		<-c
	}
	return <-c
}

// 105. Construct Binary Tree from Preorder and Inorder Traversal
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }
    m := slices.Index(inorder, preorder[0])
    return &TreeNode{
        Val: preorder[0],
        Left: buildTree(preorder[1: m + 1], inorder[:m]),
        Right: buildTree(preorder[m + 1:], inorder[m + 1:]),
    }
}

// 124. Binary Tree Maximum Path Sum

// 297. Serialize and Deserialize Binary Tree

// 208. Implement Trie (Prefix Tree)

// 211. Design Add and Search Words Data Structure

// 212. Word Search II

// 295. Find Median from Data Stream

// 39. Combination Sum
func combinationSum(candidates []int, target int) [][]int {
	ans := make([][]int, 0)
	var bt func(i int, crt int, vals []int)
    bt = func(i int, crt int, vals []int) {
        if crt == target {
            cpy := make([]int, len(vals))
            copy(cpy, vals)
            ans = append(ans, cpy)
            return
        }
        if crt > target || i == len(candidates) {
            return
        }
        vals = append(vals, candidates[i])
        bt(i, crt + candidates[i], vals)
        vals = vals[:len(vals) - 1]
        bt(i + 1, crt, vals)
    }
    bt(0, 0, make([]int, 0))
	return ans
}

// 79. Word Search
func exist(board [][]byte, word string) bool {
    rows, cols := len(board), len(board[0])
    var btk func(int, int, int) bool
    btk = func(r, c int, i int)  bool {
        if i == len(word) {
            return true
        }
        if r < 0 || r == rows || c < 0 || c == cols || board[r][c] != word[i] {
            return false
        }

        tmp := board[r][c]
        board[r][c] = '#'
        if btk(r + 1, c, i + 1) || btk(r - 1, c, i + 1) || btk(r, c + 1, i + 1) || btk(r, c - 1, i + 1) {
            return true
        }
        board[r][c] = tmp
        
        return false
    }

    for r := 0; r < rows; r++ {
        for c := 0; c < cols; c++ {
            if btk(r, c, 0){
                return true
            }
        }
    }
    return false
}

// 200. Number of Islands
func numIslands(grid [][]byte) int {
    ans := 0
    for r := range grid {
        for c := range grid[0] {
            if grid[r][c] == '1' {
                dfs(grid, r, c)
                ans++
            }
        }
    }
    return ans
}

func dfs(grid [][]byte, r int, c int) {
    rows, cols := len(grid), len(grid[0])
    if !(0 <= r && r < rows) || !(0 <= c && c < cols) || grid[r][c] == '0' {
        return
    }
    grid[r][c] = '0'

    dfs(grid, r + 1, c)
    dfs(grid, r - 1, c)
    dfs(grid, r, c + 1)
    dfs(grid, r, c - 1)
}

// 133. Clone Graph
type Node struct {
    Val int
    Neighbors []*Node
}

func cloneGraph(node *Node) *Node {
    if node == nil {
        return nil
    }

    visited := make(map[int]*Node)
    visited[node.Val] = &Node{Val: node.Val}
    q := []*Node{node}

    for len(q) > 0 {
        crt := q[0]
        q = q[1:]
        
        for _, nbr := range crt.Neighbors {
            if visited[nbr.Val] == nil {
                visited[nbr.Val] = &Node{Val: nbr.Val}
                q = append(q, nbr)
            }
            visited[crt.Val].Neighbors = append(visited[crt.Val].Neighbors, visited[nbr.Val])
        }
    }
    return visited[node.Val]
}

// 417. Pacific Atlantic Water Flow
func pacificAtlantic(heights [][]int) [][]int {
    ans := make([][]int, 0)
		rows, cols := len(heights), len(heights[0])
		p, a := make([][]bool, rows), make([][]bool, rows)
		for r := 0; r < rows; r++ {
			p[r] = make([]bool, cols)
			a[r] = make([]bool, cols)
		}

		var dfs func(r int, c int, prev int, ocean [][]bool)
		dfs = func(r int, c int, prev int, ocean [][]bool) {
			if r < 0 || c < 0 || r == rows || c == cols || heights[r][c] < prev || ocean[r][c] {
				return
			}
			ocean[r][c] = true
			dfs(r + 1, c, heights[r][c], ocean)
			dfs(r - 1, c, heights[r][c], ocean)
			dfs(r, c + 1, heights[r][c], ocean)
			dfs(r, c - 1, heights[r][c], ocean) 
		}

		for r := 0; r < rows; r++ {
			dfs(r, 0, -1, p)
			dfs(r, cols - 1, -1, a)
		}

		for c := 0; c < cols; c++ {
			dfs(0, c, -1, p)
			dfs(rows - 1, c, -1, a)
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if p[r][c] && a[r][c] {
					ans = append(ans, []int{r, c})
				}
			}
		}
		return ans
}

// 207. Course Schedule
func canFinish(numCourses int, prerequisites [][]int) bool {
    graph := make(map[int][]int)
    for _, pre := range prerequisites {
        graph[pre[1]] = append(graph[pre[1]], pre[0])
    }

    visited := make(map[int]int)

    var hasCycle func(c int) bool
    hasCycle = func(c int) bool {
        if visited[c] != 0 {
            return visited[c] == 1
        }
        visited[c] = 1
        for _, nbr := range graph[c] {
                if hasCycle(nbr) {
                    return true
                }
        }
        visited[c] = -1
        return false
        }
        for c := 0; c < numCourses; c++ {
            if hasCycle(c) {
                return false
            }
    }

    return true
}

// 323. Number of Connected Components in an Undirected Graph
func countComponents(n int, edges [][]int) int {
    graph := make(map[int][]int)
    for _, e := range edges {
        graph[e[1]] = append(graph[e[1]], e[0])
        graph[e[0]] = append(graph[e[0]], e[1])
    }    

    ans := 0
    nodes := make([]int, n)
    for i := 0; i < n; i++ {
        nodes[i] = i
    }
    var dfs func(node int) 
    dfs = func(node int) {
        if nodes[node] == -1 {
            return
        }
        nodes[node] = -1

        for _, nbr := range graph[node] {
            dfs(nbr)
        }
    }

    for _, node := range nodes {
        if node != -1 {
            dfs(node)
            ans += 1
        }
    }
    return ans
}

// 261. Graph Valid Tree
func validTree(n int, edges [][]int) bool {
    graph := make(map[int][]int)
    for _, e := range edges {
        graph[e[0]] = append(graph[e[0]], e[1])
        graph[e[1]] = append(graph[e[1]], e[0])
    }
    visited := make(map[int]bool)
    var hasCycle func(child int, parent int) bool
    hasCycle = func(child int, parent int) bool {
        if visited[child] {
            return true
        }
        visited[child] = true
        for _, nbr := range graph[child] {
            if nbr == parent {
                continue
            }
            if hasCycle(nbr, child) {
                return true
            }
        }
        return false
    }
    return !hasCycle(0, -1) && len(visited) == n
}

// 269. Alien Dictionary
func alienOrder(words []string) string {
    graph := make(map[byte][]byte)
    for _, word := range words {
        for i := range word {
            graph[word[i]] = make([]byte, 0)
        }
    }

    for i := 0; i < len(words) - 1; i++ {
        l, r := words[i], words[i + 1]
        lln, rln := len(l), len(r)
        mn := min(lln, rln)
        if lln > rln && l[:mn] == r[:mn] {
            return ""
        }
        for j := 0; j < mn; j++ {
            if l[j] != r[j] {
                graph[l[j]] = append(graph[l[j]], r[j])
                break
            }
        }
    }
    
    ans, visited := make([]byte, 0), make(map[byte]int)
    var hasCycle func(c byte) bool 
    hasCycle = func(c byte) bool {
        if visited[c] != 0 {
            return visited[c] == 1
        }
        visited[c] = 1
        for _, nbr := range graph[c] {
            if hasCycle(nbr) {
                return true
            }
        }
        visited[c] = -1
        ans = append(ans, c)
        return false
    }

    for k := range graph {
        if hasCycle(k) {
            return ""
        }
    }

    for i, j := 0, len(ans) - 1; i < j; i, j = i + 1, j - 1 {
        ans[i], ans[j] = ans[j], ans[i]
    }
    
    return string(ans[:])
}


// 70. Climbing Stairs
func climbStairs(n int) int {
    t := make([]int, n + 1)
    t[0], t[1] = 1, 1
    for i := 2; i < n + 1; i++ {
        t[i] = t[i - 1] + t[i - 2]
    }
    return t[n] 
}

// 198. House Robber
func rob1(nums []int) int {
    n := len(nums)
    if n >= 2 {
        nums[1] = max(nums[0], nums[1])
    }
    for i := 2; i < n; i++ {
        nums[i] = max(nums[i] + nums[i - 2], nums[i - 1])
    }
    return nums[n - 1]
}