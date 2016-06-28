package middleware;

import java.util.*;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 3/11/16 15:16
 */
public class Ds {

	public static void main(String[] args) {
		int[] nums = new int[4];
		for (int i = 0; i < 4; i++) {

			nums[i] = i;
		}
		System.out.println("nums = " + Arrays.toString(nums));
		permute2(nums);
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> path = new ArrayList<Integer>();
		SortedSet<Long> set = new TreeSet<Long>();
		set.add(-1l);
		System.out.println("+" + set.subSet((long) -1, (long) 0).isEmpty());
		if (!set.subSet((long) -1, (long) -1).isEmpty()) System.out.println("+" + true);
		Integer s = new Integer(1);
		path.add(s);
		path.add(0);
		String string = new String();
		result.add(path);
		System.out.println("Math.pow(n,1/3)%1 =" + (1 - ((Math.pow(64, 1.0 / 3.0)) % 1.0)));
		long nowTime = new Date().getTime();//以前的时间\r
	}

	public List<Integer> largestDivisibleSubset(int[] nums) {
		int[] dp = new int[nums.length];
		Arrays.sort(nums);
		int maxNum = 0, index = 0;
		for (int i = 0; i < nums.length; i++)
			for (int j = 0; j < i; j++)
				if (nums[i] % nums[j] == 0) {
					dp[i] = Math.max(dp[i], dp[j] + 1);
					if (dp[i] > maxNum) {
						maxNum = dp[i];
						index = i;
					}
				}
		List<Integer> result = new ArrayList<>();
		int k = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[index] % nums[i] == 0) result.add(nums[i]);
		}
		return result;
	}

	public int countNumbersWithUniqueDigits(int n) {
		int result = 1;
		int res = 9;
		for (int i = 1; i <= n; i++) {
			result += res;
			res *= 10 - i;
		}
		return result;
	}


	public double myPow(double x, int n) {
		if (n == Integer.MIN_VALUE) //-INT_MIN will cause overflow
			return myPow(x, n + 1) / x;
		if (n < 0) return 1 / myPow(x, -n);
		if (n == 0) return 1;
		double halfResult = myPow(x, n >> 1);
		return halfResult * halfResult * myPow(x, n % 2);
	}

	public double myPow2(double x, int n) {
		if (n == Integer.MIN_VALUE) //-INT_MIN will cause overflow
			return myPow(x, n + 1) / x;
		if (n < 0) return 1 / myPow(x, -n);
		if (n == 0) return 1;
		return n % 2 == 0 ? myPow(x * x, n >> 1) : myPow(x * x, n >> 1) * x;
	}


	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null) return head;
		ListNode slow = head, fast = head;
		while (fast.next != null && fast.next.next != null) {
			fast = fast.next.next;
			slow = slow.next;
		}
		fast = slow.next;//fast成为链表的后半段
		slow.next = null;//head成为链表的前半段
		fast = sortList(fast);
		head = sortList(head);
		/**前后半段链表排序后,再进行合并,等于合并两个有序的链表*/
		return merge(fast, head);
	}

	private ListNode merge(ListNode fast, ListNode head) {
		ListNode result = new ListNode(-1), cur = result;
		while (fast != null && head != null) {
			if (fast.val < head.val) {
				cur.next = fast;
				fast = fast.next;
			} else {
				cur.next = head;
				head = head.next;
			}
			cur = cur.next;
		}
		cur.next = fast == null ? head : fast;
		return result.next;
	}

	private ListNode merge2(ListNode l1, ListNode l2) {
		if (l1 == null) return l2;
		if (l2 == null) return l1;
		if (l1.val <= l2.val) {
			l1.next = merge(l1.next, l2);
			return l1;
		} else {
			l2.next = merge(l1, l2.next);
			return l2;
		}
	}


	public List<List<Integer>> permuteUnique(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		Arrays.sort(nums);
		backtrack(result, new ArrayList<Integer>(), nums, new boolean[nums.length]);
		return result;
	}

	private static void backtrack(List<List<Integer>> result, ArrayList<Integer> tempList, int[]
			nums, boolean[] used) {
		if (tempList.size() == nums.length) {
			result.add(new ArrayList<>(tempList));
		} else {
			for (int i = 0; i < nums.length; i++) {
				/**only insert duplicate element
				 when the previous duplicate element has been inserted*/
				if (used[i] || i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
				used[i] = true;
				tempList.add(nums[i]);
				backtrack(result, tempList, nums, used);
				used[i] = false;
				tempList.remove(tempList.size() - 1);
			}
		}
	}


	public List<List<Integer>> permuteUnique3(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		result.add(new ArrayList<Integer>());
		for (int i = 0; i < nums.length; i++) {
			List<List<Integer>> newres = new ArrayList<>();
			for (List<Integer> k : result) {
				for (int j = 0; j <= k.size(); j++) {
					if (j > 0 && k.get(j - 1) == nums[i]) break;
					List<Integer> list = new ArrayList<>(k);
					list.add(j, nums[i]);
					newres.add(list);
				}
			}
			result = newres;
		}
		return result;
	}

	public List<List<Integer>> permuteUnique2(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		result.add(new ArrayList<Integer>());
		for (int i = 0; i < nums.length; i++) {
			Set<List<Integer>> currentSet = new HashSet<List<Integer>>();
			for (int j = 0; j <= i; j++)
				for (List<Integer> k : result) {
					List<Integer> list = new ArrayList<>(k);
					list.add(j, nums[i]);
					currentSet.add(list);
				}
			result = new ArrayList<>(currentSet);
		}
		return result;
	}


	public List<List<Integer>> permute3(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		perm(result, nums, 0);
		return result;
	}

	public void perm(List<List<Integer>> result, int[] nums, int pos) {
		if (pos == nums.length) {
			List<Integer> list = new ArrayList<Integer>();
			for (int a : nums) list.add(a);
			result.add(list);
			return;
		}
		for (int i = pos; i < nums.length; i++) {
			swap(nums, i, pos);
			perm(result, nums, pos + 1);
			swap(nums, i, pos);
		}
	}

	private void swap(int[] nums, int i, int j) {
		int tmp = nums[i];
		nums[i] = nums[j];
		nums[j] = tmp;
	}

	public static List<List<Integer>> permute2(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		backtrack(result, new ArrayList<Integer>(), nums);
		return result;
	}


	private static void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] nums) {
		if (tempList.size() == nums.length) {
			result.add(new ArrayList<>(tempList));
		} else {
			for (int i = 0; i < nums.length; i++) {
				if (tempList.contains(nums[i])) continue; // element already exists, skip
				tempList.add(nums[i]);
				backtrack(result, tempList, nums);
				tempList.remove(tempList.size() - 1);
			}
		}
	}

	public static List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		result.add(new ArrayList<Integer>());
		for (int i = 0; i < nums.length; i++) {
			List<List<Integer>> newRes = new ArrayList<>();
			for (int j = 0; j <= i; j++)
				for (List<Integer> k : result) {
					List<Integer> list = new ArrayList<>(k);
					list.add(j, nums[i]);
					newRes.add(list);
				}
			result = newRes;
			System.out.println("tmp result = " + result);
		}
		return result;
	}

	public List<List<Integer>> subsetsWithDup2(int[] nums) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> subSets = new ArrayList<Integer>();
		result.add(subSets);
		Arrays.sort(nums);
		int len = 0, start;
		for (int i = 0; i < nums.length; i++) {
			/**nums[i-1]已经和前面的子集并在一起作为subSets了,nums[i]==nums[i-1]就不要再做同样的事情了*/
			if (i != 0 && nums[i] != nums[i - 1]) start = 0;
			else start = len;
			len = result.size();
			for (int j = start; j < len; j++) {
				List<Integer> temp = new ArrayList<>(result.get(j));
				temp.add(nums[i]);
				result.add(temp);
			}
		}
		return result;
	}

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		Arrays.sort(nums);
		getSubsetsWithDup(result, new ArrayList<Integer>(), nums, 0);
		return result;
	}

	private void getSubsetsWithDup(List<List<Integer>> result, List<Integer> list, int[] nums, int start) {
		result.add(new ArrayList<Integer>(list));
		for (int i = start; i < nums.length; i++) {
			if (i > start && nums[i] == nums[i - 1]) continue;
			list.add(nums[i]);
			getSubsets(result, list, nums, i + 1);
			list.remove(list.size() - 1);
		}
	}


	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		int n = nums.length;
		int size = (int) Math.pow(2, n);
		for (int i = 0; i < size; i++) {
			List<Integer> list = new ArrayList<Integer>();
			for (int j = 0; j < n; j++) {
				if ((i >> j & 1) == 1) {
					list.add(nums[j]);
				}
			}
			result.add(list);
		}
		return result;
	}

	public List<List<Integer>> subsets4(int[] nums) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> subSets = new ArrayList<Integer>();
		result.add(subSets);
		Arrays.sort(nums);
		int len;
		for (int i = 0; i < nums.length; i++) {
			len = result.size();
			for (int j = 0; j < len; j++) {
				List<Integer> temp = new ArrayList<Integer>(result.get(j));
				temp.add(nums[i]);
				result.add(temp);
			}
		}
		return result;
	}

	public List<List<Integer>> subsets3(int[] nums) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		int n = nums.length;
		int size = (int) Math.pow(2, n);
		for (int i = 0; i < size; i++) {
			List<Integer> list = new ArrayList<Integer>();
			for (int j = i, index = 0; j > 0; j >>= 1, index++)
			/**判断i的2进制表示自右往左第index上的数是否为1*/
				if ((j & 1) == 1)
					list.add(nums[index]);
			result.add(list);
		}
		return result;
	}


	public List<List<Integer>> subsets2(int[] nums) {
		List<List<Integer>> result = new ArrayList<>();
		Arrays.sort(nums);
		getSubsets(result, new ArrayList<Integer>(), nums, 0);
		return result;
	}

	private void getSubsets(List<List<Integer>> result, List<Integer> list, int[] nums, int start) {
		result.add(new ArrayList<Integer>(list));
		for (int i = start; i < nums.length; i++) {
			list.add(nums[i]);
			getSubsets(result, list, nums, i + 1);
			list.remove(list.size() - 1);
		}
	}

	public int maxArea(int[] height) {
		int begin = 0, end = height.length - 1, maxArea = 0;
		while (begin < end) {
			maxArea = Math.max(maxArea, Math.min(height[begin], height[end]) * (end - begin));
			if (height[begin] > height[end]) {
				int k = end;
				while (k > begin && height[k] <= height[end])
					k--;
				end = k;
			} else {
				int k = begin;
				while (k < end && height[k] <= height[begin])
					k++;
				begin = k;
			}
		}
		return maxArea;
	}

	public boolean searchMatrix2(int[][] matrix, int target) {
		if (matrix.length == 0) return false;
		int m = 0, n = matrix[0].length - 1;
		while (m < matrix.length && n >= 0) {
			int x = matrix[m][n];
			if (target == x) return true;
			else if (target < x) n--;
			else m++;
		}
		return false;
	}


	public void rotate(int[][] matrix) {
		int n = matrix.length;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n - i; j++) {
				int tmp = matrix[i][j];
				matrix[i][j] = matrix[n - 1 - j][n - i - 1];
				matrix[n - 1 - j][n - i - 1] = tmp;
			}
		for (int i = 0; i < n / 2; i++)
			for (int j = 0; j < n; j++) {
				int tmp = matrix[i][j];
				matrix[i][j] = matrix[n - 1 - i][j];
				matrix[n - 1 - i][j] = tmp;
			}
	}

	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix.length == 0) return false;
		int m = matrix.length, n = matrix[0].length;
		int low = 0, hight = m * n;
		while (low < hight) {
			int mid = low + (hight - low) / 2;
			int vlaue = matrix[mid / n][mid % n];
			if (vlaue == target) return true;
			else if (vlaue < target) low = mid + 1;
			else hight = mid;
		}
		return false;
	}

	public void sortColors(int[] nums) {
		int red = 0, blue = nums.length - 1, i = 0;
		while (i <= blue) {
			int tmp;
			/**0排到前面,2排到后面*/
			if (nums[i] == 0) {
				tmp = nums[i];
				nums[i++] = nums[red];
				nums[red++] = tmp;
			} else if (nums[i] == 2) {
				tmp = nums[i];
				nums[i] = nums[blue];
				nums[blue--] = tmp;
			} else i++;
		}
	}

	public void sortColors2(int[] nums) {
		int[] count = new int[3];
		for (int i = 0; i < nums.length; i++)
			count[nums[i]]++;
		for (int i = 0, index = 0; i < 3; i++)
			for (int j = 0; j < count[i]; j++)
				nums[index++] = i;
	}

	public void sortColors3(int[] nums) {
		int l = 0, m = 0, n = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == 0) {
				nums[n++] = 2;
				nums[m++] = 1;
				nums[l++] = 0;
			} else if (nums[i] == 1) {
				nums[n++] = 2;
				nums[m++] = 1;
			} else nums[n++] = 2;
		}
	}

	public int[] intersection(int[] nums1, int[] nums2) {
		Set<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < nums1.length; i++)
			set.add(nums1[i]);
		int[] result = new int[nums2.length];
		int k = 0;
		for (int i = 0; i < nums2.length; i++)
			if (set.contains(nums2[i])) {
				result[k++] = nums2[i];
				set.remove(nums2[i]); //一定要删除
			}
		return Arrays.copyOfRange(result, 0, k);
	}


	public int[] intersection2(int[] nums1, int[] nums2) {
		Arrays.sort(nums1);
		Arrays.sort(nums2);
		int[] result = new int[nums2.length];
		int i = 0, j = 0, k = 0;
		while (i < nums1.length && j < nums2.length) {
			if (nums1[i] < nums2[j]) {
				i++;
			} else if (nums1[i] > nums2[j]) {
				j++;
			} else {
				result[k++] = nums1[i++];
				while (i < nums1.length && nums1[i] == nums2[j]) i++;//防止重复
				j++;
			}
		}
		return Arrays.copyOfRange(result, 0, k);
	}


	public int jump(int[] nums) {
		int reach = 0;//最右能跳到哪里
		int last = 0;//上次落的的最大号阶梯,如果当前超过了,那必然是step了一下
		int step = 0;//步数
		for (int i = 0; i < nums.length; i++) {
			if (i > last) {
				step++;
				last = reach;
			}
			reach = Math.max(reach, i + nums[i]);
		}
		return step;
	}

	public int jump2(int[] nums) {
		// 初始化l[0], r[0]
		int k = 0, l = 0, r = 0;
		// 当最后位置的f值没有计算出时保持计算
		while (r < nums.length - 1) {
			int next_r = r;
			// 遍历l[k]到r[k],计算r[k+1]
			for (int i = l; i <= r; i++) next_r = Math.max(next_r, i + nums[i]);
			// 替换到k+1，以进行之后的计算
			k++;
			l = r + 1;
			r = next_r;
		}
		// 返回最后一个位置的f值
		return k;
	}

	public boolean canJump(int[] nums) {
		int reach = 1;//最右能跳到哪里
		for (int i = 0; i < reach && reach < nums.length; i++)
			reach = Math.max(reach, i + 1 + nums[i]);
		return reach >= nums.length;
	}

	public boolean canJump2(int[] nums) {
		int[] surplus = new int[nums.length];//在每个台阶,剩余的最大步数
		for (int i = 1; i < nums.length; i++) {
			surplus[i] = Math.max(surplus[i - 1], nums[i - 1]) - 1;
			if (surplus[i] < 0) return false;
		}
		return surplus[nums.length - 1] >= 0;
	}

	public int hIndex(int[] citations) {
		Arrays.sort(citations);
		int len = citations.length;
		for (int i = 0; i < len; i++) {
			if (citations[i] >= len - i)
				return len - i;
		}
		return 0;
	}


	public int hIndex2(int[] citations) {
		int len = citations.length;
		int[] count = new int[len + 1];
		for (int citation : citations) {
			if (citation >= len) count[len]++;
			else count[citation]++;
		}
		int sum = 0;
		for (int i = len; i >= 0; i--) {
			sum += count[i];
			if (sum >= i) return i;
		}
		return 0;
	}

	public int hIndex3(int[] citations) {
		int len = citations.length, low = 0, hight = len, mid;
		while (low < hight) {
			mid = (low + hight) / 2;
			if (citations[mid] >= len - mid)
				hight = mid;
			else low = mid + 1;

		}
		return len - low;
	}


	public int coinChange(int[] coins, int amount) {
		int[] dp = new int[amount + 1];
		final int INF = 0x7ffffffe;
		for (int i = 1; i <= amount; i++) dp[i] = INF;
		for (int coin : coins)
			for (int i = coin; i <= amount; i++)
				dp[i] = Math.min(dp[i], dp[i - coin] + 1);
		return dp[amount] == INF ? -1 : dp[amount];
	}

	public boolean isAdditiveNumber(String num) {
		int len = num.length();
		for (int i = 1; i <= len / 2; i++) {
			for (int j = 1; Math.max(i, j) <= len - j - i; j++) {
				String s1 = num.substring(0, i), s2 = num.substring(i, j + i);
				if (s1.charAt(0) == '0' && s1.length() > 1 || s2.charAt(0) == '0' && s2.length() > 1)
					continue;
				Long d1 = new Long(s1), d2 = new Long(s2), sum = d1 + d2;
				String next = sum.toString(), now = s1 + s2 + next;
				while (now.length() < len && num.startsWith(now)) {
					d1 = d2;
					d2 = sum;
					sum = d1 + d2;
					now += sum.toString();
				}
				if (now.equals(num)) return true;
			}
		}
		return false;
	}

	public boolean isAdditiveNumber2(String num) {
		for (int i = 1; i <= num.length() / 2; i++)
			for (int j = 1; Math.max(i, j) <= num.length() - j - i; j++)
				if (isValid(num, num.substring(0, i), num.substring(i, i + j), i + j)) return true;
		return false;
	}

	private boolean isValid(String num, String first, String second, int index) {
		if (first.length() > 1 && first.startsWith("0")
				|| second.length() > 1 && second.startsWith("0")) return false;
		if (index == num.length()) return true;
		long sum = Long.parseLong(first) + Long.parseLong(second);
		if (num.startsWith(sum + "", index))
			if (isValid(num, second, sum + "", index + (sum + "").length())) return true;
		return false;
	}

	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode result = new ListNode(0);
		ListNode pointer = result;
		int flag = 0;
		while (l1 != null || l2 != null) {
			if (l1 != null) {
				flag += l1.val;
				l1 = l1.next;
			}
			if (l2 != null) {
				flag += l2.val;
				l2 = l2.next;
			}
			pointer.next = new ListNode(flag % 10);
			pointer = pointer.next;
			flag /= 10;
		}
		pointer.next = flag == 1 ? new ListNode(1) : null;
		return result.next;
	}


	public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
		if (k < 1 || t < 0) return false;
		SortedSet<Long> set = new TreeSet<Long>();
		for (int i = 0; i < nums.length; i++) {
			if (!set.subSet((long) nums[i] - t, (long) nums[i] + t + 1).isEmpty()) return true;
			if (i >= k) set.remove((long) nums[i - k]);
			set.add((long) nums[i]);
		}
		return false;
	}

	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		List<Integer> tmp = new ArrayList<Integer>();
		dfs2(1, k, n, tmp, result);
		return result;
	}

	private void dfs2(int begin, int k, int n, List<Integer> tmp, List<List<Integer>>
			result) {
		if (k == 0)
		/**tmp是变动的,所以此处需要new一个*/
			result.add(new ArrayList<Integer>(tmp));
		/*分别i开始往下深搜*/
		for (int i = begin; i <= n; i++) {
			tmp.add(i);
			dfs2(i + 1, k - 1, n, tmp, result);//搜索下个数字
			tmp.remove(tmp.size() - 1);//回溯
		}
	}

	public List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		List<Integer> tmp = new ArrayList<Integer>();
		dfs(1, k, n, tmp, result);
		return result;
	}

	private void dfs(int begin, int k, int rest, List<Integer> tmp, List<List<Integer>> result) {
		/**注意这个终止条件*/
		if (rest == 0 && k == 0)
		/**tmp是变动的,所以此处需要new一个*/
			result.add(new ArrayList<Integer>(tmp));
		/*分别i开始往下深搜*/
		for (int i = begin; i <= 9 && i <= rest; i++) {
			tmp.add(i);
			dfs(i + 1, k - 1, rest - i, tmp, result);//搜索下个数字
			tmp.remove(tmp.size() - 1);//回溯
		}
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		Arrays.sort(candidates);
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		List<Integer> tmp = new ArrayList<Integer>();
		dfs(0, candidates, target, tmp, result);
		return result;
	}

	private void dfs(int begin, int[] candidates, int rest, List<Integer> tmp, List<List<Integer>> result) {
		if (rest == 0)
		/**tmp是变动的,所以此处需要new一个*/
			result.add(new ArrayList<Integer>(tmp));
		/*分别从candidates[i]开始往下深搜*/
		for (int i = begin; i < candidates.length && candidates[i] <= rest; i++) {
			tmp.add(candidates[i]);
			dfs(i, candidates, rest - candidates[i], tmp, result);//搜索下个数字
			tmp.remove(tmp.size() - 1);//回溯
		}
	}


	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		Arrays.sort(candidates);
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		List<Integer> tmp = new ArrayList<Integer>();
		dfs2(0, candidates, target, tmp, result);
		return result;
	}

	private void dfs2(int begin, int[] candidates, int rest, List<Integer> tmp, List<List<Integer>> result) {
		if (rest == 0)
		/**tmp是变动的,所以此处需要new一个*/
			result.add(new ArrayList<Integer>(tmp));
		/*分别从candidates[i]开始往下深搜,if candidates[i]==candidates[i-1],避免重复,跳过*/
		for (int i = begin; i < candidates.length && candidates[i] <= rest; i++) {
			if (i > begin && candidates[i] == candidates[i - 1]) continue;
			tmp.add(candidates[i]);
			dfs2(i + 1, candidates, rest - candidates[i], tmp, result);//搜索下个数字
			tmp.remove(tmp.size() - 1);//回溯
		}
	}

	public boolean isPowerOfFour(int num) {
		if (num < 0) return false;
		/**判断是否是2的幂*/
		if ((num & (num - 1)) != 0) return false;
		/**是2的幂后,判断是否是4的幂*/
		if ((num & 0x55555555) != 0) return true;
		return false;
	}

	public List<String> generateParenthesis(int n) {
		List<String> ret = new ArrayList<String>(), inner, outter;
		if (n == 0) {
			ret.add("");
			return ret;
		}
		if (n == 1) {
			ret.add("()");
			return ret;
		}
		for (int i = 0; i < n; ++i) {
			inner = generateParenthesis(i);
			outter = generateParenthesis(n - i - 1);
			for (int j = 0; j < inner.size(); ++j) {
				for (int k = 0; k < outter.size(); ++k) {
					ret.add("(" + inner.get(j) + ")" + outter.get(k));
				}
			}
		}
		return ret;
	}

	public class LRUCache extends LinkedHashMap<Integer, Integer> {
		private int maxcapacity;

		public LRUCache(int capacity) {
			super(capacity, 0.75f, true);
			this.maxcapacity = capacity;
		}

		public int get(int key) {
			Integer value = super.get(key);
			return value == null ? -1 : value;
		}

		public void set(int key, int value) {
			super.put(key, value);
		}

		protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
			return size() > maxcapacity;
		}
	}


	public class LRUCache2 {


		Map<Integer, Integer> map = new LinkedHashMap<Integer, Integer>();
		//		Map<Integer, Integer> mapUse = new HashMap<Integer, Integer>();
		List<Integer> list = new ArrayList<Integer>();
		int capacityMax = 0;
		int capacityTmp = 0;
		int index = 0;

		public LRUCache2(int capacity) {
			this.capacityMax = capacity;
		}

		public int get(int key) {
			if (map.containsKey(key)) {
				list.remove(Integer.valueOf(key));
				list.add(key);
				System.out.println("index = " + index);
				return map.get(key);
			}
			return -1;
		}

		public void set(int key, int value) {
			if (capacityTmp < capacityMax) {
				map.put(key, value);
				capacityTmp++;
			} else if (capacityTmp == capacityMax) {
				if (map.containsKey(key)) map.put(key, value);
				else {
					map.remove(list.get(index++));
					map.put(key, value);
				}
				list.add(Integer.valueOf(key));
			}
		}
	}

	public class Solution {
		List<String> parenthesisList = new ArrayList<String>();

		public List<String> generateParenthesis(int n) {
			generateLeftsAndRights("", n, n);
			return parenthesisList;
		}

		private void generateLeftsAndRights(String subList, int left, int right) {
			if (left == 0 && right == 0) {
				parenthesisList.add(subList);
				return;
			}
			if (left > 0) generateLeftsAndRights(subList + "(", left - 1, right);
			if (right > 0 && left < right)
				generateLeftsAndRights(subList + ")", left, right - 1);
		}
	}

	int integerBreak2(int n) {
		int[] result = new int[n + 1];
		result[0] = 1;
		result[1] = 1;
		for (int i = 2; i <= n; i++) {
			result[i] = -1;
			for (int j = 1; j < i; j++) {
				result[i] = Math.max(j * result[i - j], Math.max(result[i], j * (i - j)));
			}
		}
		return result[n];
	}

	public int integerBreak(int n) {
		if (n < 4) return n - 1;
		int result = 1;
		while (n > 4) {
			result *= 3;
			n -= 3;
		}
		return n * result;
	}

	public int minPathSum2(int[][] grid) {
		int row = grid.length;
		int column = grid[0].length;
		int[][] path = new int[row][column];
		path[0][0] = grid[0][0];
		for (int i = 1; i < row; i++)
			path[i][0] = path[i - 1][0] + grid[i][0];
		for (int i = 1; i < column; i++)
			path[0][i] = path[0][i - 1] + grid[0][i];
		for (int i = 1; i < row; i++) {
			for (int j = 1; j < column; j++) {
				path[i][j] = Math.min(path[i][j - 1], path[i - 1][j]) + grid[i][j];
			}
		}
		return path[row - 1][column - 1];
	}

	public int minPathSum(int[][] grid) {
		int row = grid.length;
		int column = grid[0].length;
		int[] path = new int[column];
		path[0] = 0;
		for (int i = 1; i < column; i++)
			path[i] = Integer.MAX_VALUE;
		//path[j]的值为第i行第j列当前从左上角到达此点的最小值.
		for (int i = 0; i < row; i++) {
			path[0] = path[0] + grid[i][0];
			for (int j = 1; j < column; j++) {
				path[j] = Math.min(path[j - 1], path[j]) + grid[i][j];
			}
		}
		return path[column - 1];
	}


	public int nthSuperUglyNumber(int n, int[] primes) {
		int[] nums = new int[n];
		nums[0] = 1;
		int[] idx = new int[primes.length];
		int cout = 1;
		while (cout < n) {
			int minNum = Integer.MAX_VALUE;
			for (int i = 0; i < primes.length; i++) {
				minNum = Math.min(nums[idx[i]] * primes[i], minNum);
			}
			for (int i = 0; i < primes.length; i++) {
				if (nums[idx[i]] * primes[i] == minNum) idx[i]++;
			}
			nums[cout++] = minNum;
		}
		return nums[n - 1];
	}

	public static int nthUglyNumber(int n) {
		SortedSet<Long> set = new TreeSet<Long>();
		long ithNum = 1;
		set.add(ithNum);
		for (int i = 0; i < n; i++) {
			ithNum = set.first();
			set.add(ithNum * 2);
			set.add(ithNum * 3);
			set.add(ithNum * 5);
			set.remove(ithNum);
		}
		return (int) ithNum;
	}


	public static int nthUglyNumber2(int n) {
		int l1 = 0, l2 = 0, l3 = 0;
		int[] nums = new int[n];
		nums[0] = 1;
		int i = 1;
		while (i < n) {
			int minNum = Math.min(Math.min(nums[l1] * 2, nums[l2] * 3), nums[l3] * 5);
			if (nums[l1] * 2 == minNum) l1++;
			if (nums[l2] * 3 == minNum) l2++;
			if (nums[l3] * 5 == minNum) l3++;
			nums[i++] = minNum;
		}
		return nums[n - 1];
	}

	public int[][] generateMatrix(int n) {
		int[][] matrix = new int[n][n];
		int count = 1;
		int begin = 0, end = n - 1;
		while (begin > end) {
			for (int i = begin; i < end; i++) matrix[begin][i] = count++;
			for (int i = begin; i < end; i++) matrix[i][end] = count++;
			for (int i = end; i > begin; i--) matrix[end][i] = count++;
			for (int i = end; i > begin; i--) matrix[i][begin] = count++;
			begin++;
			end--;
		}
		if (begin == end) matrix[begin][end] = count;
		return matrix;
	}


	public List<Integer> spiralOrder(int[][] matrix) {
		List<Integer> list = new ArrayList<Integer>();
		int row = matrix.length - 1, column = row < 0 ? -1 : matrix[0].length - 1;
		for (int x = 0, y = 0; x <= row && y <= column; x++, y++) {
			//输出最外圈的第一行
			for (int i = y; i <= column; i++) {
				list.add(matrix[x][i]);
			}
			//输出最外圈的最右列
			for (int i = x + 1; i <= row; i++) {
				list.add(matrix[i][column]);
			}
			//输出最外圈的最底行,要注意防止和第一行重复
			for (int i = column - 1; i >= y && x != row; i--) {
				list.add(matrix[row][i]);
			}
			//输出最外圈的最左列,要注意防止和最右列重复
			for (int i = row - 1; i > x && y != column; i--) {
				list.add(matrix[i][y]);
			}
			row--;
			column--;
		}
		return list;
	}

	public int findMin2(int[] nums) {
		int low = 0, hight = nums.length - 1, mid;
		while (low < hight && nums[low] >= nums[hight]) {
			mid = low + (hight - low) / 2;
			if (nums[mid] > nums[hight]) low = mid + 1;
			if (nums[mid] == nums[hight]) low = low + 1;
			if (nums[mid] < nums[hight]) hight = mid;
		}
		return nums[low];
	}

	public int findMin(int[] nums) {
		int low = 0, hight = nums.length - 1, mid;
		while (low < hight) {
			mid = low + (hight - low) / 2;
			if (nums[mid] > nums[low]) low = mid + 1;
			else hight = mid;
		}
		return nums[hight];
	}

	public ListNode detectCycle(ListNode head) {
		if (head == null || head.next == null) return null;
		ListNode slow = head, fast = head;
		while (slow != null && fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				fast = head;
				while (fast != slow) {
					fast = fast.next;
					slow = slow.next;
				}
				return fast;
			}
		}
		return null;
	}

	public boolean hasCycle(ListNode head) {
		if (head == null || head.next == null) return false;
		ListNode fast = head;
		while (head != null && fast != null && fast.next != null) {
			fast = fast.next.next;
			head = head.next;
			if (fast == head) return true;
		}
		return false;
	}

	public int rob(TreeNode root) {
		if (root == null) return 0;
		if (root.left == null && root.right == null) return root.val;
		int left = rob(root.left);
		int right = rob(root.right);
		int leftSon = 0, rightSon = 0;
		if (root.left != null)
			leftSon = rob(root.left.left) + rob(root.left.right);
		if (root.right != null)
			rightSon = rob(root.right.left) + rob(root.right.right);
		return Math.max(left + right, root.val + leftSon + rightSon);
	}

	public int searchInsert(int[] nums, int target) {
		int hight = nums.length, low = 0, mid;
		while (hight > low) {
			mid = low + (hight - low) / 2;
			if (nums[mid] == target) return mid;
			if (nums[mid] > target)
				hight = mid;
			else
				low = mid + 1;
		}
		return low;
	}

	public int rob2(int[] nums) {
		if (nums.length <= 0) return 0;
		if (nums.length == 1) return nums[0];

		//打劫第1家到倒数第2家的最大值
		int a = nums[0], b = Math.max(nums[1], nums[0]);
		for (int i = 2; i < nums.length - 1; i++) {
			int tmp = b;
			b = Math.max(a + nums[i], b);
			a = tmp;
		}
		if (nums.length <= 2) return b;

		//打劫第2家到最后一家的最大值
		int c = nums[1], d = Math.max(nums[1], nums[2]);
		for (int i = 3; i < nums.length; i++) {
			int tmp = d;
			d = Math.max(c + nums[i], d);
			c = tmp;
		}
		return Math.max(b, d);
	}


	public int rob(int[] nums) {
		if (nums.length <= 0) return 0;
		if (nums.length == 1) return nums[0];
		int a = nums[0], b = Math.max(nums[1], nums[0]);
		for (int i = 2; i < nums.length; i++) {
			int tmp = b;
			b = Math.max(a + nums[i], b);
			a = tmp;
		}
		return b;
	}

	public String intToRoman(int num) {
		String[] str = new String[]{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
		int[] val = new int[]{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
		StringBuilder roman = new StringBuilder("");
		for (int i = 0; i < val.length; i++) {
			while (num >= val[i]) {
				roman.append(str[i]);
				num = num - val[i];
			}
		}
		return roman.toString();
	}

	public List<Integer> preorderTraversal(TreeNode root) {

		return null;
	}


	public int missingNumber(int[] nums) {
		int result = 0;
		for (int i = 0; i < nums.length; i++) {
			result = result ^ nums[i];
		}
		return result ^ nums.length;
	}

	public int missingNumber2(int[] nums) {
		int result = (1 + nums.length) * nums.length / 2;
		for (int i = 0; i < nums.length; i++) {
			result -= nums[i];
		}
		return result;
	}


	public int[] productExceptSelf(int[] nums) {
		if (nums.length == 0) return nums;
		int len = nums.length, right = 1;
		int[] left = new int[len];
		left[0] = 1;
		for (int i = 1; i < len; i++) {
			left[i] = left[i - 1] * nums[i - 1];
		}
		for (int i = len - 1; i >= 0; i--) {
			left[i] = left[i] * right;
			right *= nums[i];
		}
		return left;
	}

	public int maxProfit4(int[] prices) {
		if (prices.length <= 1) return 0;
		int sell = 0, buy = -prices[0], presell = 0;
		for (int i = 1; i < prices.length; i++) {
			int tmp = sell;//昨天卖的最高收入
			sell = Math.max(sell, buy + prices[i]);
			buy = Math.max(buy, presell - prices[i]);//今天若买入的最高收入
			presell = tmp;//前天卖的最高收入
		}
		return sell;
	}

	public int maxProfit3(int[] prices) {
		if (prices.length == 0) return 0;
		int minbuy = prices[0], profit = 0;
		int[] front = new int[prices.length];
		front[0] = 0;
		int[] back = new int[prices.length];
		back[prices.length - 1] = 0;
		for (int i = 1; i < prices.length; i++) {
			minbuy = Math.min(prices[i], minbuy);
			front[i] = Math.max(front[i - 1], prices[i] - minbuy);
		}
		for (int i = prices.length - 2, maxSell = prices[prices.length - 1]; i > 0; i--) {
			maxSell = Math.max(prices[i], maxSell);
			back[i] = Math.max(back[i - 1], maxSell - prices[i]);
		}
		for (int i = 0; i < prices.length; i++) {
			profit = Math.max(front[i] + back[i], profit);
		}
		return profit;
	}

	public int maxProfit2(int[] prices) {
		int profit = 0;
		for (int i = 1; i < prices.length; i++) {
			int price = prices[i] - prices[i - 1];
			profit += Math.max(0, price);
		}
		return profit;
	}

	public int maxProfit(int[] prices) {
		if (prices.length == 0) return 0;
		int minbuy = prices[0], profit = 0;
		for (int i = 1; i < prices.length; i++) {
			minbuy = Math.min(prices[i], minbuy);
			profit = Math.max(profit, prices[i] - minbuy);
		}
		return profit;
	}

	public int singleNumber2(int[] nums) {
		int one = 0, two = 0, three = 0;
		for (int i = 0; i < nums.length; i++) {
			two = one & nums[i] | two;
			one = one ^ nums[i];
			three = one & three;
			one = one & ~three;
			two = two & ~three;
		}
		return one;
	}

	public int threeSumClosest(int[] nums, int target) {
		Arrays.sort(nums);
		int len = nums.length, flag = Integer.MAX_VALUE;
		for (int i = 0; i < len - 2; i++) {
			int left = i + 1, right = len - 1;
			while (left < right) {
				int sum = nums[i] + nums[left] + nums[right];
				flag = Math.abs(sum - target) < Math.abs(flag) ? sum - target : flag;
				if (sum < target) left++;//数据小,继续找后边大的数据.
				else right--;//数据大,继续找前边小的数据.
			}
		}
		return target + flag;
	}

	public List<List<Integer>> fourSum(int[] nums, int target) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		int len = nums.length;
		Arrays.sort(nums);
		for (int k = 0; k < len - 3; k++) {
			if (k > 0 && nums[k] == nums[k - 1]) continue;
			for (int i = k + 1; i < len - 2; i++) {
				if (i != k + 1 && nums[i] == nums[i - 1]) continue;
				int left = i + 1, right = len - 1;
				while (left < right) {
					int sum = nums[k] + nums[i] + nums[left] + nums[right];
					if (sum == target) {
						List<Integer> list = new ArrayList<Integer>();
						list.add(nums[k]);
						list.add(nums[i]);
						list.add(nums[left]);
						list.add(nums[right]);
						result.add(list);
						/**去掉重复的数据,并且继续收缩找中间的数据*/
						while (left++ < right && nums[left] == nums[left - 1]) ;
						while (left < --right && nums[right] == nums[right + 1]) ;
					} else if (sum < target)
						left++;//数据小,继续找后边大的数据.
					else right--;//数据大,继续找前边小的数据.
				}
			}
		}
		return result;
	}

	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		int len = nums.length;
		Arrays.sort(nums);
		for (int i = 0; i < len - 2 && nums[i] <= 0; i++) {
			if (i > 0 && nums[i] == nums[i - 1]) continue;
			int left = i + 1, right = len - 1;
			while (left < right) {
				if (nums[i] + nums[left] + nums[right] == 0) {
					List<Integer> list = new ArrayList<Integer>();
					list.add(nums[i]);
					list.add(nums[left]);
					list.add(nums[right]);
					result.add(list);
					/**去掉重复的数据,并且继续收缩找中间的数据*/
					while (left++ < right && nums[left] == nums[left - 1]) ;
					while (left < --right && nums[right] == nums[right + 1]) ;
				} else if (nums[i] + nums[left] + nums[right] < 0)
					left++;//数据小,继续找后边大的数据.
				else right--;//数据大,继续找前边小的数据.

			}
		}
		return result;
	}


	public int[] singleNumber3(int[] nums) {
		int k = 0;
		for (int i = 0; i < nums.length; i++) {
			k = k ^ nums[i];
		}
		int[] result = new int[2];
		int n = k & (-k);//求一个数k的二进制1的最低位
		for (int i = 0; i < nums.length; i++) {
			if ((n & nums[i]) == 0) {
				result[0] = result[0] ^ nums[i];
			}
			result[1] = result[1] ^ nums[i];
		}
		return result;
	}

	public int singleNumber(int[] nums) {
		int k = 0;
		for (int i = 0; i < nums.length; i++) {
			k = k ^ nums[i];
		}
		return k;
	}


	/**
	 * 求一个数x的二进制最低位
	 */
	static int lowbit(int x) {
		return x & (-x);
	}

	/**
	 * 求一个数x的二进制最高位
	 */
	static int highbit(int x) {
		return (int) Math.pow(2, (int) (Math.log(x) / Math.log(2)));
	}

	public int[] countBits(int num) {
		int[] nums = new int[num + 1];
		nums[0] = 0;
		for (int i = 1; i < nums.length; i++) {
			nums[i] = nums[i & (i - 1)] + 1;
		}
		return nums;
	}

	public int[] countBits2(int num) {
		int[] nums = new int[num + 1];
		nums[0] = 0;
		for (int i = 1; i < nums.length; i++) {
			nums[i] = nums[i >> 1] + i % 2;
		}
		return nums;
	}

	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> listNode = new ArrayList<List<Integer>>();
		if (root == null) return listNode;
		List<Integer> node = new ArrayList<Integer>();
		node.add(root.val);
		listNode.add(node);
		if (root.left == null && root.right == null) return listNode;
		List<List<Integer>> leftList = levelOrder(root.left);
		List<List<Integer>> rightList = levelOrder(root.right);
		int len = Math.min(rightList.size(), leftList.size());
		int other = Math.max(rightList.size(), leftList.size());
		int i;
		for (i = 0; i < len; i++) {
			leftList.get(i).addAll(rightList.get(i));
		}
		if (rightList.size() > i) {
			for (int j = i; j < other; j++) {
				leftList.add(j, rightList.get(j));
			}
		}
		listNode.addAll(leftList);
		return listNode;
	}


	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> listNode = new ArrayList<List<Integer>>();
		if (root == null) return listNode;
		List<Integer> node = new ArrayList<Integer>();
		node.add(root.val);
		listNode.add(node);
		if (root.left == null && root.right == null) return listNode;
		List<List<Integer>> leftList = levelOrderBottom(root.left);
		List<List<Integer>> rightList = levelOrderBottom(root.right);
		int dif = Math.abs(rightList.size() - leftList.size());
		if (rightList.size() > leftList.size()) {
			for (int i = 0; i < dif; i++) {
				leftList.add(i, rightList.get(i));
			}
		}
		for (int i = dif; i < leftList.size(); i++) {
			if (rightList.size() < leftList.size())
				leftList.get(i).addAll(rightList.get(i - dif));
			else leftList.get(i).addAll(rightList.get(i));
		}
		listNode.addAll(0, leftList);
		return listNode;
	}

	public boolean isSymmetric(TreeNode root) {
		if (root == null) return true;
		return mirror(root.left, root.right);
	}

	public boolean mirror(TreeNode p, TreeNode q) {
		if (p == null && q == null) return true;
		if (p == null || q == null) return false;
		return p.val == q.val && mirror(p.left, q.right) && mirror(p.right, q.left);
	}

	public boolean isSymmetric2(TreeNode root) {
		Stack<TreeNode> nodeStack = new Stack<TreeNode>();
		if (root == null) return true;
		nodeStack.push(root.left);
		nodeStack.push(root.right);
		while (!nodeStack.isEmpty()) {
			TreeNode q = nodeStack.pop();
			TreeNode p = nodeStack.pop();
			if (p == null && q == null) continue;
			if (p == null || q == null) return false;
			if (p.val != q.val) return false;
			nodeStack.push(p.left);
			nodeStack.push(q.right);
			nodeStack.push(p.right);
			nodeStack.push(q.left);
		}
		return true;
	}


	public List<String> binaryTreePaths(TreeNode root) {
		List<String> paths = new ArrayList<String>();
		if (root == null) return paths;
		if (root.left == null && root.right == null) {
			paths.add(root.val + "");
			return paths;
		}
		List<String> left = binaryTreePaths(root.left);
		List<String> right = binaryTreePaths(root.right);
		for (String tmp : left) {
			paths.add(root.val + "->" + tmp);
		}
		for (String tmp : right) {
			paths.add(root.val + "->" + tmp);
		}
		return paths;
	}

	public static String getHint(String secret, String guess) {
		int a = 0, b = 0;
		for (int i = 0; i < guess.length(); i++) {
			if (guess.charAt(i) == secret.charAt(i))
				a++;
		}
		for (int i = 0; i < guess.length(); i++) {
			if (secret.contains(guess.charAt(i) + "")) {
				secret = secret.replaceFirst(guess.charAt(i) + "", "");
				b++;
			}
		}
		return a + "A" + (b - a) + "B";
	}

	public boolean isPalindrome(String s) {
		s = s.toLowerCase().replaceAll("[^a-z|^0-9]", "");
		int len = s.length();
		for (int i = 0; i < len / 2; i++) {
			if (s.charAt(i) != s.charAt(len - 1 - i))
				return false;
		}
		return true;
	}

	public boolean isPalindrome(ListNode head) {
		if (head == null || head.next == null) return true;
		ListNode fast = head, slow = head;//利用快慢指针,寻找中间节点
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		/*** Start:对链表后半部分进行反转 By yuanyuan.niu*/
		ListNode pre = slow, cursor = slow.next;
		while (cursor != null) {
			ListNode tmp = cursor.next;
			cursor.next = pre;
			pre = cursor;
			cursor = tmp;
		}
		slow.next = null;
		/**end反转链表结束,pre作为链表后半部分反转后的头节点*/
		while (pre != null) {
			if (head.val != pre.val) return false;
			pre = pre.next;
			head = head.next;
		}
		return true;
	}

	public boolean isPalindrome(int x) {
		if (x < 0) return false;
		int newNum = 0;
		int oldNum = x;
		while (x != 0) {
			newNum = newNum * 10 + x % 10;
			x = x / 10;
		}
		return newNum == oldNum;
	}

	public boolean isBalanced(TreeNode root) {
		return depth(root) >= 0;
	}

	public int depth(TreeNode root) {
		if (root == null) return 0;
		int highL = depth(root.left);
		int highR = depth(root.right);
		if (Math.abs(highL - highR) > 1 || highL < 0 || highR < 0) return -1;
		return Math.max(highL, highR) + 1;
	}

	public int minDepth(TreeNode root) {
		if (root == null) return 0;
		int rightDepth = minDepth(root.right);
		int leftDepth = minDepth(root.left);
		if (rightDepth != 0 && leftDepth != 0)
			return Math.min(leftDepth, rightDepth) + 1;
		return rightDepth != 0 ? rightDepth + 1 : leftDepth + 1;
	}

	public boolean hasPathSum(TreeNode root, int sum) {
		if (root == null) return false;
		if (root.left == null && root.right == null) return root.val == sum ? true : false;
		return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
	}

	public ListNode swapPairs(ListNode head) {
		if (head == null || head.next == null) return head;
		ListNode result = head.next;
		ListNode pre = new ListNode(-1);
		while (head != null && head.next != null) {
			ListNode even = head.next;
			head.next = even.next;
			even.next = head;
			pre.next = even;
			pre = head;
			head = head.next;
		}
		return result;
	}

	public ListNode swapPairs2(ListNode head) {
		if (head == null || head.next == null) return head;
		ListNode even = head.next;
		head.next = swapPairs(even.next);
		even.next = head;
		return even;
	}


	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		} else
			return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
	}


	public int myAtoi(String str) {
		str = str.trim();
		if (str.length() == 0) return 0;
		int flag = str.charAt(0) == '-' ? -1 : 1;
		int i = str.charAt(0) == '+' || str.charAt(0) == '-' ? 1 : 0;
		//注意value的类型,如果是int,下面value > Integer.MAX_VALUE就不准确了,因为value最多只能表示MAX_VALUE
		long value = 0;
		for (int j = i; j < str.length(); j++) {
			if (str.charAt(j) >= '0' && str.charAt(j) <= '9') {
				value = 10 * value + str.charAt(j) - '0';
				if (value > Integer.MAX_VALUE)
					return flag == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
			} else break;
		}
		return (int) value * flag;
	}

	public static int compareVersion(String version1, String version2) {
		String[] nums1 = version1.split("\\.");
		String[] nums2 = version2.split("\\.");
		for (int i = 0; i < Math.max(nums1.length, nums2.length); i++) {
			int num1 = i < nums1.length ? Integer.valueOf(nums1[i]) : 0;
			int num2 = i < nums2.length ? Integer.valueOf(nums2[i]) : 0;
			if (num1 > num2) return 1;
			if (num1 < num2) return -1;
		}
		return 0;
	}


	public int[] twoSum(int[] nums, int target) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		int[] result = new int[2];
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(nums[i])) {
				result[0] = map.get(target - nums[i]);
				result[1] = i;
				return result;
			} else
				map.put(target - nums[i], i);
		}
		return result;
	}

	public int reverse(int x) {
		long sum = 0;
		int flag = x < 0 ? -1 : 1;
		x = Math.abs(x);
		while (x != 0) {
			int x1 = x % 10;
			sum = sum * 10 + x1;
			x /= 10;
		}
		if (sum * flag > Integer.MAX_VALUE || sum * flag < Integer.MIN_VALUE)
			return 0;
		return (int) (sum * flag);
	}

	public String convert(String s, int numRows) {
		StringBuffer result = new StringBuffer("");
		if (s.length() <= 1 || numRows <= 1) return s;
		for (int i = 0; i < numRows; i++) {
			for (int j = 0, index = i; index < s.length(); j++, index = (2 * numRows - 2) * j + i) {
				result.append(s.charAt(index));//竖着一列的字符定位
				if (i == 0 || i == numRows - 1) continue;//第一行和最后一行对角线上没有字符
				if (index + (numRows - i - 1) * 2 < s.length())
					result.append(s.charAt(index + (numRows - i - 1) * 2));//斜着的在对角线上的字符定位
			}
		}
		return result.toString();
	}

	public int countPrimes(int n) {
		boolean[] isPrimes = new boolean[n];
		for (int i = 2; i * i < n; i++) {
			if (!isPrimes[i])
				for (int j = i * i; j < n; j += i)
					isPrimes[j] = true;
		}
		int count = 0;
		for (int i = 2; i < n; i++)
			if (!isPrimes[i]) count++;
		return count;
	}

	public List<String> summaryRanges(int[] nums) {
		List<String> ranges = new ArrayList<String>();
		if (nums.length == 0) return ranges;
		int start = 0;
		int end = 0;
		for (int i = 0; i < nums.length; i++) {
			if (i + 1 < nums.length && nums[i] == nums[i + 1] - 1) end++;
			else {
				if (start == end) ranges.add(String.valueOf(nums[start]));
				else ranges.add(nums[start] + "->" + nums[end]);
				start = ++end;
			}
		}
		return ranges;
	}

	public int strStr2(String haystack, String needle) {
		return haystack.indexOf(needle);
	}

	public int strStr(String haystack, String needle) {
		if (needle.isEmpty()) return 0;
		int len1 = haystack.length();
		int len2 = needle.length();
		for (int i = 0; i <= len1 - len2; i++) {
			for (int j = 0; j < len2; j++) {
				if (haystack.charAt(i + j) != needle.charAt(j)) break;
				if (j == len2 - 1) return i;
			}
		}
		return -1;
	}

	public String addBinary(String a, String b) {
		int len = Math.max(a.length(), b.length());
		StringBuffer aNew = new StringBuffer(a);
		StringBuffer bNew = new StringBuffer(b);
		aNew.reverse();
		bNew.reverse();
		StringBuffer c = new StringBuffer("");
		int carry = 0;
		for (int i = 0; i < len; i++) {
			int ai = a.length() > i ? aNew.charAt(i) - '0' : 0;
			int bi = b.length() > i ? bNew.charAt(i) - '0' : 0;
			int value = (ai + bi + carry) % 2;
			carry = (ai + bi + carry) / 2;
			c.append(value);
		}
		if (carry == 1) c.append(carry);
		return c.reverse().toString();
	}

	public String longestCommonPrefix(String[] strs) {
		if (strs.length == 0) return "";
		if (strs.length == 1) return strs[0];
		StringBuffer prefix = new StringBuffer("");
		for (int i = 0; i < strs[0].length(); i++) {
			prefix.append(strs[0].charAt(i));
			for (int j = 1; j < strs.length; j++)
				if (strs[j].length() < prefix.length() || strs[j].charAt(i) != strs[0].charAt(i))
					return prefix.substring(0, prefix.length() - 1);
		}
		return prefix.toString();
	}

	public ListNode removeElements(ListNode head, int val) {
		ListNode cur = new ListNode(-1);
		cur.next = head;
		ListNode tmp = cur;
		while (cur.next != null) {
			if (cur.next.val == val) cur.next = cur.next.next;
			else cur = cur.next;
		}
		return tmp.next;
	}

	public static String countAndSay(int n) {
		if (n == 0) return null;
		String[] nums = new String[n - 1];
		nums[0] = "1";
		for (int i = 0; i < n - 1; i++) {
			int count = 0;
			char num = nums[i].charAt(0);
			StringBuffer say = new StringBuffer("");
			for (int j = 0; j < nums[i].length(); j++) {
				if (nums[i].charAt(j) == num)
					count++;
				else {
					say.append(count).append(num);
					num = nums[i].charAt(j);
					count = 1;
				}
			}
			nums[i + 1] = say.append(count).append(num).toString();
		}
		return nums[n - 1];
	}

	public boolean isIsomorphic(String s, String t) {
		if (s.length() != t.length()) return false;
		HashMap<Character, Integer> mapS = new HashMap<Character, Integer>();
		HashMap<Character, Integer> mapT = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			if (!mapS.containsKey(s.charAt(i)))
				mapS.put(s.charAt(i), s.charAt(i) - t.charAt(i));
			if (!mapT.containsKey(t.charAt(i)))
				mapT.put(t.charAt(i), s.charAt(i) - t.charAt(i));
			if (mapS.get(s.charAt(i)) != s.charAt(i) - t.charAt(i)) return false;
			if (mapT.get(t.charAt(i)) != s.charAt(i) - t.charAt(i)) return false;
		}
		return true;
	}

	public int lengthOfLastWord(String s) {
		if (s.trim().equals("")) return 0;
		String[] ss = s.split(" ");
		return ss[ss.length - 1].length();
	}

	public static boolean wordPattern(String pattern, String str) {
		String[] strs = str.split(" ");
		System.out.println("strs = " + Arrays.toString(strs));
		if (pattern.length() != strs.length) return false;
		HashMap<Character, String> mapS = new HashMap<Character, String>();
		HashMap<String, Character> mapT = new HashMap<String, Character>();
		for (int i = 0; i < pattern.length(); i++) {
			if (!mapS.containsKey(pattern.charAt(i)))
				mapS.put(pattern.charAt(i), strs[i]);
			if (!mapT.containsKey(strs[i]))
				mapT.put(strs[i], pattern.charAt(i));
			if (!mapS.get(pattern.charAt(i)).equals(strs[i])) return false;
			if (!mapT.get(strs[i]).equals(pattern.charAt(i))) return false;
		}
		return true;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		for (int i = m; i < m + n; i++)
			nums1[i] = nums2[i - m];
		Arrays.sort(nums1);
	}

	public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
		int plus = (C - A) * (D - B) + (G - E) * (H - F);
		int surplus = (Math.min(G, C) - Math.max(E, A)) * (Math.min(D, H) - Math.max(B, F));
		int area = E > C || G < A || F > D || H < B ? plus : plus - surplus;
		return area;
	}

	public static int reverseBits(int n) {
		int res = 0;
		for (int i = 0; i < 32; i++) {
			res = (res << 1) ^ (n & 1);
			n = n >> 1;
		}
		return res;
	}

	public boolean containsNearbyDuplicate(int[] nums, int k) {
		HashMap<Integer, Integer> numMap = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (!numMap.containsKey(nums[i]) || i - numMap.get(nums[i]) > k)
				numMap.put(nums[i], i);
			else return true;
		}
		return false;
	}

	public void merge2(int[] nums1, int m, int[] nums2, int n) {
		int p = m - 1;
		int q = n - 1;
		int cur = m + n - 1;
		while (q >= 0)
			nums1[cur--] = p >= 0 && nums1[p] > nums2[q] ? nums1[p--] : nums2[q--];
	}


	public boolean isValidSudoku(char[][] board) {
		List<Set<Character>> square = new ArrayList<Set<Character>>();
		List<Set<Character>> row = new ArrayList<Set<Character>>();
		List<Set<Character>> column = new ArrayList<Set<Character>>();
		for (int i = 0; i < 9; i++) {
			square.add(new HashSet<Character>());
			row.add(new HashSet<Character>());
			column.add(new HashSet<Character>());
		}
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.') continue;
				if (square.get(i / 3 * 3 + j / 3).contains(board[i][j]) || row.get(i).contains
						(board[i][j])
						|| column
						.get(j).contains(board[i][j])) return false;
				square.get(i / 3 * 3 + j / 3).add(board[i][j]);
				row.get(i).add(board[i][j]);
				column.get(j).add(board[i][j]);
			}
		}
		return true;
	}

	public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		if (headA == null || headB == null) return null;
		ListNode tmpA = headA;
		ListNode tmpB = headB;
		int lenA = 0;
		int lenB = 0;
		while (tmpA != null) {
			tmpA = tmpA.next;
			lenA++;
		}
		while (tmpB != null) {
			tmpB = tmpB.next;
			lenB++;
		}
		if (tmpA != tmpB) return null;
		int dif = lenA - lenB;
		tmpA = headA;
		tmpB = headB;
		while (dif > 0) {
			tmpA = tmpA.next;
			dif--;
		}
		while (dif < 0) {
			tmpB = tmpB.next;
			dif++;
		}

		while (tmpA != null) {
			if (tmpA.equals(tmpB))
				return tmpA;
			else {
				tmpA = tmpA.next;
				tmpB = tmpB.next;
			}
		}
		return tmpA;
	}


	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode cur1 = head;
		ListNode cur2 = new ListNode(-1);
		cur2.next = head;
		while (cur1 != null) {
			cur1 = cur1.next;
			if (n != 0)
				n--;
			else cur2 = cur2.next;
		}
		if (cur2.next == head) return head.next;
		cur2.next = cur2.next.next;
		return head;
	}

	public static boolean isValid(String s) {
		String[] strings = s.split("");
		Stack<String> stack = new Stack<String>();
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("]", "[");
		map.put(")", "(");
		map.put("}", "{");
		for (String tmp : strings) {
			if (tmp.equals("{") || tmp.equals("(") || tmp.equals("[")) {
				stack.push(tmp);
				continue;
			}
			if (map.containsKey(tmp)) {
				if (stack.isEmpty() || !map.get(tmp).equals(stack.peek()))
					return false;
				stack.pop();
			}
		}
		if (stack.isEmpty()) return true;
		return false;
	}

	public static List<List<Integer>> generate(int numRows) {
		List<List<Integer>> triangle = new LinkedList<List<Integer>>();
		if (numRows <= 0) return triangle;
		List<Integer> lineNew1 = new ArrayList<Integer>();
		lineNew1.add(1);
		triangle.add(lineNew1);
		if (numRows == 1) return triangle;
		List<Integer> lineNew2 = new ArrayList<Integer>();
		lineNew2.add(1);
		lineNew2.add(1);
		triangle.add(lineNew2);
		if (numRows == 2) return triangle;
		for (int i = 2; i < numRows; i++) {
			List<Integer> lineNew = new ArrayList<Integer>();
			List<Integer> line = triangle.get(i - 1);
			for (int j = 0; j < i - 1; j++) {
				int num = line.get(j) + line.get(j + 1);
				lineNew.add(j, num);
			}
			lineNew.add(0, 1);
			lineNew.add(i, 1);
			triangle.add(lineNew);
		}
		return triangle;
	}

	public static List<Integer> getRow(int rowIndex) {
		List<Integer> line = new ArrayList<Integer>();
		for (int i = 0; i <= rowIndex; i++) {
			line.add(1);
		}
		for (int i = 1; i < rowIndex; i++) {
			for (int j = 1; j <= rowIndex - i; j++) {
				int num = line.get(j) + line.get(j - 1);
				line.set(j, num);
			}
		}
		return line;
	}

	public int[] plusOne(int[] digits) {
		for (int i = digits.length - 1; i >= 0; i--) {
			if (digits[i] == 9) digits[i] = 0;
			else {
				digits[i]++;
				return digits;
			}
		}
		int[] newDigits = Arrays.copyOf(digits, digits.length + 1);
		newDigits[0] = 1;
		newDigits[digits.length] = 0;
		return newDigits;
	}

	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		ListNode head = new ListNode(-1);
		ListNode temp = head;
		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				temp.next = l1;
				l1 = l1.next;
			} else {
				temp.next = l2;
				l2 = l2.next;
			}
			temp = temp.next;
		}
		if (l1 != null) {
			temp.next = l1;
		} else {
			temp.next = l2;
		}
		return head.next;

	}


	/**
	 * Definition for a binary tree node. * public class TreeNode { * int val; * TreeNode left; * TreeNode right; * TreeNode(int x) { val = x; } * }
	 */
	public class Solution2 {
		/**
		 * 判断是否平衡二叉树 * 看左右子树高度差是否超过1 * 不超过1则分别判断左右子树是否平衡二叉树
		 */
		public boolean isBalanced(TreeNode root) {
			if (root == null) return true;
			if (dep(0, root) > -10) return true;
			return false;
		} //计算树的高度

		private int dep(int dep, TreeNode root) {
			if (root == null) {
				return dep - 1;
			}
			int dep1 = dep(dep + 1, root.left);
			int dep2 = dep(dep + 1, root.right);
			//如果高度差超过1,返回-10 //则以后的dep返回值均为-10
			if (Math.abs(dep1 - dep2) > 1) {
				return -10;
			}
			return dep1 > dep2 ? dep1 : dep2;
		}
	}

	public int removeElement(int[] nums, int val) {
		int count = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] != val) {
				nums[count++] = nums[i];
			}
		}
		return count;
	}

	public int removeDuplicates(int[] nums) {
		int count = 0;
		Arrays.sort(nums);
		int len = nums.length - 1;
		if (len <= 0) return len + 1;
		for (int i = 0; i < len; i++) {
			if (nums[i] != nums[i + 1]) {
				nums[count++] = nums[i];
			}
		}
		nums[count++] = nums[len];
		return count;
	}


	public boolean isPowerOfTwo(int n) {
		return n > 0 && (n & (n - 1)) == 0;
	}

	public boolean isPowerOfThree(int n) {
		double log = Math.log10(n) / Math.log10(3);
		return log - (int) log == 0;
	}

	public boolean isHappy(int n) {
		HashSet<Integer> set = null;
		while (set.add(n)) {
			int newNum = 0;
			while (n != 0) {
				newNum = (int) +Math.pow(n % 10, 2);
				n = n / 10;
			}
			n = newNum;
		}
		return n == 1;
	}

	public boolean isUgly(int num) {
		if (num < 1)
			return false;
		if (num % 2 == 0) return isUgly(num / 2);
		if (num % 3 == 0) return isUgly(num / 3);
		if (num % 5 == 0) return isUgly(num / 5);
		return num == 1;
	}

	public ListNode oddEvenList(ListNode head) {
		if (head == null) return head;
		ListNode temp = head.next;
		ListNode even = head.next;
		ListNode odd = head;
		while (even != null && even.next != null) {
			odd.next = even.next;
			odd = even.next;
			even.next = odd.next;
			even = even.next;
		}
		odd.next = temp;
		return head;
	}

	public long climbStairs(long n) {
		if (n <= 2) return n;
		long f1 = 1;
		long f2 = 2;
		long f3 = 0;
		int i = 3;
		while (i++ <= n) {
			f3 = f1 + f2;
			f1 = f2;
			f2 = f3;
		}
		return f3;
	}

	public ListNode deleteDuplicates(ListNode head) {
		if (head == null) return head;
		ListNode temp = head;
		while (temp != null && temp.next != null) {
			if (temp.next.val == temp.val) {
				temp.next = temp.next.next;
				continue;
			}
			temp = temp.next;
		}
		return head;
	}

	public int romanToInt(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);
		int value = map.get(s.charAt(0));
		for (int i = 1; i < s.length(); i++) {
			if (map.get(s.charAt(i)) > map.get(s.charAt(i - 1))) {
				value = value + map.get(s.charAt(i)) - 2 * map.get(s.charAt(i - 1));
			} else {
				value = value + map.get(s.charAt(i));
			}
		}
		return value;
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || p == null || q == null)
			return null;
		if (Math.max(p.val, q.val) < root.val)
			return lowestCommonAncestor(root.left, p, q);
		if (Math.min(p.val, q.val) > root.val)
			return lowestCommonAncestor(root.right, p, q);
		return root;
	}

	public ListNode reverseList(ListNode head) {
		if (head == null || head.next == null) {
			return head;
		}
		Stack<ListNode> listNodeStack = new Stack<ListNode>();
		while (head != null) {
			listNodeStack.push(head);
			head = head.next;
		}
		head = listNodeStack.pop();
		ListNode lastNode = head;
		while (!listNodeStack.empty()) {
			lastNode.next = listNodeStack.pop();
			lastNode = lastNode.next;
		}
		lastNode.next = null;
		return head;
	}

	/**
	 * Definition for singly-linked list.
	 */
	public class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
		}
	}

	class MyQueue {
		Stack<Integer> stackFront = new Stack<Integer>();
		Stack<Integer> stackBack = new Stack<Integer>();

		// Push element x to the back of queue.
		public void push(int x) {
			stackBack.push(x);
		}

		// Removes the element from in front of queue.
		public void pop() {
			if (!stackFront.isEmpty()) stackFront.pop();
			else {
				while (!stackBack.isEmpty()) stackFront.push(stackBack.pop());
				stackFront.pop();
			}
		}

		// Get the front element.
		public int peek() {
			if (!stackFront.isEmpty()) return stackFront.peek();
			else {
				while (!stackBack.isEmpty()) stackFront.push(stackBack.pop());
				return stackFront.peek();
			}
		}

		// Return whether the queue is empty.
		public boolean empty() {
			return stackBack.empty() && stackFront.empty();
		}
	}


	/**
	 * Definition for a binary tree node.
	 */
	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}

	class NumArray {
		int[] sums;

		public NumArray(int[] nums) {
			sums = new int[nums.length];
			System.arraycopy(nums, 0, sums, 0, nums.length);
			for (int i = 1; i < nums.length; i++)
				sums[i] = sums[i - 1] + nums[i];
		}

		public int sumRange(int i, int j) {
			if (i < 0 || j < 0 || i >= sums.length || j >= sums.length) return 0;
			return i == 0 ? sums[j] : sums[j] - sums[i - 1];
		}
	}


	class MyStack {
		Queue<Integer> queueFront = new LinkedList<Integer>();
		Queue<Integer> queueBack = new LinkedList<Integer>();

		// Push element x onto stack.
		public void push(int x) {
			queueFront.add(x);
		}

		// Removes the element on top of the stack.
		public void pop() {
			while (queueFront.size() > 1) {
				queueBack.add(queueFront.remove());
			}
			queueFront.remove();
			queueFront = queueBack;
			queueBack = new LinkedList<Integer>();

		}


		// Get the top element.
		public int top() {
			while (queueFront.size() > 1) {
				queueBack.add(queueFront.remove());
			}
			int top = queueFront.peek();
			queueBack.add(queueFront.remove());
			queueFront = queueBack;
			queueBack = new LinkedList<Integer>();
			return top;
		}

		// Return whether the stack is empty.
		public boolean empty() {
			return queueBack.isEmpty() && queueFront.isEmpty();
		}
	}

	public class Solution3 {
		int maxsum = Integer.MIN_VALUE;

		public int maxPathSum(TreeNode root) {
			maxsum(root);
			return maxsum;
		}

		private int maxsum(TreeNode root) {
			if (root == null) return 0;
			int left = maxsum(root.left);
			int right = maxsum(root.right);
			int curmax = root.val + Math.max(0, left) + Math.max(0, right);
			maxsum = Math.max(maxsum, curmax);
			return Math.max(root.val, Math.max(root.val + left, root.val + right));
		}
	}

	/**
	 * Definition for a binary tree node.
	 * public class TreeNode {
	 * int val;
	 * TreeNode left;
	 * TreeNode right;
	 * TreeNode(int x) { val = x; }
	 * }
	 */
	public class Solution8 {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> path = new ArrayList<Integer>();

		public List<List<Integer>> pathSum(TreeNode root, int sum) {
			dfs(root, sum);
			return result;
		}

		private void dfs(TreeNode root, int sum) {
			if (root == null) return;
			path.add(root.val);
			sum -= root.val;
			if (sum == 0 && root.left == null && root.right == null)
			/**一定要注意这里要new一个新的list,因为path在变化.**/
				result.add(new ArrayList<Integer>(path));
			path.addAll(path);
			dfs(root.left, sum);
			dfs(root.right, sum);
			/**注意这里的remove,因为计算新的path的时候,path要清空,所以回归的时候,把本次放进的元素移除**/
			path.remove(path.size() - 1);
		}
	}
}


