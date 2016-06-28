package leetcode;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 1/15/16 17:08
 */
public class Solution237 {
	public void deleteNode(ListNode node) {
		if (node == null) return;
		node.val = node.next.val;
		node.next = node.next.next;
	}
}