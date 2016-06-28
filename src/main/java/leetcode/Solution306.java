package leetcode;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 1/15/16 17:19
 */
public class Solution306 {
	public static boolean isAdditiveNumber(String num) {
		//枚举前两个数的位置，因为前两个数决定了num是否为Additive
		for(int i = 0; i < num.length(); i++) {
			for(int j = i + 1; j < num.length() - i - 1; j++) {
				String first = num.substring(0, i + 1);
				String second = num.substring(i + 1, j + 1);
				System.out.println("valid(j + 1, num, first, second)   =    "+"valid("+(j + 1)+","+
						num+","+
						first+","+second+")");
				if(valid(j + 1, num, first, second))
					return true;
			}
		}
		return false;
	}

	private static boolean valid(int start, String num, String first, String second) {
		if(start == num.length())
			return true;
		long f = Long.parseLong(first);
		long s = Long.parseLong(second);
		if(!Long.toString(f).equals(first) || !Long.toString(s).equals(second))
			return false;
		long sum = f + s;
		String sumS = Long.toString(sum);
		if(start + sumS.length() > num.length())
			return false;
		String third = num.substring(start, start + sumS.length());
		long t = Long.parseLong(third);
		if(!Long.toString(t).equals(third) || t != sum)
			return false;
		System.out.println(" valid(start + sumS.length(), num, second, third)   =    "+"valid("+
				(start + sumS.length())+","+
				num+","+
				second+","+third+")");
		return valid(start + sumS.length(), num, second, third);
	}

	public  static  void main(String[] args) {
		System.out.println(isAdditiveNumber("112358"));;
	}

}