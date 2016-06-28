package StaticDemo;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/6/15 22:32
 */
public class StaticDemo {
	static int a = 42;
	static int b = 99;
	static void callme() {

		System.out.println("a = " + a);
	}
}

class StaticByName {

	public static void main(String args[]) {
		StaticDemo.callme();
		System.out.println("b = " + StaticDemo.b);
	}
}
