package multithreading;

import java.util.Date;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/3/15 16:11
 */
class TimePrinterRunnable
		implements Runnable {
	int pauseTime;
	String name;
	public TimePrinterRunnable(int x, String n) {
		pauseTime = x;
		name = n;
	}
	public void run() {
		while(true) {
			try {
				System.out.println(name + ":" + new
						Date(System.currentTimeMillis()));
				Thread.sleep(pauseTime);
			} catch(Exception e) {
				System.out.println(e);
			}
		}
	}
	static public void main(String args[]) {
		Thread t1 = new Thread (new TimePrinter(1000, "Fast Guy"));
		t1.start();
		Thread t2 = new Thread (new TimePrinter(3000, "Slow Guy"));
		t2.start();

	}
}
