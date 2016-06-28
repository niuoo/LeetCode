package multithreading;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/4/15 10:20
 */
public class Probe extends Thread{
public Probe(){}
public void run() {
//	while (true) {
		Thread[] x = new Thread[100];
		Thread.enumerate(x);
		System.out.println("Thread.enumerate(x) = "+Thread.enumerate(x));
		for (int i = 0; i < 100; i++) {
			Thread t = x[i];
			System.out.println("i = "+i);
			if (t == null)
			{
				break;}
			else
				System.out.println(t.getName() + "\t" + t.getPriority()
						+ "\t" + t.isAlive() + "\t" + t.isDaemon());
		}
//	}
}

	public static class Broken
	{   private  long x;
		Broken()
		{   new Thread()
		{   public void run()
			{   x = -1;
			}
		}.start();
			x = 0;
			System.out.println("x = "+x);
		}

	}
	public static void main(String [] args) throws InterruptedException {
		Thread thread2 =  new Probe();
		try {
			Thread.sleep(200);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		thread2.wait(33);
		System.out.println("thread main = " + thread2.getName());
		thread2.start();
	}

}