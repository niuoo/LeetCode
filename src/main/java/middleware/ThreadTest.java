package middleware;

import java.util.concurrent.CountDownLatch;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/7/15 10:16
 */
class ThreadTest implements Runnable{
	private static CountDownLatch startCdl; // 用于启动所有连接线程的闸门
	private static CountDownLatch doneCdl;// 所有连接工作都结束的控制器
	public ThreadTest(CountDownLatch startCdl,CountDownLatch doneCdl) {
		this.startCdl=startCdl;
		this.doneCdl=doneCdl;
	}

	public void run() {
		try {
			System.out.println("startCdl.waitbefore();"+startCdl.getCount());
			startCdl.await();
			System.out.println("startCdl.waitafter();" + startCdl.getCount());
			System.out.println(
					Thread.currentThread().getName() + " has been working!!!!");
			// 此处需要代码清单一的那些连接操作

			doneCdl.countDown();
			System.out.println("doneCdl.countDown();"+doneCdl.getCount());
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}


class CountDownLatchDemo1 {
	public static void main(String[] args) {
		CountDownLatch startCdl = new CountDownLatch(1);// 启动的闸门值为 1
		CountDownLatch doneCdl = new CountDownLatch(100);// 连接的总数为 100
		for(int i=1; i <=100; i ++){
			ThreadTest tt = new ThreadTest(startCdl,doneCdl);
			new Thread(tt,"Thread"+i).start();
		}
		// 记录所有连接线程的开始时间
		long start = System.nanoTime();
		// 所有线程虽然都已建立，并 start。但只有等闸门打开才都开始运行。
		startCdl.countDown();
		try {
			System.out.println("iiiiii");
			doneCdl.await();// 主线程等待所有连接结束
			// 连接达到峰值后，执行一些测试逻辑代码

		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// 记录所有连接线程的结束时间
		long end = System.nanoTime();
		System.out.println("The task takes time(ms): "+(end-start)/100000);

	}
}