package multithreading;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/3/15 17:19
 */
public class BSync {
	int totalThreads;
	int currentThreads;
	public BSync(int x) {
		totalThreads = x;
		currentThreads = 0;
	}
	public synchronized void waitForAll() {
		currentThreads++;
		if(currentThreads < totalThreads) {
			try {
				wait();
			} catch (Exception e) {}
		}
		else {
			currentThreads = 0;
			notifyAll();
		}
	}
}
