package middleware;

import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.Exchanger;
import java.util.concurrent.TimeUnit;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/7/15 10:42
 */
// 假设建立连接的类是 Connection
class Connection {
	private String connName;

	private String ipAddress;

	public Connection(String connName, String ipAddString) {
		this.connName = connName;
		this.ipAddress = ipAddString;
	}


}
/*
。为了实现项目后期的连接缓存队列 TCPIPQ 的测试，Exchanger 可能会被使用。
   Exchanger 可以实现两组线程互相交换一些共享资源的功能。
 */
public class ConnectionQueueDemo {

	// 使用交换器实现连接器与释放连接器之间资源的共享
	private static Exchanger<LinkedList<Connection>> exconn =
			new Exchanger<LinkedList<Connection>>();

	// 连接器
	public class Connector implements Runnable {

		private LinkedList<Connection> connQueue;

		private String ipAddress;

		public Connector(LinkedList<Connection> connQueue, String ipAddress) {
			this.connQueue = connQueue;
			this.ipAddress = ipAddress;
		}

		public void run() {
			boolean flag = true;
			while (flag) {
				// 每次连接随机的 1~2 个连接。
				Random random = new Random();
				int connNumb =  2; // 得到随机的 1~2 个连接数
//				int connNumb = (random.nextInt()) % 2 + 1; // 得到随机的 1~2 个连接数

				if (connNumb > 1) {
					System.out.println("Connector creates 2 connection!");
				} else {
					System.out.println("Connector creates 1 connection!");
				}
				for (int i = 0; i < connNumb; i++) {
					Connection conn = new Connection("Connector", getIpAddress());
					connQueue.add(conn);
				}
				// 休息 1 秒
				try {
					TimeUnit.SECONDS.sleep(1);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
				try {
					// 交换给释放连接器，让释放连接器工作！
					connQueue = (LinkedList<Connection>) exconn.exchange(connQueue);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				if (connQueue.size() == 6) {
					System.out.println("The connection queue is full!! The programme is end!");
					// 当队列满时，可以加入一些测试逻辑代码

					flag = false;
					System.exit(0);
				} else {
					System.out.println("After Disconnector, the size of the queue is " + connQueue.size());
				}
			}


		}

		public String getIpAddress() {
			return ipAddress;
		}
	}

	// 释放连接器
	public class Disconnector implements Runnable {
		private LinkedList<Connection> connQueue;

		public Disconnector(LinkedList<Connection> connQueue) {
			this.connQueue = connQueue;
		}

		public void run() {
			boolean flag = true;
			while (flag) {
				System.out.println("Disconnector disconnects 1 connection!");
				if (!connQueue.isEmpty())
					connQueue.remove(0);
				// 休息 1 秒
				try {
					TimeUnit.SECONDS.sleep(1);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
				try {
					// 交换给连接器，让连接器工作！
					connQueue = (LinkedList<Connection>) exconn.exchange(connQueue);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				if (connQueue.size() == 0) {
					System.out.println("There is no connection in the queue!");
				} else {
					System.out.println(
							"After Connector, the size of the queue is " + connQueue.size());
				}
			}
		}
	}
//在 main 函数里是具体的 Demo 实现。新建了连接器和释放连接器两个线程，它们共享一个连接缓存队列。
// 由于，连接器每次随机的连接的连接数要大于释放连机器释放的连接数，所以最后，连接队列会满。
	public static void main(String[] args) {
		LinkedList<Connection> connQueue = new LinkedList<Connection>();
		ConnectionQueueDemo connectionQueueDemo = new ConnectionQueueDemo();
		new Thread(connectionQueueDemo.new Connector(connQueue, "192.168.1.1")).start();
		new Thread(connectionQueueDemo.new Disconnector(connQueue)).start();
	}
}
