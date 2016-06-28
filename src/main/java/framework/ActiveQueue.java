package framework;

import java.util.Stack;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/7/15 23:33
 */
//MethodRequest接口定义
interface MethodRequest {
	public void call();
}

//ActiveQueue定义，其实就是一个producer/consumer队列
class ActiveQueue {
	public ActiveQueue() {
		_queue = new Stack();
	}

	public synchronized void enqueue(MethodRequest mr) {
		System.out.println("入队");
		while (_queue.size() > QUEUE_SIZE) {
			try {
				System.out.println("队列满");
				wait();

			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		_queue.push(mr);
		notifyAll();
		System.out.println("Leave Queue");
	}

	public synchronized MethodRequest dequeue() {
		MethodRequest mr;

		while (_queue.empty()) {
			try {
				System.out.println("queue is empty");
				wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		mr = (MethodRequest) _queue.pop();
		notifyAll();
		System.out.println("pop "+_queue.size());
		System.out.println("");
		return mr;
	}

	private Stack _queue;
	private final static int QUEUE_SIZE = 3;
}

//ActiveObject的定义
class ActiveObject extends Thread {
	public ActiveObject() {
		_queue = new ActiveQueue();
		start();
	}

	public void enqueue(MethodRequest mr) {
		_queue.enqueue(mr);
	}

	public void run() {
		while (true) {
			MethodRequest mr = _queue.dequeue();
			mr.call();
		}
	}

	private ActiveQueue _queue;

	public static void main(String[] args) {
//		Service s = new ServiceImp();
		Service s = new ServiceProxy();
		/*for (int i=0;i<5;i++){
			System.out.println("第"+ i+" 次循环");
			s.sayHello();
		}

*/

		Client c = new Client(s);

		c.requestService();


	}
}

class SayHello2 implements MethodRequest {
	public SayHello2(Service s) {
		_service = s;
	}

	public void call() {
		_service.sayHello();
	}

	private Service _service;
}


class ServiceProxy implements Service {
	public ServiceProxy() {
		_service = new ServiceImp();
		_active_object = new ActiveObject();
	}

	public void sayHello() {
		MethodRequest mr = new SayHello2(_service);
		_active_object.enqueue(mr);
	}

	private Service _service;
	private ActiveObject _active_object;
}


