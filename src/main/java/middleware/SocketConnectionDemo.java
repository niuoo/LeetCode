package middleware;

import java.io.BufferedOutputStream;
import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.util.HashMap;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/6/15 23:33
 */
class Utility { // 工具类
	public byte[] getHeadData(String... parms){

		return new byte[0];
	}
	public byte[] getBodyData(String... parms){

		return new byte[0];
	}
	public byte[] getFootData(String... parms){
		HashMap hashMap;

		return new byte[0];
	}
}
class SocketClient extends Utility implements Runnable{ // 实现 Runnable 接口
	String hostName;
	int port;
	public SocketClient(String hostName,int port) {
		this.hostName=hostName;
		this.port=port;

	}
	// 线程启动后的连接，并发送数据的
	public void run() {
		try {
			InetAddress inetAddress = InetAddress.getByName(hostName);
			Socket socket = new Socket(inetAddress,port);
			OutputStream os = socket.getOutputStream();
			BufferedOutputStream bos = new BufferedOutputStream(os);
			byte[] sendHeadData = getHeadData("aaa","bbb","ccc");
			byte[] sendBodyData = getBodyData("ddd","eee","fff");
			byte[] sendFootData = getFootData("ggg","hhh","iii");
			bos.write(sendHeadData);
			bos.write(sendBodyData);
			bos.write(sendFootData);
			bos.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

// 测试的主线程，演示测试的一些场景
public class SocketConnectionDemo {

	public int connectionNumb; // 连接数

	public String hostName;// 连接的 IMS Connect 的域名

	public int port;// 连接的端口

	public SocketConnectionDemo(int connectionNumb,String hostName, int port){
		this.connectionNumb=connectionNumb;
		this.hostName=hostName;
		this.port=port;
	}

	public static void main(String[] args) {
		int connectionNumb = 50;
		String hostName = "ec32181.vmec.svl.com";
		int port = 9999;
		SocketConnectionDemo scd = new SocketConnectionDemo(connectionNumb,hostName,port);

	}

	// 最简单的测试场景
	public void  testScenario1(){
		for(int i=1; i<=connectionNumb; i++){
			SocketClient sc = new SocketClient(hostName, port);
			Thread clientT = new Thread(sc,"Client"+i);
			clientT.start();
		}
		// 测试的具体逻辑

	}
}