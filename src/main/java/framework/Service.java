package framework;

/**
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 12/7/15 14:45
 */
interface Service
{
	public void sayHello();
}
class ServiceImp implements Service
{
	public void sayHello() {

		System.out.println("Hello World!");
	}
}
class Client
{
	public Client(Service s) {
		_service = s;
	}
	public void requestService() {
		_service.sayHello();
	}
	private Service _service;
}