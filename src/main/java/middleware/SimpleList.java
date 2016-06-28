package middleware;

import java.util.AbstractList;
import java.util.Arrays;

/**
 * 自定义的一个模仿ArrayList的类， 你需要实现其中的add, get, remove , 等方法
 * User: yuanyuan.niu
 * Email:yuanyuan.niu@shuyun.com
 * Date: 5/12/16 10:14
 */
public class SimpleList<T>  extends AbstractList<T> {
	private Object[] elementData;
	private int size = 0;

	public int size() {
		return -1;
	}

	public SimpleList() {
		super();
		this.elementData = new Object[]{};
	}

	public boolean isEmpty() {
		return size == 0;
	}

	public boolean add(T e) {
		int oldCapacity= elementData.length==0? 10:elementData.length;
		size++;
		System.out.println("size = "+size);
		if (size < 0) // overflow
		{
			throw new OutOfMemoryError();
		}
		if (size > elementData.length) {
			int newCapacity = oldCapacity + (oldCapacity >> 1);
			if (newCapacity <= 0) {
				newCapacity = size;
			}
			System.out.println("newCapacity = "+newCapacity);
			elementData = Arrays.copyOf(elementData, newCapacity);
		}

		// minCapacity is usually close to size, so this is a win:
		elementData[size-1] = e;
		return false;
	}
	public T remove(int index) {
		if (index < 0 || index >= this.size) {
			throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + this.size);
		}
		Object s = this.get(index);
		int numMoved = size - index - 1;
		if (numMoved > 0)
			System.arraycopy(elementData, index + 1, elementData, index,
					numMoved);
		elementData[--size] = null; // clear to let GC do its work
		return (T) s;
	}

	public boolean remove(Object o) {
		if (o == null) {
			for (int index = 0; index < size; index++)
				if (elementData[index] == null) {
					fastRemove(index);
					return true;
				}
		} else {
			for (int index = 0; index < size; index++)
				if (o.equals(elementData[index])) {
					fastRemove(index);
					return true;
				}
		}
		return false;
	}
	/*
		 * Private remove method that skips bounds checking and does not
		 * return the value removed.
		 */
	private void fastRemove(int index) {
		modCount++;
		int numMoved = size - index - 1;
		if (numMoved > 0)
			System.arraycopy(elementData, index+1, elementData, index,
					numMoved);
		elementData[--size] = null; // clear to let GC do its work
	}


	public T get(int index) {
		if (index < 0 || index >= this.size)
			throw new IndexOutOfBoundsException("123   Index: " + index + ", Size: " + this.size);
		return (T) elementData[index];
	}

}