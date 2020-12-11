/*
rgbd-tracker
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library.
*/

#if !defined(__LINKED_LIST_H__)
#define __LINKED_LIST_H
#include <stdio.h>
/*
LinkedList implementation
	* Simple forward linked list to avoid stl
author: ttykkala
*/

class LinkedNode {
public:
	LinkedNode(void *obj) {object = obj; next = NULL;}
	~LinkedNode() {};
	LinkedNode *next;
	void *object;
};

class LinkedList {
public:
	LinkedList() {first = NULL; last = NULL; cursor=NULL;}
	~LinkedList() {deleteNodes();}
	void *firstNode()
	{
		if (first == NULL) return NULL;
		cursor = first;
		return first->object;
	}
	void *nextNode()
	{
		if (cursor == NULL) return NULL;
		if (cursor->next == NULL) return NULL;
		cursor = cursor->next;
		return cursor->object;
	}
	void addElement(void *obj) {
		if (first == NULL)
		{
			first = new LinkedNode(obj);
			last = first;
			cursor = first;
		} else {
			LinkedNode *new_node = new LinkedNode(obj);
			last->next = new_node;
			last = new_node;
		}
	}
	void deleteNodes()
	{
		while (first)
		{
			LinkedNode *dead_node = first;
			first=first->next;
			delete dead_node;
		}
	}
private:
	LinkedNode *first;  // first pointer of the list
	LinkedNode *last;   // last pointer for adding new elements
	LinkedNode *cursor; // temp pointer for looping
};

#endif
