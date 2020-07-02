---
layout: post
title: 两个链表的首个公共节点
category: LeetCode
tags: Markdown
description:
---
## 描述



## 思路

首先分别遍历两个链表得到两个链表的长度。    
然后两个指针分别指向链表头节点，指向长链表的指针向前移动n位，使长链表剩余长度和端链表相同，然后同时移动两个指针，两个指针重合的节点就是首个公共节点。
