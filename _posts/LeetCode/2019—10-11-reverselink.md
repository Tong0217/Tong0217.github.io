---
layout: post
title: 一次遍历实现链表反转
category: LeetCode
tags:
description:
---
### 思路
将链表分为左右两个链表，左链表初始为NULL，右链表为给定链表，每次见右链表的头移到做链表的头部，直到右链表为空。

    ListNode* reverseList(ListNode* head) {
        if(head==NULL || head->next==NULL) return head;
        ListNode *p=NULL;
        ListNode *q=head;
        while(q){
            head=q->next;
            q->next=p;
            p=q,q=head;
        }
        return p;
    }
