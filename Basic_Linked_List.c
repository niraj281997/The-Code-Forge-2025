//
//  Linked_list.c
//  C
//
//  Created by Niraj Gohel on 01/06/25.
//
#include "Linked_list.h"
#ifdef Linkedlist
void linked_list_main(void)
{
    LLNode *head = nullptr;
    operation op = DEFAULT;
    int counter =0;
    while (INFINITE) {
        printf("%s\n%s\n",Option_op,Choice_op);
        scanf("%d", &op);
        switch (op) {
            case ADD_FIRST:
                f_SLL_add_first(&head);
                break;
            case ADD_LAST:
                f_SLL_add_last(&head);
                break;
            case PRINT:
                f_SLL_check_null(&head,  f_SLL_print);
                break;
            case DELETE_FIRST:
                f_SLL_check_null(&head,f_SLL_delete_first);
                break;
            case REVERSE_LIST:
                f_SLL_check_null_with_return(&head,f_SLL_reverse);
                break;
            case FIND_MID:
                f_SLL_check_null_with_return(&head, f_SLL_mid_element);
                break;
            case CHECK_CIRCULAR:
                f_SLL_check_null(&head,f_SLL_check_circular);
                break;
            case CREATE_CIRCULAR:
                f_SLL_check_null(&head, f_SLL_create_circular);
                break;
            case EXIT:
                f_SLL_exit(&head);
                exit(0);
                break;
            default:
                if(counter)
                    exit(0);
                printf("%s",Not_op);
                counter++;
                break;
        }
       
    }
}


static void f_SLL_check_null(LLNode** head, void (*f_pointer)(LLNode**))
{
    if(*head==nullptr)
    {
        printf("%s\n",Not_op);
        return;
    }
    else
    {
        f_pointer(head);
        printf("\n");
    }
    return;
}

static void f_SLL_check_null_with_return(LLNode** head, LLNode* (*f_pointer)(LLNode**))
{
    if(*head==nullptr)
    {
        printf("%s\n",Not_op);
        return;
    }
    else
    {
        if(circular_flag)
        {
            printf("%s",CLtrue);
            return;
        }
        else{
            f_pointer(head);
            printf("\n");
        }
    }
    return;
}



static LLNode* f_SLL_reverse(LLNode **head)
{
    LLNode *curr = *head,*prev = nullptr,*next = nullptr;
    while(curr!=nullptr)
    {
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    *head = prev;
    return prev;
}

static LLNode* f_SLL_mid_element(LLNode **head)
{
    LLNode*turtle = *head, *rabbit =*head;
    while(rabbit!=nullptr && rabbit->next!=nullptr)
    {
        turtle = turtle->next->next;
        rabbit = rabbit->next;
    }
    return rabbit;
}


static void f_SLL_check_circular(LLNode** head)
{
    LLNode*rabbit = *head, *turtle = *head;
    circular_flag =1;
    while(rabbit!=nullptr)
    {
        rabbit = rabbit->next->next;
        turtle = turtle->next;
        if(rabbit==turtle)
        {
            printf("%s",CLtrue);
            printf("Want to break it ? 1 Yes 0 No");
            int flag = 0;
            scanf("%d",&flag);
            if(flag)
                f_SLL_remove_circular(head);
            return;
        }
    }
    printf("%s",CLfalse);
    return;
}

LLNode* f_create(void)
{
    LLNode* temp =nullptr;
    temp = (LLNode*)calloc(1, sizeof(LLNode));
    if(temp==nullptr)
    {
        printf("%s\n",Memory_leak);
        return nullptr;
    }
    printf("%s", Enter_data);
    scanf("%d",&(temp->num));
    
    return temp;
}

static void f_SLL_print(LLNode **head)
{
    if(*head==nullptr)
    {
        printf("\n");
        return;
    }
    else
    {
        printf("%d ->",(*head)->num);
        f_SLL_print(&((*head)->next));
    }
    
    return;
}
static void f_SLL_create_circular(LLNode **head)
{
    LLNode *temp = *head;
    while (temp->next!=nullptr) {
        temp=temp->next;
    }
    temp->next= *head;
    return;
}


static void f_SLL_add_first(LLNode ** head)
{
    if(*head==nullptr)
        *head=f_create();
    else
    {
        LLNode* temp = *head;
        *head = f_create();
        (*head)->next = temp;
    }
}

static void f_SLL_add_last(LLNode **head)
{
    if(*head==nullptr)
        *head=f_create();
    else
        f_SLL_add_last(&((*head)->next));
}

static void f_SLL_delete_first(LLNode **head)
{
    LLNode*temp = *head;
    *head = (*head)->next;
    free(temp);
    return;
}

static void f_SLL_delete_last(LLNode **head)
{
    if(!((*head)->next))
    {
        free(*head);
        *head=nullptr;
        return;
    }
    else
    {
        f_SLL_delete_last(&((*head)->next));
    }
    return;
}


static void f_SLL_exit(LLNode **head)
{
    if(*head==nullptr)
        return;
    else
    {
        f_SLL_exit((LLNode**)&((*head)->next));
        free(*head);
        *head=nullptr;
    }
}

static void f_SLL_remove_circular(LLNode **head)
{
    LLNode * fast = *head , *slow = *head;
    while (fast!=nullptr && fast->next!=nullptr) {
        fast= fast->next->next;
        slow = slow->next;
        if(fast==slow)
        {
            break;
        }
      if(fast!=slow)
          return;
        slow = *head;
        LLNode * prev = nullptr;
        while(slow!=fast)
        {
            prev = fast;
            slow = slow->next;
            fast = fast->next;
        }
        prev->next=nullptr;
    }
    circular_flag = 0;
    return;
    
}
#endif
