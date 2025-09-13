# 🧠 Virtual Functions in Object-Oriented Programming

## 📘 What is a Virtual Function?

A **virtual function** is a member function in a base class that you expect to override in derived classes. It enables **runtime polymorphism**, allowing the program to decide at runtime which function to invoke based on the actual object type.
##🔍 Why Runtime Binding Is Required
When you use a base class pointer or reference to refer to a derived class object, the compiler doesn’t know at compile time which version of the function to call. Without runtime binding, it would default to the base class version — even if the actual object is of the derived type.

### 🧠 Example Without Virtual Function (Static Binding)
```cpp
class Base {
public:
    void show() {
        cout << "Base class show" << endl;
    }
};

class Derived : public Base {
public:
    void show() {
        cout << "Derived class show" << endl;
    }
};

int main() {
    Base* ptr;
    Derived d;
    ptr = &d;
    ptr->show(); // Output: Base class show ❌
}

```
### ✅ Example With Virtual Function (Runtime Binding)
```cpp
class Base {
public:
    virtual void show() {
        cout << "Base class show" << endl;
    }
};

class Derived : public Base {
public:
    void show() override {
        cout << "Derived class show" << endl;
    }
};

int main() {
    Base* ptr;
    Derived d;
    ptr = &d;
    ptr->show(); // Output: Derived class show ✅
}

```
# 🧩 Related Concepts

## 🔗 Inheritance
Inheritance allows a class (child) to acquire properties and behaviors from another class (parent).

Diagram Idea:A base class with two derived classes branching from it.


## 🔄 Function Overriding

Function overriding occurs when a derived class provides its own implementation of a method already defined in its base class.

## 🎭 Polymorphism

Polymorphism means “many forms.” In OOP, it allows objects of different types to be treated through a common interface. Virtual functions are the key to runtime polymorphism.

## 🧪 Pure Virtual Functions & Abstract Classes

A pure virtual function is declared by assigning = 0 in the base class. It has no implementation in the base class and must be overridden in derived classes.
```
class AbstractBase {
public:
    virtual void show() = 0; // Pure virtual function
};
```
Any class containing a pure virtual function becomes an abstract class and cannot be instantiated.

## 🖼️ Visualizing Virtual Functions

Diagram Idea:A base class with a virtual method speak(), and two derived classes overriding it.

[(Insert image: virtual_function_diagram.png)](https://bing.com/th/id/BCEI.81240555-4217-4e14-8977-98a866b7853a.png)<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/ce1d0103-3cc9-4f57-87cb-9451ce00b828" />


This shows how the correct method is chosen at runtime depending on the actual object type, not the pointer type.

## ✅ Summary Table

| Concept              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Virtual Function      | Enables runtime method dispatch                                             |
| Function Overriding   | Redefines base class method in derived class                                |
| Inheritance           | Allows one class to inherit from another                                    |
| Polymorphism          | Treats different objects through a common interface                         |
| Pure Virtual Function | Declared with `= 0`, forces derived class to implement                      |
| Abstract Class        | Contains at least one pure virtual function, cannot be instantiated         |

## 📎 Notes

Use the virtual keyword in the base class to enable polymorphism.

Use override in the derived class to make your intent explicit.

Abstract classes are great for defining interfaces.

Virtual functions are resolved at runtime, not compile time.
