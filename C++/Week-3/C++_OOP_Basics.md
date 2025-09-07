# C++ Classes, Constructors, Destructors, and Member Functions

This document explains the core building blocks of **Object-Oriented Programming (OOP)** in C++.  
We’ll cover **classes, constructors, destructors, and member functions** with examples and theory.

---

## 1. Classes & Objects
- A **class** is a user-defined type that groups related data (**attributes**) and behavior (**functions**).
- An **object** is an instance of a class.
- Classes improve code organization, reusability, and encapsulation.

### Example
```cpp
class Car {
public:
    std::string brand;
    int speed;

    void drive() {
        std::cout << brand << " is driving at " << speed << " km/h\n";
    }
};

int main() {
    Car c1;              // object
    c1.brand = "Tesla";
    c1.speed = 100;
    c1.drive();
}
```

---

## 2. Constructors
Constructors are **special member functions** that run when an object is created.  
They initialize the object’s state.

### Types of Constructors

#### Default Constructor
```cpp
class Car {
public:
    Car() { std::cout << "Car created\n"; }
};
```
- Generated automatically by the compiler if you don’t define any constructor.  
- If you define **any constructor**, the default one won’t be provided unless you explicitly write:
  ```cpp
  Car() = default;
  ```

#### Parameterized Constructor
```cpp
Car(std::string b, int s) : brand(b), speed(s) {}
```
- Allows initialization with custom values.

#### Copy Constructor
```cpp
Car(const Car& other) {
    brand = other.brand;
    speed = other.speed;
}
```
- Creates a new object as a copy of an existing one.  
- Called when:
  - Passing objects by value  
  - Returning objects by value  
  - Explicitly copying (`Car c2 = c1;`)  

#### Move Constructor (C++11+)
```cpp
Car(Car&& other) noexcept {
    brand = std::move(other.brand);
    speed = other.speed;
}
```
- Transfers resources from a temporary object (**rvalue**).  
- Improves performance by avoiding unnecessary copies.  

### Why Constructors Matter
- Without proper constructors, objects may be left in an **uninitialized or invalid state**.  
- Choosing between **copy** and **move** semantics is crucial for performance in modern C++.  

---

## 3. Destructors
A **destructor** runs automatically when an object is destroyed (goes out of scope or is deleted).  
It is mainly used for **cleanup**: releasing memory, closing files, freeing handles, etc.

### Example
```cpp
class Car {
public:
    ~Car() {
        std::cout << "Car destroyed\n";
    }
};
```

### Key Points
- Only **one destructor** per class.  
- Cannot take arguments or return values.  
- If you expect polymorphic behavior (deleting through a base-class pointer), mark destructors as **virtual**:
```cpp
class Base {
    virtual ~Base() {}
};
```

---

## 4. Member Functions

### What Are Member Functions?
- Functions defined inside a class that describe its **behavior**.  
- They can access and modify the class’s data members.

Example:
```cpp
class Car {
public:
    std::string brand;
    int speed;

    void drive() {
        std::cout << brand << " is driving at " << speed << " km/h\n";
    }
};
```

### Types of Member Functions

#### Instance Member Functions
- Operate on specific objects.  
- Can access data members and other member functions.
```cpp
class Car {
public:
    void drive() {
        std::cout << "Car is driving\n";
    }
};
```

#### Static Member Functions
- Belong to the **class itself**, not an object.  
- Cannot access non-static members.  
- Called with `ClassName::functionName()`.
```cpp
class Car {
public:
    static void info() {
        std::cout << "Cars exist!\n";
    }
};
Car::info();
```

#### Const Member Functions
- Declared with `const` at the end.  
- Cannot modify data members.  
- Useful when working with `const` objects.
```cpp
class Car {
public:
    std::string brand;
    void show() const {
        std::cout << brand;
    }
};
```

#### Inline Member Functions
- Defined inside the class body.  
- Compiler may inline them to avoid call overhead.  
- Best for short, frequently used functions.
```cpp
class Car {
public:
    void horn() { std::cout << "Beep!\n"; }
};
```

---

## 5. Summary
- **Classes** group data and behavior.  
- **Constructors** initialize objects, with special types (default, parameterized, copy, move).  
- **Destructors** clean up resources when objects are destroyed.  
- **Member functions** define object behavior: instance, static, const, and inline.  

Together, these features form the foundation of **OOP in C++**.  
