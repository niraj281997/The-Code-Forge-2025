
# Move Constructor, Rvalue References, and `constexpr` in C++

## 1. Move Constructor

A **move constructor** is a special constructor in C++ that transfers resources from one object to another instead of copying them.  
This is especially useful when working with dynamic memory or large data structures.

### Example:
```cpp
class Node {
    int* ptr;
public:
    Node(int val) {
        ptr = new int(val);
    }

    // Move constructor
    Node(Node&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr; // leave source in safe state
    }

    ~Node() { delete ptr; }
};
```

### Why use it?
- Avoids expensive deep copies.
- Transfers ownership instead of duplicating data.
- Critical for performance in STL containers (like `std::vector`, `std::string`).

---

## 2. Rvalue and Rvalue References

### What is an Rvalue?
- **Lvalue**: Has a name, an address, persists in memory.  
  Example: `int x = 5;` → `x` is an lvalue.
- **Rvalue**: Temporary object, does not have a persistent memory address.  
  Example: `x + 2` or `42`.

```cpp
int a = 10;
int b = a + 5; // (a + 5) is an rvalue
```

### What does `&&` mean?
- `&&` declares an **rvalue reference**.
- It can bind to temporaries (rvalues).

```cpp
int&& r = 5; // r takes ownership of temporary value 5
```

### Why rvalue references?
They enable move semantics (move constructors, move assignment), so we can reuse resources instead of copying.

---

## 3. `constexpr`

### What it means
`constexpr` tells the compiler to evaluate a function or variable **at compile time if possible**.

### Example with variable
```cpp
constexpr int size = 10;
int arr[size]; // array of size 10
```

### Example with function
```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int val = square(5); // evaluated at compile-time
```

If you call it with runtime values, it still works but executes at runtime.

### Key points
- Makes code faster by precomputing values at compile time.
- Used in template arguments, array sizes, and optimizations.
- Local variables inside a `constexpr` function don’t need to be `constexpr`, unless you want them to be usable as compile-time constants.

---

## Summary
- **Move constructor**: Transfers ownership of resources, avoids deep copies.  
- **Rvalue (`&&`)**: Represents temporaries, used for move semantics.  
- **`constexpr`**: Enables compile-time evaluation for constants and functions.

## More on Rvalues

### 1. Lvalue vs Rvalue Recap
- **Lvalue**: Something that has an identifiable location in memory (can appear on the left-hand side of `=`).
  ```cpp
  int x = 10;   // x is an lvalue
  ```

- **Rvalue**: A temporary object or literal that does not have a stable memory address (usually appears on the right-hand side).
  ```cpp
  int y = x + 5;   // (x + 5) is an rvalue
  int z = 42;      // 42 is an rvalue literal
  ```

### 2. Visualizing Lvalue and Rvalue

```
Memory:

  [ x ] ---> 10     (lvalue: has storage)
        +
        5           (rvalue: temporary, no name)
        =
  [ y ] ---> 15
```

Here, `x` and `y` are lvalues because they occupy storage in memory.  
`(x + 5)` is an rvalue because it exists only during evaluation.

---

### 3. Binding Rules

- **Lvalues can bind to references (&):**
  ```cpp
  int a = 10;
  int& ref = a;   // OK
  ```

- **Rvalues cannot bind to normal references, but they can bind to rvalue references (&&):**
  ```cpp
  int&& rref = 20;  // OK
  int& ref = 20;    // ERROR (20 is an rvalue)
  ```

---

### 4. Rvalue References in Functions

They are useful when writing functions that take advantage of temporaries.

```cpp
void process(const string& s) {
    cout << "Lvalue reference called
";
}

void process(string&& s) {
    cout << "Rvalue reference called
";
}

int main() {
    string str = "Hello";
    process(str);        // Lvalue version
    process("World");    // Rvalue version (temporary string)
}
```

This technique is called **function overloading with rvalue references** and is used in *move semantics* and *perfect forwarding*.

---

### 5. Practical Importance of Rvalues
- Avoid unnecessary copies (especially with large objects like vectors, strings).
- Enable move semantics in STL containers (`std::vector`, `std::string`, etc.).
- Foundation for **modern C++ features** like `std::move` and `std::forward`.

---

### 6. `std::move`

`std::move` is not actually moving anything. It just **casts an lvalue into an rvalue reference**, so that move semantics can be triggered.

```cpp
vector<int> v1 = {1,2,3};
vector<int> v2 = std::move(v1); // resources moved, v1 is now empty
```

Here, `v1` becomes an rvalue (thanks to `std::move`), so the move constructor of `vector` is invoked instead of copying.
