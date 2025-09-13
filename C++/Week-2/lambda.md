
# 🔹 Lambda Functions in C++

## What is a Lambda?
A lambda function is basically an anonymous function—a function without a name—that you can define right where you need it.  
Instead of writing a full function or functor, you can just drop a lambda inline.

## Syntax
```cpp
[capture_list](parameters) -> return_type {
    // function body
};
```

- **Capture list [ ]**: Decides what variables from the surrounding scope the lambda can use.  
- **Parameters ( )**: Like a normal function’s arguments.  
- **Return type ->**: Optional; usually deduced automatically.  
- **Body { }**: The actual code.  

## There are different ways to capture:
- `[ ]` – captures nothing.  
- `[x]` – captures variable `x` by value.  
- `[&x]` – captures `x` by reference.  
- `[=]` – captures all variables in scope by value.  
- `[&]` – captures all variables by reference.  
- `[=, &z]` – captures everything by value, but `z` by reference.  
- `[&, x]` – captures everything by reference, but `x` by value.  

## Key Features
- Inline, short-lived functions — no need for boilerplate.  
- Closures — lambdas can “remember” captured variables.  
- Useful with STL algorithms (`sort`, `for_each`, `remove_if`, etc.).  
- Can be stored in `std::function` for reuse.  

---

# Difference Between Inline Function and Lambda Function in C++

## 🔹 1. Inline Function

### Definition
An **inline function** is just a **normal named function** that the compiler *may* expand at the call site instead of doing a function call.  
It was introduced to avoid the overhead of function calls for very small functions.

### Example
```cpp
inline int add(int a, int b) {
    return a + b;
}

int main() {
    int x = add(2, 3);  // compiler may replace this with "2 + 3"
    return 0;
}
```

### Key Points
- **Has a name** (`add` here).
- Defined with the `inline` keyword (though the compiler may inline any small function, keyword or not).
- Exists mainly as a performance hint in older C++ (today compilers optimize aggressively anyway).
- Cannot “capture” local variables — it only works with its parameters.

---

## 🔹 2. Lambda Function

### Definition
A **lambda function** is an **anonymous function object** (no name by itself) that can capture variables from the surrounding scope.

### Example
```cpp
int factor = 2;
auto multiply = [factor](int x) {
    return x * factor;
};

int y = multiply(5); // uses captured 'factor'
```

### Key Points
- **Anonymous** (no identifier unless you assign to a variable).
- Can **capture variables** from outer scope (`[ ]` capture list).
- Compiles into a hidden class with `operator()`.
- Much more flexible than inline functions, especially for passing into STL algorithms.

---

## 🔹 3. Comparison Table

| Feature              | Inline Function                        | Lambda Function                         |
|-----------------------|----------------------------------------|------------------------------------------|
| Naming               | Has a name (e.g., `add`)              | Anonymous (unless assigned to a variable) |
| Scope Capture        | ❌ Cannot capture outer variables      | ✅ Can capture by value or reference      |
| Purpose              | Reduce function call overhead          | Define short, inline custom behavior      |
| Type                 | Regular function                      | Compiler-generated functor (object with `operator()`) |
| Parameters           | Normal function parameters            | Parameters + optional capture list        |
| STL Use              | Rarely used with STL algorithms       | Designed to work with STL (`sort`, `for_each`, etc.) |
| Modern Relevance     | Mostly obsolete (compilers inline automatically) | Widely used in modern C++                |

---

## 🔹 4. When to Use

- **Inline functions**:  
  Good for tiny, reusable, *named* utilities (though in practice, today just use `constexpr` or `static inline` helpers — the compiler will inline anyway).  

- **Lambda functions**:  
  Perfect for *one-off logic*, especially when passing behavior into algorithms, threading, event handling, or when you need closures (capturing local state).  

---

👉 In short:  
- Inline = “named shortcut for a function call.”  
- Lambda = “anonymous, flexible function object that can also close over local variables.”  
