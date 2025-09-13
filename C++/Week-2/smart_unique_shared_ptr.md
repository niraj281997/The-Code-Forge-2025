# 🔹 Smart Pointers in C++

## 0. Why Smart Pointers Were Introduced

Before C++11, programmers had to manage memory manually with `new` and `delete`.  
This caused several **common problems**:

1. **Memory Leaks** – Forgetting to call `delete` after `new` meant memory was never released.
   ```cpp
   int* p = new int(10);
   // forgot delete -> leak!
   ```

2. **Dangling Pointers** – Accessing memory after it was freed.
   ```cpp
   int* p = new int(5);
   delete p;
   cout << *p; // ❌ undefined behavior
   ```

3. **Double Deletes** – Accidentally deleting the same pointer twice.
   ```cpp
   int* p = new int(7);
   delete p;
   delete p;  // ❌ crash / undefined behavior
   ```

4. **Exception Safety Issues** – If an exception was thrown before `delete`, memory leaks occurred.

5. **Complex Ownership** – In large codebases, it was unclear who owned the memory and who should free it.

👉 To solve these, C++11 introduced **Smart Pointers** that use RAII (Resource Acquisition Is Initialization).  
They automatically manage memory and release resources when no longer needed.


---

## 1. `std::unique_ptr`

### What is it?
- A smart pointer that owns a resource **exclusively**.
- Cannot be copied, only **moved**.
- Automatically deletes the managed object when it goes out of scope.
- Very lightweight, minimal overhead.

### Example:
```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    auto p1 = make_unique<int>(42);
    cout << *p1 << endl;

    // auto p2 = p1;  // ❌ Error: cannot copy
    auto p2 = move(p1);  // ✅ Ownership transferred
    cout << *p2 << endl;

    return 0;
}
```

### Key Points:
- Prefer `make_unique` instead of `new`.
- Best choice when you want single, exclusive ownership.
- Eliminates manual `delete` calls.


---

## 2. `std::shared_ptr`

### What is it?
- A smart pointer that allows **multiple owners** of the same resource.
- Uses **reference counting** internally.
- The resource is destroyed only when the last `shared_ptr` goes out of scope.

### Example:
```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    auto sp1 = make_shared<int>(100);
    auto sp2 = sp1;  // both share ownership

    cout << *sp1 << " " << *sp2 << endl;
    cout << "Use count: " << sp1.use_count() << endl; // 2

    return 0;
}
```

### Key Points:
- `make_shared` is preferred (allocates object + control block in one step).
- Good for shared ownership in graphs, trees, or passing objects around.
- Overhead: needs reference counting (heavier than `unique_ptr`).


---

## 3. Comparison

| Feature        | `unique_ptr`        | `shared_ptr`         |
|----------------|---------------------|----------------------|
| Ownership      | Exclusive           | Shared               |
| Copyable       | ❌ No               | ✅ Yes                |
| Overhead       | Minimal             | Reference counter    |
| Best Use Case  | Default choice      | Shared ownership     |

---

## 4. Best Practices
- Use `unique_ptr` by default.  
- Use `shared_ptr` **only when necessary** (shared ownership).  
- Avoid raw `new`/`delete` when possible.  
